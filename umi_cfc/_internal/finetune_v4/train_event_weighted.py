#!/usr/bin/env python3
"""隔离的逐帧 event-weight SmolVLA 训练入口。

本文件不修改 LeRobot 源码。它先消费 ``--event-sidecar``，再在当前 Python
进程内替换训练脚本实际延迟导入的 sample-weighter 工厂，并将加权损失固定为
``mean(event_weight * per_sample_flow_loss)``。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


MIN_WEIGHT = 0.9
MAX_WEIGHT = 1.1


class EventWeightError(ValueError):
    pass


def _consume_wrapper_args(argv: Sequence[str]) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--event-sidecar", type=Path, required=True)
    ns, remainder = parser.parse_known_args(list(argv))
    if any(arg == "--event-sidecar" or arg.startswith("--event-sidecar=") for arg in remainder):
        raise EventWeightError("--event-sidecar 只能提供一次")
    return ns.event_sidecar.expanduser().resolve(), remainder


def load_event_weights(path: Path) -> "Any":
    """读取严格的 index -> event_weight 表，返回按 index 排列的 float32 数组。"""
    import numpy as np
    import pyarrow.parquet as pq

    if not path.is_file():
        raise EventWeightError(f"event sidecar 不存在: {path}")
    table = pq.read_table(path, columns=["index", "event_weight"])
    if table.num_rows == 0:
        raise EventWeightError("event sidecar 为空")
    indices = table.column("index").to_numpy(zero_copy_only=False)
    weights = table.column("event_weight").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    if indices.ndim != 1 or weights.ndim != 1 or len(indices) != len(weights):
        raise EventWeightError("index/event_weight 必须是一维且等长")
    if not np.issubdtype(indices.dtype, np.integer):
        raise EventWeightError("index 必须是整数")
    expected = np.arange(len(indices), dtype=indices.dtype)
    if not np.array_equal(indices, expected):
        raise EventWeightError("index 必须唯一、完整并严格覆盖 0..N-1")
    if not np.isfinite(weights).all():
        raise EventWeightError("event_weight 含 NaN/Inf")
    if weights.min() < np.float32(MIN_WEIGHT) or weights.max() > np.float32(MAX_WEIGHT):
        raise EventWeightError(f"event_weight 必须位于 [{MIN_WEIGHT}, {MAX_WEIGHT}]")
    if float(weights.max()) == float(weights.min()):
        raise EventWeightError("event_weight 必须有变异，不能全部相同")
    return weights


@dataclass
class EventSidecarWeighter:
    weights: Any
    device: Any
    source: Path

    def compute_batch_weights(self, batch: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        import torch

        if "index" not in batch:
            raise EventWeightError("训练 batch 缺少全局 index")
        indices = torch.as_tensor(batch["index"], dtype=torch.long)
        if indices.ndim != 1:
            raise EventWeightError("batch index 必须是一维")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= len(self.weights)):
            raise EventWeightError("batch index 超出 event sidecar 覆盖范围")
        # 只做查表和 device 转移；严禁按 batch 重新归一化。
        selected = torch.as_tensor(self.weights[indices.cpu().numpy()], dtype=torch.float32, device=self.device)
        return selected, {
            "type": "event_sidecar",
            "mean_weight": float(selected.detach().mean().cpu()) if selected.numel() else math.nan,
            "min_weight": float(selected.detach().min().cpu()) if selected.numel() else math.nan,
            "max_weight": float(selected.detach().max().cpu()) if selected.numel() else math.nan,
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "event_sidecar",
            "source": str(self.source),
            "num_frames": len(self.weights),
            "global_min": float(self.weights.min()),
            "global_max": float(self.weights.max()),
            "global_mean": float(self.weights.mean()),
        }


def event_weighted_mean(per_sample_loss: Any, weights: Any) -> Any:
    """L = mean(w_i * loss_i)，不除以 batch 权重和。"""
    if per_sample_loss.ndim != 1 or weights.ndim != 1 or per_sample_loss.shape != weights.shape:
        raise EventWeightError("per-sample loss 与 event weight 必须是一维同形")
    return (per_sample_loss * weights).mean()


def _make_update_policy(torch: Any, train_module: Any):
    def update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator,
                      lr_scheduler=None, lock=None, sample_weighter=None):
        started = time.perf_counter()
        policy.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sample_weights = weight_stats = None
        if sample_weighter is not None:
            sample_weights, weight_stats = sample_weighter.compute_batch_weights(batch)
        with accelerator.accumulate(policy):
            with accelerator.autocast():
                if sample_weights is None:
                    loss, output_dict = policy(batch)
                else:
                    per_sample_loss, output_dict = policy(batch, reduction="none")
                    loss = event_weighted_mean(per_sample_loss, sample_weights)
                    output_dict = {} if output_dict is None else output_dict
                    for key, value in weight_stats.items():
                        output_dict[f"sample_weight_{key}"] = value
            accelerator.backward(loss)
            grad_norm = None
            if accelerator.sync_gradients and grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            with lock if lock is not None else nullcontext():
                optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
        if accelerator.sync_gradients and train_module.has_method(
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"
        ):
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
        train_metrics.loss = loss.item()
        if grad_norm is not None:
            train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]
        train_metrics.update_s = time.perf_counter() - started
        if torch.cuda.is_available():
            train_metrics.gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        if output_dict:
            train_metrics.update_metrics(output_dict)
        return train_metrics, output_dict
    return update_policy


def main(argv: Sequence[str] | None = None) -> None:
    sidecar, forwarded = _consume_wrapper_args(sys.argv[1:] if argv is None else argv)
    weights = load_event_weights(sidecar)

    # cfg.sample_weighting 非空才进入上游 loss 分支；uniform 仅作为配置开关，
    # 工厂在本进程内被替换，日志/统计均明确标记 event_sidecar，绝不冒充 RABC。
    if not any(arg.startswith("--sample_weighting.") for arg in forwarded):
        forwarded.append("--sample_weighting.type=uniform")
    import torch
    import lerobot.utils.sample_weighting as weighting_module
    from lerobot.scripts import lerobot_train as train_module

    def factory(config, policy, device, dataset_root=None, dataset_repo_id=None):
        del policy, dataset_root, dataset_repo_id
        if config is None or config.type != "uniform":
            raise EventWeightError("wrapper 要求 sample_weighting.type=uniform 作为启用开关")
        return EventSidecarWeighter(weights=weights, device=device, source=sidecar)

    weighting_module.make_sample_weighter = factory
    train_module.update_policy = _make_update_policy(torch, train_module)
    sys.argv = [sys.argv[0], *forwarded]
    print(f"EVENT_SIDECAR_WEIGHTING enabled source={sidecar} frames={len(weights)} "
          f"range=[{weights.min():.6f},{weights.max():.6f}] loss=mean(w*flow_loss)", file=sys.stderr)
    train_module.main()


if __name__ == "__main__":
    main()
