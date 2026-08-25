"""Extract one shared, frozen DINOv2 feature cache for the Stage-0 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F


TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "handover_block",
    "lift_pot",
    "move_can_pot",
    "place_container_plate",
]
CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
BACKBONE = "vit_small_patch14_dinov2.lvd142m"


def video_features(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    width, height = 640, 480
    frame_bytes = width * height * 3
    process = subprocess.Popen(
        [
            "/usr/bin/ffmpeg",
            "-v",
            "error",
            "-threads",
            "4",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    outputs: list[np.ndarray] = []
    frames: list[np.ndarray] = []

    def flush() -> None:
        if not frames:
            return
        array = np.stack(frames, axis=0)
        tensor = torch.from_numpy(array).to(device=device, non_blocking=True)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
        tensor = F.interpolate(
            tensor, size=(224, 224), mode="bicubic", align_corners=False
        )
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tensor = (tensor - mean) / std
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            feature = model(tensor)
        outputs.append(feature.float().cpu().numpy().astype(np.float16))
        frames.clear()

    while True:
        chunks = bytearray()
        while len(chunks) < frame_bytes:
            chunk = process.stdout.read(frame_bytes - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        if not chunks:
            break
        if len(chunks) != frame_bytes:
            process.kill()
            raise RuntimeError(
                f"Truncated RGB frame from {path}: {len(chunks)}/{frame_bytes} bytes"
            )
        frame = np.frombuffer(chunks, dtype=np.uint8).reshape(height, width, 3)
        frames.append(frame)
        if len(frames) >= batch_size:
            flush()
    flush()
    return_code = process.wait()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if return_code != 0:
        raise RuntimeError(f"FFmpeg failed for {path}: {stderr.strip()}")
    if not outputs:
        raise RuntimeError(f"No frames decoded from: {path}")
    return np.concatenate(outputs, axis=0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    data_root = Path("/home/user/etsf_stage0/data/robotwin_clean")
    output_root = Path("/home/user/etsf_stage0/stage0/features")
    output_root.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for shared DINOv2 feature extraction")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)

    model = timm.create_model(
        BACKBONE, pretrained=True, num_classes=0, img_size=224
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rng = random.Random(20260825)
    manifest: dict[str, object] = {
        "backbone": BACKBONE,
        "backbone_output_dim_per_view": int(model.num_features),
        "views": CAMERAS,
        "image_size": [224, 224],
        "preprocessing": "RGB, bicubic 224x224, ImageNet mean/std",
        "frozen": True,
        "split_seed": 20260825,
        "tasks": {},
    }

    started = time.time()
    processed = 0
    for task in TASKS:
        episodes = sorted((data_root / task / "data/chunk-000").glob("episode_*.parquet"))
        episode_ids = [int(path.stem.split("_")[-1]) for path in episodes]
        shuffled = episode_ids.copy()
        rng.shuffle(shuffled)
        eval_ids = sorted(shuffled[:10])
        train_ids = sorted(shuffled[10:])
        manifest["tasks"][task] = {
            "episode_count": len(episode_ids),
            "train_episode_ids": train_ids,
            "eval_episode_ids": eval_ids,
        }

        task_output = output_root / task
        task_output.mkdir(parents=True, exist_ok=True)
        for episode_id in episode_ids:
            output_path = task_output / f"episode_{episode_id:06d}.npz"
            if output_path.exists():
                with np.load(output_path) as cached:
                    if cached["features"].ndim == 2:
                        processed += 1
                        continue

            view_features = []
            for camera in CAMERAS:
                video_path = (
                    data_root
                    / task
                    / "videos/chunk-000"
                    / camera
                    / f"episode_{episode_id:06d}.mp4"
                )
                view_features.append(
                    video_features(video_path, model, device, args.batch_size)
                )
            lengths = {feature.shape[0] for feature in view_features}
            if len(lengths) != 1:
                raise RuntimeError(
                    f"View length mismatch for {task}/{episode_id}: {lengths}"
                )
            features = np.concatenate(view_features, axis=1)
            np.savez_compressed(output_path, features=features)
            processed += 1
            print(
                f"FEATURE_PROGRESS={processed}/300 task={task} episode={episode_id} "
                f"frames={features.shape[0]} dim={features.shape[1]} "
                f"elapsed_s={time.time() - started:.1f}",
                flush=True,
            )

    weight_candidates = list(
        Path("/home/user/.cache/huggingface/hub").glob(
            "models--timm--vit_small_patch14_dinov2.lvd142m/**/model.safetensors"
        )
    )
    if weight_candidates:
        manifest["backbone_weight_path"] = str(weight_candidates[0])
        manifest["backbone_weight_sha256"] = sha256_file(weight_candidates[0])
    manifest["feature_extraction_seconds"] = time.time() - started
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("FEATURE_EXTRACTION_COMPLETE=" + json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
