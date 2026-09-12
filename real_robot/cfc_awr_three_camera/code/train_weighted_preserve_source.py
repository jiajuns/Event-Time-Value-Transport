#!/usr/bin/env python3
"""AWR SmolVLA training with an audited source and byte-exact processor preservation."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import shutil
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve_checkpoint_stats(kwargs):
    result = copy.deepcopy(kwargs)
    result.pop("dataset_stats", None)
    for group, step in (("preprocessor_overrides", "normalizer_processor"),
                        ("postprocessor_overrides", "unnormalizer_processor")):
        if group in result and step in result[group]:
            result[group][step].pop("stats", None)
    return result


def forwarded_path(arguments, option):
    prefix = option + "="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return Path(argument[len(prefix):]).resolve()
        if argument == option and index + 1 < len(arguments):
            return Path(arguments[index + 1]).resolve()
    raise ValueError(f"missing required forwarded option: {option}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    options, forwarded = parser.parse_known_args()
    source = options.source.resolve()
    if source.parent.name != "050000":
        raise ValueError("source must be checkpoint 050000")
    if sha256(source / "model.safetensors") != options.expected_source_sha:
        raise ValueError("source actor SHA256 mismatch")
    output_dir = forwarded_path(forwarded, "--output_dir")
    sys.argv = [sys.argv[0], *forwarded]

    import torch
    from safetensors.torch import load_file
    from lerobot.scripts import lerobot_train as training
    from finetune_v4 import train_event_weighted as weighted

    original_loss = weighted.event_weighted_mean

    def checked_loss(losses, weights):
        loss = original_loss(losses, weights)
        if not torch.isfinite(loss) or float(loss.detach()) > 100.0:
            raise ValueError("non-finite or implausibly large normalized flow loss")
        return loss

    weighted.event_weighted_mean = checked_loss
    original_processors = training.make_pre_post_processors

    def fixed_processors(*, policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
        if Path(pretrained_path).resolve() != source:
            raise ValueError("training pretrained_path differs from audited source")
        pre, post = original_processors(
            policy_cfg=policy_cfg,
            pretrained_path=pretrained_path,
            pretrained_revision=pretrained_revision,
            **preserve_checkpoint_stats(kwargs),
        )
        pairs = (
            (pre, "policy_preprocessor_step_5_normalizer_processor.safetensors"),
            (post, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"),
        )
        for pipeline, filename in pairs:
            expected = load_file(str(source / filename))
            states = [step.state_dict() for step in pipeline.steps if hasattr(step, "_tensor_stats")]
            if len(states) != 1 or states[0].keys() != expected.keys():
                raise ValueError("processor statistics schema changed")
            if any(not torch.equal(states[0][key].cpu(), value) for key, value in expected.items()):
                raise ValueError("source checkpoint normalization was overridden")
        print("AUDITED_NORMALIZERS: source pre/postprocessor tensors preserved", flush=True)
        return pre, post

    training.make_pre_post_processors = fixed_processors
    weighted.main()

    processor_files = (
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    checkpoints = sorted((output_dir / "checkpoints").glob("*/pretrained_model"))
    if not checkpoints:
        raise ValueError("training produced no deployable checkpoints")
    for checkpoint in checkpoints:
        for filename in processor_files:
            shutil.copyfile(source / filename, checkpoint / filename)
            if sha256(checkpoint / filename) != sha256(source / filename):
                raise ValueError(f"failed to restore source processor: {checkpoint / filename}")
    print(f"RESTORED_SOURCE_PROCESSORS: {len(checkpoints)} checkpoints are byte-exact", flush=True)


if __name__ == "__main__":
    main()
