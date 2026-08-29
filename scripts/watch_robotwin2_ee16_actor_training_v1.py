#!/usr/bin/env python3
"""Convert the full public EE16 dataset, validate it, then train SmolVLA.

This launcher is intended for the remote 4090 host.  It does not read critic
labels.  Policy training starts only after all 2,750 public expert episodes are
materialized with the reviewed 16-D state/action contract.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DATASET_ID = "robotwin2_move_can_pot_5emb_ee16_full2750"
DATASET_ROOT = Path("/home/user/.cache/huggingface/lerobot") / DATASET_ID
ROBOTWIN_ROOT = Path("/home/user/etsf_stage0/RoboTwin")
LEROBOT_ROOT = Path("/home/user/etsf_stage0/lerobot")
PYTHON = Path("/home/user/etsf_stage0/.venv_lerobot_smolvla_v044/bin/python")
TRAIN = Path("/home/user/etsf_stage0/.venv_lerobot_smolvla_v044/bin/lerobot-train")
BASE_MODEL = Path("/home/user/etsf_smolvla_models/smolvla_base_c83c3163")
VLM_METADATA = Path("/home/user/etsf_stage0/offline_assets/smolvlm2_500m_metadata")
OFFLINE_BASE_MODEL = Path(
    "/home/user/etsf_smolvla_models/"
    "smolvla_base_c83c3163_offline_smolvlm_metadata_v1"
)
OUTPUT_ROOT = Path(
    "/home/user/etsf_smolvla_models/"
    "smolvla_robotwin2_move_can_pot_5emb_ee16_full2750_20k_20260830"
)
CONVERT_LOG = Path(
    "/home/user/robotwin2_move_can_pot_5emb_ee16_full2750_"
    "lerobot_convert_20260830.log"
)
TRAIN_LOG = Path(
    "/home/user/smolvla_robotwin2_move_can_pot_5emb_ee16_"
    "full2750_20k_20260830.train.log"
)
STATE_PATH = Path(
    "/home/user/smolvla_robotwin2_move_can_pot_5emb_ee16_"
    "full2750_20k_20260830.watcher_state.json"
)
TRAIN_PID_PATH = Path(
    "/home/user/smolvla_robotwin2_move_can_pot_5emb_ee16_"
    "full2750_20k_20260830.train.pid"
)

INPUT_FEATURES = {
    "observation.state": {"type": "STATE", "shape": [16]},
    "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
}
OUTPUT_FEATURES = {"action": {"type": "ACTION", "shape": [16]}}
RENAME_MAP = {
    "observation.images.cam_high": "observation.images.camera1",
    "observation.images.cam_left_wrist": "observation.images.camera2",
    "observation.images.cam_right_wrist": "observation.images.camera3",
}


def compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def prepare_offline_base_model() -> None:
    """Create a read-only checkpoint view whose tokenizer is local.

    Model tensors are hard-linked to the reviewed base checkpoint, so this
    neither downloads nor duplicates the 900 MB weight file.  Only the small
    processor JSON is materialized with its tokenizer path rewritten.
    """

    processor_name = "policy_preprocessor.json"

    def validate() -> None:
        value = json.loads((OFFLINE_BASE_MODEL / processor_name).read_text())
        tokenizer_steps = [
            step
            for step in value.get("steps", [])
            if step.get("registry_name") == "tokenizer_processor"
        ]
        if (
            len(tokenizer_steps) != 1
            or tokenizer_steps[0].get("config", {}).get("tokenizer_name")
            != str(VLM_METADATA)
            or not os.path.samefile(
                OFFLINE_BASE_MODEL / "model.safetensors",
                BASE_MODEL / "model.safetensors",
            )
        ):
            raise RuntimeError("offline SmolVLA checkpoint view changed")

    if OFFLINE_BASE_MODEL.exists():
        validate()
        return
    temporary = Path(str(OFFLINE_BASE_MODEL) + ".partial")
    if temporary.exists():
        raise FileExistsError(f"incomplete offline checkpoint view exists: {temporary}")
    shutil.copytree(BASE_MODEL, temporary, copy_function=os.link)
    processor_path = temporary / processor_name
    value = json.loads(processor_path.read_text(encoding="utf-8"))
    tokenizer_steps = [
        step
        for step in value.get("steps", [])
        if step.get("registry_name") == "tokenizer_processor"
    ]
    if len(tokenizer_steps) != 1:
        raise RuntimeError("base SmolVLA processor has no unique tokenizer step")
    tokenizer_steps[0]["config"]["tokenizer_name"] = str(VLM_METADATA)
    # Break the hard link before writing so the reviewed source checkpoint is
    # never modified.
    processor_path.unlink()
    processor_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    for path in temporary.iterdir():
        path.chmod(0o444)
    temporary.chmod(0o555)
    os.replace(temporary, OFFLINE_BASE_MODEL)
    validate()


def write_state(status: str, **extra: object) -> None:
    payload = {
        "format": "etsf_robotwin2_ee16_actor_training_watcher_v2",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "dataset_root": str(DATASET_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "conversion_log": str(CONVERT_LOG),
        "training_log": str(TRAIN_LOG),
        "final_checkpoint": str(
            OUTPUT_ROOT / "checkpoints" / "020000" / "pretrained_model"
        ),
        **extra,
    }
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE_PATH)


def validate_dataset() -> dict[str, object]:
    info_path = DATASET_ROOT / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    state_shape = features.get("observation.state", {}).get("shape")
    action_shape = features.get("action", {}).get("shape")
    episode_count = info.get("total_episodes")
    visual_keys = sorted(
        key for key in features if key.startswith("observation.images.")
    )
    expected_visual_keys = sorted(RENAME_MAP)
    if episode_count != 2750:
        raise RuntimeError(f"expected 2750 episodes, found {episode_count!r}")
    if state_shape != [16] or action_shape != [16]:
        raise RuntimeError(
            f"expected state/action [16], found {state_shape!r}/{action_shape!r}"
        )
    if info.get("fps") != 15:
        raise RuntimeError(f"expected 15 fps, found {info.get('fps')!r}")
    if visual_keys != expected_visual_keys:
        raise RuntimeError(
            f"expected three actor cameras {expected_visual_keys!r}, found {visual_keys!r}"
        )
    return {
        "episodes": episode_count,
        "frames": info.get("total_frames"),
        "state_shape": state_shape,
        "action_shape": action_shape,
        "fps": info.get("fps"),
        "visual_keys": visual_keys,
    }


def conversion_command() -> list[str]:
    return [
        str(PYTHON),
        "XPolicyLab/scripts/transform_lerobot_v30_format.py",
        "RoboTwin2_move_can_pot_EE16.*.etsf_ee16_15hz",
        "--repo_id",
        DATASET_ID,
        "--data_type",
        "RoboDojo",
        "--data_version",
        "v1.0",
        "--max_episode",
        "500",
        "--resolution",
        "240x320",
    ]


def training_command() -> list[str]:
    return [
        str(TRAIN),
        f"--policy.path={OFFLINE_BASE_MODEL}",
        # The complete SmolVLA checkpoint below supplies every VLM weight.
        # Construct the identical architecture from the reviewed local
        # config/tokenizer bundle, then let policy.from_pretrained load the
        # checkpoint instead of attempting an unnecessary Hub download.
        f"--policy.vlm_model_name={VLM_METADATA}",
        "--policy.load_vlm_weights=false",
        f"--policy.input_features={compact_json(INPUT_FEATURES)}",
        f"--policy.output_features={compact_json(OUTPUT_FEATURES)}",
        "--policy.adapt_to_pi_aloha=false",
        f"--rename_map={compact_json(RENAME_MAP)}",
        f"--dataset.repo_id={DATASET_ID}",
        f"--dataset.root={DATASET_ROOT}",
        "--dataset.video_backend=pyav",
        "--batch_size=16",
        "--steps=20000",
        f"--output_dir={OUTPUT_ROOT}",
        "--job_name=smolvla_robotwin2_5emb_ee16_full2750_20k",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        "--num_workers=4",
        "--eval_freq=0",
        "--log_freq=50",
        "--save_checkpoint=true",
        "--save_freq=5000",
        "--seed=20260830",
    ]


def main() -> int:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/home/user/.cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": f"{LEROBOT_ROOT / 'src'}:{ROBOTWIN_ROOT}",
            "CUDA_VISIBLE_DEVICES": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    for required in (
        PYTHON,
        TRAIN,
        BASE_MODEL,
        BASE_MODEL / "model.safetensors",
        VLM_METADATA,
        VLM_METADATA / "config.json",
        VLM_METADATA / "tokenizer.json",
        ROBOTWIN_ROOT,
        LEROBOT_ROOT,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"training output already exists: {OUTPUT_ROOT}")
    prepare_offline_base_model()
    if DATASET_ROOT.exists():
        dataset_contract = validate_dataset()
        conversion_exit_code = 0
        dataset_reused_after_strict_validation = True
    else:
        convert = conversion_command()
        write_state("conversion_running", conversion_command=convert)
        with CONVERT_LOG.open("w", encoding="utf-8") as stream:
            conversion = subprocess.run(
                convert,
                cwd=ROBOTWIN_ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        write_state("conversion_exited", conversion_exit_code=conversion.returncode)
        if conversion.returncode != 0:
            raise RuntimeError(f"conversion failed with exit code {conversion.returncode}")
        dataset_contract = validate_dataset()
        conversion_exit_code = conversion.returncode
        dataset_reused_after_strict_validation = False
    free_bytes = shutil.disk_usage("/home/user").free
    if free_bytes < 20 * 1024**3:
        raise RuntimeError(f"less than 20 GiB free before training: {free_bytes}")

    train = training_command()
    with TRAIN_LOG.open("w", encoding="utf-8") as stream:
        training = subprocess.Popen(
            train,
            cwd=LEROBOT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        TRAIN_PID_PATH.write_text(f"{training.pid}\n", encoding="utf-8")
        write_state(
            "training_running",
            conversion_exit_code=conversion_exit_code,
            dataset_contract=dataset_contract,
            dataset_reused_after_strict_validation=dataset_reused_after_strict_validation,
            offline_vlm_metadata=str(VLM_METADATA),
            offline_checkpoint_weights=str(OFFLINE_BASE_MODEL / "model.safetensors"),
            free_bytes_before_training=free_bytes,
            training_pid=training.pid,
            training_command=train,
            input_features=INPUT_FEATURES,
            output_features=OUTPUT_FEATURES,
            rename_map=RENAME_MAP,
            batch_size=16,
            steps=20000,
            save_freq=5000,
        )
        training_exit_code = training.wait()

    final_checkpoint = OUTPUT_ROOT / "checkpoints" / "020000" / "pretrained_model"
    if training_exit_code != 0 or not final_checkpoint.is_dir():
        write_state(
            "training_failed",
            training_exit_code=training_exit_code,
            dataset_contract=dataset_contract,
        )
        return training_exit_code or 1
    write_state(
        "complete",
        training_exit_code=training_exit_code,
        dataset_contract=dataset_contract,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_state("failed", error=f"{type(error).__name__}: {error}")
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
