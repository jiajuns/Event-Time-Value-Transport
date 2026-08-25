import json
from pathlib import Path

from huggingface_hub import snapshot_download


TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "handover_block",
    "lift_pot",
    "move_can_pot",
    "place_container_plate",
]


def main() -> None:
    output_dir = Path("/home/user/etsf_stage0/data/robotwin_clean")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="StarVLA/RoboTwin-Clean",
        repo_type="dataset",
        revision="main",
        allow_patterns=[f"{task}/**" for task in TASKS],
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        max_workers=8,
    )
    manifest = {
        "repo_id": "StarVLA/RoboTwin-Clean",
        "revision": "main",
        "tasks": TASKS,
        "note": "The published snapshot contains 50 successful episodes per selected task.",
    }
    (output_dir / "stage0_selection.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
