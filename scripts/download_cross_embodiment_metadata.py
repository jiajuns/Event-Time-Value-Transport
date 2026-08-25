"""Download trajectory-metadata-only subsets for cross-embodiment speed audit."""

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


REPOS = [
    "Avada11/RoboTwin-AgileX",
    "Avada11/RoboTwin-Panda",
    "Avada11/RoboTwin-X5",
]


def main() -> None:
    root = Path("/home/user/etsf_stage0/cross_embodiment_public_data")
    root.mkdir(parents=True, exist_ok=True)
    for repo_id in REPOS:
        local_dir = root / repo_id.split("/")[-1]
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
            allow_patterns=["*/demo*/_traj_data/*.pkl"],
            local_dir=local_dir,
            max_workers=8,
        )
        print(f"DOWNLOADED {repo_id} -> {local_dir}", flush=True)

    piper_root = root / "robotwin_piper_dual_click_bell"
    for filename in ["meta/info.json", "meta/episodes/chunk-000/file-000.parquet"]:
        hf_hub_download(
            repo_id="hongxiaoy/robotwin_piper_dual_click_bell",
            repo_type="dataset",
            revision="main",
            filename=filename,
            local_dir=piper_root,
        )
    print("DOWNLOADED Piper click_bell metadata", flush=True)


if __name__ == "__main__":
    main()
