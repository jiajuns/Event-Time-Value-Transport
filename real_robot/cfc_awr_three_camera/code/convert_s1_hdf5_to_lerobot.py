#!/usr/bin/env python3
"""Convert Astribot raw recorder HDF5 (grasp task) -> LeRobot v3.0 dataset (video-encoded).

Raw format (Orin recorder):
  time (T,)
  joints_dict/joints_position_state   (T,25)  [radians; gripper dims 14,22 = motor multi-turn rad]
  joints_dict/joints_position_command (T,25)  [radians; gripper dims = 0-100 percent]
  poses_dict/astribot_gripper_left|right (T,1) [0-100 percent]
  images_dict/{head,left,right}/rgb   JPEG byte stream + rgb_size (T,)
"""
import argparse, json, os, sys, time
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import cv2, h5py, numpy as np
from lerobot.datasets import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig, DepthEncoderConfig

GRIPPER_LEFT, GRIPPER_RIGHT = 14, 22
STATE_DIM = ACTION_DIM = 25
IMG_SIZE = 256

RAW_CAM_KEYS = ["head", "left", "right"]
DATASET_CAM_KEYS = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]
CAM_MAP = dict(zip(RAW_CAM_KEYS, DATASET_CAM_KEYS))


def resolve_raw_cam_keys(f) -> dict:
    """兼容两种采集器目录名: 旧版 {head,left,right} / RGBD 新版 {head_rgbd,left_wrist_rgbd,right_wrist_rgbd}。"""
    g = f["images_dict"]
    if "head_rgbd" in g:
        return {"head": "head_rgbd", "left": "left_wrist_rgbd", "right": "right_wrist_rgbd"}
    return {"head": "head", "left": "left", "right": "right"}


def resize_with_pad(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Aspect-preserving resize, centered on black square canvas (RGB)."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    rh, rw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - rh) // 2, (size - rw) // 2
    canvas[top:top + rh, left:left + rw] = resized
    return canvas


def resize_depth(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """头部深度 512x720 uint16 mm -> (size,size,1) uint16, 等比缩放+居中黑边。"""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    rh, rw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size, 1), dtype=np.uint16)
    top, left = (size - rh) // 2, (size - rw) // 2
    canvas[top:top + rh, left:left + rw, 0] = resized
    return canvas


def load_camera_frames(f, cam: str) -> list[np.ndarray]:
    """读取一集相机帧, 兼容两种记录器格式:
    1) vlen 数据集: 每元素一帧 JPEG 字节
    2) 扁平字节流: rgb 为 (total_bytes,) uint8, 按 rgb_size 累计偏移切片
    """
    g = f["images_dict"][cam]
    sizes = np.asarray(g["rgb_size"], dtype=np.int64)
    raw = g["rgb"]
    frames = []

    def decode(jpg: np.ndarray):
        jpg = np.asarray(jpg, dtype=np.uint8).ravel()
        img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise RuntimeError(f"{cam}: JPEG decode failed (len={len(jpg)}, magic={bytes(jpg[:4]).hex()})")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if h5py.check_dtype(vlen=raw.dtype) is not None:
        for t in range(len(sizes)):
            frames.append(decode(np.asarray(raw[t], dtype=np.uint8)))
    else:
        blob = np.asarray(raw, dtype=np.uint8).ravel()
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        if offsets[-1] > len(blob):
            raise RuntimeError(f"{cam}: rgb_size 总和 {int(offsets[-1])} 超过字节流长度 {len(blob)}")
        for t in range(len(sizes)):
            frames.append(decode(blob[int(offsets[t]):int(offsets[t + 1])]))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/chefmate/Astribot/grasp_raw")
    ap.add_argument("--root", default="/home/chefmate/Astribot/grasp_lerobot_v3")
    ap.add_argument("--repo-id", default="astribot/grasp")
    ap.add_argument("--task", default="grasp the target object")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--glob", default="grasp_episode_*.hdf5")
    ap.add_argument(
        "--cameras",
        default="head,left,right",
        help="comma-separated cameras selected from: head,left,right",
    )
    ap.add_argument("--skip", default="", help="comma-separated source episode indices to skip")
    ap.add_argument("--swap-wrist", action="store_true",
                    help="交换左右腕相机(修正装反的硬件; 重训时部署端需同步开 --swap-wrist)")
    ap.add_argument("--with-depth", action="store_true",
                    help="HDF5 含头部深度(images_dict/head/depth uint16 512x720 mm)时, 加 base_0_depth 观测")
    args = ap.parse_args()

    selected_cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    unknown_cameras = sorted(set(selected_cameras) - set(RAW_CAM_KEYS))
    if not selected_cameras or unknown_cameras or len(selected_cameras) != len(set(selected_cameras)):
        raise ValueError(
            f"invalid --cameras={args.cameras!r}; expected a non-empty unique subset of {RAW_CAM_KEYS}"
        )

    if args.swap_wrist:
        RAW_CAM_KEYS[1], RAW_CAM_KEYS[2] = RAW_CAM_KEYS[2], RAW_CAM_KEYS[1]
        global CAM_MAP
        CAM_MAP = dict(zip(RAW_CAM_KEYS, DATASET_CAM_KEYS))
        print("⚠️ --swap-wrist: 左右腕相机交换后写入数据集", flush=True)

    skip = {int(x) for x in args.skip.split(",") if x.strip()}
    paths = sorted(
        [p for p in __import__("glob").glob(os.path.join(args.data_dir, args.glob))],
        key=lambda p: int(os.path.basename(p).split("_episode_")[1].split(".")[0]),
    )
    print(f"found {len(paths)} episodes; skipping {sorted(skip) if skip else 'none'}")

    features = {
        CAM_MAP[c]: {"dtype": "video", "shape": (IMG_SIZE, IMG_SIZE, 3), "names": ["height", "width", "channels"]}
        for c in selected_cameras
    }
    features["observation.state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": ["qpos"]}
    features["action"] = {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["action"]}
    if args.with_depth:
        features["observation.images.base_0_depth"] = {
            "dtype": "video", "shape": (IMG_SIZE, IMG_SIZE, 1),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True, "depth_unit": "mm"},
        }

    if os.path.exists(args.root):
        raise FileExistsError(f"{args.root} exists - refusing to overwrite. Move/delete it first.")

    rgb_encoder = RGBEncoderConfig(vcodec="h264", crf=23, preset="fast")

    create_kwargs = dict(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        robot_type="astribot_s1",
        use_videos=True,
        rgb_encoder=rgb_encoder,
    )
    if args.with_depth:
        create_kwargs["depth_encoder"] = DepthEncoderConfig()
    dataset = LeRobotDataset.create(**create_kwargs)

    mapping = []
    t0 = time.time()
    for new_idx, path in enumerate(paths):
        src_idx = int(os.path.basename(path).split("_episode_")[1].split(".")[0])
        if src_idx in skip:
            print(f"skip source episode {src_idx}")
            continue
        with h5py.File(path, "r") as f:
            state = np.asarray(f["joints_dict/joints_position_state"], dtype=np.float32).copy()
            action = np.asarray(f["joints_dict/joints_position_command"], dtype=np.float32).copy()
            ts = np.asarray(f["time"], dtype=np.float64)
            # 夹爪: 旧记录器有 0-100% 列(poses_dict/astribot_gripper_*), 覆盖 motor rad;
            # 新采集器(RGBD 版)无该列, 保留 SDK 原始关节值(rad), 训练/部署同一单位即可
            if "poses_dict/astribot_gripper_left" in f:
                state[:, GRIPPER_LEFT] = np.asarray(f["poses_dict/astribot_gripper_left"], dtype=np.float32)[:, 0]
                state[:, GRIPPER_RIGHT] = np.asarray(f["poses_dict/astribot_gripper_right"], dtype=np.float32)[:, 0]
            raw_keys = resolve_raw_cam_keys(f)
            cams = {c: load_camera_frames(f, raw_keys[c]) for c in selected_cameras}
            depth = None
            head_key = raw_keys["head"]
            if args.with_depth and f"images_dict/{head_key}/depth" in f:
                dset = f[f"images_dict/{head_key}/depth"]
                if dset.ndim == 3:
                    # 最终版采集器: (T,512,720) uint16 固定数组
                    depth = np.asarray(dset, dtype=np.uint16).copy()
                elif dset.ndim == 1 and h5py.check_dtype(vlen=dset.dtype) is not None:
                    # 中间版本: (T,) vlen uint8 字节流, 逐帧解析 raw uint16 512x720
                    depth = [np.frombuffer(np.asarray(dset[i], np.uint8), np.uint16).reshape(512, 720)
                             for i in range(len(dset))]

        T = len(ts)
        assert all(len(cams[c]) == T for c in selected_cameras)
        for t in range(T):
            frame = {
                "observation.state": state[t],
                "action": action[t],
                "task": args.task,
            }
            for c in selected_cameras:
                frame[CAM_MAP[c]] = resize_with_pad(cams[c][t])
            if depth is not None:
                frame["observation.images.base_0_depth"] = resize_depth(depth[t])
            dataset.add_frame(frame)
        dataset.save_episode()
        mapping.append({"source_episode": src_idx, "dataset_episode": new_idx, "frames": T})
        del state, action, ts, cams
        print(f"  ep{src_idx} -> dataset ep{new_idx}: {T} frames ({time.time()-t0:.0f}s elapsed)", flush=True)

    dataset.finalize()
    with open(os.path.join(args.root, "conversion_mapping.json"), "w") as fh:
        json.dump(
            {"task": args.task, "fps": args.fps, "cameras": selected_cameras, "episodes": mapping},
            fh,
            indent=2,
        )
    print(f"DONE. dataset at {args.root}; {len(mapping)} episodes, "
          f"{sum(m['frames'] for m in mapping)} frames total, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
