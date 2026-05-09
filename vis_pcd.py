"""
Point Cloud Visualizer — GT vs Predicted (LeRobot Dataset)
===========================================================
Lifts depth + color from two datasets (ground-truth and predicted)
into 3D RGB point clouds and streams the whole episode into rerun.io.
Both datasets share the same camera calibration.

Usage:
    python vis_pcd.py \\
        --gt    xiaochyVera/pick_red_mug_human \\
        --pred  xiaochyVera/pick_red_mug_human_madepth \\
        --episode 0 \\
        --calib   cam_calibration.json
"""

import argparse
import json

import numpy as np
import rerun as rr
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

FRONT_COLOR = "observation.images.cam_azure_kinect_front.color"
LEFT_COLOR  = "observation.images.cam_azure_kinect_left.color"
FRONT_DEPTH = "observation.images.cam_azure_kinect_front.transformed_depth"
LEFT_DEPTH  = "observation.images.cam_azure_kinect_left.transformed_depth"


def load_calib(path: str):
    with open(path) as f:
        d = json.load(f)

    def parse(cam):
        K    = np.array(cam["intrinsic"],  dtype=np.float64)
        dist = np.array(cam["distortion"], dtype=np.float64)
        C2W  = np.array(cam["extrinsic"],  dtype=np.float64)
        return K, dist, C2W

    K0, dist0, C2W0 = parse(d["cam0"])
    K1, dist1, C2W1 = parse(d["cam1"])
    return K0, dist0, C2W0, K1, dist1, C2W1


def depth_to_metres(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze().float().numpy()
    if arr.max() > 100.0:
        arr = arr / 1000.0
    return arr


def color_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) float32 [0,1] → (H,W,3) uint8"""
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def depth_to_pointcloud(
    depth_m: np.ndarray,
    color_rgb: np.ndarray,
    K: np.ndarray,
    C2W: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject depth → world-space XYZ with RGB colors. Returns (N,3), (N,3)."""
    H, W = depth_m.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    valid = (depth_m > min_depth) & (depth_m < max_depth)
    d = depth_m[valid].astype(np.float32)

    pts_cam = np.stack([
        (us[valid] - cx) / fx * d,
        (vs[valid] - cy) / fy * d,
        d,
        np.ones_like(d),
    ], axis=-1)  # (N, 4)
    pts_world = (C2W @ pts_cam.T).T  # (N, 4)

    return pts_world[:, :3].astype(np.float32), color_rgb[valid]


def build_pcd_for_frame(frame, has_front, has_left, K0, C2W0, K1, C2W1, max_depth):
    """Merge front+left point clouds for one frame. Returns (xyz, rgb)."""
    all_xyz, all_rgb = [], []

    if has_front:
        color = color_to_hwc_uint8(frame[FRONT_COLOR])
        depth = depth_to_metres(frame[FRONT_DEPTH])
        xyz, rgb = depth_to_pointcloud(depth, color, K0, C2W0, max_depth=max_depth)
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    if has_left:
        color = color_to_hwc_uint8(frame[LEFT_COLOR])
        depth = depth_to_metres(frame[LEFT_DEPTH])
        xyz, rgb = depth_to_pointcloud(depth, color, K1, C2W1, max_depth=max_depth)
        all_xyz.append(xyz)
        all_rgb.append(rgb)

    return np.concatenate(all_xyz), np.concatenate(all_rgb)


def visualize(args):
    K0, _, C2W0, K1, _, C2W1 = load_calib(args.calib)

    print(f"Loading GT dataset  : {args.gt}")
    gt_ds   = LeRobotDataset(args.gt,   tolerance_s=0.0004)
    print(f"Loading pred dataset: {args.pred}")
    pred_ds = LeRobotDataset(args.pred, tolerance_s=0.0004)

    ep = args.episode
    gt_start = int(gt_ds.episode_data_index["from"][ep].item())
    gt_end   = int(gt_ds.episode_data_index["to"][ep].item())
    pred_start = int(pred_ds.episode_data_index["from"][ep].item())
    n_frames = min(gt_end - gt_start, int(pred_ds.episode_data_index["to"][ep].item()) - pred_start)
    print(f"Episode {ep}: {n_frames} frames")

    gt_has_front   = FRONT_COLOR in gt_ds.features   and FRONT_DEPTH in gt_ds.features
    gt_has_left    = LEFT_COLOR  in gt_ds.features   and LEFT_DEPTH  in gt_ds.features
    pred_has_front = FRONT_COLOR in pred_ds.features and FRONT_DEPTH in pred_ds.features
    pred_has_left  = LEFT_COLOR  in pred_ds.features and LEFT_DEPTH  in pred_ds.features

    # ── Rerun init ────────────────────────────────────────────────────────────
    # rr.save() replaces the sink, so spawn and save are mutually exclusive.
    # Default: live viewer. Pass --output to write a .rrd file instead.
    rr.init("map_anything/pcd_compare", spawn=args.output is None)
    if args.output:
        rr.save(args.output)
        print(f"Saving to {args.output} — open later with: rerun {args.output}")

    # Static camera entities (same for both datasets — shared calibration)
    first_gt = gt_ds[gt_start]

    if gt_has_front:
        H0, W0 = color_to_hwc_uint8(first_gt[FRONT_COLOR]).shape[:2]
        rr.log("world/cam_front",
               rr.Transform3D(translation=C2W0[:3, 3], mat3x3=C2W0[:3, :3]),
               rr.Pinhole(image_from_camera=K0, width=W0, height=H0),
               static=True)

    if gt_has_left:
        H1, W1 = color_to_hwc_uint8(first_gt[LEFT_COLOR]).shape[:2]
        rr.log("world/cam_left",
               rr.Transform3D(translation=C2W1[:3, 3], mat3x3=C2W1[:3, :3]),
               rr.Pinhole(image_from_camera=K1, width=W1, height=H1),
               static=True)

    # ── Frame loop ────────────────────────────────────────────────────────────
    for i in range(n_frames):
        rr.set_time_sequence("frame", i)

        gt_frame   = gt_ds[gt_start     + i]
        pred_frame = pred_ds[pred_start + i]

        # Color images from GT (same in both datasets; log once per frame)
        if gt_has_front:
            rr.log("world/cam_front/color",
                   rr.Image(color_to_hwc_uint8(gt_frame[FRONT_COLOR])))
        if gt_has_left:
            rr.log("world/cam_left/color",
                   rr.Image(color_to_hwc_uint8(gt_frame[LEFT_COLOR])))

        # GT point cloud — solid white
        if gt_has_front or gt_has_left:
            xyz, rgb = build_pcd_for_frame(
                gt_frame, gt_has_front, gt_has_left, K0, C2W0, K1, C2W1, args.max_depth)
            rr.log("world/gt/points", rr.Points3D(xyz, colors=np.full((len(xyz), 3), 255, dtype=np.uint8)))

        # Predicted point cloud — solid black
        if pred_has_front or pred_has_left:
            xyz, rgb = build_pcd_for_frame(
                pred_frame, pred_has_front, pred_has_left, K0, C2W0, K1, C2W1, args.max_depth)
            rr.log("world/pred/points", rr.Points3D(xyz, colors=np.tile([0, 255, 0], (len(xyz), 1)).astype(np.uint8)))

    print("Done. Rerun viewer should be open.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gt",        required=True,
                   help="Ground-truth LeRobot dataset repo ID")
    p.add_argument("--pred",      required=True,
                   help="Predicted LeRobot dataset repo ID")
    p.add_argument("--episode",   type=int, default=0,
                   help="Episode index (default: 0)")
    p.add_argument("--calib",     default="cam_calibration.json",
                   help="Calibration JSON (default: cam_calibration.json)")
    p.add_argument("--max_depth", type=float, default=5.0,
                   help="Max depth in metres to include (default: 5.0)")
    p.add_argument("--output",    default=None,
                   help="Save .rrd file to this path (default: ep000.rrd)")
    args = p.parse_args()
    visualize(args)
