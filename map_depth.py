"""
MapAnything Depth Replacement — LeRobot Dataset
================================================
Replaces transformed_depth in both Azure Kinect cameras with
MapAnything metric depth predictions using known camera poses.

Note: MapAnything camera_poses uses cam2world (C2W) convention —
same as the calibration JSON, NO inversion needed.

Setup:
    git clone https://github.com/facebookresearch/map-anything
    cd map-anything
    conda create -n mapanything python=3.12 -y
    conda activate mapanything
    pip install -e ".[all]"
    pip install lerobot opencv-python

Run:
    python mapanything_replace_depth.py \\
        --source xiaochyVera/pick_red_mug_human \\
        --target xiaochyVera/pick_red_mug_human_madepth \\
        --calib  cam_calibration.json
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# CC BY-NC model (trained on 13 datasets, best indoor performance)
# For Apache 2.0 use: "facebook/map-anything-apache"
MODEL_ID = "facebook/map-anything"

AUTO_FIELDS  = {"episode_index", "frame_index", "index", "task_index", "timestamp"}
FRONT_COLOR  = "observation.images.cam_azure_kinect_front.color"
LEFT_COLOR   = "observation.images.cam_azure_kinect_left.color"
FRONT_DEPTH  = "observation.images.cam_azure_kinect_front.transformed_depth"
LEFT_DEPTH   = "observation.images.cam_azure_kinect_left.transformed_depth"


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

def load_calib(path: str):
    """
    Load calibration JSON.

    intrinsic  : (3,3) K matrix
    extrinsic  : (4,4) cam-to-world — MapAnything uses C2W directly, no inversion
    distortion : (8,)  rational model [k1,k2,p1,p2,k3,k4,k5,k6]
    """
    with open(path) as f:
        d = json.load(f)

    def parse(cam):
        K    = np.array(cam["intrinsic"],  dtype=np.float32)  # (3,3)
        dist = np.array(cam["distortion"], dtype=np.float32)  # (8,)
        C2W  = np.array(cam["extrinsic"],  dtype=np.float32)  # (4,4) cam→world
        return K, dist, C2W  # NOTE: C2W, not inverted

    K0, dist0, C2W0 = parse(d["cam0"])   # front
    K1, dist1, C2W1 = parse(d["cam1"])   # left
    return K0, dist0, C2W0, K1, dist1, C2W1


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def lerobot_color_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """
    LeRobot: (C, H, W) float32 [0,1] RGB
    → MapAnything: (H, W, 3) uint8 [0,255] RGB
    """
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def undistort(img_rgb: np.ndarray, K: np.ndarray, dist: np.ndarray, K_new: np.ndarray) -> np.ndarray:
    """Undistort RGB image using 8-coeff rational model."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    bgr_ud = cv2.undistort(bgr, K, dist, None, K_new)
    return cv2.cvtColor(bgr_ud, cv2.COLOR_BGR2RGB)  # back to RGB for MapAnything


def get_K_new(K, dist, h, w):
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    return K_new


# ──────────────────────────────────────────────────────────────────────────────
# Depth output helper
# ──────────────────────────────────────────────────────────────────────────────

def depth_m_to_lerobot_tensor(depth_m: np.ndarray, H: int, W: int) -> torch.Tensor:
    """
    MapAnything depth (H,W) float32 metres
    → LeRobot: (1, H, W) float32 in [0, 65.535]
      so that * 1000 → uint16 millimetres (Azure Kinect convention)
    """
    depth_resized = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_clipped = np.clip(depth_resized, 0, 65.535)
    return torch.from_numpy(depth_clipped).float().unsqueeze(0)  # (1, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# MapAnything inference for one frame pair
# ──────────────────────────────────────────────────────────────────────────────

def predict_depth_pair(
    model,
    img0_rgb: np.ndarray,    # (H, W, 3) uint8 RGB, undistorted, front
    img1_rgb: np.ndarray,    # (H, W, 3) uint8 RGB, undistorted, left
    K0_new: np.ndarray,      # (3,3) updated intrinsics for undistorted front
    K1_new: np.ndarray,      # (3,3) updated intrinsics for undistorted left
    C2W0: np.ndarray,        # (4,4) cam0 cam-to-world
    C2W1: np.ndarray,        # (4,4) cam1 cam-to-world
    device: str,
):
    """
    Build MapAnything view dicts with image + intrinsics + pose for both cams,
    run inference, return (depth0_m, depth1_m) as float32 (H,W) numpy arrays.
    """
    from mapanything.utils.image import preprocess_inputs

    views = [
        {
            # Front camera — image + calibration + pose
            "img": img0_rgb,                                      # (H, W, 3) uint8
            "intrinsics": torch.tensor(K0_new, dtype=torch.float32),  # (3,3)
            "camera_poses": torch.tensor(C2W0, dtype=torch.float32),  # (4,4) C2W
            "is_metric_scale": torch.tensor([True], device=device),
        },
        {
            # Left camera — image + calibration + pose
            "img": img1_rgb,
            "intrinsics": torch.tensor(K1_new, dtype=torch.float32),
            "camera_poses": torch.tensor(C2W1, dtype=torch.float32),
            "is_metric_scale": torch.tensor([True], device=device),
        },
    ]

    processed = preprocess_inputs(views)

    with torch.no_grad():
        predictions = model.infer(
            processed,
            memory_efficient_inference = True,
            use_amp                    = True,
            amp_dtype                  = "bf16",
            apply_mask                 = True,
            mask_edges                 = True,
            apply_confidence_mask      = False,
            ignore_calibration_inputs  = False,
            ignore_pose_inputs         = False,
            ignore_depth_inputs        = False,
        )

    # predictions: list of dicts, one per view; depth_z has shape (B, H, W, 1)
    depth0_m = predictions[0]["depth_z"][0, :, :, 0].cpu().numpy().astype(np.float32)
    depth1_m = predictions[1]["depth_z"][0, :, :, 0].cpu().numpy().astype(np.float32)
    return depth0_m, depth1_m


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def replace_depth(args):
    try:
        from mapanything.models import MapAnything
    except ImportError:
        raise ImportError(
            "\n[ERROR] mapanything not found.\n"
            "  git clone https://github.com/facebookresearch/map-anything\n"
            "  cd map-anything && pip install -e '.[all]'\n"
        )

    # ── Calibration ───────────────────────────────────────────────────────────
    print(f"Loading calibration: {args.calib}")
    K0, dist0, C2W0, K1, dist1, C2W1 = load_calib(args.calib)
    print(f"  cam0 (front) fx={K0[0,0]:.1f}  C2W loaded (no inversion needed)")
    print(f"  cam1 (left)  fx={K1[0,0]:.1f}  C2W loaded (no inversion needed)")

    # ── Source dataset ────────────────────────────────────────────────────────
    print(f"\nLoading source dataset: {args.source}")
    source_dataset = LeRobotDataset(args.source, tolerance_s=0.0004)
    source_meta    = LeRobotDatasetMetadata(args.source)

    has_front_depth = FRONT_DEPTH in source_dataset.features
    has_left_depth  = LEFT_DEPTH  in source_dataset.features

    print(f"  fps={source_dataset.fps}  episodes={source_meta.info['total_episodes']}")
    print(f"  front depth present={has_front_depth}")
    print(f"  left  depth present={has_left_depth}")

    # ── Target dataset ────────────────────────────────────────────────────────
    print(f"\nCreating target dataset: {args.target}")

    # Ensure both depth features are always present in the target
    target_features = dict(source_dataset.features)

    existing_depth_spec = (
        source_dataset.features.get(FRONT_DEPTH)
        or source_dataset.features.get(LEFT_DEPTH)
    )
    if existing_depth_spec is None:
        raise RuntimeError(
            "Source dataset has neither front nor left depth — cannot infer feature spec. "
            "Add at least one depth stream to the source dataset first."
        )

    if FRONT_DEPTH not in target_features:
        print(f"  [INFO] Adding missing feature: {FRONT_DEPTH}")
        target_features[FRONT_DEPTH] = existing_depth_spec

    if LEFT_DEPTH not in target_features:
        print(f"  [INFO] Adding missing feature: {LEFT_DEPTH}")
        target_features[LEFT_DEPTH] = existing_depth_spec


    target_dataset = LeRobotDataset.create(
        repo_id  = args.target,
        fps      = source_dataset.fps,
        features = target_features,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {MODEL_ID} on {device} ...")
    model = MapAnything.from_pretrained(MODEL_ID).to(device)
    model.eval()
    print("Model ready.\n")

    # ── Episode loop ──────────────────────────────────────────────────────────
    total_episodes = source_meta.info["total_episodes"]

    for ep_idx in range(total_episodes):
        start = source_dataset.episode_data_index["from"][ep_idx].item()
        end   = source_dataset.episode_data_index["to"][ep_idx].item()
        print(f"Episode {ep_idx:03d}  frames [{start}, {end})")

        # Compute K_new from actual image resolution (once per episode)
        first  = source_dataset[start]
        img0   = lerobot_color_to_hwc_uint8(first[FRONT_COLOR])
        img1   = lerobot_color_to_hwc_uint8(first[LEFT_COLOR])
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        K0_new = get_K_new(K0, dist0, h0, w0)
        K1_new = get_K_new(K1, dist1, h1, w1)

        for idx in tqdm(range(start, end), desc=f"  ep {ep_idx:03d}"):
            frame = source_dataset[idx]

            # Copy all non-auto fields
            frame_data = {
                k: v for k, v in frame.items()
                if k not in AUTO_FIELDS and k in source_dataset.features
            }
            frame_data["task"] = source_meta.tasks[frame["task_index"].item()]

            # Undistort color images
            img0_rgb = lerobot_color_to_hwc_uint8(frame[FRONT_COLOR])
            img1_rgb = lerobot_color_to_hwc_uint8(frame[LEFT_COLOR])
            img0_ud  = undistort(img0_rgb, K0, dist0, K0_new)
            img1_ud  = undistort(img1_rgb, K1, dist1, K1_new)

            # MapAnything inference
            depth0_m, depth1_m = predict_depth_pair(
                model,
                img0_ud, img1_ud,
                K0_new, K1_new,
                C2W0, C2W1,
                device,
            )

            # Replace depth fields
            frame_data[FRONT_DEPTH] = depth_m_to_lerobot_tensor(depth0_m, h0, w0)
            frame_data[LEFT_DEPTH]  = depth_m_to_lerobot_tensor(depth1_m, h1, w1)

            # Fix up image tensor formats for add_frame
            for key in list(frame_data.keys()):
                if key.startswith("observation.images.cam_azure_kinect"):
                    if key.endswith(".color"):
                        frame_data[key] = (
                            frame_data[key].permute(1, 2, 0) * 255
                        ).to(torch.uint8)
                    elif key.endswith(".transformed_depth"):
                        frame_data[key] = (
                            frame_data[key].permute(1, 2, 0) * 1000
                        ).to(torch.uint16)
                    elif key.endswith(".goal_gripper_proj"):
                        frame_data[key] = (
                            frame_data[key].permute(1, 2, 0) * 255
                        ).to(torch.uint8)

            if "observation.images.cam_wrist" in frame_data:
                frame_data["observation.images.cam_wrist"] = (
                    frame_data["observation.images.cam_wrist"].permute(1, 2, 0) * 255
                ).to(torch.uint8)

            target_dataset.add_frame(frame_data)

        target_dataset.save_episode()
        print(f"  Episode {ep_idx:03d} saved.")

    print(f"\nDone! {len(target_dataset)} frames written.")

    if args.push_to_hub:
        print(f"Pushing to hub: {args.target} ...")
        target_dataset.push_to_hub(repo_id=args.target)

    return target_dataset


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source",      required=True,
                   help="Source LeRobot repo ID")
    p.add_argument("--target",      required=True,
                   help="Target LeRobot repo ID")
    p.add_argument("--calib",       default="cam_calibration.json",
                   help="Calibration JSON (cam0=front, cam1=left)")
    p.add_argument("--push_to_hub", action="store_true",
                   help="Push finished dataset to HuggingFace Hub")
    args = p.parse_args()
    replace_depth(args)
