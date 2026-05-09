"""
Depth Comparison — LeRobot Datasets
=====================================
Compare predicted depth (DA3) vs ground-truth depth for a single episode.

Usage:
    python compare_depth.py \\
        --pred  xiaochyVera/pick_red_mug_human_1_ss_da3 \\
        --gt    xiaochyVera/pick_red_mug_human_1_ss \\
        --demo  0
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

FRONT_DEPTH = "observation.images.cam_azure_kinect_front.transformed_depth"
LEFT_DEPTH  = "observation.images.cam_azure_kinect_left.transformed_depth"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def depth_to_metres(t: torch.Tensor) -> np.ndarray:
    """
    Convert a depth tensor returned by LeRobotDataset to a float32 (H, W) array in metres.

    LeRobot stores transformed_depth as uint16 mm, decoded back as a float tensor.
    Typical shapes: (1, H, W) or (H, W, 1). Values either in mm (>200) or metres (<10).
    """
    arr = t.squeeze().float().numpy()
    if arr.max() > 100.0:          # stored as millimetres
        arr = arr / 1000.0
    return arr


def compute_metrics(pred_m: np.ndarray, gt_m: np.ndarray, valid_min=0.1, valid_max=10.0):
    """
    Compute MAE, RMSE, and AbsRel over valid pixels (GT in [valid_min, valid_max]).
    Returns (mae, rmse, absrel, abs_err_map).
    """
    mask = (gt_m > valid_min) & (gt_m < valid_max)
    if mask.sum() == 0:
        nan = float("nan")
        return nan, nan, nan, np.zeros_like(gt_m)

    p = pred_m[mask]
    g = gt_m[mask]
    err = np.abs(p - g)
    mae    = float(err.mean())
    rmse   = float(np.sqrt((err ** 2).mean()))
    absrel = float((err / g).mean())

    abs_err_map = np.full_like(gt_m, np.nan)
    abs_err_map[mask] = np.abs(pred_m[mask] - gt_m[mask])
    return mae, rmse, absrel, abs_err_map


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def compare(args):
    tolerance_s = 0.0004
    print(f"Loading pred dataset : {args.pred}")
    pred_ds = LeRobotDataset(args.pred, tolerance_s=tolerance_s)
    print(f"Loading gt   dataset : {args.gt}")
    gt_ds   = LeRobotDataset(args.gt,   tolerance_s=tolerance_s)

    ep = args.demo
    start = int(pred_ds.episode_data_index["from"][ep].item())
    end   = int(pred_ds.episode_data_index["to"][ep].item())
    n_frames = end - start
    print(f"Episode {ep}: frames [{start}, {end})  ({n_frames} frames)")

    has_front = FRONT_DEPTH in pred_ds.features and FRONT_DEPTH in gt_ds.features
    has_left  = LEFT_DEPTH  in pred_ds.features and LEFT_DEPTH  in gt_ds.features

    if not (has_front or has_left):
        raise ValueError("Neither front nor left depth found in both datasets.")

    cameras = []
    if has_front:
        cameras.append(("front", FRONT_DEPTH))
    if has_left:
        cameras.append(("left",  LEFT_DEPTH))

    # Per-frame metrics storage: {cam_name: {"mae": [], "rmse": [], "absrel": []}}
    metrics = {name: {"mae": [], "rmse": [], "absrel": []} for name, _ in cameras}

    # Pick ~6 evenly spaced frames for the visual panel
    sample_indices = np.linspace(0, n_frames - 1, min(6, n_frames), dtype=int).tolist()
    samples = {name: [] for name, _ in cameras}   # list of (gt, pred, err) tuples

    for local_idx in range(n_frames):
        global_idx = start + local_idx
        pred_frame = pred_ds[global_idx]
        gt_frame   = gt_ds[global_idx]

        for cam_name, key in cameras:
            pred_m = depth_to_metres(pred_frame[key])
            gt_m   = depth_to_metres(gt_frame[key])
            mae, rmse, absrel, err_map = compute_metrics(pred_m, gt_m)
            metrics[cam_name]["mae"].append(mae)
            metrics[cam_name]["rmse"].append(rmse)
            metrics[cam_name]["absrel"].append(absrel)

            if local_idx in sample_indices:
                samples[cam_name].append((gt_m, pred_m, err_map))

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'Camera':<8}  {'MAE (m)':<10}  {'RMSE (m)':<10}  {'AbsRel':<10}")
    print("-" * 44)
    for cam_name, _ in cameras:
        mae_arr    = np.array(metrics[cam_name]["mae"])
        rmse_arr   = np.array(metrics[cam_name]["rmse"])
        absrel_arr = np.array(metrics[cam_name]["absrel"])
        print(
            f"{cam_name:<8}  "
            f"{np.nanmean(mae_arr):<10.4f}  "
            f"{np.nanmean(rmse_arr):<10.4f}  "
            f"{np.nanmean(absrel_arr):<10.4f}"
        )

    # ── Plot 1: per-frame metric curves ───────────────────────────────────────
    n_cams = len(cameras)
    fig, axes = plt.subplots(3, n_cams, figsize=(6 * n_cams, 10), squeeze=False)
    fig.suptitle(f"Depth Error — Episode {ep}  |  pred: {args.pred.split('/')[-1]}", fontsize=13)
    metric_labels = [("mae", "MAE (m)"), ("rmse", "RMSE (m)"), ("absrel", "AbsRel")]

    for col, (cam_name, _) in enumerate(cameras):
        for row, (metric_key, ylabel) in enumerate(metric_labels):
            ax = axes[row][col]
            vals = np.array(metrics[cam_name][metric_key])
            ax.plot(vals, linewidth=1.2)
            ax.set_xlabel("Frame")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{cam_name} — {ylabel}")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = f"depth_curves_ep{ep}.png"
    plt.savefig(curves_path, dpi=150)
    print(f"\nSaved: {curves_path}")

    # ── Plot 2: sample depth maps + error ─────────────────────────────────────
    for cam_name, _ in cameras:
        n_samples = len(samples[cam_name])
        if n_samples == 0:
            continue

        fig, axes = plt.subplots(3, n_samples, figsize=(3.5 * n_samples, 9))
        if n_samples == 1:
            axes = axes[:, np.newaxis]
        fig.suptitle(
            f"{cam_name} depth samples — Episode {ep}  "
            f"(gt / pred / |err|, metres)",
            fontsize=12,
        )
        row_labels = ["GT (m)", "Pred (m)", "|Error| (m)"]

        for col, (gt_m, pred_m, err_map) in enumerate(samples[cam_name]):
            vmin = np.nanpercentile(gt_m[gt_m > 0.1], 2)
            vmax = np.nanpercentile(gt_m[gt_m > 0.1], 98)
            err_vmax = np.nanpercentile(err_map[~np.isnan(err_map)], 95) if not np.all(np.isnan(err_map)) else 1.0

            for row, (data, cmap, v0, v1, label) in enumerate([
                (gt_m,   "plasma", vmin, vmax,     row_labels[0]),
                (pred_m, "plasma", vmin, vmax,     row_labels[1]),
                (err_map,"hot",    0,    err_vmax,  row_labels[2]),
            ]):
                ax = axes[row][col]
                im = ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if col == 0:
                    ax.set_ylabel(label, fontsize=9)
                frame_no = start + sample_indices[col]
                ax.set_title(f"frame {frame_no}", fontsize=8)
                ax.axis("off")

        plt.tight_layout()
        maps_path = f"depth_maps_{cam_name}_ep{ep}.png"
        plt.savefig(maps_path, dpi=150)
        print(f"Saved: {maps_path}")

    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pred",  required=True,  help="Predicted depth dataset repo ID")
    p.add_argument("--gt",    required=True,  help="Ground-truth depth dataset repo ID")
    p.add_argument("--demo",  type=int, default=0, help="Episode / demo index (default: 0)")
    args = p.parse_args()
    compare(args)
