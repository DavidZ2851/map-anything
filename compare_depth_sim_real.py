"""
Compare metric depth difference between a real and a sim MKV depth file.
Frames are matched by relative position (0→1) since the two files may have
different durations / frame counts.

Usage:
    python compare_depth_sim_real.py \\
        --real  kinect_depth.mkv \\
        --sim   sim_depth.mkv    \\
        --n-samples 6
"""

import argparse

import av
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# MKV depth reader
# ──────────────────────────────────────────────────────────────────────────────

def _video_duration(path: str) -> float:
    with av.open(path) as c:
        stream = c.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        return float(c.duration) / 1_000_000


def read_frame_at(path: str, target_ts: float) -> np.ndarray:
    """Decode the closest frame to target_ts and return (H,W) float32 metres."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        seek_pts = int(target_ts / float(stream.time_base))
        container.seek(seek_pts, stream=stream, backward=True)
        for packet in container.demux(stream):
            for frame in packet.decode():
                fmt = frame.format.name
                if fmt in ("gray16le", "gray16be"):
                    arr = frame.to_ndarray(format="gray16le")
                else:
                    arr = frame.to_ndarray(format="gray16le")
                return arr.astype(np.float32) / 1000.0   # mm → metres
    raise RuntimeError(f"No frame decoded at t={target_ts:.3f}s in {path}")


def sample_frames(path: str, n: int) -> tuple[list[np.ndarray], list[float]]:
    """Sample n frames evenly by relative position (independent of duration)."""
    duration = _video_duration(path)
    # avoid hitting the very last pts which may be past stream end
    timestamps = np.linspace(0.0, duration * 0.98, n).tolist()
    frames = [read_frame_at(path, t) for t in timestamps]
    return frames, timestamps


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, gt: np.ndarray,
                    valid_min=0.1, valid_max=10.0):
    """MAE, RMSE, AbsRel over valid GT pixels. Returns scalars + error map."""
    mask = (gt > valid_min) & (gt < valid_max)
    abs_err = np.full_like(gt, np.nan)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan, abs_err

    p, g = pred[mask], gt[mask]
    diff = np.abs(p - g)
    mae    = float(diff.mean())
    rmse   = float(np.sqrt((diff ** 2).mean()))
    absrel = float((diff / g).mean())
    abs_err[mask] = np.abs(pred[mask] - gt[mask])
    return mae, rmse, absrel, abs_err


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(real_frames, sim_frames, all_metrics, args):
    n = len(real_frames)

    # ── Figure 1: depth maps + error ──────────────────────────────────────────
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 9))
    if n == 1:
        axes = axes[:, np.newaxis]
    fig.suptitle(
        f"Depth comparison — real: {args.real.split('/')[-1]}  |  sim: {args.sim.split('/')[-1]}",
        fontsize=12,
    )

    for col in range(n):
        real_m = real_frames[col]
        sim_m  = sim_frames[col]
        _, _, _, err_map = all_metrics[col]

        valid = real_m[(real_m > 0.1) & (real_m < 10.0)]
        vmin = float(np.percentile(valid, 2))  if valid.size else 0
        vmax = float(np.percentile(valid, 98)) if valid.size else 5
        err_vmax = float(np.nanpercentile(err_map, 95)) if not np.all(np.isnan(err_map)) else 1.0

        mae, rmse, absrel, _ = all_metrics[col]

        for row, (data, cmap, v0, v1, rlabel) in enumerate([
            (real_m, "plasma", vmin,  vmax,     "Real (m)"),
            (sim_m,  "plasma", vmin,  vmax,     "Sim (m)"),
            (err_map,"hot",    0,     err_vmax,  "|Diff| (m)"),
        ]):
            ax = axes[row][col]
            im = ax.imshow(data, cmap=cmap, vmin=v0, vmax=v1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if col == 0:
                ax.set_ylabel(rlabel, fontsize=9)
            if row == 2:
                ax.set_title(
                    f"MAE={mae:.3f}m  RMSE={rmse:.3f}m\nAbsRel={absrel:.3f}",
                    fontsize=7,
                )
            ax.axis("off")

    plt.tight_layout()
    maps_path = "depth_error_maps.png"
    plt.savefig(maps_path, dpi=150)
    print(f"Saved: {maps_path}")

    # ── Figure 2: per-sample metric bars ──────────────────────────────────────
    maes    = [m[0] for m in all_metrics]
    rmses   = [m[1] for m in all_metrics]
    absrels = [m[2] for m in all_metrics]
    xs = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Per-sample depth error metrics", fontsize=12)

    for ax, vals, ylabel in zip(axes,
                                 [maes, rmses, absrels],
                                 ["MAE (m)", "RMSE (m)", "AbsRel"]):
        ax.bar(xs, vals, color="steelblue", alpha=0.8)
        ax.axhline(np.nanmean(vals), color="red", linestyle="--",
                   label=f"mean={np.nanmean(vals):.3f}")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"s{i}" for i in xs])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    bars_path = "depth_error_bars.png"
    plt.savefig(bars_path, dpi=150)
    print(f"Saved: {bars_path}")

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def compare(args):
    print(f"Sampling {args.n_samples} frames from real: {args.real}")
    real_frames, _ = sample_frames(args.real, args.n_samples)

    print(f"Sampling {args.n_samples} frames from sim : {args.sim}")
    sim_frames, _ = sample_frames(args.sim, args.n_samples)

    print(f"\nReal shape: {real_frames[0].shape}  "
          f"range [{real_frames[0].min():.2f}, {real_frames[0].max():.2f}] m")
    print(f"Sim  shape: {sim_frames[0].shape}  "
          f"range [{sim_frames[0].min():.2f}, {sim_frames[0].max():.2f}] m")

    all_metrics = []
    print(f"\n{'Sample':<8}  {'MAE (m)':<10}  {'RMSE (m)':<10}  {'AbsRel':<10}")
    print("-" * 44)
    for i, (real_m, sim_m) in enumerate(zip(real_frames, sim_frames)):
        mae, rmse, absrel, err_map = compute_metrics(sim_m, real_m)
        all_metrics.append((mae, rmse, absrel, err_map))
        print(f"{i:<8}  {mae:<10.4f}  {rmse:<10.4f}  {absrel:<10.4f}")

    maes    = np.array([m[0] for m in all_metrics])
    rmses   = np.array([m[1] for m in all_metrics])
    absrels = np.array([m[2] for m in all_metrics])
    print("-" * 44)
    print(f"{'mean':<8}  {np.nanmean(maes):<10.4f}  "
          f"{np.nanmean(rmses):<10.4f}  {np.nanmean(absrels):<10.4f}")

    plot_results(real_frames, sim_frames, all_metrics, args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--real",      required=True,  help="Real depth MKV file")
    p.add_argument("--sim",       required=True,  help="Sim depth MKV file")
    p.add_argument("--n-samples", type=int, default=6,
                   help="Number of frame pairs to sample (default: 6)")
    args = p.parse_args()
    compare(args)
