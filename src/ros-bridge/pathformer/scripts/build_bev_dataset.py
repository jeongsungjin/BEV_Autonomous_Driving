#!/usr/bin/env python3

"""Generate BEV raster sequence dataset for Pathformer training.

Usage:
    python build_bev_dataset.py \
        --bev_dir ~/.ros/bev_rasters \
        --pose_csv poses.csv \
        --seq_len 8 --pred_len 12 \
        --out_path bev_dataset.npz

Requirements:
    - numpy, pandas
    - Raster .npy files named <timestamp>.npy (float seconds)
    - pose CSV with columns: timestamp,x,y (same time base)
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_bev_rasters(bev_dir: Path, stride: int = 1, resize: tuple = None):
    files = sorted(bev_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy raster files found in {bev_dir}")
    # Apply stride
    files = files[::stride]
    timestamps = np.array([float(f.stem) for f in files])
    bev_arrays = []
    for f in files:
        arr = np.load(f)
        if resize is not None:
            import cv2
            C, H, W = arr.shape
            new_w, new_h = resize
            resized = np.zeros((C, new_h, new_w), dtype=arr.dtype)
            for c in range(C):
                resized[c] = cv2.resize(arr[c], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            arr = resized
        bev_arrays.append(arr)
    bev_stack = np.stack(bev_arrays)  # (N, C, H, W)
    return timestamps, bev_stack


def load_pose_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    if not set(["timestamp", "x", "y"]).issubset(df.columns):
        raise ValueError("CSV must contain columns: timestamp, x, y")
    df = df.sort_values("timestamp")
    return df["timestamp"].values, df[["x", "y"]].values


def interpolate_positions(target_ts, pose_ts, pose_xy):
    """Linearly interpolate x, y to match target timestamps."""
    xs = np.interp(target_ts, pose_ts, pose_xy[:, 0])
    ys = np.interp(target_ts, pose_ts, pose_xy[:, 1])
    return np.stack([xs, ys], axis=1)


def build_sequences(bev, positions, seq_len, pred_len, channel_weights):
    N = bev.shape[0]
    samples = N - seq_len - pred_len + 1
    if samples <= 0:
        raise ValueError("Not enough frames to build any sequence."
                         f" Need >={seq_len + pred_len}, got {N}")
    X_list = []
    Y_list = []
    for i in range(samples):
        X_seq = bev[i:i + seq_len]  # (seq_len, C, H, W)
        # apply channel weights once per frame
        w = np.array(channel_weights, dtype=np.float32).reshape(1, 3, 1, 1)  # broadcast over seq_len
        X_seq = np.clip(X_seq * w, 0, 255).astype(np.uint8)
        start_pos = positions[i]
        future_pos = positions[i + seq_len:i + seq_len + pred_len] - start_pos  # relative trajectory
        X_list.append(X_seq)
        Y_list.append(future_pos)
    return np.stack(X_list), np.stack(Y_list)  # X:(M,seq_len,C,H,W), Y:(M,pred_len,2)


def main():
    parser = argparse.ArgumentParser("Build BEV dataset for Pathformer")
    parser.add_argument("--bev_dir", type=str, required=True, help="Directory of raster .npy files")
    parser.add_argument("--pose_csv", type=str, required=True, help="CSV with timestamp,x,y")
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--frame_stride", type=int, default=1, help="Use every k-th frame to reduce dataset size")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("W", "H"), help="Resize BEV raster to WxH")
    parser.add_argument("--pos_scale", type=float, default=20.0, help="Divide future positions by this scale factor")
    parser.add_argument("--channel_weights", nargs=3, type=float, metavar=("W_DET","W_DA","W_LL"), default=[1.0,1.0,3.0], help="Weights for det, da, ll channels")
    parser.add_argument("--out_path", type=str, default="bev_dataset.npz")
    args = parser.parse_args()

    bev_dir = Path(os.path.expanduser(args.bev_dir))
    pose_csv = Path(os.path.expanduser(args.pose_csv))

    print(f"Loading BEV rasters from {bev_dir} ...")
    ts_bev, bev = load_bev_rasters(bev_dir, args.frame_stride, tuple(args.resize) if args.resize else None)
    print(f"Loaded {bev.shape[0]} frames, shape per frame {bev.shape[1:]}.")

    print(f"Loading pose CSV {pose_csv} ...")
    ts_pose, pose_xy = load_pose_csv(pose_csv)

    print("Interpolating poses to raster timestamps ...")
    positions = interpolate_positions(ts_bev, ts_pose, pose_xy)

    print("Building sequences ...")
    X, Y = build_sequences(bev, positions, args.seq_len, args.pred_len, args.channel_weights)
    if args.pos_scale != 0:
        Y = Y / args.pos_scale
    print(f"Generated dataset: X{X.shape}, Y{Y.shape}")

    np.savez_compressed(args.out_path, X=X, Y=Y, pos_scale=np.array([args.pos_scale], dtype=np.float32))
    print(f"Saved dataset to {args.out_path}")


if __name__ == "__main__":
    main() 