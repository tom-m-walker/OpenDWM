"""
Convert NuScenes-format .bin point cloud files to .pcd (binary).

The .bin files written by LidarDiffusionPipeline.save_results() are raw
float32 with shape [N, 5] (x, y, z, 0, 0). This script walks an output
directory, converts every .bin to a .pcd alongside it, and optionally
removes the originals.

Usage:
    # Convert all .bin files under a directory tree:
    python examples/bin_to_pcd.py -i output/lidar_nuscenes_eval

    # Also delete the .bin files after conversion:
    python examples/bin_to_pcd.py -i output/lidar_nuscenes_eval --delete-bin
"""

import argparse
import glob
import os
import struct
import numpy as np


def write_pcd_binary(path: str, xyz: np.ndarray):
    """Write xyz (float32, shape [N, 3]) as a binary PCD file."""
    n = len(xyz)
    xyz = xyz.astype(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(xyz.tobytes())


def convert(bin_path: str, delete_after: bool = False):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    xyz = points[:, :3]
    pcd_path = bin_path.replace(".bin", ".pcd")
    write_pcd_binary(pcd_path, xyz)
    if delete_after:
        os.remove(bin_path)
    return pcd_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True,
                        help="Root directory to search for .bin files.")
    parser.add_argument("--delete-bin", action="store_true",
                        help="Remove .bin files after conversion.")
    args = parser.parse_args()

    bin_files = glob.glob(
        os.path.join(args.input_dir, "**", "*.bin"), recursive=True)

    if not bin_files:
        print("No .bin files found under", args.input_dir)
    else:
        for bf in sorted(bin_files):
            pcd = convert(bf, delete_after=args.delete_bin)
            print(f"{bf} -> {pcd}")
        print(f"Converted {len(bin_files)} file(s).")
