import numpy as np
import struct
import torch
import dwm.functional


def preprocess_points(batch, device):
    mhv = dwm.functional.make_homogeneous_vector
    return [
        [
            (mhv(p_j.to(device)) @ t_j.permute(1, 0))[:, :3]
            for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
        ]
        for p_i, t_i in zip(
            batch["lidar_points"], batch["lidar_transforms"].to(device))
    ]


def postprocess_points(batch, ego_space_points):
    return [
        [
            (
                dwm.functional.make_homogeneous_vector(p_j.cpu()) @
                torch.linalg.inv(t_j).permute(1, 0)
            )[:, :3]
            for p_j, t_j in zip(p_i, t_i.flatten(0, 1))
        ]
        for p_i, t_i in zip(
            ego_space_points, batch["lidar_transforms"])
    ]


def voxels2points(grid_size, voxels):
    interval = torch.tensor([grid_size["interval"]])
    min = torch.tensor([grid_size["min"]])
    return [
        [
            torch.nonzero(v_j).flip(-1).cpu() * interval + min
            for v_j in v_i
        ]
        for v_i in voxels
    ]


def points_to_range_image(
    points: np.ndarray,
    height: int = 64,
    width: int = 1024,
    fov_up: float = 10.0,
    fov_down: float = -30.0,
    max_range: float = 80.0,
) -> np.ndarray:
    """
    Spherical projection of a point cloud to a range image.

    Args:
        points: (N, 3) float32 array of x, y, z coordinates.
        height: number of vertical bins.
        width:  number of horizontal bins.
        fov_up: upper elevation limit in degrees (default 10° for NuScenes).
        fov_down: lower elevation limit in degrees (default -30° for NuScenes).
        max_range: range used to normalise the uint8 output (metres).

    Returns:
        uint8 array of shape (height, width) with range values mapped to
        [0, 255]. Pixels with no return are 0.
    """
    fov_up_rad = np.radians(fov_up)
    fov_down_rad = np.radians(fov_down)
    fov_rad = abs(fov_up_rad) + abs(fov_down_rad)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    valid = r > 0.0
    x, y, z, r = x[valid], y[valid], z[valid], r[valid]
    if len(r) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    # Horizontal angle: clockwise from +x axis → [0, 1)
    yaw = -np.arctan2(y, x)
    u = (0.5 * (yaw / np.pi + 1.0) * width).astype(np.int32)
    u = np.clip(u, 0, width - 1)

    # Vertical angle: map elevation to row index
    pitch = np.arcsin(np.clip(z / r, -1.0, 1.0))
    v = ((1.0 - (pitch - fov_down_rad) / fov_rad) * height).astype(np.int32)
    v = np.clip(v, 0, height - 1)

    range_image = np.zeros((height, width), dtype=np.float32)
    # When multiple points map to the same pixel keep the closest
    order = np.argsort(r)[::-1]
    range_image[v[order], u[order]] = r[order]

    return np.clip(range_image / max_range * 255.0, 0, 255).astype(np.uint8)


def write_pcd_binary(path: str, xyz: np.ndarray) -> None:
    """Write an (N, 3) float32 array as a binary PCD file."""
    xyz = xyz.astype(np.float32)
    n = len(xyz)
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
