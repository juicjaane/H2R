"""
Unit test for surface.py plane fitting and coordinate transforms.
No camera required. Run:  python scripts/test_surface_math.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.calibration.surface import (
    pixel_to_3d, fit_plane, build_table_transform,
    cam_to_table, table_to_cam,
)

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(name, condition, detail=""):
    tag = PASS if condition else FAIL
    print(f"  [{tag}]  {name}" + (f"  ({detail})" if detail else ""))
    return condition


def test_pixel_to_3d():
    print("\n--- pixel_to_3d ---")
    # At principal point, z should equal depth
    p = pixel_to_3d(320, 240, 1.0, fx=600, fy=600, cx=320, cy=240)
    check("principal point gives (0,0,1)", np.allclose(p, [0, 0, 1], atol=1e-9))

    # Right of principal point
    p = pixel_to_3d(320 + 60, 240, 1.0, fx=600, fy=600, cx=320, cy=240)
    check("60px right gives X=0.1m", np.isclose(p[0], 0.1, atol=1e-9))


def test_fit_plane_horizontal():
    print("\n--- fit_plane: perfect horizontal table ---")
    # 4 corners of a flat table at Z=1.0m in camera frame
    # (in camera frame, Z=1.0 means 1m away from camera)
    corners = np.array([
        [-0.3, -0.2, 1.0],
        [ 0.3, -0.2, 1.0],
        [ 0.3,  0.2, 1.0],
        [-0.3,  0.2, 1.0],
    ])
    plane = fit_plane(corners)

    # Normal should be [0, 0, -1] (pointing toward camera = -Z direction in cam frame)
    check("normal ≈ [0,0,-1]",
          np.allclose(np.abs(plane["normal"]), [0, 0, 1], atol=1e-10),
          f"got {plane['normal'].round(6)}")

    # All residuals should be ~0
    check("all residuals < 0.01mm",
          plane["max_residual_m"] < 1e-4,
          f"max={plane['max_residual_m']:.2e}")

    # Origin = centroid = [0, 0, 1]
    check("origin = centroid",
          np.allclose(plane["origin"], [0, 0, 1.0], atol=1e-10),
          f"got {plane['origin'].round(6)}")


def test_fit_plane_noisy():
    print("\n--- fit_plane: 4 corners with ±3mm noise ---")
    rng = np.random.default_rng(42)
    corners = np.array([
        [-0.3, -0.2, 1.0],
        [ 0.3, -0.2, 1.0],
        [ 0.3,  0.2, 1.0],
        [-0.3,  0.2, 1.0],
    ])
    noisy = corners + rng.normal(0, 0.003, corners.shape)
    plane = fit_plane(noisy)
    check("noisy corners: max residual < 5mm",
          plane["max_residual_m"] < 0.005,
          f"got {plane['max_residual_m']*1000:.2f}mm")


def test_build_table_transform():
    print("\n--- build_table_transform ---")
    corners = np.array([
        [-0.3, -0.2, 1.0],   # TL
        [ 0.3, -0.2, 1.0],   # TR
        [ 0.3,  0.2, 1.0],   # BR
        [-0.3,  0.2, 1.0],   # BL
    ])
    plane = fit_plane(corners)
    T = build_table_transform(corners, plane)

    check("T is 4×4", T.shape == (4, 4))

    # The origin (centroid) should map to (0,0,0) in table frame
    origin_in_table = cam_to_table(plane["origin"], T)
    check("origin → (0,0,0) in table frame",
          np.allclose(origin_in_table, [0, 0, 0], atol=1e-9),
          f"got {origin_in_table.round(6)}")

    # TL corner in table frame: should have y≈0, z≈0
    tl_table = cam_to_table(corners[0], T)
    check("TL z≈0 in table frame",
          abs(tl_table[2]) < 1e-8,
          f"z={tl_table[2]:.2e}")

    # TR-TL distance should be preserved
    tr_table = cam_to_table(corners[1], T)
    tl_table = cam_to_table(corners[0], T)
    dist_cam   = np.linalg.norm(corners[1] - corners[0])
    dist_table = np.linalg.norm(tr_table - tl_table)
    check("TL→TR distance preserved",
          np.isclose(dist_cam, dist_table, atol=1e-9),
          f"cam={dist_cam:.4f}m  table={dist_table:.4f}m")

    # X-axis of table should align with TL→TR
    tl_t = cam_to_table(corners[0], T)
    tr_t = cam_to_table(corners[1], T)
    direction = (tr_t - tl_t)
    direction = direction / np.linalg.norm(direction)
    check("TL→TR aligns with table X-axis",
          np.allclose(direction, [1, 0, 0], atol=1e-9),
          f"got {direction.round(6)}")


def test_roundtrip():
    print("\n--- cam↔table roundtrip ---")
    corners = np.array([
        [-0.3, -0.2, 1.0],
        [ 0.3, -0.2, 1.0],
        [ 0.3,  0.2, 1.0],
        [-0.3,  0.2, 1.0],
    ])
    plane = fit_plane(corners)
    T = build_table_transform(corners, plane)

    pts = np.random.default_rng(0).uniform(-0.5, 0.5, (10, 3)) + [0, 0, 1]
    for p in pts:
        p_table = cam_to_table(p, T)
        p_back  = table_to_cam(p_table, T)
        if not np.allclose(p, p_back, atol=1e-9):
            check("roundtrip", False, f"max err {np.max(np.abs(p - p_back)):.2e}")
            return
    check("10 random points roundtrip < 1e-9", True)


if __name__ == "__main__":
    test_pixel_to_3d()
    test_fit_plane_horizontal()
    test_fit_plane_noisy()
    test_build_table_transform()
    test_roundtrip()
    print("\nDone.\n")
