from __future__ import annotations

import numpy as np

from handumi.processing.tag_mask import corners_from_row, dilate_quad, mask_tags


def _image():
    img = np.full((120, 160, 3), 200, dtype=np.uint8)
    img[40:80, 60:100] = 0  # black "tag"
    return img


def test_mask_fill_removes_tag_with_border_colour():
    img = _image()
    quad = np.array([[60, 40], [99, 40], [99, 79], [60, 79]], dtype=float)
    out = mask_tags(img, [quad], method="fill", margin=0.1)
    assert out[60, 80].tolist() == [200, 200, 200]
    assert (out[0:30] == 200).all()  # untouched elsewhere
    assert img[60, 80, 0] == 0  # input not modified


def test_mask_blur_and_noise_change_region_only():
    img = _image()
    quad = np.array([[60, 40], [99, 40], [99, 79], [60, 79]], dtype=float)
    for method in ("blur", "noise"):
        out = mask_tags(img, [quad], method=method, margin=0.0)
        assert (out[:30] == img[:30]).all()
        assert not (out[45:75, 65:95] == img[45:75, 65:95]).all()


def test_dilate_and_corners_from_row():
    quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    big = dilate_quad(quad, 0.5)
    assert np.allclose(big.mean(axis=0), [5, 5]) and np.isclose(big[1, 0] - big[0, 0], 15.0)
    row = {
        "observation.apriltag.left_corners_px": np.concatenate([quad.reshape(-1), np.full(24, np.nan)]).astype(np.float32),
        "observation.apriltag.right_corners_px": np.full(32, np.nan, dtype=np.float32),
        "observation.apriltag.world_corners_px": np.concatenate([quad.reshape(-1) + 100, quad.reshape(-1) + 200, np.full(48, np.nan)]).astype(np.float32),
    }
    quads = corners_from_row(row)
    assert len(quads) == 3 and quads[0][0].tolist() == [100, 100]  # world quads come first
