"""Fisheye calibration coverage: a centre-only set of views produces a confidently wrong D on a 160 deg lens.

These pin the two things that make the intrinsics trustworthy rather than merely produced: the coverage accounting
that refuses a centre-only set, and the provenance that ties a calibration to one physical camera and one frozen
imaging preset."""
import numpy as np
import pytest
import yaml
from handumi_collector.pose.calibration import write_fisheye
from handumi_collector.tools.calibrate_fisheye import DIST_BINS, GRID, coverage_of, coverage_report


def _board(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)


def test_coverage_places_a_board_by_centroid_and_apparent_size():
    assert coverage_of(_board(100, 100, 200, 200), (1920, 1080))[:2] == (0, 0)      # top-left cell
    assert coverage_of(_board(1700, 950, 1900, 1050), (1920, 1080))[:2] == (2, 2)   # bottom-right cell
    assert coverage_of(_board(100, 100, 200, 200), (1920, 1080))[2] == "far"        # small in frame => far
    assert coverage_of(_board(700, 300, 1700, 900), (1920, 1080))[2] == "near"      # large in frame => near


def test_coverage_report_names_exactly_what_is_missing():
    cov = dict(grid=[[0] * GRID for _ in range(GRID)], dist={d: 0 for d in DIST_BINS})
    cov["grid"][1][1] = 3; cov["dist"]["mid"] = 3                                   # the centre-only mistake
    cells, dists = coverage_report(cov)
    assert len(cells) == GRID * GRID - 1 and "r1c1" not in cells
    assert dists == ["near", "far"]
    for r in range(GRID):
        for c in range(GRID):
            cov["grid"][r][c] = 1
    for d in DIST_BINS:
        cov["dist"][d] = 1
    assert coverage_report(cov) == ([], [])


def test_intrinsics_file_carries_the_imaging_state_it_is_valid_under(tmp_path):
    """Intrinsics are tied to the resolution AND the exposure they were measured under. A file that does not say which
    camera and which settings produced it cannot be checked against a live device later."""
    p = write_fisheye("left", np.eye(3), np.zeros(4), (1920, 1080), source="test",
                      extra=dict(product_name="FisheyeCamLeft", reported_serial="UC684", resolution=[1920, 1080],
                                 fps=30, imaging_preset={"exposure-time-abs": "20", "gain": "0"},
                                 calibration_version="exposure_v001", coverage_complete=True),
                      cal_dir=tmp_path)
    d = yaml.safe_load(p.read_text())
    assert d["side"] == "left" and d["image_size"] == [1920, 1080] and d["model"] == "kannala_brandt"
    assert d["product_name"] == "FisheyeCamLeft" and d["reported_serial"] == "UC684"
    assert d["imaging_preset"]["exposure-time-abs"] == "20" and d["calibration_version"] == "exposure_v001"
    assert d["coverage_complete"] is True
    p2 = write_fisheye("left", np.eye(3), np.zeros(4), (1920, 1080), cal_dir=tmp_path)   # extra stays optional
    assert yaml.safe_load(p2.read_text())["side"] == "left" and p2 != p


def test_valid_radius_masks_the_extrapolated_periphery():
    """A fisheye calibrated from a board that never reaches the image corners must not be used out there."""
    import numpy as np
    from handumi_collector.pose.virtual_view import VirtualPinholeView

    K = np.array([[780.0, 0, 960.0], [0, 780.0, 540.0], [0, 0, 1.0]])
    D = np.array([-0.02, -0.004, 0.003, -0.001])
    K_t = np.array([[120.0, 0, 320.0], [0, 120.0, 240.0], [0, 0, 1.0]])  # virtual view wide enough to reach past r=900

    unmasked = VirtualPinholeView(dict(model="kannala_brandt", K=K, D=D), K_t, (640, 480))
    assert unmasked.valid_radius_px is None and unmasked.valid_fraction == 1.0

    masked = VirtualPinholeView(dict(model="kannala_brandt", K=K, D=D, valid_radius_px=900.0), K_t, (640, 480))
    assert 0.0 < masked.valid_fraction < 1.0, "a wide virtual view must lose its extrapolated corners"

    img = np.full((1080, 1920, 3), 200, np.uint8)
    out_m, out_u = masked.render(img), unmasked.render(img)
    assert (out_m == 0).any(), "masked region must render as border, not as guessed pixels"
    assert (out_m == 0).sum() > (out_u == 0).sum()
    # the centre, which the board did cover, is untouched
    assert (out_m[220:260, 300:340] == out_u[220:260, 300:340]).all()


def test_frozen_wrist_calibrations_declare_a_valid_radius():
    import yaml
    from pathlib import Path

    for side in ("left", "right"):
        p = Path(f"configs/calibration/fisheye_{side}_v001.yaml")
        if not p.exists():
            continue
        d = yaml.safe_load(p.read_text())
        assert d.get("valid_radius_px"), f"{p} has no valid_radius_px — the periphery would be silently extrapolated"
        vr = d["valid_radius"]
        assert vr["valid_radius_px"] <= vr["r_observed_px"] + 1e-6, "cannot trust further out than anything was observed"
        assert vr["valid_radius_px"] < vr["r_corner_px"], "board never reached the image corner; claiming it did is wrong"


def _fisheye_intr(valid_radius_px=None):
    import numpy as np
    d = dict(model="kannala_brandt", K=np.array([[780.0, 0, 960.0], [0, 780.0, 540.0], [0, 0, 1.0]]),
             D=np.array([-0.02, -0.004, 0.003, -0.001]), image_size=(1920, 1080))
    if valid_radius_px is not None:
        d["valid_radius_px"] = valid_radius_px
    return d


def test_vo_rectifier_is_unchanged_where_nothing_is_masked():
    """Switching the remap to float32 so the source radius stays readable must not alter the image."""
    import numpy as np
    from handumi_collector.pose.backends.opencv_vo import FisheyeRectifier

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    a = FisheyeRectifier(_fisheye_intr(), out_size=(640, 480), fov_deg=90.0)
    b = FisheyeRectifier(_fisheye_intr(valid_radius_px=900.0), out_size=(640, 480), fov_deg=90.0)
    assert b.valid_fraction == 1.0, "a 90 deg rectification never reaches r=900, so nothing should be cut"
    assert np.array_equal(a.rectify(img), b.rectify(img))


def test_vo_rectifier_cuts_the_extrapolated_periphery_before_detection():
    """The wrist VO path — not just the virtual view — must refuse the region where D was extrapolated."""
    import numpy as np
    from handumi_collector.pose.backends.opencv_vo import FisheyeRectifier, OpenCvVoBackend

    wide = FisheyeRectifier(_fisheye_intr(valid_radius_px=900.0), out_size=(640, 480), fov_deg=160.0)
    assert 0.0 < wide.valid_fraction < 1.0, "a 160 deg rectification does reach past r=900"
    assert wide.valid_mask is not None and (wide.valid_mask == 0).any()

    unguarded = FisheyeRectifier(_fisheye_intr(), out_size=(640, 480), fov_deg=160.0)
    assert unguarded.valid_mask is None and unguarded.valid_fraction == 1.0

    # the mask is eroded, so the black seam itself is outside the detectable area
    ys, xs = np.where(wide.valid_mask > 0)
    r = np.hypot(wide._maps[0][ys, xs] - 960.0, wide._maps[1][ys, xs] - 540.0)
    assert r.max() <= 900.0, "every detectable pixel must come from a measured source radius"

    vo = OpenCvVoBackend(rect_fov_deg=160.0, rect_size=(640, 480))
    vo.initialize(intrinsics=_fisheye_intr(valid_radius_px=900.0))
    gray = np.full((480, 640), 127, np.uint8)
    gray[::7, ::7] = 255                      # texture everywhere, including the masked corners
    pts = vo._detect(gray, None)
    if len(pts):
        u = pts[:, 0].astype(int); v = pts[:, 1].astype(int)
        assert (vo._rect.valid_mask[v, u] > 0).all(), "detector placed features in the extrapolated region"


def test_cross_view_keypoints_never_come_from_the_extrapolated_region():
    """Section 11: the frozen valid_radius must bind on the cross-view path too, before PnP sees anything."""
    import cv2
    import numpy as np
    from handumi_collector.pose.cross_view import match_features

    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)   # texture everywhere
    mask = np.zeros((480, 640), np.uint8)
    mask[120:360, 160:480] = 255                                 # the only region a feature may be taken from

    _, pts_b, _, n_kp_b = match_features(img, img, mask_b=mask)
    assert n_kp_b > 0, "the masked region must still yield keypoints, or the test proves nothing"
    kb = cv2.SIFT_create(nfeatures=3000).detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
    xy = np.array([k.pt for k in kb])
    assert ((xy[:, 0] >= 160) & (xy[:, 0] < 480) & (xy[:, 1] >= 120) & (xy[:, 1] < 360)).all(), \
        "SIFT placed a keypoint outside the mask — the extrapolated periphery would reach PnP"

    _, _, _, n_unmasked = match_features(img, img)
    assert n_unmasked > n_kp_b, "without the mask the whole frame is used, so the mask is doing real work"
