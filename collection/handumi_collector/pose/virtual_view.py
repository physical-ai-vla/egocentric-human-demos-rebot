"""C922-like virtual wrist view from the Arducam fisheye (design freeze 2026-09-10):

    Arducam 160° RAW (kept as-is for VIO)  →  fisheye calibration  →  virtual camera with K_target (+ D_target: the robot wrist
    C922's own mild distortion, so the rendered image looks like a RAW C922 frame)  →  optional pure rotation R (virtual camera
    orientation relative to the Arducam)  →  shared X-VLA wrist observation with the robot wrist C922.

Intrinsics/FOV/distortion and a pure rotation offset are matched exactly; a translation offset between the two optical centres
cannot be (parallax) — keep the physical mount pose similar (tools.view_match_qa measures the residual E_view in pixels).
Target K/D/size/R live in a versioned file configs/calibration/virtual_wrist_vNNN.yaml (per side; never hard-coded)."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .calibration import load_versioned, save_versioned

C922_1080P_K = np.array([[1577.7277664, 0.0, 937.6144422], [0.0, 1586.1210734, 440.7252291], [0.0, 0.0, 1.0]])   # calibration/c922_2026-09-03.yaml (head C922)
SIDES = ("left", "right")


def c922_like_target(width: int = 640, height: int = 480, K_1080p: np.ndarray = C922_1080P_K) -> np.ndarray:
    """Default target: the measured C922 1080p intrinsics scaled to a 4:3 640×480 mode keeping the vertical FOV
    (horizontal centre crop). This is an ESTIMATE — measure the robot wrist C922 at 640×480 (tools.calibrate_c922) → v002."""
    s = height / 1080.0; crop_x = (1920 - 1080 * width / height) / 2
    return np.array([[K_1080p[0, 0] * s, 0, (K_1080p[0, 2] - crop_x) * s], [0, K_1080p[1, 1] * s, K_1080p[1, 2] * s], [0, 0, 1]])


def _per_side(v, default):
    """Accept a single value or {left:…, right:…}."""
    if isinstance(v, dict) and set(v) & set(SIDES): return {s: (np.asarray(v[s], np.float64) if v.get(s) is not None else default) for s in SIDES}
    a = default if v is None else np.asarray(v, np.float64)
    return {s: a for s in SIDES}


def load_virtual_wrist(which: str = "latest", cal_dir: Path | None = None) -> tuple[str | None, dict]:
    """→ (version, {K_target: {side: 3x3}, D_target: {side: (n,) or None}, size, rpy_deg: {side: [3]}, source})."""
    v, d = load_versioned("virtual_wrist", which, cal_dir)
    if not d:
        return None, dict(K_target={s: c922_like_target() for s in SIDES}, D_target={s: None for s in SIDES}, size=(640, 480),
                          rpy_deg={s: [0, 0, 0] for s in SIDES}, source="default c922_like_target() estimate")
    K = _per_side(d["K_target"], c922_like_target()); K = {s: k.reshape(3, 3) for s, k in K.items()}
    Dt = d.get("D_target"); D = {s: None for s in SIDES}
    if Dt is not None:
        D = {s: (np.asarray(Dt[s], np.float64).reshape(-1) if isinstance(Dt, dict) and Dt.get(s) is not None else (None if isinstance(Dt, dict) else np.asarray(Dt, np.float64).reshape(-1))) for s in SIDES}
    return v, dict(K_target=K, D_target=D, size=tuple(int(x) for x in d["size"]), rpy_deg={s: list((d.get("rpy_deg") or {}).get(s, [0, 0, 0])) for s in SIDES}, source=d.get("source", ""))


def write_virtual_wrist(K_target, size, *, D_target=None, rpy_deg=None, source: str = "", notes: str = "", cal_dir: Path | None = None) -> Path:
    """K_target / D_target: one array or {left, right}. Writes the next virtual_wrist_vNNN.yaml."""
    def enc(v):
        if v is None: return None
        if isinstance(v, dict): return {s: (None if v.get(s) is None else np.asarray(v[s], float).tolist()) for s in SIDES}
        return np.asarray(v, float).tolist()
    return save_versioned("virtual_wrist", dict(schema="handumi_virtual_wrist/v2", K_target=enc(K_target), D_target=enc(D_target), size=[int(size[0]), int(size[1])],
                                                rpy_deg=rpy_deg or {s: [0, 0, 0] for s in SIDES}, source=source, notes=notes), cal_dir=cal_dir)


def _rays_from_target(K_t, D_t, size) -> np.ndarray:
    """Unit-free normalised rays (x, y, 1) for every target pixel, undoing the target's own distortion when given."""
    w, h = size; u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    D = np.zeros(5) if D_t is None else np.asarray(D_t, np.float64).reshape(-1)
    n = cv2.undistortPoints(pts, np.asarray(K_t, np.float64), D).reshape(-1, 2)
    return np.concatenate([n, np.ones((len(n), 1))], axis=1)



def valid_radius_maps(map_x, map_y, K_src, valid_radius_px, *, erode_px: int = 16):
    """Given a remap that pulls from a fisheye source, mark which target pixels are fed from a source radius where the
    distortion model was actually MEASURED, and blank the rest.

    Returns (map_x, map_y, mask, valid_fraction) with the out-of-range entries sent off-image so cv2.remap renders them
    as border.  The mask is eroded by `erode_px` because the resulting black boundary is itself a superb corner: feature
    detectors would happily latch onto the seam and report motion of an artefact.  Pass the mask to the detector rather
    than relying on the blanking alone -- that is the whole point of masking BEFORE feature extraction instead of
    discarding bad features afterwards, by which time a biased bearing has already entered PnP.
    """
    if valid_radius_px is None:
        return map_x, map_y, np.full(map_x.shape, 255, np.uint8), 1.0
    K_src = np.asarray(K_src, np.float64)
    r = np.hypot(map_x - K_src[0, 2], map_y - K_src[1, 2])
    inside = r <= float(valid_radius_px)
    mx, my = map_x.copy(), map_y.copy()
    mx[~inside] = -1e4; my[~inside] = -1e4
    mask = (inside.astype(np.uint8)) * 255
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        mask = cv2.erode(mask, k)
    return mx, my, mask, float(inside.mean())


class VirtualPinholeView:
    """Remap a fisheye (kannala_brandt) or pinhole source into a virtual camera with K_target / D_target / size / rotation R
    (R = orientation of the virtual camera in the source camera frame; identity = same optical axis)."""

    def __init__(self, src_intrinsics: dict, K_target, size: tuple[int, int], rpy_deg=(0, 0, 0), D_target=None) -> None:
        K = np.asarray(src_intrinsics["K"], np.float64); D = np.asarray(src_intrinsics.get("D", np.zeros(4)), np.float64).reshape(-1)
        self.K_target = np.asarray(K_target, np.float64).reshape(3, 3); self.D_target = None if D_target is None else np.asarray(D_target, np.float64).reshape(-1)
        self.size = (int(size[0]), int(size[1])); self.rpy_deg = tuple(float(x) for x in rpy_deg)
        self.R = Rotation.from_euler("xyz", np.radians(np.asarray(rpy_deg, float))).as_matrix()
        model = src_intrinsics.get("model", "kannala_brandt")
        rays_v = _rays_from_target(self.K_target, self.D_target, self.size)          # rays in the VIRTUAL camera frame
        rays_s = (self.R @ rays_v.T).T                                                 # same rays expressed in the SOURCE camera frame
        rays_s = rays_s.reshape(-1, 1, 3)
        rvec = tvec = np.zeros(3)
        if model == "pinhole":
            Dsrc = np.zeros(5); Dsrc[:min(5, len(D))] = D[:5]
            px, _ = cv2.projectPoints(rays_s, rvec, tvec, K, Dsrc)
        else:
            behind = rays_s[:, 0, 2] <= 1e-6
            px, _ = cv2.fisheye.projectPoints(rays_s, rvec, tvec, K, D[:4])
            px[behind] = -1e4
        px = px.reshape(self.size[1], self.size[0], 2).astype(np.float32)
        # Drop target pixels fed by a source radius where D was extrapolated, not measured (see estimate_valid_radius in
        # scripts/charuco_calibrate.py). A 160 deg board calibration never reaches the image corners, and the 9th-order
        # polynomial out there is confidently wrong with nothing in the reprojection error to show it. Sourcing a feature
        # from that region yields a biased bearing and so a biased pose, silently -- so the pixels are removed instead.
        self.valid_radius_px = src_intrinsics.get("valid_radius_px")
        self._map_x, self._map_y, self.valid_mask, self.valid_fraction = valid_radius_maps(
            px[..., 0].copy(), px[..., 1].copy(), K, self.valid_radius_px)

    def render(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self._map_x, self._map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.size[0] / (2 * self.K_target[0, 0]))))


def fit_rotation(p_virtual: np.ndarray, p_robot: np.ndarray, K_t, D_t=None, rpy0_deg=(0, 0, 0)) -> dict:
    """Residual pure rotation that maps points seen in the current virtual view onto the robot camera's pixels:
    minimise Σ‖proj(K_t, D_t, R_fit · ray(p_virtual)) − p_robot‖². Returns rpy_deg to ADD to the current rpy (compose), plus
    before/after E_view. Translation (parallax) cannot be absorbed — a large residual after the fit means: move the mount."""
    K_t = np.asarray(K_t, np.float64); D = np.zeros(5) if D_t is None else np.asarray(D_t, np.float64).reshape(-1)
    pv = np.asarray(p_virtual, np.float64).reshape(-1, 1, 2); pr = np.asarray(p_robot, np.float64).reshape(-1, 2)
    n = cv2.undistortPoints(pv, K_t, D).reshape(-1, 2); rays = np.concatenate([n, np.ones((len(n), 1))], axis=1)
    def proj(rpy):
        R = Rotation.from_euler("xyz", np.radians(rpy)).as_matrix(); r = (R @ rays.T).T.reshape(-1, 1, 3)
        q, _ = cv2.projectPoints(r, np.zeros(3), np.zeros(3), K_t, D); return q.reshape(-1, 2)
    e0 = float(np.linalg.norm(proj(np.zeros(3)) - pr, axis=1).mean())
    sol = least_squares(lambda x: (proj(x) - pr).ravel(), np.asarray(rpy0_deg, float), method="lm" if len(pr) >= 3 else "trf")
    e1 = float(np.linalg.norm(proj(sol.x) - pr, axis=1).mean())
    return dict(rpy_delta_deg=sol.x.tolist(), E_view_before_px=e0, E_view_after_px=e1, n_points=int(len(pr)))


def compose_rpy(rpy_current_deg, rpy_delta_deg) -> list[float]:
    """rpy of R_current · R_delta (delta expressed in the current virtual frame)."""
    R = Rotation.from_euler("xyz", np.radians(rpy_current_deg)) * Rotation.from_euler("xyz", np.radians(rpy_delta_deg))
    return np.degrees(R.as_euler("xyz")).tolist()
