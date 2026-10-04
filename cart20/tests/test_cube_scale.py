"""Known-size cube scale: ray-cast a red cube of known edge into a pointmap and recover the scale (incl. Sim3 s_k)."""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ego_cart20 import cube_scale as CS
from scipy.spatial.transform import Rotation


def render(edge=0.05, units=1.0, yaw=35, pitch=-30, dist=0.25, occlude=0.0, hw=(192, 256), f=180.0, seed=0):
    """camera at origin looking +z; cube of `edge` metres rotated by yaw/pitch, centre at z=dist. Pointmap in MASt3R
    units = metres * units. occlude: fraction of the image (right side) covered by a grey 'hand' in front."""
    H, W = hw; v, u = np.mgrid[0:H, 0:W]; d = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u, float)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    R = Rotation.from_euler("yx", [yaw, pitch], degrees=True).as_matrix(); c = np.array([0, 0, dist])
    o = -c @ R; dl = d @ R                                                     # ray in cube frame
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-edge / 2 - o) / dl; t2 = (edge / 2 - o) / dl
    tn = np.nanmax(np.minimum(t1, t2), -1); tf = np.nanmin(np.maximum(t1, t2), -1); hit = (tn <= tf) & (tn > 0)
    X = np.where(hit[..., None], d * tn[..., None], d * (dist + 0.3) / d[..., 2:3])
    rgb = np.full((H, W, 3), 120, np.uint8); rgb[hit] = (200, 20, 25)
    rng = np.random.default_rng(seed); X = X + rng.normal(0, 0.0004, X.shape)
    if occlude:
        cut = int(W * (1 - occlude)); X[:, cut:] = d[:, cut:] * 0.08 / d[:, cut:, 2:3]; rgb[:, cut:] = (180, 150, 130)
    return (X * units).astype(np.float32), np.full((H, W), 5.0, np.float32), rgb


def test_recovers_scale_with_units_and_sim3():
    X, C, I = render(units=2.7)                                     # MASt3R units = 2.7 x metres
    s_k = 0.8                                                       # keyframe Sim3 scale: world = 0.8 * cam
    r = CS.frame_scale(X, C, I, s_k, None, 2)
    assert r["ok"], r
    want = 1.0 / (2.7 * 0.8)                                        # metres per world unit
    assert abs(r["s_frame"] / want - 1) < 0.08, (r["s_frame"], want)


def test_episode_median_and_partial_occlusion_rejected():
    X, C, I = render(units=1.0, occlude=0.55)                       # hand covers most of the cube
    r = CS.frame_scale(X, C, I, 1.0, None, 2)
    assert not r["ok"] or abs(r["s_frame"] - 1.0) < 0.15            # either rejected or still right, never silently wrong
    import tempfile
    frames = [render(units=2.0, yaw=y, pitch=p, seed=i) for i, (y, p) in enumerate([(30, -25), (40, -35), (25, -30), (45, -20)])]
    with tempfile.TemporaryDirectory() as d:
        np.savez(pathlib.Path(d) / "kf.npz", X_cam=np.stack([f[0] for f in frames]).astype(np.float16), C=np.stack([f[1] for f in frames]),
                 img=np.stack([f[2] for f in frames]), T_WC=np.tile([0, 0, 0, 0, 0, 0, 1, 1.0], (4, 1)), frame_idx=np.arange(4), stride=2,
                 hw=np.array([384, 512]), K=np.eye(3))
        e = CS.episode_scale(pathlib.Path(d) / "kf.npz")
    assert e["valid"] and abs(e["s_cube"] - 0.5) / 0.5 < 0.08, CS.strip(e)


def test_no_cube_is_invalid():
    X, C, I = render(); I[:] = 120
    assert not CS.frame_scale(X, C, I, 1.0, None, 2)["ok"]




def test_pnp_gate_directions():
    """>= MIN_FRAMES valid frames passes, fewer fails; small spread / residual pass (guards against a flipped inequality)"""
    from ego_cart20 import cube_pnp as CP
    rng = np.random.default_rng(1); n = CP.MIN_FRAMES; k = 2.0                     # world units per metre
    c = np.array([0.3, -0.1, 0.5]); p = rng.normal(0, 0.3, (n, 3))
    R = np.tile(np.eye(3), (n, 1, 1)); t = (c - p) / k                              # metric cube-centre vectors, R = I
    u = np.einsum("nij,nj->ni", R, t); fs = CP.fit_scale(p, u)
    assert abs(fs["s"] - 1 / k) < 1e-6 and fs["centre_resid_m"] < 1e-6
    assert n == 10 and CP.MAX_REL_SPREAD == 0.10 and CP.MAX_CENTRE_RESID_M == 0.02


if __name__ == "__main__":
    for f in (test_recovers_scale_with_units_and_sim3, test_episode_median_and_partial_occlusion_rejected, test_no_cube_is_invalid, test_pnp_gate_directions): f(); print("PASS", f.__name__)
