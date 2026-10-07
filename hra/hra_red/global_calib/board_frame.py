"""[2026-10-06 user] shared TABLE frame from a fixed checkerboard (8x5 inner corners, 25 mm squares).
  cam            detect the board in the global frame -> T_cam_board (solvePnP with global_cam_intrinsics.json); writes an image with
                 the inner corners labelled (index = row * 8 + col, origin = corner 0) -> board_labels.jpg
  touch <idx>    the right gripper tip is ON inner corner <idx>: record the FK TCP (base, m)
  solve          T_base_board from the touched corners (Kabsch, >= 3), T_base_cam = T_base_board inv(T_cam_board); checks the earlier
                 taped-cube samples (samples.json) against the camera model
Board frame: x along the 8-corner rows, y along the 5-corner columns, z = x cross y; units m."""
import json, pathlib, sys
import cv2, numpy as np, requests
OUT = pathlib.Path.home() / "c8/hra_red/global_calib"; BR = "http://localhost:8021"; SQ = 0.025; P = (8, 5)
sys.path.insert(0, str(pathlib.Path.home() / "holobrain-mac-model"))
OBJ = np.zeros((40, 3)); OBJ[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * SQ
ST = OUT / "board_state.json"


def state():
    return json.load(open(ST)) if ST.exists() else {}


def cam():
    I = json.load(open(OUT / "global_cam_intrinsics.json")); K = np.array(I["K"]); D = np.array(I["dist"])
    img = cv2.imdecode(np.frombuffer(requests.get(BR + "/frame/middle", timeout=10).content, np.uint8), 1)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); ok, c = cv2.findChessboardCorners(g, P)
    if not ok:
        cv2.imwrite(str(OUT / "board_labels.jpg"), img); print("board NOT detected -- whole board visible? (board_labels.jpg = current frame)"); return
    c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (3, 30, 0.01))
    ok, rv, tv = cv2.solvePnP(OBJ, c, K, D)
    pr, _ = cv2.projectPoints(OBJ, rv, tv, K, D); e = np.linalg.norm(pr.reshape(-1, 2) - c.reshape(-1, 2), axis=1)
    R, _ = cv2.Rodrigues(rv); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tv.ravel()
    s = state(); s.update(T_cam_board=T.tolist(), corners_px=c.reshape(-1, 2).tolist(), pnp_reproj_px=float(e.mean())); json.dump(s, open(ST, "w"), indent=1)
    for i, (u, v) in enumerate(c.reshape(-1, 2)):
        col = (0, 0, 255) if i in (0, 7, 32, 39) else (0, 160, 0)
        cv2.circle(img, (int(u), int(v)), 4, col, -1); cv2.putText(img, str(i), (int(u) + 4, int(v) - 4), 0, 0.4, col, 1)
    cv2.drawFrameAxes(img, K, D, rv, tv, 0.05)
    cv2.imwrite(str(OUT / "board_labels.jpg"), img)
    print(f"board detected: camera->board distance {np.linalg.norm(tv):.3f} m, PnP reproj {e.mean():.2f} px; labels in board_labels.jpg (red = corners 0, 7, 32, 39)")


def touch(idx):
    import infer_core_v4 as IC, eef_kin
    k = object.__new__(IC.V4Inferencer); k.kin = eef_kin.Kin({"max_joint_delta": None})
    m, _ = k._tcp_mat(requests.get(BR + "/observe", timeout=10).json()["joints_rad"]); p = m[1][:3, 3]
    s = state(); s.setdefault("touch", {})[str(idx)] = p.tolist(); json.dump(s, open(ST, "w"), indent=1)
    print(f"corner {idx}: tip at base {np.round(p * 1000, 1)} mm  ({len(s['touch'])} corners recorded)")


def solve():
    s = state(); t = s.get("touch", {})
    if len(t) < 3: print("need >= 3 touched corners"); return
    ids = sorted(int(i) for i in t); A = OBJ[ids]; B = np.array([t[str(i)] for i in ids])
    ca, cb = A.mean(0), B.mean(0); U, S_, Vt = np.linalg.svd((A - ca).T @ (B - cb)); Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ Dg @ U.T; tvec = cb - R @ ca
    Tbb = np.eye(4); Tbb[:3, :3] = R; Tbb[:3, 3] = tvec
    res = np.linalg.norm((A @ R.T + tvec) - B, axis=1)
    out = dict(T_base_board=Tbb.tolist(), touch_residual_mm=(res * 1000).round(1).tolist(), board_up_in_base=R[:, 2].round(3).tolist())
    if "T_cam_board" in s:
        Tbc = Tbb @ np.linalg.inv(np.array(s["T_cam_board"])); out["T_base_cam"] = Tbc.tolist(); out["cam_pos_base_m"] = Tbc[:3, 3].round(3).tolist()
        smp = OUT / "samples.json"
        if smp.exists():
            I = json.load(open(OUT / "global_cam_intrinsics.json")); K = np.array(I["K"]); Dd = np.array(I["dist"])
            Tcb = np.linalg.inv(Tbc); rv, _ = cv2.Rodrigues(Tcb[:3, :3])
            S = [r for r in json.load(open(smp)) if r["uv"] is not None and 15 < r["uv"][1] < 465 and r["area"] >= 400]
            pts = np.array([np.array(r["T"])[:3, 3] for r in S]); uv = np.array([r["uv"] for r in S])
            pr, _ = cv2.projectPoints(pts, rv, Tcb[:3, 3], K, Dd); e = np.linalg.norm(pr.reshape(-1, 2) - uv, axis=1)
            out["taped_cube_check_px_p50"] = float(np.median(e)); out["taped_cube_check_px_max"] = float(e.max())
    s["solve"] = out; json.dump(s, open(ST, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("T_")}, indent=1))


if __name__ == "__main__":
    {"cam": cam, "solve": solve}.get(sys.argv[1], lambda: touch(int(sys.argv[2])))()
