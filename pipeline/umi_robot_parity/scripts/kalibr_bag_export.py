"""stereo_imu_record take -> Kalibr camera-IMU JOINT calibration inputs (2026-10-16).

  bag       <take>/kalibr/export/calib.bag  (ROS1 bag v2.0, uncompressed, written by the tiny writer below; no ROS needed on the Mac)
            cameras are matched to v004 by USB location (meta.json), not by file name; a take without C1 exports mono (cam0 only)
            /cam0/image_raw = LEFT  (C0, 0x03110000) mono8 at the saved size, header.stamp = cam_left_t_ns  (host monotonic)
            /cam1/image_raw = RIGHT (C1, 0x03140000) mono8, header.stamp = cam_right_t_ns (own stamps; L/R are free-running)
            /imu0           = XIAO imu_right.npz, device_us -> host ns (fit_device_to_host), + --pre-offset-ms (default 0)
  yamls     camchain.yaml (KB4 = pinhole-equi from stereo_trackcam_v004, intrinsics / save_scale, T_cn_cnm1 = T_CR_CL),
            imu.yaml (imu_noise_xiao_right_v001 x --noise-scale, update_rate = measured), aprilgrid.yaml (6x6 tag36h11, 25 mm, spacing 0.3 = board_corners())
  check     --check: OpenCV AprilTag detection + PnP with v004 on every --check-step-th frame of both cams -> reprojection sanity
  Clock: IMU stamps are on the HOST clock with NO camera offset applied (unless --pre-offset-ms), so Kalibr's timeshift_cam_imu
  (t_imu = t_cam + shift) is the whole offset; kalibr_to_camimu.py turns it into our offset_ms = pre_offset_ms - 1000*shift.
    python scripts/kalibr_bag_export.py datasets/stereo_imu/<take> [--check] [--max-seconds 120]"""
from __future__ import annotations
import argparse, json, struct, sys
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from handumi_collector.pose.timing import fit_device_to_host   # noqa: E402
from trackcam_board_lever import board_corners                   # noqa: E402

STEREO = "configs/calibration/stereo_trackcam_v005.yaml"
NOISE = "configs/calibration/imu_noise_xiao_right_v001.yaml"
GRID = dict(target_type="aprilgrid", tagCols=6, tagRows=6, tagSize=0.025, tagSpacing=0.3)   # == board_corners(size=0.025, spacing=0.3, cols=6); id0 bottom-left, ids along +x then rows up
# The PRINTED board is outputs/aprilgrid_6x6_25mm.pdf with a 1-bit black tag border (OpenCV markerBorderBits=1 finds it, 2 finds nothing on the
# 10-16 ruler take). Kalibr's AprilgridOptions default is blackTagBorder=2, so run_kalibr.sh patches it via KALIBR_BLACK_TAG_BORDER (default 1).
# Our tags are also drawn in the standard (OpenCV/AprilTag) orientation, while kalibr_create_target_pdf rotates every tag 180 deg -> without
# ~/vio/kalibr/patch_kalibr.py (KALIBR_TAG_ROT180=1) Kalibr mis-assigns corners: ~70 px residuals, camera-only calibration diverges;
# with it, camera-only Kalibr on 20261015_220411_calib = 0.52/0.47 px.
BLACK_TAG_BORDER = 1
TAG_ROT180 = 1

# ---------------------------------------------------------------- minimal ROS1 bag v2.0 writer
HDR_DEF = "MSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n"
SEP = "=" * 80 + "\n"
IMAGE = ("sensor_msgs/Image", "060021388200f6f0f447d0fcd9c64743",
         "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n" + SEP + HDR_DEF)
IMU = ("sensor_msgs/Imu", "6a62c6daae103f4ff57a132d6f95cec2",
       "std_msgs/Header header\ngeometry_msgs/Quaternion orientation\nfloat64[9] orientation_covariance\ngeometry_msgs/Vector3 angular_velocity\n"
       "float64[9] angular_velocity_covariance\ngeometry_msgs/Vector3 linear_acceleration\nfloat64[9] linear_acceleration_covariance\n" + SEP + HDR_DEF
       + SEP + "MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n" + SEP + "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n")


def _t(ns: int) -> bytes: return struct.pack("<II", ns // 1_000_000_000, ns % 1_000_000_000)
def _s(x: str) -> bytes: b = x.encode(); return struct.pack("<I", len(b)) + b


def _hdr(fields: dict) -> bytes:
    out = b""
    for k, v in fields.items():
        f = k.encode() + b"=" + v; out += struct.pack("<I", len(f)) + f
    return out


def _rec(fields: dict, data: bytes) -> bytes:
    h = _hdr(fields); return struct.pack("<I", len(h)) + h + struct.pack("<I", len(data)) + data


def ser_header(seq: int, ns: int, frame: str) -> bytes: return struct.pack("<I", seq) + _t(ns) + _s(frame)


def ser_image(seq: int, ns: int, frame: str, img: np.ndarray) -> bytes:
    h, w = img.shape
    return ser_header(seq, ns, frame) + struct.pack("<II", h, w) + _s("mono8") + struct.pack("<BI", 0, w) + struct.pack("<I", h * w) + np.ascontiguousarray(img, np.uint8).tobytes()


def ser_imu(seq: int, ns: int, gyro, acc) -> bytes:
    cov0 = struct.pack("<9d", -1, 0, 0, 0, 0, 0, 0, 0, 0); z9 = struct.pack("<9d", *([0.0] * 9))
    return ser_header(seq, ns, "imu0") + struct.pack("<4d", 0, 0, 0, 1) + cov0 + struct.pack("<3d", *map(float, gyro)) + z9 + struct.pack("<3d", *map(float, acc)) + z9


class BagWriter:
    """Chunked, uncompressed ROS1 bag v2.0 (records: 0x03 bag header, 0x05 chunk, 0x04 index, 0x07 connection, 0x06 chunk info)."""
    CHUNK = 8 << 20

    def __init__(self, path: Path):
        self.f = open(path, "wb"); self.f.write(b"#ROSBAG V2.0\n"); self.hdr_pos = self.f.tell(); self._bag_header(0, 0, 0)
        self.conns: list[tuple[str, tuple]] = []; self.chunks = []; self._new_chunk()

    def _bag_header(self, index_pos: int, nconn: int, nchunk: int):
        h = _hdr({"op": b"\x03", "index_pos": struct.pack("<Q", index_pos), "conn_count": struct.pack("<I", nconn), "chunk_count": struct.pack("<I", nchunk)})
        pad = 4096 - (4 + len(h) + 4); self.f.write(struct.pack("<I", len(h)) + h + struct.pack("<I", pad) + b" " * pad)

    def _conn_rec(self, cid: int) -> bytes:
        topic, (typ, md5, mdef) = self.conns[cid]
        return _rec({"op": b"\x07", "conn": struct.pack("<I", cid), "topic": topic.encode()},
                    _hdr({"topic": topic.encode(), "type": typ.encode(), "md5sum": md5.encode(), "message_definition": mdef.encode(), "callerid": b"/kalibr_bag_export"}))

    def _new_chunk(self):
        self.buf = bytearray(); self.idx: dict[int, list] = {}; self.t0 = None; self.t1 = None; self.seen: set[int] = set()

    def add_conn(self, topic: str, spec: tuple) -> int:
        self.conns.append((topic, spec)); return len(self.conns) - 1

    def write(self, cid: int, ns: int, data: bytes):
        if cid not in self.seen: self.buf += self._conn_rec(cid); self.seen.add(cid)
        self.idx.setdefault(cid, []).append((ns, len(self.buf)))
        self.buf += _rec({"op": b"\x02", "conn": struct.pack("<I", cid), "time": _t(ns)}, data)
        self.t0 = ns if self.t0 is None else min(self.t0, ns); self.t1 = ns if self.t1 is None else max(self.t1, ns)
        if len(self.buf) >= self.CHUNK: self._flush()

    def _flush(self):
        if not self.idx: return
        pos = self.f.tell(); self.f.write(_rec({"op": b"\x05", "compression": b"none", "size": struct.pack("<I", len(self.buf))}, bytes(self.buf)))
        for cid, ent in self.idx.items():
            self.f.write(_rec({"op": b"\x04", "ver": struct.pack("<I", 1), "conn": struct.pack("<I", cid), "count": struct.pack("<I", len(ent))},
                              b"".join(_t(ns) + struct.pack("<I", off) for ns, off in ent)))
        self.chunks.append((pos, self.t0, self.t1, {c: len(e) for c, e in self.idx.items()})); self._new_chunk()

    def close(self):
        self._flush(); index_pos = self.f.tell()
        for cid in range(len(self.conns)): self.f.write(self._conn_rec(cid))
        for pos, t0, t1, cnt in self.chunks:
            self.f.write(_rec({"op": b"\x06", "ver": struct.pack("<I", 1), "chunk_pos": struct.pack("<Q", pos), "start_time": _t(t0), "end_time": _t(t1),
                               "count": struct.pack("<I", len(cnt))}, b"".join(struct.pack("<II", c, n) for c, n in cnt.items())))
        self.f.seek(self.hdr_pos); self._bag_header(index_pos, len(self.conns), len(self.chunks)); self.f.close()


# ---------------------------------------------------------------- export
def load_cams(sc: int):
    st = yaml.safe_load((ROOT / STEREO).read_text()); cams = {}
    for k, side in enumerate(("left", "right")):
        c = st[side]; K = np.array([[c["fx"] / sc, 0, c["cx"] / sc], [0, c["fy"] / sc, c["cy"] / sc], [0, 0, 1.0]]); D = np.array([c["k1"], c["k2"], c["k3"], c["k4"]], float)
        cams[side] = (K, D)
    return st, cams


def camchain(st: dict, sc: int, w: int, h: int, sides: list[str]) -> dict:
    out = {}
    for k, side in enumerate(sides):
        c = st[side]
        d = dict(camera_model="pinhole", intrinsics=[float(c["fx"]) / sc, float(c["fy"]) / sc, float(c["cx"]) / sc, float(c["cy"]) / sc],
                 distortion_model="equidistant", distortion_coeffs=[float(c[x]) for x in ("k1", "k2", "k3", "k4")], resolution=[w, h],
                 rostopic=f"/cam{k}/image_raw", cam_overlaps=[1 - k] if len(sides) == 2 else [])
        if k == 1: d["T_cn_cnm1"] = [[float(v) for v in row] for row in st["T_CR_CL"]]
        out[f"cam{k}"] = d
    return out


def map_sides(st: dict, meta: dict) -> dict:
    """v004 side ('left' = C0, 'right' = C1) -> take file prefix ('cam_left'/'cam_right'), matched by USB location (10-15 takes had L/R swapped)."""
    m = {}
    for side in ("left", "right"):
        for tk in ("left", "right"):
            if str(meta.get(tk, {}).get("location", "")).lower() == str(st[side]["location"]).lower(): m[side] = f"cam_{tk}"
    if "left" not in m: raise SystemExit(f"C0 ({st['left']['location']})가 이 테이크에 없음: {meta.get('left')}, {meta.get('right')}")
    return m


def xcorr_lag_ms(t_cam_ns: np.ndarray, poses: dict, t_imu_ns: np.ndarray, gyro: np.ndarray, max_gap: int, search_ms: float = 300.0) -> tuple[float, float]:
    """Board-PnP angular speed vs gyro norm, demeaned cross-correlation on a 5 ms grid within +-search_ms.
    Returns (lag_ms, peak_corr); lag > 0 = camera stamps LATER than IMU host stamps (= our offset_ms convention)."""
    from scipy.spatial.transform import Rotation as Rot
    ks = sorted(poses); w, tw = [], []
    for a_, b_ in zip(ks[:-1], ks[1:]):
        if b_ - a_ > max_gap: continue
        dt = (t_cam_ns[b_] - t_cam_ns[a_]) / 1e9
        w.append(np.linalg.norm(Rot.from_matrix(poses[b_][:3, :3] @ poses[a_][:3, :3].T).as_rotvec()) / dt); tw.append((t_cam_ns[a_] + t_cam_ns[b_]) / 2.0)
    if len(w) < 50: return float("nan"), float("nan")
    w, tw = np.array(w), np.array(tw); g = np.arange(tw[0], tw[-1], 5e6)
    cw = np.interp(g, tw, w); gw = np.interp(g, t_imu_ns.astype(float), np.linalg.norm(gyro, axis=1)); cw -= cw.mean(); gw -= gw.mean()
    L = int(search_ms / 5); n = len(g); best = max(range(-L, L + 1), key=lambda k: float(np.dot(cw[max(k, 0):n + min(k, 0)], gw[max(-k, 0):n - max(k, 0)])))
    k = best; c = float(np.corrcoef(cw[max(k, 0):n + min(k, 0)], gw[max(-k, 0):n - max(k, 0)])[0, 1])
    return 5.0 * best, round(c, 3)


def board_check(video: Path, n: int, K, D, step: int):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  (lambda p: (setattr(p, "cornerRefinementMethod", cv2.aruco.CORNER_REFINE_SUBPIX), p)[-1])(cv2.aruco.DetectorParameters()))
    cap = cv2.VideoCapture(str(video)); errs, ntags, poses = [], [], {}
    for k in range(n):
        ok, f = cap.read()
        if not ok: break
        if k % step: continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); corners, ids, _ = det.detectMarkers(g)
        if ids is None: ntags.append(0); continue
        keep = [(c.reshape(4, 2), int(i)) for c, i in zip(corners, ids.ravel()) if i < GRID["tagRows"] * GRID["tagCols"]]; ntags.append(len(keep))
        if len(keep) < 7: continue   # Kalibr: minTagsForValidObs = max(rows, cols) + 1
        obj = np.vstack([board_corners(i, GRID["tagSize"], GRID["tagSpacing"], GRID["tagCols"]) for _, i in keep]); img = np.vstack([c for c, _ in keep]).astype(np.float64)
        und = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, D.reshape(4, 1)).reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(obj, und, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        if not ok: continue
        rv, tv = cv2.solvePnPRefineLM(obj, und, np.eye(3), None, rv, tv)
        proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rv, tv, K, D.reshape(4, 1)); e = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        errs.append(e); T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = tv.ravel(); poses[k] = T
    e = np.concatenate(errs) if errs else np.zeros(0)
    return dict(frames_checked=len(ntags), frames_valid_ge7tags=len(poses), tags_median=float(np.median(ntags)) if ntags else 0.0,
                reproj_median_px=round(float(np.median(e)), 3) if len(e) else None, reproj_p95_px=round(float(np.percentile(e, 95)), 3) if len(e) else None), poses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--out", default=None, help="default <take>/kalibr/export")
    ap.add_argument("--pre-offset-ms", type=float, default=0.0, help="added to IMU host stamps before writing (keep 0: Kalibr estimates the full offset)")
    ap.add_argument("--acc-delay-ms", type=float, default=0.0, help="accelerometer lags the gyro by this much: acc written at each gyro stamp = acc(t + delay)")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="multiplies the 4 noise numbers of the IMU yaml")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="crop to the first N s (0 = all)"); ap.add_argument("--cam-stride", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--check", action="store_true", help="OpenCV board detection + reprojection sanity with v004"); ap.add_argument("--check-step", type=int, default=5)
    a = ap.parse_args()
    E = Path(a.take).resolve(); out = Path(a.out) if a.out else E / "kalibr" / "export"; out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((E / "meta.json").read_text()); sc = int(meta["save_scale"]); st, cams = load_cams(sc)
    fmap = map_sides(st, meta); sides = [x for x in ("left", "right") if x in fmap]   # sides[k] -> /cam{k}
    ts = {s_: np.load(E / f"{fmap[s_]}_t_ns.npy").astype(np.int64) for s_ in sides}
    z = np.load(E / "imu_right.npz"); fit = fit_device_to_host(z["device_us"], z["host_ns"]); ti = fit.to_host_ns(z["device_us"]).astype(np.int64) + int(round(a.pre_offset_ms * 1e6))
    o = np.argsort(ti, kind="stable"); ti, gy, ac = ti[o], np.asarray(z["gyro"], float)[o], np.asarray(z["acc"], float)[o]
    keep = np.concatenate([[True], np.diff(ti) > 0]); ti, gy, ac = ti[keep], gy[keep], ac[keep]
    if a.acc_delay_ms:   # 10-16: board/accel offset ~57 ms vs board/gyro ~44 ms on 190404 -> the XIAO accel stream lags its gyro
        ac = np.stack([np.interp(ti + a.acc_delay_ms * 1e6, ti, ac[:, j]) for j in range(3)], 1)
    t_start = max([ts[s_][0] for s_ in sides] + [ti[0]]); t_end = min([ts[s_][-1] for s_ in sides] + [ti[-1]])
    if a.max_seconds > 0: t_end = min(t_end, t_start + int(a.max_seconds * 1e9))
    rate = 1e6 / float(np.median(np.diff(np.sort(np.asarray(z["device_us"], np.float64)))))
    print(f"[export] 테이크 {E.name}: " + ", ".join(f"cam{k}={fmap[s_]}({st[s_]['location']}) {len(ts[s_])}프레임" for k, s_ in enumerate(sides))
          + f", IMU {len(ti)} 샘플 {rate:.0f} Hz (클럭 잔차 {fit.residual_std_ms:.2f} ms), 겹치는 구간 {(t_end - t_start) / 1e9:.1f} s", flush=True)
    if len(sides) == 1: print("[export] 주의: C1 카메라가 없어 모노(cam0만)로 내보냄", flush=True)

    bw = BagWriter(out / "calib.bag"); cid = {s_: bw.add_conn(f"/cam{k}/image_raw", IMAGE) for k, s_ in enumerate(sides)}; ci = bw.add_conn("/imu0", IMU)
    caps = {s_: cv2.VideoCapture(str(E / f"{fmap[s_]}.mp4")) for s_ in sides}
    nxt = {s_: 0 for s_ in sides}; nimg = {s_: 0 for s_ in sides}; size = None
    j = int(np.searchsorted(ti, t_start)); nimu = 0
    # merge in time order so the bag is chronological (not required by rosbag, but keeps chunks tight)
    while True:
        cand = [(int(ts[s_][nxt[s_]]), s_) for s_ in sides if nxt[s_] < len(ts[s_])]
        if not cand: break
        t, s_ = min(cand); k = nxt[s_]; nxt[s_] += 1
        ok, f = caps[s_].read()
        if not ok: nxt[s_] = len(ts[s_]); continue
        while j < len(ti) and ti[j] <= min(t, t_end):
            bw.write(ci, int(ti[j]), ser_imu(nimu, int(ti[j]), gy[j], ac[j])); nimu += 1; j += 1
        if t < t_start or t > t_end or k % a.cam_stride: continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f; size = g.shape[::-1]
        bw.write(cid[s_], t, ser_image(nimg[s_], t, f"cam{sides.index(s_)}", g)); nimg[s_] += 1
    while j < len(ti) and ti[j] <= t_end:
        bw.write(ci, int(ti[j]), ser_imu(nimu, int(ti[j]), gy[j], ac[j])); nimu += 1; j += 1
    bw.close()
    w, h = size; print(f"[export] calib.bag: " + ", ".join(f"cam{k} {nimg[s_]}" for k, s_ in enumerate(sides)) + f", imu0 {nimu} 메시지, {(out / 'calib.bag').stat().st_size / 1e9:.2f} GB", flush=True)

    noise = yaml.safe_load((ROOT / NOISE).read_text())["imu_noise"]
    imu = dict(rostopic="/imu0", update_rate=round(rate, 2),
               **{k: float(noise[k]) * a.noise_scale for k in ("accelerometer_noise_density", "accelerometer_random_walk", "gyroscope_noise_density", "gyroscope_random_walk")})
    (out / "camchain.yaml").write_text(yaml.safe_dump(camchain(st, sc, w, h, sides), sort_keys=False, default_flow_style=None))
    (out / "imu.yaml").write_text(yaml.safe_dump(imu, sort_keys=False)); (out / "aprilgrid.yaml").write_text(yaml.safe_dump(GRID, sort_keys=False))
    em = dict(take=str(E), save_scale=sc, image_size=[w, h], stereo_calib=STEREO, cams={f"cam{k}": dict(v004_side=s_, file=fmap[s_], location=st[s_]["location"], n=nimg[s_]) for k, s_ in enumerate(sides)},
              imu_noise=NOISE, noise_scale=a.noise_scale, pre_offset_ms=a.pre_offset_ms, acc_delay_ms=a.acc_delay_ms, black_tag_border=BLACK_TAG_BORDER, tag_rot180=TAG_ROT180,
              clock_fit=fit.to_dict(), t_start_ns=int(t_start), t_end_ns=int(t_end), n_imu=nimu, imu_rate_hz=round(rate, 2),
              convention="bag stamps: cam = own host ns; imu = host ns + pre_offset_ms. Kalibr shift: t_imu = t_cam + shift -> offset_ms = pre_offset_ms - 1000*shift")
    if a.check:
        chk = {}; P = {}
        for k, s_ in enumerate(sides):
            K, D = cams[s_]; name = f"cam{k}_{s_}"; chk[name], P[s_] = board_check(E / f"{fmap[s_]}.mp4", len(ts[s_]), K, D, a.check_step)
            print(f"[check] {name}: 확인 {chk[name]['frames_checked']} 프레임 중 7태그 이상 {chk[name]['frames_valid_ge7tags']}, 태그 중앙 {chk[name]['tags_median']:.0f}, "
                  f"재투영 중앙 {chk[name]['reproj_median_px']} px p95 {chk[name]['reproj_p95_px']} px", flush=True)
        if len(sides) == 2:   # stereo consistency: T_CR_CL vs board poses of near-simultaneous pairs (|dt| <= 5 ms)
            tl, tr = ts["left"], ts["right"]; T_RL = np.asarray(st["T_CR_CL"], float); dr, dtr = [], []
            for kl, TL in P["left"].items():
                kr = int(np.argmin(np.abs(tr - tl[kl])))
                if kr in P["right"] and abs(int(tr[kr]) - int(tl[kl])) <= 5e6:
                    Ed = np.linalg.inv(T_RL) @ P["right"][kr] @ np.linalg.inv(TL)
                    dr.append(np.degrees(np.linalg.norm(cv2.Rodrigues(Ed[:3, :3])[0]))); dtr.append(np.linalg.norm(Ed[:3, 3]) * 1000)
            if dr:
                chk["stereo_vs_T_CR_CL"] = dict(pairs=len(dr), rot_err_deg_median=round(float(np.median(dr)), 3), trans_err_mm_median=round(float(np.median(dtr)), 2))
                print(f"[check] 스테레오 일관성 (보드 PnP 쌍 {len(dr)}): T_CR_CL 대비 회전 {np.median(dr):.2f}° / 이동 {np.median(dtr):.1f} mm (중앙)", flush=True)
            else: print("[check] 스테레오 일관성: 5 ms 안의 L/R 동시 보드 쌍 없음", flush=True)
        # camera-IMU time offset from board angular speed vs gyro (host clocks, IMU WITHOUT pre_offset) -> Kalibr time prior.
        # Kalibr's own prior (un-demeaned full correlation of a 100 knot/s spline) went to -5.39 s on cam0 of 20261016_190404_calib.
        t_host = ti - int(round(a.pre_offset_ms * 1e6)); pri = {}
        for k_, s_ in enumerate(sides):
            lag, cc = xcorr_lag_ms(ts[s_], P[s_], t_host, gy, max_gap=2 * a.check_step)
            pri[f"cam{k_}"] = dict(offset_ms=lag, corr=cc, kalibr_prior_s=round((a.pre_offset_ms - lag) / 1000.0, 4))
            print(f"[check] cam{k_} 보드-자이로 상관: 오프셋 {lag:.0f} ms (상관 {cc}) -> Kalibr 시간 prior {(a.pre_offset_ms - lag) / 1000.0:+.3f} s", flush=True)
        em["time_prior"] = pri
        em["board_check_opencv"] = chk
    (out / "export_meta.json").write_text(json.dumps(em, indent=1, ensure_ascii=False))
    print(f"[export] 완료 -> {out}", flush=True)


if __name__ == "__main__":
    main()
