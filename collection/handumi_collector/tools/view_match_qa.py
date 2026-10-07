"""Same-scene human/robot wrist-view matching QA (design step 5).

    robot wrist C922 frame  |  HandUMI Arducam RAW  |  HandUMI C922-like virtual view (K_target + D_target + rpy)

Reports the residual per *group*, because the groups say WHICH physical thing is wrong:
    E_view_all / median / p95                     the residual distribution over this scene's landmarks. THE GATE IS THE MEAN
                                                  (E_view_all_px vs thresholds.pass_px/warn_px); median and p95 are diagnostic
                                                  only, so mean 6.9 / p95 8.9 is a correct PASS, not a bug
    E_view_center / E_view_edge                   edge >> center  → wrong K_target / distortion (FOV model)
    E_view_near / E_view_far                      near >> far     → PARALLAX: the two optical centres differ → move the MOUNT
    scale_ratio_mean / _std                       ≠ 1             → wrong K_target focal length
    rotation_fit.rpy_delta_deg                    large           → MOUNT_MISMATCH (do not absorb a mount error in rpy)

`--fit-rpy` is only for the *residual* rotation after the physical mount is matched: a fit larger than
`thresholds.mount_mismatch_rpy_deg` is reported as MOUNT_MISMATCH and the proposal is marked not-recommended.
Correspondences: `--pick` (manual, trusted, 7-10 points with near/far labels), `--points file.json`, or the colour-blob detector.
Output: <out>/view_match.json (with full calibration provenance), side_by_side.png, overlay_robot_virtual.png, virtual.png.

    python -m handumi_collector.tools.view_match_qa --side left --out qa_left --robot-index 1 --human-index 0 --pick
    python -m handumi_collector.tools.view_match_qa --side left --out qa_left --robot r.png --human raw.png --points pts.json --fit-rpy
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import yaml
from ..config import DEFAULT_CONFIG_DIR
from ..pose.calibration import SideCalibration
from ..pose.episode_io import write_json
from ..pose.virtual_view import VirtualPinholeView, compose_rpy, fit_rotation, load_virtual_wrist


def load_spec(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or DEFAULT_CONFIG_DIR / "view_match.yaml").read_text())


# --------------------------------------------------------------------------------------------- correspondences
def detect_landmarks(img_bgr: np.ndarray, spec: dict) -> dict[str, dict]:
    """{name: {center, w, h, area, depth}} for the largest blob of each configured colour. Missing ones stay absent (never faked)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV); out = {}
    for name, s in spec["landmarks"].items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in zip(s["hsv_lo"], s["hsv_hi"]): mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= float(s.get("min_area", 100))]
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea); x, y, w, h = cv2.boundingRect(c); m = cv2.moments(c)
        out[name] = dict(center=[m["m10"] / m["m00"], m["m01"] / m["m00"]], w=int(w), h=int(h), area=float(cv2.contourArea(c)), depth=s.get("depth"))
    return out


def points_to_landmarks(points: dict) -> tuple[dict, dict]:
    """{"robot": [[u,v],…], "virtual": [[u,v],…], "depth": ["near",…], "names": [...]} → (lm_robot, lm_virtual).
    Sizes are unknown for hand-picked points, so scale ratios are simply not reported for them."""
    r, v = points["robot"], points["virtual"]
    if len(r) != len(v): raise ValueError(f"points: {len(r)} robot vs {len(v)} virtual")
    names = points.get("names") or [f"p{i}" for i in range(len(r))]
    depth = points.get("depth") or [None] * len(r)
    lm_r = {n: dict(center=list(a), depth=d) for n, a, d in zip(names, r, depth)}
    lm_v = {n: dict(center=list(b), depth=d) for n, b, d in zip(names, v, depth)}
    return lm_r, lm_v


def pick_points(robot_bgr: np.ndarray, virtual_bgr: np.ndarray, spec: dict) -> dict:
    """Manual correspondence picking: click the SAME feature in the robot frame (left) then the virtual frame (right).
    Keys: n = mark the next pair near, f = far, u = undo last, s = skip the suggested name, q/ENTER = done."""
    names = list(spec.get("picking", {}).get("points", [])) or [f"p{i}" for i in range(10)]
    depths = list(spec.get("picking", {}).get("depths", [])) + [None] * len(names)
    pairs: list[dict] = []; pending: list[float] | None = None; idx = 0; depth_override = None
    h = max(robot_bgr.shape[0], virtual_bgr.shape[0]); wr = robot_bgr.shape[1]
    state = dict(click=None)

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN: state["click"] = (x, y)
    win = "pick correspondences  (LEFT: robot | RIGHT: virtual)   n/f = near/far, u = undo, s = skip, q = done"
    cv2.namedWindow(win); cv2.setMouseCallback(win, on_mouse)
    while True:
        canvas = np.concatenate([robot_bgr, virtual_bgr], axis=1).copy()
        for p in pairs:
            a = (int(p["robot"][0]), int(p["robot"][1])); b = (int(p["virtual"][0]) + wr, int(p["virtual"][1]))
            col = (0, 255, 0) if p.get("depth") == "near" else (0, 200, 255)
            for c in (a, b): cv2.drawMarker(canvas, c, col, cv2.MARKER_CROSS, 14, 2)
            cv2.line(canvas, a, b, col, 1); cv2.putText(canvas, p["name"][:14], (a[0] + 5, a[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        if pending is not None: cv2.drawMarker(canvas, (int(pending[0]), int(pending[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        nm = names[idx] if idx < len(names) else f"p{idx}"
        d = depth_override or (depths[idx] if idx < len(depths) else None)
        msg = f"[{len(pairs)}] next: {nm} ({d or 'depth?'})  -> click ROBOT (left)" if pending is None else f"[{len(pairs)}] {nm}: now click VIRTUAL (right)"
        cv2.putText(canvas, msg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, canvas); k = cv2.waitKey(20) & 0xFF
        if state["click"] is not None:
            x, y = state["click"]; state["click"] = None
            if pending is None and x < wr: pending = [float(x), float(y)]
            elif pending is not None and x >= wr:
                pairs.append(dict(name=nm, robot=pending, virtual=[float(x - wr), float(y)], depth=d)); pending = None; idx += 1; depth_override = None
        if k in (ord("n"), ord("f")): depth_override = "near" if k == ord("n") else "far"
        if k == ord("u") and pairs: pairs.pop(); idx = max(0, idx - 1)
        if k == ord("s"): idx += 1
        if k in (ord("q"), 13): break
    cv2.destroyWindow(win)
    return dict(names=[p["name"] for p in pairs], robot=[p["robot"] for p in pairs], virtual=[p["virtual"] for p in pairs], depth=[p["depth"] for p in pairs])


# --------------------------------------------------------------------------------------------- metrics
def compare(lm_robot: dict, lm_virtual: dict, image_size: tuple[int, int], spec: dict) -> dict:
    """Per-landmark errors + the grouped E_view metrics (all / center / edge / near / far) and scale ratios."""
    th = spec.get("thresholds", {}); w, h = image_size
    centre = np.array([(w - 1) / 2, (h - 1) / 2]); half_diag = float(np.hypot(w, h) / 2)
    common = sorted(set(lm_robot) & set(lm_virtual)); per = {}
    for n in common:
        a, b = lm_robot[n], lm_virtual[n]
        pa, pb = np.asarray(a["center"], float), np.asarray(b["center"], float)
        r_frac = float(np.linalg.norm(pa - centre) / half_diag)
        d = dict(center_err_px=float(np.linalg.norm(pb - pa)), dx_px=float(pb[0] - pa[0]), dy_px=float(pb[1] - pa[1]),
                 radius_frac=round(r_frac, 3), region="edge" if r_frac > float(th.get("edge_radius_frac", 0.6)) else "center",
                 depth=a.get("depth") or b.get("depth"))
        if a.get("w") and b.get("w"): d["width_ratio"] = b["w"] / max(a["w"], 1); d["height_ratio"] = b["h"] / max(a["h"], 1)
        per[n] = d
    def mean_of(pred, key="center_err_px"):
        v = [p[key] for p in per.values() if pred(p) and p.get(key) is not None]
        return (float(np.mean(v)) if v else None), len(v)
    E_all, n_all = mean_of(lambda p: True)
    errs = np.array([p["center_err_px"] for p in per.values()], float) if per else np.zeros(0)
    E_c, n_c = mean_of(lambda p: p["region"] == "center"); E_e, n_e = mean_of(lambda p: p["region"] == "edge")
    E_n, n_n = mean_of(lambda p: p["depth"] == "near"); E_f, n_f = mean_of(lambda p: p["depth"] == "far")
    ratios = [p[k] for p in per.values() for k in ("width_ratio", "height_ratio") if p.get(k) is not None]
    return dict(E_view_all_px=E_all, E_view_median_px=(float(np.median(errs)) if len(errs) else None),
                E_view_p95_px=(float(np.percentile(errs, 95)) if len(errs) else None),
                E_view_center_px=E_c, E_view_edge_px=E_e, E_view_near_px=E_n, E_view_far_px=E_f,
                counts=dict(all=n_all, center=n_c, edge=n_e, near=n_n, far=n_f),
                near_far_ratio=(None if not (E_n and E_f) else float(E_n / E_f)),
                scale_ratio_mean=(float(np.mean(ratios)) if ratios else None), scale_ratio_std=(float(np.std(ratios)) if ratios else None),
                max_err_px=(float(max(p["center_err_px"] for p in per.values())) if per else None),
                missing_in_robot=sorted(set(lm_virtual) - set(lm_robot)), missing_in_virtual=sorted(set(lm_robot) - set(lm_virtual)),
                per_landmark=per)


def verdict_and_flags(m: dict, spec: dict, rot: dict | None) -> tuple[str | None, list[str]]:
    th = spec.get("thresholds", {}); flags: list[str] = []
    E = m["E_view_all_px"]
    v = None if E is None else ("PASS" if E < float(th.get("pass_px", 8)) else "WARN" if E < float(th.get("warn_px", 20)) else "REJECT")
    nf = m.get("near_far_ratio")
    if nf is not None and nf > float(th.get("near_far_ratio_warn", 2.0)):
        flags.append(f"PARALLAX: E_near/E_far = {nf:.1f} > {th.get('near_far_ratio_warn')} — the optical centres differ; move the MOUNT (rotation cannot fix this)")
    if m["E_view_edge_px"] and m["E_view_center_px"] and m["E_view_edge_px"] > 2.5 * max(m["E_view_center_px"], 1e-6):
        flags.append("FOV_MODEL: edge error >> centre error — K_target / distortion is wrong (re-measure the robot C922 at its recording resolution)")
    sr, tol = m.get("scale_ratio_mean"), float(th.get("scale_ratio_tol", 0.08))
    if sr is not None and abs(sr - 1.0) > tol:
        flags.append(f"SCALE: apparent size ratio {sr:.3f} (tol ±{tol}) — K_target focal length does not match the robot camera")
    if rot:
        lim = float(th.get("mount_mismatch_rpy_deg", 5.0)); mx = float(np.max(np.abs(rot["rpy_delta_deg"])))
        if mx > lim:
            flags.append(f"MOUNT_MISMATCH: the fit wants {mx:.1f}° (> {lim}°) — fix the physical mount orientation first; rpy is only for a small residual")
            rot["recommended"] = False
        else:
            rot["recommended"] = rot["E_view_after_px"] < (rot["E_view_before_px"] * 0.7)
    if v in ("WARN", "REJECT") and not flags:
        flags.append("uniform offset with no parallax/FOV signature — check the residual rotation (--fit-rpy) and the landmark accuracy")
    return v, flags


# --------------------------------------------------------------------------------------------- run
def report_text(res: dict) -> str:
    """The compact human-readable block quoted in reports / the H120 write-up."""
    def f(x, u="px", n=1): return "—" if x is None else f"{x:.{n}f} {u}".strip()
    m = res; rot = m.get("rotation_fit") or {}
    sr = "—" if m.get("scale_ratio_mean") is None else f"{m['scale_ratio_mean']:.2f} ± {m['scale_ratio_std']:.2f}"
    rpy = "—" if not rot.get("rpy_delta_deg") else "[" + ", ".join(f"{v:.1f}" for v in rot["rpy_delta_deg"]) + "] deg"
    lines = [f"side {res['side']}   scene {res.get('scene') or '-'}   mount {res['provenance'].get('mount_revision') or '-'}"
             f"   fisheye {res['provenance']['source_fisheye_intrinsics']}   target {res['provenance']['target_virtual_wrist']}",
             f"E_view mean       {f(m['E_view_all_px'])}", f"E_view median     {f(m['E_view_median_px'])}",
             f"E_view p95        {f(m['E_view_p95_px'])}", f"near              {f(m['E_view_near_px'])}",
             f"far               {f(m['E_view_far_px'])}",
             "near/far          " + f(m.get("near_far_ratio"), "", 2),
             f"center            {f(m['E_view_center_px'])}", f"edge              {f(m['E_view_edge_px'])}",
             f"scale ratio       {sr}", f"fit RPY           {rpy}",
             f"points            {m['counts']['all']} ({m['counts']['near']} near / {m['counts']['far']} far)",
             f"verdict           {res['verdict'] or '—'}   (gate: {res.get('gate', {}).get('rule', 'n/a')}; median/p95 diagnostic only)",
             f"flags             {'; '.join(res['flags']) if res['flags'] else 'NONE'}"]
    return "\n".join(lines)


def run(robot_bgr: np.ndarray, human_raw_bgr: np.ndarray, side: str, *, out: Path, cal_dir: Path | None = None, spec: dict | None = None,
        points: dict | None = None, fit_rpy: bool = False, mount_revision: str = "", notes: str = "", scene: str = "") -> dict:
    spec = spec or load_spec()
    cal = SideCalibration(side, cal_dir=cal_dir)
    if cal.intrinsics is None: raise RuntimeError(f"fisheye intrinsics for {side} missing — run tools.calibrate_fisheye first")
    vver, vw = load_virtual_wrist(cal_dir=cal_dir)
    K_t, D_t, rpy = vw["K_target"][side], vw["D_target"][side], vw["rpy_deg"][side]
    view = VirtualPinholeView(cal.intrinsics, K_t, vw["size"], rpy, D_target=D_t)
    virt = view.render(human_raw_bgr)
    if robot_bgr.shape[:2] != virt.shape[:2]:
        robot_bgr = cv2.resize(robot_bgr, (virt.shape[1], virt.shape[0]))
    lm_r, lm_v = points_to_landmarks(points) if points else (detect_landmarks(robot_bgr, spec), detect_landmarks(virt, spec))
    m = compare(lm_r, lm_v, (virt.shape[1], virt.shape[0]), spec)
    rot = None
    if fit_rpy:
        common = sorted(set(lm_r) & set(lm_v))
        if len(common) >= 3:
            rot = fit_rotation([lm_v[n]["center"] for n in common], [lm_r[n]["center"] for n in common], K_t, D_t)
            rot["rpy_proposed_deg"] = compose_rpy(rpy, rot["rpy_delta_deg"])
        else:
            rot = dict(error=f"need >= 3 correspondences for a rotation fit, have {len(common)}")
    v, flags = verdict_and_flags(m, spec, rot if rot and "error" not in rot else None)
    th = spec.get("thresholds", {})
    gate = dict(metric="E_view_all_px", statistic="mean", pass_px=float(th.get("pass_px", 8.0)), warn_px=float(th.get("warn_px", 20.0)),
                rule=f"mean < {float(th.get('pass_px', 8.0))} px", diagnostic_only=["E_view_median_px", "E_view_p95_px"],
                note="engineering acceptance criterion; median/p95 are diagnostics and never change the verdict")
    res = dict(schema="handumi_view_match_qa/v2", side=side, scene=scene, verdict=v, flags=flags, gate=gate,
               provenance=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), source_fisheye_intrinsics=cal.versions["fisheye"],
                               target_virtual_wrist=vver or "DEFAULT_ESTIMATE (no virtual_wrist_vNNN.yaml)", K_target=np.asarray(K_t).tolist(),
                               target_distortion=(None if D_t is None else np.asarray(D_t).tolist()), virtual_rotation_rpy_deg=list(rpy),
                               virtual_size=list(vw["size"]), hfov_virtual_deg=round(view.hfov_deg(), 2), mount_revision=mount_revision,
                               correspondences="manual/points" if points else "colour-blob detector", n_correspondences=m["counts"]["all"], notes=notes),
               **{k: x for k, x in m.items() if k != "per_landmark"}, rotation_fit=rot, per_landmark=m["per_landmark"])
    res["report"] = report_text(res)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(res["report"] + "\n")
    h = virt.shape[0]; raw_small = cv2.resize(human_raw_bgr, (int(human_raw_bgr.shape[1] * h / human_raw_bgr.shape[0]), h))
    def annot(img, lm, color, title):
        o = img.copy()
        for n, d in lm.items():
            u, vv = int(round(d["center"][0])), int(round(d["center"][1]))
            cv2.drawMarker(o, (u, vv), color, cv2.MARKER_CROSS, 16, 2); cv2.putText(o, n[:14], (u + 6, vv - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(o, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2); return o
    panel = np.concatenate([annot(robot_bgr, lm_r, (0, 255, 0), "robot C922"), annot(raw_small, {}, (0, 0, 0), "Arducam RAW"),
                            annot(virt, lm_v, (0, 200, 255), "C922-like virtual")], axis=1)
    cv2.imwrite(str(out / "side_by_side.png"), panel)
    cv2.imwrite(str(out / "overlay_robot_virtual.png"), cv2.addWeighted(robot_bgr, 0.5, virt, 0.5, 0))
    cv2.imwrite(str(out / "virtual.png"), virt); cv2.imwrite(str(out / "robot.png"), robot_bgr)
    if points: write_json(points, out / "points.json")
    write_json(res, out / "view_match.json")
    return res


def _grab(index: int, w: int, h: int, warmup: int = 15) -> np.ndarray:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION); cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(3, w); cap.set(4, h)
    ok, f = False, None
    for _ in range(warmup): ok, f = cap.read()
    cap.release()
    if not ok: raise RuntimeError(f"camera {index}: no frame")
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="human(virtual) vs robot wrist view matching QA")
    ap.add_argument("--side", required=True, choices=["left", "right"]); ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default=None, help="robot wrist C922 image"); ap.add_argument("--human", default=None, help="HandUMI Arducam RAW image")
    ap.add_argument("--robot-index", type=int, default=None); ap.add_argument("--human-index", type=int, default=None)
    ap.add_argument("--robot-size", default="640x480"); ap.add_argument("--human-size", default="1920x1080")
    ap.add_argument("--pick", action="store_true", help="pick correspondences manually (recommended: 7-10 points)")
    ap.add_argument("--points", default=None, help='JSON {"robot": [[u,v],…], "virtual": [[u,v],…], "depth": [...], "names": [...]}')
    ap.add_argument("--fit-rpy", action="store_true", help="fit the RESIDUAL rotation (only after the physical mount is matched)")
    ap.add_argument("--mount-revision", default="", help="e.g. handumi_left_mount_v2 — recorded in the provenance block")
    ap.add_argument("--scene", default="", help="layout id, e.g. layout_03"); ap.add_argument("--notes", default="")
    ap.add_argument("--cal-dir", default=None); ap.add_argument("--spec", default=None)
    a = ap.parse_args(argv)
    rw, rh = (int(x) for x in a.robot_size.split("x")); hw, hh = (int(x) for x in a.human_size.split("x"))
    robot = cv2.imread(a.robot) if a.robot else _grab(a.robot_index, rw, rh)
    human = cv2.imread(a.human) if a.human else _grab(a.human_index, hw, hh)
    if robot is None or human is None: print("could not read the robot/human image"); return 2
    spec = load_spec(a.spec); cal_dir = Path(a.cal_dir) if a.cal_dir else None
    points = json.loads(Path(a.points).read_text()) if a.points else None
    if a.pick:
        from ..pose.calibration import SideCalibration as _SC
        cal = _SC(a.side, cal_dir=cal_dir)
        if cal.intrinsics is None: print(f"fisheye intrinsics for {a.side} missing — run tools.calibrate_fisheye first"); return 2
        vver, vw = load_virtual_wrist(cal_dir=cal_dir)
        virt = VirtualPinholeView(cal.intrinsics, vw["K_target"][a.side], vw["size"], vw["rpy_deg"][a.side], D_target=vw["D_target"][a.side]).render(human)
        r0 = cv2.resize(robot, (virt.shape[1], virt.shape[0])) if robot.shape[:2] != virt.shape[:2] else robot
        points = pick_points(r0, virt, spec)
        if not points["robot"]: print("no correspondences picked"); return 2
    r = run(robot, human, a.side, out=Path(a.out), cal_dir=cal_dir, spec=spec, points=points, fit_rpy=a.fit_rpy,
            mount_revision=a.mount_revision, notes=a.notes, scene=a.scene)
    print(r["report"])
    print()
    for n, p in r["per_landmark"].items():
        sr = f"  w×{p['width_ratio']:.2f} h×{p['height_ratio']:.2f}" if p.get("width_ratio") else ""
        print(f"  {n:22s} {p['region']:6s} {str(p['depth'] or '-'):5s} err {p['center_err_px']:6.1f} px  dx {p['dx_px']:+6.1f} dy {p['dy_px']:+6.1f}{sr}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
