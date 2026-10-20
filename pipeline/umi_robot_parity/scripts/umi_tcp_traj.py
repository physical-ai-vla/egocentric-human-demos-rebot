"""UMI TCP trajectory from a tag_rig_record take (2026-10-20, replay parity A = "UMI measured"). Per colour frame with >= 1 tag of that UMI:
T_C_rig (scripts/tag_rig_fuse.py, rotation imu|tag, optional depth range) -> TCP point p_C = T_C_rig * tcp_in_rig_m (touch-TCP calibration,
configs/calibration/umi_tag_rig_v001.json), then into the robot base with T_B_C (configs/calibration/robot_cam_v001.json) when it exists.
For --proto axes takes it also writes the stroke segments (hand, axis, direction, repeat, outward / return / hold windows in host ns).
Robot base convention used for the expected directions: +X 앞, +Y 왼쪽, +Z 위 (reBot URDF base, right READY TCP ~ (0.45, -0.30, 0.28)).
    .venv/bin/python scripts/umi_tcp_traj.py datasets/tag_rig/<take> --hand 오른손 [--rot imu|tag] [--depth] [--robot-cal <json>]
-> <take>/umi_traj_<right|left>.npz (t_ns, p_C, R_C_tcp, ntags[, p_B, R_B_tcp]) + <take>/segments.json   (unwritable take -> <take>_derived/)"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from tag_rig_fuse import fuse_take, SIDE      # noqa: E402
DIRS = {"앞으로": ("X", +1), "뒤로": ("X", -1), "왼쪽으로": ("Y", +1), "오른쪽으로": ("Y", -1), "위로": ("Z", +1), "아래로": ("Z", -1)}


def segments(meta: dict) -> list[dict]:
    """axes protocol -> strokes. sched entries: hold '<hand>, <ax>축, <r>회, 멈추세요' -> move '<hand> <dir> <n>센티 천천히' -> hold -> return"""
    t0 = int(meta["t_start_monotonic_ns"]); S = meta["protocol"]; out = []
    ns = lambda e, a=0.0: int(t0 + (e["t0"] + a * e["dur"]) * 1e9)
    for i, e in enumerate(S):
        w = e["text"].split()
        if "센티 천천히" not in e["text"] or w[1] not in DIRS: continue
        hold = S[i - 1]["text"].split(","); ax, sg = DIRS[w[1]]
        out.append(dict(hand=w[0], axis=ax, sign=sg, dir=w[1], rep=int(hold[2].strip().rstrip("회")), cm=float(w[2].replace("센티", "")),
                        hold_pre=[ns(S[i - 1]), ns(S[i - 1], 1)], out=[ns(e), ns(e, 1)], hold_mid=[ns(S[i + 1]), ns(S[i + 1], 1)], back=[ns(S[i + 2]), ns(S[i + 2], 1)],
                        hold_post=[ns(S[i + 3]), ns(S[i + 3], 1)] if i + 3 < len(S) else None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--hand", default="오른손", choices=list(SIDE)); ap.add_argument("--rot", default="imu", choices=["imu", "tag"])
    ap.add_argument("--depth", action="store_true"); ap.add_argument("--tag-size", type=float, default=0.048)
    ap.add_argument("--rig", default=str(ROOT / "configs/calibration/umi_tag_rig_v001.json")); ap.add_argument("--robot-cal", default=str(ROOT / "configs/calibration/robot_cam_v001.json"))
    a = ap.parse_args(); E = Path(a.take); meta = json.loads((E / "meta.json").read_text())
    rig = json.loads(Path(a.rig).read_text()); name, side = SIDE[a.hand]; tcp = np.array(rig[name]["tcp_in_rig_m"])
    F = fuse_take(E, a.hand, rig, a.tag_size, use_depth=a.depth, rot_source=a.rot)
    if F is None: sys.exit(f"{a.hand} 유미 태그가 한 번도 안 보임")
    T = F["T_C_rig"]; p_C = T[:, :3, :3] @ tcp + T[:, :3, 3]
    res = dict(t_ns=F["t_ns"], p_C=p_C, R_C_tcp=T[:, :3, :3], ntags=F["ntags"], rot=a.rot, depth=a.depth)
    if Path(a.robot_cal).exists():
        cal = json.loads(Path(a.robot_cal).read_text()); TBC = np.array(cal["T_B_C"])
        res.update(p_B=p_C @ TBC[:3, :3].T + TBC[:3, 3], R_B_tcp=TBC[:3, :3] @ T[:, :3, :3], robot_cal=a.robot_cal)
        print(f"로봇 베이스 좌표로 변환 ({a.robot_cal})")
    else: print(f"[주의] 로봇 캘리브레이션 없음 ({a.robot_cal}) -- 카메라 좌표만 저장 (replay_parity 는 --cam-axes 로 근사 비교만 가능)")
    out = E if os.access(E, os.W_OK) else E.parent / (E.name + "_derived"); out.mkdir(exist_ok=True)
    np.savez(out / f"umi_traj_{side}.npz", **res)
    gaps = np.diff(F["t_ns"]) / 1e9
    print(f"{a.hand}: 프레임 {len(p_C)}/{meta['frames']} ({len(p_C) / max(meta['frames'], 1) * 100:.0f}%), 태그 2개 이상 {np.mean(F['ntags'] >= 2) * 100:.0f}%, "
          f"최대 끊김 {gaps.max() if len(gaps) else 0:.2f}s, 태그-융합 회전차 중앙 {np.median(F['tag_vs_fused_deg']):.1f}° -> {out / f'umi_traj_{side}.npz'}")
    if meta.get("proto") == "axes":
        seg = segments(meta); (out / "segments.json").write_text(json.dumps(seg, ensure_ascii=False, indent=1)); print(f"구간 {len(seg)}개 -> {out / 'segments.json'}")


if __name__ == "__main__":
    main()
