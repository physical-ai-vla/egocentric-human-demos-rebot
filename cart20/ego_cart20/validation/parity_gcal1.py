#!/usr/bin/env python3
"""Cross-check against the frozen ego_relcart20task_rel16_v4_nopseudoq_gcal1 (same human MASt3R tracks, same gcal1 gripper).
At the SAME physical instants (gcal1 row = left-clock raw frame start+idx), compare
  action 0:20  (A_k = inv(T_t) T(t + k UMI_DT), REL16 + g)   expected ~equal (gcal1 pairs R to the nearest left frame, we
               interpolate R on its own clock -> sub-frame differences only)
  state 0:20   (RELCART20 task anchor)                        same, plus the anchor-instant difference
usage: parity_gcal1.py <raw_root> <episode_id> [...]"""
import glob, json, pathlib, sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ego_cart20.io.raw_episode_loader import load_raw_episode
from ego_cart20.convert import convert_episode
from ego_cart20.labels.build_cart20 import build_cart20_chunk
from ego_cart20.labels.build_state20 import build_state20_relcart20
from ego_cart20.geometry.transforms import pose9_to_T
from ego_cart20.geometry.rotation6d import rotation_angle
G1 = pathlib.Path.home() / "c8/rel16ego/ego_relcart20task_rel16_v4_nopseudoq_gcal1"; SH = pathlib.Path.home() / "c8/rel16ego/rel16ego_shards"


def gcal1_rows(src):
    for s in ("rel16ego_train", "rel16ego_val"):
        e_idx = 0
        for f in sorted((SH / s).glob("*.rows.npz")):
            z = np.load(f); n = len(z["starts"])
            if f.name[:-9] == src: return s, list(range(e_idx, e_idx + n)), z
            e_idx += n
    raise KeyError(src)


def main(raw_root, eps):
    out = {}
    for e in eps:
        raw = load_raw_episode(pathlib.Path(raw_root) / e); ep = convert_episode(raw); tr = ep["_tracks"]
        s, eidx, z = gcal1_rows(e)
        tL = raw.arms["left"].t_s
        tabs = [pq.read_table(f).to_pandas() for f in sorted(glob.glob(str(G1 / s / "data/*/*.parquet")))]
        import pandas as pd; df = pd.concat(tabs); dp, dr, dg, sp, sg = [], [], [], [], []
        for k, ei in enumerate(eidx):
            d = df[df.episode_index == ei].sort_values("frame_index")
            rows = z["starts"][k] + z["idx"][k]
            for i, r in enumerate(rows):
                t = tL[r]; c, ok = build_cart20_chunk(tr["left"], tr["right"], t)
                if not ok: continue
                A = np.stack(d.action.iloc[i]).astype(np.float64)[:, :20]
                for o in (0, 10):
                    dp.append(np.linalg.norm(A[:, o:o + 3] - c[:, o:o + 3], axis=1).max() * 1e3)
                    dr.append(np.degrees(rotation_angle(np.swapaxes(pose9_to_T(A[:, o:o + 9])[:, :3, :3], -1, -2) @ pose9_to_T(c[:, o:o + 9].astype(np.float64))[:, :3, :3])).max())
                    dg.append(np.abs(A[:, o + 9] - c[:, o + 9]).max())
                S = np.asarray(d["observation.state"].iloc[i], np.float64)
                TL, gL, _ = tr["left"].sample([t]); TR, gR, _ = tr["right"].sample([t])
                mine = build_state20_relcart20(ep["_anchor"]["left"], TL[0], gL[0], ep["_anchor"]["right"], TR[0], gR[0])
                sp.append(max(np.linalg.norm(S[0:3] - mine[0:3]), np.linalg.norm(S[9:12] - mine[9:12])) * 1e3); sg.append(np.abs(S[18:] - mine[18:]).max())
        pc = lambda x: dict(n=len(x), p50=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))
        out[e] = dict(action_pos_mm=pc(dp), action_rot_deg=pc(dr), action_grip_abs=pc(dg), state_pos_mm=pc(sp), state_grip_abs=pc(sg),
                      anchor_t_s=float(ep["metadata"]["task_start_time_s"]))
        print(e, json.dumps(out[e]), flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
