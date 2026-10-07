"""[2026-09-28] Item 6: DEPLOYMENT state94 (+ binary gripper actuator path, prompts) == dataset, for REL16-v2B.

The runtime path is exercised as deployed: infer_core_v4 with V4_STATE_MODE=umi94, observations pushed into the
timestamped ring buffer via push_obs (FK + frame fix, raw gripper -> width), state built by build_state_umi76 at
t with the history interpolated at t - 50.05 ms. Recorded teleop joints are replayed at their true 30 Hz timestamps
(frame/30 s), so any difference is the runtime's semantics, not timing. Compared against the stored v2B state94 row.

Also: grip_to_cmd for V4_GRIPPER=binary uses the frozen gripper_contract_v2.json (1 -> OPEN 42, 0 -> CLOSED 0);
the deploy UI's six prompt strings are byte-identical to the dataset's tasks.parquet.
"""
import glob, os, sys, json
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model")); sys.path.insert(0, os.path.expanduser("~/umi_bridge/umi76"))
os.environ["V4_STATE_MODE"] = "umi94"; os.environ["V4_GRIPPER"] = "binary"; os.environ["V4_ACTION_MODE"] = "umi"
import infer_core_v4 as IC
import derive_v2 as DV

L = os.path.expanduser("~/holobrain-data/lerobot"); N = "/home/bh-aiteam/holobrain-data/lerobot"
DS = f"{L}/r380_umi94_rel16_v2B"
REMAP = {f"{N}/rebot_3stack_R150_headview": f"{L}/src_rebot_3stack_R150_headview",
         f"{N}/rebot_3stack_R30_day4_headview": f"{L}/src_rebot_3stack_R30_day4_headview",
         f"{N}/rebot_3stack_center675_s96": f"{L}/src_rebot_3stack_center675_s96"}
DV.REMAP.update(REMAP)
src_of = DV.row_sources(DS, os.path.expanduser("~/umi_bridge/rel16_audit/r380/r675_rbp.json"))


def col(t, c):
    a = t.column(c).combine_chunks(); return a.storage if isinstance(a, pa.ExtensionArray) else a


tabs = [pq.read_table(f) for f in sorted(glob.glob(f"{DS}/data/chunk-*/*.parquet"))]
D = {c: np.concatenate([np.asarray(col(t, c).to_pylist()) for t in tabs]) for c in ("observation.state", "episode_index", "frame_index")}
S, EP, FR = D["observation.state"], D["episode_index"], D["frame_index"]
_cache = {}


def src_q(root, e):
    if (root, e) not in _cache:
        r = REMAP[root]; out = []
        for f in sorted(glob.glob(f"{r}/data/**/*.parquet", recursive=True)):
            t = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"])
            ep = np.asarray(col(t, "episode_index").to_numpy()); m = ep == e
            if m.any():
                fr = np.asarray(col(t, "frame_index").to_numpy())[m]; q = np.asarray(col(t, "observation.state").to_pylist())[m]
                out.append((fr, q))
        fr = np.concatenate([o[0] for o in out]); q = np.concatenate([o[1] for o in out])
        _cache[(root, e)] = q[np.argsort(fr)].astype(np.float64)
    return _cache[(root, e)]


class K:          # the runtime methods, without loading a model
    kin = IC.eef_kin.Kin()
    _tcp_mat = IC.V4Inferencer._tcp_mat; push_obs = IC.V4Inferencer.push_obs
    _interp_at = IC.V4Inferencer._interp_at; build_state_umi76 = IC.V4Inferencer.build_state_umi76


rng = np.random.default_rng(1)
errs = {"HEAD": [], "FRONT": []}; blocks = {}
OFFS = {"self_hist_pos": [0, 1, 2, 38, 39, 40], "self_hist_rot": list(range(12, 18)) + list(range(50, 56)),
        "cross_pos": list(range(6, 12)) + list(range(44, 50)), "cross_rot": list(range(24, 36)) + list(range(62, 74)),
        "grip": [36, 37, 74, 75], "tcp18": list(range(76, 94))}
for pool, rows in (("HEAD", np.flatnonzero(EP < 180)), ("FRONT", np.flatnonzero(EP >= 180))):
    for i in rng.choice(rows, 40, replace=False):
        root, se, *_ = src_of[int(EP[i])]
        q = src_q(root, se); i0 = 2 * int(FR[i]) + 2
        k = K()
        for j in range(max(0, i0 - 6), i0 + 1):              # 200 ms of real 30 Hz history into the ring buffer
            qq = q[j].copy(); qq[IC.ARM_IDX] = np.radians(qq[IC.ARM_IDX])
            k.push_obs(qq, t=j / 30.0)
        st, _, _ = k.build_state_umi76()
        d = np.abs(st.astype(np.float64) - S[i].astype(np.float64))
        errs[pool].append(d.max())
        for nm, ix in OFFS.items():
            blocks.setdefault(nm, []).append(d[ix].max())
for p, v in errs.items():
    print(f"state94 runtime vs dataset {p:5s}: max|diff| med {np.median(v):.2e}  max {np.max(v):.2e}  (n {len(v)})")
for nm, v in blocks.items():
    print(f"    block {nm:14s} max {np.max(v):.2e}")
print(f"gripper actuator: contract file {IC.GRIP_CONTRACT_FILE}; grip_to_cmd(1.0)={IC.grip_to_cmd(1.0)} grip_to_cmd(0.0)={IC.grip_to_cmd(0.0)} "
      f"grip_to_cmd(0.49)={IC.grip_to_cmd(0.49)} grip_to_cmd(0.51)={IC.grip_to_cmd(0.51)}")
import importlib.util
spec = importlib.util.spec_from_file_location("ui", os.path.expanduser("~/holobrain-mac-model/mac_v4_smoke_ui.py"))
src = open(spec.origin).read()
ui_tasks = [l.split('", "', 1)[1].rsplit('")', 1)[0] for l in src.splitlines() if l.strip().startswith('("R') or l.strip().startswith('("B') or l.strip().startswith('("P')][:6]
ds_tasks = set(pd.read_parquet(f"{DS}/meta/tasks.parquet").index)
print(f"prompts: {sum(t in ds_tasks for t in ui_tasks)}/6 deploy-UI order strings byte-identical to tasks.parquet ({len(ds_tasks)} dataset tasks)")
json.dump({"state94_max": {p: float(np.max(v)) for p, v in errs.items()}, "blocks": {k: float(np.max(v)) for k, v in blocks.items()},
           "grip_cmd": [IC.grip_to_cmd(1.0), IC.grip_to_cmd(0.0)], "prompts_identical": int(sum(t in ds_tasks for t in ui_tasks))},
          open(os.path.expanduser("~/umi_bridge/rel16_audit/r380/runtime_parity_v2b.json"), "w"), indent=1)
