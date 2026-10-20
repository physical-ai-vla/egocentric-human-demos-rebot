"""Visual-teleop episode -> common DiT action contract (2026-10-14, spec 32).

Per timestep 20-D relative Cartesian action  [left dxyz(3) rot6d(6) | right dxyz(3) rot6d(6) | g_L | g_R], g 0 = closed / 1 = open,
computed by the EXISTING contract exporter ego_teleop/recorder/umi_contract_export.py (ego_cart20 relpose: UMI-relative,
rot6d = first two ROWS, FK -> reBot dataset-TCP axes via umi_robot_frame) -- the same contract the UMI/ego data uses.
Source = robot-EXECUTED TCP: FK of the OBSERVED joints (obs_T); a preview/mock run without observation falls back to FK of the
accepted command (exec_T) and says so in meta. Never the human target. Gripper label = OBSERVED jaw (g_from_raw), else g_exec.
The per-step 20-D action is action[:, 0, :] (k = 1); the full (n, H, 20) chunk is kept too.
Clock + horizon + validation come from Agent A's SHARED contract (~/laundry_system/contracts/action_contract.py: ACTION_DT_S 0.05 =
20 Hz, DEFAULT_HORIZON 32, rot6d rows, dataset-TCP axes x = approach = reBot FK z), imported, never redefined. Every chunk is checked
with its validate_action_chunk and the first one is cross-checked against its encode_action_chunk.
LeRobot v3 parquet/video writing is left to the shared writer once Agent B fixes the feature schema; this writes contract.npz."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sys
from ego_teleop.recorder.umi_contract_export import export_umi_contract
from ego_teleop.transforms.umi_robot_frame import fk_to_dataset_tcp

SHARED_CONTRACT_ROOT = Path.home() / "laundry_system"


def shared_contract():
    """Agent A's action contract module (read-only import)."""
    if str(SHARED_CONTRACT_ROOT) not in sys.path: sys.path.insert(0, str(SHARED_CONTRACT_ROOT))
    from contracts import action_contract
    return action_contract

SIDES = ("left", "right")


def export_episode(ep_dir: str | Path, *, horizon: int | None = None, dt_s: float | None = None, skill_id: str | None = None,
                   max_gap_s: float = 0.1) -> dict:
    ep = Path(ep_dir).expanduser(); a = dict(np.load(ep / "ticks.npz", allow_pickle=False)); meta = json.loads((ep / "metadata.json").read_text())
    t = a["t_ns"].astype(np.float64) / 1e9
    src = "obs_T" if all(np.isfinite(a[f"obs_T_{s}"]).all(axis=(1, 2)).any() for s in SIDES) else "exec_T"
    T, valid, g, ts = {}, {}, {}, {}
    for s in SIDES:
        Ts = a[f"{src}_{s}"]; ok = np.isfinite(Ts).all(axis=(1, 2))
        gs = a[f"obs_g_{s}"] if np.isfinite(a[f"obs_g_{s}"]).any() else a[f"g_exec_{s}"]
        T[s], valid[s], g[s], ts[s] = Ts[ok], np.ones(int(ok.sum()), bool), gs[ok], t[ok]
    ac = shared_contract()
    if horizon is None: horizon = int(ac.DEFAULT_HORIZON)
    if dt_s is None: dt_s = float(ac.ACTION_DT_S)
    out = export_umi_contract(ts, T, valid, g, horizon=horizon, dt_s=dt_s, max_gap_s=max_gap_s, data_source=meta.get("data_source", "VISUAL_TELEOP"))
    act = out["action"]
    bad = [i for i in range(len(act)) if not ac.validate_action_chunk(np.asarray(act[i], np.float32), horizon=horizon).ok]
    if bad: raise ValueError(f"{len(bad)} chunks fail the shared contract validation (first: {ac.validate_action_chunk(np.asarray(act[bad[0]], np.float32), horizon=horizon).reason})")
    xcheck = None
    if len(act):                       # cross-check chunk 0 against the shared encoder (same executed poses, same clock)
        i0 = int(out["action_index"][0]) if "action_index" in out else 0; tc = np.asarray(out["t"], np.float64)
        def pose_at(s, tq):
            j = int(np.argmin(np.abs(ts[s] - tq))); return fk_to_dataset_tcp(T[s][j]), float(g[s][j])
        if np.all(np.abs(np.diff(ts["left"])) < max_gap_s) and np.all(np.abs(np.diff(ts["right"])) < max_gap_s):
            t0 = tc[i0]; fut = [t0 + k * dt_s for k in range(1, horizon + 1)]
            near = lambda s, tq: float(np.min(np.abs(ts[s] - tq)))
            if max(near(s, x) for s in SIDES for x in [t0] + fut) < 1e-6:   # exact sample times only (no interpolation ambiguity)
                (TL0, _), (TR0, _) = pose_at("left", t0), pose_at("right", t0)
                FL = [pose_at("left", x) for x in fut]; FR = [pose_at("right", x) for x in fut]
                ref = ac.encode_action_chunk(TL0, np.array([a for a, _ in FL]), TR0, np.array([a for a, _ in FR]), np.array([b for _, b in FL]), np.array([b for _, b in FR]))
                xcheck = float(np.max(np.abs(ref - act[0])))
    np.savez_compressed(ep / "contract.npz", action_chunk=act, action20=act[:, 0, :] if len(act) else np.zeros((0, 20)),
                        **{k: v for k, v in out.items() if k not in ("action", "meta") and isinstance(v, np.ndarray)})
    m = dict(out.get("meta", {}), label_source=src, contract_version=ac.CONTRACT_VERSION, shared_encoder_max_abs_diff=xcheck, skill_id=skill_id or meta.get("skill_id"), data_source=meta.get("data_source"),
             dt_s=dt_s, horizon=horizon, action20_layout="[L dxyz rot6d | R dxyz rot6d | gL gR], k=1, g 0=closed 1=open")
    (ep / "contract_meta.json").write_text(json.dumps(m, indent=1, default=str)); return m
