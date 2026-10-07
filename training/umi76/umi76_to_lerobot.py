"""[2026-09-25] HEAD180 UMI76 converter: UMI zarr (absolute TCP) -> LeRobot dataset, upstream bimanual
UMI observation semantics at upstream's physical timing. Contract: ~/umi_bridge/umi76/UMI76_CONTRACT.md.

Derived from umi_to_lerobot_v4_3cam.py; the video decoding and dataset writing are unchanged and only the
state/action construction is new.

OLD (this file's ancestor): UMI zarr -> body-frame delta.

This is the canonical transform for v4. R150 and, later, HandUMI both pass through THIS converter, so
"same pipeline" means the same UMI -> delta -> LeRobot contract, not the same runtime Dataset object.
X-VLA's training code is not modified at all.

Per stored frame t (at the UMI step rate, 15 Hz):

    action[t]            = [ pos3 + rot6d + gripper ]  per arm, from  inv(T_t) @ T_{t+1}      -> 20D
    observation.state[t] = per arm [ prev_rel pos3+rot6d, gripper, wrt pos3+rot6d ]           -> 38D
                           prev_rel = inv(T_t) @ T_{t-1}          (the current pose relative to
                                                                   itself is identity, so it is not
                                                                   stored -- it carries no information)
                           wrt      = inv(T_other_t) @ T_this_t   (the cross-arm term v3 also uses)
    images               = left_wrist (arm 0), right_wrist (arm 1), 224x224, straight from the zarr

So all geometry is baked here and X-VLA sees plain tensors. A 16-step chunk pulled by delta_timestamps
is exactly [a_t, a_{t+1}, ..., a_{t+15}], the consecutive body-frame increments.

Episode edges are handled by dropping, never by inventing:
  * the LAST frame of an episode has no t+1, so it is not written (every stored frame has a real action)
  * the FIRST frame has no t-1, so prev_rel is identity and `state_prev_valid` is 0 for that frame
No identity action is ever written as if it were a real label.
"""
import argparse, json, os, pathlib, sys, time
import numpy as np, zarr, av, cv2, pandas as pd, glob
from scipy.spatial.transform import Rotation as Rot, Slerp
from concurrent.futures import ThreadPoolExecutor
_POOL = ThreadPoolExecutor(max_workers=2)
sys.path.insert(0, "/home/bh-aiteam/umi_bridge")
import umi_action as UA

sys.path.insert(0, "/home/bh-aiteam/universal_manipulation_interface")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()
from umi.common.pose_util import pose_to_mat, mat_to_pose10d
from r150_v4_instructions import episode_table, R150_LEROBOT

SOFT_DP_MM, SOFT_DTH_DEG = 40.0, 12.0
HARD_DP_MM, HARD_DTH_DEG = 70.0, 40.0
N_ROBOT = 2
# state layout. full38 keeps the cross-arm term; slim20 drops it so the state is shape-compatible with
# the pretrained xvla-base action encoder (max_state_dim=20 -- setting 38 grows action_encoder.fc from
# [30, 73728] to [30, 92160] and strict load fails). slim20 = per arm [prev_rel pos3+rot6d, gripper].
STATE_MODE = "full38"
# "delta" = the v4 per-frame body increment; "umi" = the shared current-anchor chunk (umi_action.build_chunk)
ACTION_MODE = "delta"
# "current" keeps the original (and frozen) semantics: action gripper = grip[t], identical to the state's
# gripper. "next" = grip[t+1], which puts the gripper on the same instant as the pose target inv(T_t)@T_{t+1}.
GRIPPER_TARGET = "current"
STATE_PER_ARM = 19

# Upstream timing, in physical seconds. sampler.py:148 walks back `idx * obs_down_sample_steps` on a
# ~59.94 Hz buffer with every latency term zero (dataset_frequeny: 0), so both the observation pair and the
# action targets are spaced 3/59.94 apart. Our zarr is 30 Hz, where that is 1.5 frames, so the offsets are
# interpolated rather than indexed -- 200 ms (3 stored frames) and 66.7 ms (1 stored frame) are both wrong.
UMI_DT = 3.0 / 59.94          # 50.05 ms
RAW_HZ = 30.0
OBS_HORIZON = 2               # low_dim_obs_horizon
A_HORIZON = 16                # action_horizon
STATE76 = 76
ACTION_PER_ARM = 10         # pos3 + rot6d + gripper


def body_delta(future_mat, base_mat):
    prev = np.concatenate([base_mat[None], future_mat[:-1]], axis=0)
    return np.linalg.inv(prev) @ future_mat


def delta_magnitudes(d):
    dp = np.linalg.norm(d[..., :3, 3], axis=-1) * 1000.0
    tr = np.clip((np.trace(d[..., :3, :3], axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return dp, np.degrees(np.arccos(tr))


GLOBAL_KEY = "observation.images.global"


def _source_episode_meta(root=None):
    root = pathlib.Path(root) if root is not None else R150_LEROBOT
    fs = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).sort_values("episode_index")
    return d.reset_index(drop=True)


def decode_global(row, n, out_res=(224, 224), root=None, transform=None):
    """Decode this episode's global frames straight from the source mp4.

    Same policy as the UMI exporter used for the wrist views: no crop, INTER_AREA to 224x224, and a
    half-frame tolerance on from_timestamp (a tight comparison drops the intended first frame in 35 of
    150 episodes and the segment then runs one short).

    `transform` applies to the GLOBAL view only. It exists for one pool: R675 was recorded with the third
    camera in front of the table rather than behind the operator's head, and a FRONT->HEAD affine fit came
    out at about +175.6 deg (rot180 residual 7.7 px, against 58.7 px for a mirror and 145.6 px for both).
    Rotating R675's global stream by 180 deg therefore puts both pools in the same global slot. It is never
    applied to a wrist view, and it is never a mirror: a mirror would flip handedness against the state and
    action, which are left untouched.
    """
    chunk = int(np.asarray(row[f"videos/{GLOBAL_KEY}/chunk_index"]).ravel()[0])
    fidx = int(np.asarray(row[f"videos/{GLOBAL_KEY}/file_index"]).ravel()[0])
    t0 = float(np.asarray(row[f"videos/{GLOBAL_KEY}/from_timestamp"]).ravel()[0])
    root = pathlib.Path(root) if root is not None else R150_LEROBOT
    path = root / "videos" / GLOBAL_KEY / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.mp4"
    frames = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]; st.thread_type = "AUTO"
        c.seek(int(max(t0 - 0.5, 0) / float(st.time_base)), stream=st)
        tol = 0.5 * float(st.average_rate.denominator) / float(st.average_rate.numerator) if st.average_rate else 1/60
        for f in c.decode(st):
            if float(f.pts * st.time_base) < t0 - tol:
                continue
            im = cv2.resize(f.to_ndarray(format="rgb24"), out_res, interpolation=cv2.INTER_AREA)
            if transform == "rot180":
                im = cv2.rotate(im, cv2.ROTATE_180)
            elif transform not in (None, "none"):
                raise ValueError(f"unknown global transform {transform!r}")
            frames.append(im)
            if len(frames) >= n:
                break
    if len(frames) < n:
        raise RuntimeError(f"global short segment {path.name}: {len(frames)}/{n}")
    return np.stack(frames[:n]).astype(np.uint8)


def _state_per_arm():
    return {"full38": 19, "slim20": 10, "s32": 16}[STATE_MODE]


# joints for state34, read once from the source dataset the zarr came from
_JOINTS = None


def _joints_all():
    global _JOINTS
    if _JOINTS is None:
        import pyarrow.parquet as _pq
        srcd = "/home/bh-aiteam/holobrain-data/lerobot/rebot_3stack_R150_headview"
        _d = _pq.read_table(sorted(pathlib.Path(srcd, "data").rglob("*.parquet"))[0]).to_pandas()
        _JOINTS = np.stack([np.asarray(v, np.float64) for v in _d["observation.state"].values])
    return _JOINTS


def features(image_hw=(224, 224)):
    h, w = image_hw
    return {
        "observation.images.global": {"dtype": "video", "shape": (h, w, 3),
                                      "names": ["height", "width", "channels"]},
        "observation.images.left_wrist": {"dtype": "video", "shape": (h, w, 3),
                                          "names": ["height", "width", "channels"]},
        "observation.images.right_wrist": {"dtype": "video", "shape": (h, w, 3),
                                           "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (STATE76,), "names": None},
        # Always a chunk here. The ancestor branched on ACTION_MODE and would silently emit a flat 20-vector
        # if --action-mode was left at its default, which the trainer accepts and the twins would not share.
        "action": {"dtype": "float32", "shape": (A_HORIZON, ACTION_PER_ARM * N_ROBOT), "names": None},
        "state_prev_valid": {"dtype": "float32", "shape": (1,), "names": None},
    }


def _traj(z, r, s, e):
    """Absolute poses, widths and timestamps for one arm over one episode, at the raw 30 Hz cadence."""
    idx = np.arange(s, e)
    pos = np.asarray(z["data"][f"robot{r}_eef_pos"].oindex[idx], np.float64)
    rv = np.asarray(z["data"][f"robot{r}_eef_rot_axis_angle"].oindex[idx], np.float64)
    w = np.asarray(z["data"][f"robot{r}_gripper_width"].oindex[idx], np.float64)[:, 0]
    tau = np.arange(len(idx)) / RAW_HZ
    return tau, pos, Rot.from_rotvec(rv), w


def _sample(tau, pos, rot, w, t):
    """Pose and width at arbitrary physical times: position and width linear, rotation slerp."""
    t = np.clip(np.asarray(t, np.float64), tau[0], tau[-1])
    p = np.stack([np.interp(t, tau, pos[:, k]) for k in range(3)], axis=-1)
    q = Slerp(tau, rot)(t)
    ww = np.interp(t, tau, w)
    m = np.zeros((len(t), 4, 4))
    m[:, :3, :3] = q.as_matrix()
    m[:, :3, 3] = p
    m[:, 3, 3] = 1.0
    return m, ww


def episode_frames(z, s, e, stride):
    """One record per query row: UMI76 state and a 16-step current-anchor action chunk.

    Query rows sit on the stored 15 Hz grid so each one lines up with a video frame, but every pose the
    state and the action refer to is interpolated at upstream's physical offsets, not at stored indices.
    A row is kept only if its history and all 16 targets fall inside the episode -- nothing is padded,
    because X-VLA does not read action_is_pad and would train on invented targets as if they were real.
    """
    tr = {r: _traj(z, r, s, e) for r in range(N_ROBOT)}
    tau = tr[0][0]
    q_local = np.arange(0, len(tau), stride)
    t_q = q_local / RAW_HZ
    ok = (t_q - UMI_DT >= tau[0] - 1e-9) & (t_q + A_HORIZON * UMI_DT <= tau[-1] + 1e-9)
    q_local, t_q = q_local[ok], t_q[ok]
    if len(t_q) < 3:
        return None, "too short for the umi76 window"
    n = len(t_q)

    cur, hist, wc, wh, act_m, act_w = {}, {}, {}, {}, {}, {}
    for r in range(N_ROBOT):
        ta, pos, rot, w = tr[r]
        cur[r], wc[r] = _sample(ta, pos, rot, w, t_q)
        hist[r], wh[r] = _sample(ta, pos, rot, w, t_q - UMI_DT)
        tk = (t_q[:, None] + (np.arange(A_HORIZON) + 1)[None, :] * UMI_DT).reshape(-1)
        m, ww = _sample(ta, pos, rot, w, tk)
        act_m[r] = m.reshape(n, A_HORIZON, 4, 4)
        act_w[r] = ww.reshape(n, A_HORIZON)

    inv_cur = {r: np.linalg.inv(cur[r]) for r in range(N_ROBOT)}
    state = np.zeros((n, STATE76), np.float32)
    # Upstream packing: keys sorted alphabetically (timm_obs_encoder.py:169), each reshaped (T, D) -> flat,
    # so timestep-major with the older frame first. Do not re-order for slim20 compatibility.
    OFF = {"r0_pos": 0, "r0_pos_wrt": 6, "r0_rot": 12, "r0_rot_wrt": 24, "r0_grip": 36,
           "r1_pos": 38, "r1_pos_wrt": 44, "r1_rot": 50, "r1_rot_wrt": 62, "r1_grip": 74}
    for r in range(N_ROBOT):
        o = 1 - r
        self_h = mat_to_pose10d(inv_cur[r] @ hist[r])          # (n,9)
        self_c = mat_to_pose10d(inv_cur[r] @ cur[r])           # identity by construction
        cross_h = mat_to_pose10d(inv_cur[o] @ hist[r])
        cross_c = mat_to_pose10d(inv_cur[o] @ cur[r])
        k = f"r{r}_"
        state[:, OFF[k + "pos"]:OFF[k + "pos"] + 3] = self_h[:, :3]
        state[:, OFF[k + "pos"] + 3:OFF[k + "pos"] + 6] = self_c[:, :3]
        state[:, OFF[k + "pos_wrt"]:OFF[k + "pos_wrt"] + 3] = cross_h[:, :3]
        state[:, OFF[k + "pos_wrt"] + 3:OFF[k + "pos_wrt"] + 6] = cross_c[:, :3]
        state[:, OFF[k + "rot"]:OFF[k + "rot"] + 6] = self_h[:, 3:9]
        state[:, OFF[k + "rot"] + 6:OFF[k + "rot"] + 12] = self_c[:, 3:9]
        state[:, OFF[k + "rot_wrt"]:OFF[k + "rot_wrt"] + 6] = cross_h[:, 3:9]
        state[:, OFF[k + "rot_wrt"] + 6:OFF[k + "rot_wrt"] + 12] = cross_c[:, 3:9]
        state[:, OFF[k + "grip"]] = wh[r]
        state[:, OFF[k + "grip"] + 1] = wc[r]

    action = np.zeros((n, A_HORIZON, ACTION_PER_ARM * N_ROBOT), np.float32)
    dp_all, dth_all = [], []
    for r in range(N_ROBOT):
        A = inv_cur[r][:, None] @ act_m[r]                      # A[k] = inv(T_t) @ T(t+(k+1)dt)
        a9 = mat_to_pose10d(A.reshape(-1, 4, 4)).reshape(n, A_HORIZON, 9)
        off = r * ACTION_PER_ARM
        action[:, :, off:off + 9] = a9
        action[:, :, off + 9] = act_w[r]
        step = np.linalg.inv(A[:, :-1].reshape(-1, 4, 4)) @ A[:, 1:].reshape(-1, 4, 4)
        dp, dth = delta_magnitudes(np.concatenate([A[:, 0], step], axis=0))
        dp_all.append(dp); dth_all.append(dth)

    prev_valid = np.ones((n, 1), np.float32)                    # history always exists here, by the mask
    dp_all = np.concatenate(dp_all); dth_all = np.concatenate(dth_all)
    stats = dict(n=n, dp_p99=float(np.percentile(dp_all, 99)), dp_max=float(dp_all.max()),
                 dth_p99=float(np.percentile(dth_all, 99)), dth_max=float(dth_all.max()),
                 soft=int(((dp_all > SOFT_DP_MM) | (dth_all > SOFT_DTH_DEG)).sum()),
                 hard=int(((dp_all > HARD_DP_MM) | (dth_all > HARD_DTH_DEG)).sum()))
    return dict(idx=q_local + s, state=state, action=action, prev_valid=prev_valid), stats


def _parse_source(spec):
    """`zarr=<path>,lerobot=<root>[,global_transform=rot180]` -> dict, with the keys checked.

    Sources are concatenated in the order given and each keeps its own instruction table and its own global
    video root, so R180 = R150 + R30 is one conversion pass rather than a parquet merge: episode indices,
    statistics and provenance then come out right by construction.
    """
    d = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        d[k.strip()] = v.strip()
    unknown = set(d) - {"zarr", "lerobot", "global_transform", "episodes_json", "select"}
    if unknown:
        raise SystemExit(f"--source: unknown key(s) {sorted(unknown)} in {spec!r}")
    for k in ("zarr", "lerobot"):
        if k not in d:
            raise SystemExit(f"--source: missing {k}= in {spec!r}")
        if not pathlib.Path(d[k]).exists():
            raise SystemExit(f"--source: {k}={d[k]} does not exist")
    d.setdefault("global_transform", None)
    d.setdefault("episodes_json", None)
    # [2026-09-28] select=<json>: convert only the zarr episodes listed in its "zarr_episodes" (zarr numbering,
    # i.e. BEFORE episodes_json maps them to the pool). Without it every episode is converted, as before.
    d.setdefault("select", None)
    if d["select"] and not pathlib.Path(d["select"]).exists():
        raise SystemExit(f"--source: select={d['select']} does not exist")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default="/home/bh-aiteam/umi_bridge/r150_umi.zarr",
                    help="single-source shorthand, paired with the default LeRobot root; "
                         "ignored when --source is given")
    ap.add_argument("--source", action="append", default=[],
                    help="zarr=<path>,lerobot=<root>[,global_transform=rot180]; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="rebot/r150_umi_v4")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--stride", type=int, default=2, help="zarr frames per stored frame (30 fps -> 15 Hz)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--gripper-target", choices=["current", "next"], default="current",
                help="which instant the action gripper refers to (see GRIPPER_TARGET)")
    ap.add_argument("--action-mode", choices=["delta", "umi"], default="delta")
    ap.add_argument("--state-mode", choices=["full38", "slim20", "s32"], default="full38")
    ap.add_argument("--drop-hard", action="store_true", help="skip episodes containing a hard-gate delta")
    a = ap.parse_args()

    global STATE_MODE, GRIPPER_TARGET, ACTION_MODE
    STATE_MODE = a.state_mode
    GRIPPER_TARGET = a.gripper_target
    ACTION_MODE = a.action_mode
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    sources = [_parse_source(x) for x in a.source] or [
        dict(zarr=a.zarr, lerobot=str(R150_LEROBOT), global_transform=None)]

    out = pathlib.Path(a.out)
    # image_writer_threads is the single biggest lever here: without it add_frame writes each PNG inline and
    # costs 4.3 s per 300 frames, with 8 threads it hands off and costs 0.01 s. Measured on this box.
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=a.fps, features=features(),
                               root=str(out), robot_type="rebot_b601", use_videos=True,
                               image_writer_threads=int(os.environ.get("CONV_WRITER_THREADS", "8")))

    t0 = time.time(); kept = dropped = 0; agg = []
    for si, src in enumerate(sources):
        print(f"[source {si}] zarr={src['zarr']}  lerobot={src['lerobot']}  "
              f"global_transform={src['global_transform']}", flush=True)
        z = zarr.open(src["zarr"], "r")
        ends = np.asarray(z["meta"]["episode_ends"][:], int)
        starts = np.concatenate([[0], ends[:-1]])
        tab = episode_table(pathlib.Path(src["lerobot"]))
        src_meta = _source_episode_meta(src["lerobot"])
        lut = dict(zip(tab["episode_index"].tolist(), tab["instruction"].tolist()))
        # A zarr built from a SUBSET of a pool is numbered 0..n-1 in the order the subset was added, which is
        # not the pool's episode numbering. Without this map, episode 37 of the zarr would be given episode
        # 37's instruction and episode 37's global video -- both belonging to a different demonstration.
        ep_map = None
        if src["episodes_json"]:
            ep_map = json.loads(pathlib.Path(src["episodes_json"]).read_text())["episodes"]
            if len(ep_map) != len(ends):
                raise SystemExit(f"episodes_json lists {len(ep_map)} episodes but the zarr holds {len(ends)}")
            print(f"  subset mapping: zarr 0..{len(ends)-1} -> pool episodes "
                  f"{ep_map[0]}..{ep_map[-1]}", flush=True)
        n_ep = len(ends) if a.episodes == 0 else min(a.episodes, len(ends))
        ep_iter = list(range(n_ep))
        if src["select"]:
            ep_iter = [int(e) for e in json.loads(pathlib.Path(src["select"]).read_text())["zarr_episodes"]]
            if len(set(ep_iter)) != len(ep_iter) or min(ep_iter) < 0 or max(ep_iter) >= len(ends):
                raise SystemExit(f"select={src['select']}: zarr_episodes must be unique and within 0..{len(ends)-1}")
            n_ep = len(ep_iter)
            print(f"  select: {n_ep} of {len(ends)} zarr episodes from {src['select']}", flush=True)
        for ep in ep_iter:
            rec, st = episode_frames(z, starts[ep], ends[ep], a.stride)
            if rec is None:
                print(f"  ep {ep:3d} skipped: {st}"); dropped += 1; continue
            if a.drop_hard and st["hard"] > 0:
                print(f"  ep {ep:3d} dropped: {st['hard']} hard-gate deltas "
                      f"(dp_max {st['dp_max']:.1f} mm, dth_max {st['dth_max']:.1f} deg)")
                dropped += 1; continue
            src_ep = ep if ep_map is None else int(ep_map[ep])
            task = lut[src_ep]
            imgs = {r: z["data"][f"camera{r}_rgb"] for r in range(N_ROBOT)}
            # the two wrist reads (JpegXl decode, ~1.1 s) and the global mp4 decode (~2.7 s) are independent;
            # run them together instead of one after the other
            wrist_fut = _POOL.submit(lambda: {r: imgs[r][rec["idx"]] for r in range(N_ROBOT)})
            # global comes from the source mp4 at the SAME (episode, frame) index as the wrist views; the
            # alignment was verified directly (MAE 0.44 / corr 0.9999, and +1 frame raises MAE 17x).
            row = src_meta[src_meta["episode_index"] == src_ep].iloc[0]
            g_all = decode_global(row, int(ends[ep] - starts[ep]),
                                  root=src["lerobot"], transform=src["global_transform"])
            wrist = wrist_fut.result()
            for i, fi in enumerate(rec["idx"]):
                ds.add_frame({
                    "observation.images.global": g_all[int(fi - starts[ep])],
                    "observation.images.left_wrist": wrist[0][i],
                    "observation.images.right_wrist": wrist[1][i],
                    "observation.state": rec["state"][i],
                    "action": rec["action"][i],
                    "state_prev_valid": rec["prev_valid"][i],
                    "task": task,
                })
            # the three camera keys encode in separate processes: 5.17 -> 4.46 s per 300-frame
            # episode, measured. Not the 3x the three streams suggest, because the encode is only
            # part of save_episode and each worker pays process startup.
            ds.save_episode(parallel_encoding=True)
            kept += 1; agg.append(st)
            if ep % 10 == 0 or ep == ep_iter[-1]:
                print(f"  s{si} ep {ep:3d}/{n_ep}  frames {st['n']:5d}  dp_p99 {st['dp_p99']:6.2f} mm  "
                      f"soft {st['soft']:3d}  hard {st['hard']:2d}  [{time.time()-t0:6.1f}s]", flush=True)

    prov = dict(converter="umi_to_lerobot_v4.py",
                obs_semantics="prev_rel inv(T_t) @ T_{t-1} + gripper + wrt, baked",
                sources=[dict(zarr=x["zarr"], lerobot=x["lerobot"], global_transform=x["global_transform"],
                              **({"select": x["select"],
                                  "select_zarr_episodes": json.loads(pathlib.Path(x["select"]).read_text())["zarr_episodes"]}
                                 if x["select"] else {})) for x in sources],
                source_zarr=sources[0]["zarr"],
                global_transform=sources[0]["global_transform"],
                camera_domain_merge=" + ".join(
                    f"{pathlib.Path(x['lerobot']).name}"
                    + (f"({x['global_transform']} global)" if x["global_transform"] else "")
                    for x in sources), repo_id=a.repo_id, fps=a.fps, stride=a.stride,
                episodes_kept=kept, episodes_dropped=dropped,
                frames=int(sum(s["n"] for s in agg)),
                gates=dict(soft_dp_mm=SOFT_DP_MM, soft_dth_deg=SOFT_DTH_DEG,
                           hard_dp_mm=HARD_DP_MM, hard_dth_deg=HARD_DTH_DEG,
                           soft_total=int(sum(s["soft"] for s in agg)),
                           hard_total=int(sum(s["hard"] for s in agg))),
                instruction_source=[str(pathlib.Path(x["lerobot"]) / "meta") for x in sources],
                state_mode=STATE_MODE, gripper_target=GRIPPER_TARGET, action_mode=ACTION_MODE,
                action_semantics=("umi current-anchor inv(T_t)@T_(t+k), k=1..16"
                                  if ACTION_MODE == "umi" else "body-frame delta inv(T_t)@T_(t+1)"),
                # hard guard for every consumer: the label is ALREADY relative to the query pose, so no
                # loader, collator or trainer may relativise or horizon-expand it a second time
                pose_repr=("current_anchor_relative" if ACTION_MODE == "umi" else "body_delta"),
                already_relativized=(ACTION_MODE == "umi"),
                horizon=(UA.CHUNK if ACTION_MODE == "umi" else 1),
                state_semantics="UMI-style local relative proprioception "
                                "(prev_rel = inv(T_t) @ T_(t-1) + current width), "
                                "not the exact upstream UMI lowdim schema",
                state_dim=STATE76,
                action_dim=ACTION_PER_ARM * N_ROBOT)
    (out / "v4_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
