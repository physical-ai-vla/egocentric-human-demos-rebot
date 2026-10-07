"""[2026-09-22] Real-robot inference core for the r150-umi-v4 X-VLA checkpoint (body-frame SE(3) deltas).

Same seat as infer_core_umi.UmiInferencer, and deliberately the same conventions (eef_kin FK/IK, FLIP_IDX,
DQ/FK guards, the umi_schema gripper scale). What differs is the action semantics, and that difference is the
whole reason this file exists:

    v3 (UMI)  action[k] is relative to ONE base, the current pose:  T_k = base @ rel_k        (cumulative)
    v4 (ours) action[k] is relative to the PREVIOUS step:           T_k = T_{k-1} @ dT_k      (incremental)

Deploying an incremental chunk through the cumulative path collapses the trajectory onto its first step -- the
A3 checkpoint already cost a day to that exact confusion -- so the composition here is sequential, and it is a
RIGHT multiplication because the delta lives in the body frame (`inv(T_t) @ T_{t+1}`). A left multiplication
would apply the rotation about the base frame and send the arm somewhere else entirely.

observation.state is rebuilt the way the converter wrote it: per arm [prev_rel pos3+rot6d, gripper width],
prev_rel = inv(T_t) @ T_{t-1}, which needs the two most recent observations.

Safety for the first real-robot runs: every per-step delta is clamped to V4_CLAMP_MM / V4_CLAMP_DEG BEFORE it
is composed (scaling translation and rotation angle, direction preserved), and the gripper is HELD at its
current raw count unless V4_GRIPPER=predict. The point of the first test is direction and magnitude of the
arm motion, not the task.

[2026-09-23] V4_ACTION_MODE=umi switches the decode to the UMI current-anchor chunk the B180/B663 runs are
trained on. It is not a variant of the composition below, it is the absence of one:

    delta   T_k = T_{k-1} @ dT_k        each step on the previous PREDICTED pose
    umi     T_k = T_now   @ A_k         every step on the SAME measured pose

The forbidden form is `T_{k-1} @ A_k`, which would treat a cumulative displacement as an increment and run
the arm k times too far. There is no sequential path in the umi branch at all, so it cannot be written.

The clamp scales with the horizon in umi mode. A_k is the displacement over k steps, not one step, so
clamping every A_k to the per-step limit would flatten the trajectory into a sphere around T_now; the
envelope is k * V4_CLAMP_MM instead.

Env: V4_CKPT, V4_DEVICE (mps), V4_ACTION_MODE (delta|umi), V4_CLAMP_MM (3), V4_CLAMP_DEG (1),
     V4_N_ACTION (1), V4_GRIPPER (hold|predict), V4_DQ_MAX (0.6), V4_FK_MAX_MM (60), V4_TASK,
     V4_GLOBAL_ROT180 (0) -- rotate the GLOBAL feed 180 deg to match the training convention.
"""
import collections, json, os, sys, base64, math, time, numpy as np, cv2, torch

sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model/umi_pkg"))
from scipy.spatial.transform import Rotation as Rot, Slerp
import eef_kin
from umi.common.pose_util import pose_to_mat, mat_to_pose10d, pose10d_to_mat

# [2026-09-30 user] REL-only ablation checkpoints (HEAD180-RELCART20-RELONLY-V3 / R312C-RELCART20-RELONLY-V4) were trained with
# action channels 20:32 hard-zeroed at the transformer input AND output. Deployment must apply the SAME contract at every denoising
# step, or the unsupervised 20:32 values re-enter the transformer. V4_RELONLY=1 installs it (the training plugin's own function).
# Off by default: joint/FK checkpoints are unaffected. Such checkpoints must be run with pure Pink (the dq head is unsupervised).
RELONLY = os.environ.get("V4_RELONLY", "0") == "1"
if RELONLY:
    sys.path.insert(0, os.path.expanduser("~/umi_bridge/rel16_audit/relonly"))
    import rel16_aux_relonly as _RO
    _RO.install_zero_mask()
    if os.environ.get("IK_BACKEND", "numerical") in ("pinkdq", "policydq"):
        raise SystemExit("[infer_core_v4] V4_RELONLY=1 with IK_BACKEND=pinkdq/policydq: the dq head is unsupervised -- use pink")

DEVICE = os.environ.get("V4_DEVICE", "mps")
CKPT = os.environ.get("V4_CKPT", os.path.expanduser("~/holobrain-mac-model/ckpt_v4base_of3000"))
# The chunk length is not carried by any weight -- X-VLA uses it only to size the initial flow sample and to
# pad targets -- so a checkpoint will happily emit any number of rows. That is not permission to ask a
# 16-trained checkpoint for 32: rows past its training horizon are extrapolation with nothing to check them
# against. V4_CHUNK is for checkpoints TRAINED at that horizon; left at 0 the checkpoint's own value stands.
CHUNK_OVERRIDE = int(os.environ.get("V4_CHUNK", "0"))
DTYPE = os.environ.get("V4_DTYPE", "fp32")
if DTYPE not in ("fp32", "fp16", "bf16"):   # [2026-10-02 user] bf16 added (weights + X-VLA input cast via config.dtype)
    raise SystemExit(f"V4_DTYPE must be fp32, fp16 or bf16, not {DTYPE!r}")
ACTION_MODE = os.environ.get("V4_ACTION_MODE", "delta")
if ACTION_MODE not in ("delta", "umi"):
    raise SystemExit(f"V4_ACTION_MODE must be delta or umi, not {ACTION_MODE!r}")

# robot_service camera key -> the dataset's camera key -> what the policy was trained to call it
CAM_MAP = {"middle": "global", "left": "left_wrist", "right": "right_wrist"}
RENAME = {"observation.images.global": "observation.images.image",
          "observation.images.left_wrist": "observation.images.image2",
          "observation.images.right_wrist": "observation.images.image3"}

ARM_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]; GRIP_IDX = [6, 13]
FLIP_IDX = [0, 1, 5, 7, 8, 12]               # follower/observation frame -> leader frame, as /execute_step wants
DQ_MAX = float(os.environ.get("V4_DQ_MAX", "0.6"))
IK_LOG = os.path.expanduser(os.environ.get("V4_IK_LOG", "~/v4_ik_waypoints.jsonl"))
IK_BACKEND = os.environ.get("IK_BACKEND", "numerical")   # [2026-09-29] numerical (PyRoki) | learned (LIK0) | pink | [2026-10-01] diffik (one Jacobian step)
# [2026-10-03 user] IK_BACKEND=curobo: horizon-aware cuRobo IK on the 5090 (curobo_ik_server.py, CUDA only -> remote). One request per
# chunk = every predicted waypoint from the measured q; the server picks one continuous joint path (GPU multi-seed IK + DP over the
# waypoints). Same TCP/frame as pink (dataset frame, no R_DS_TO_FK). No silent fallback: a failed request stops the cycle.
CUROBO_URL = os.environ.get("V4_CUROBO_URL", "http://100.64.0.5:8790")
# [2026-10-03 user "추론 5090 통해서"] V4_REMOTE_INFER=<url of xvla_infer_server.py on the 5090>: the policy forward (pre + flow sampling
# + post) runs there on V4_REMOTE_CKPT (the same checkpoint, a path on the 5090); cameras, state, IK and execution stay here. The
# local model is still loaded (ablation tools, fallback when V4_REMOTE_INFER is cleared at runtime).
# [2026-10-03 user "rtc 구현 + rtc만 켜고 끌 수 있게"] Real-Time Chunking (Black et al. 2025, PI; LeRobot policies/rtc) for X-VLA.
# The new chunk is generated as an inpainting of the previous chunk's not-yet-executed part, so consecutive chunks join without a jump.
# X-VLA's sampler predicts the CLEAN action x^ at every flow step (x_t = noise*t + x^*(1-t), t: 1 -> 0), so LeRobot's velocity-form
# guidance v_g = v - gw * J^T(W*(Y - x^)) becomes, with x^ = x_t - t*v:  x^_g = x^ + t * gw * J^T(W*(Y - x^)),  J = d x^ / d x_t
# (one backward through the action transformer per flow step; the VLM encoding is computed once, no grad).
#   Y   previous chunk targets (absolute base-frame poses + widths, as executed) re-expressed in the NEW observation's current-anchor
#       frame (umi action: A_k = M_now^-1 T_prev), normalised with the checkpoint's own MEAN_STD action stats; channels 20:32 = 0.
#   alignment  prev index j* = the previous target nearest the measured TCP (both arms); new step i <-> prev step j*+1+i
#   W   1 for the first V4_RTC_DELAY steps, linear decay to 0 at V4_RTC_HORIZON (LeRobot LINEAR schedule), 0 after / past the leftover;
#       channels 20:32 never guided.  gw = min(c * inv_r2, V4_RTC_MAX_GW), tau = 1 - t (LeRobot formula).
# No previous chunk (first inference of a /run, RESET, older than V4_RTC_MAX_AGE_S) -> plain sampling. Remote inference: the server runs the same sampler (10-06).
RTC_ON = os.environ.get("V4_RTC", "0") == "1"
RTC_DELAY = int(os.environ.get("V4_RTC_DELAY", "2"))
RTC_HORIZON = int(os.environ.get("V4_RTC_HORIZON", "8"))
RTC_MAX_GW = float(os.environ.get("V4_RTC_MAX_GW", "10.0"))
RTC_MAX_AGE_S = float(os.environ.get("V4_RTC_MAX_AGE_S", "10.0"))


def _rtc_weights(d, h, total):
    w = torch.zeros(total); d = max(0, min(d, total)); h = max(d, min(h, total))
    w[:d] = 1.0
    if h > d:
        w[d:h] = torch.linspace(1, 0, h - d + 2)[1:-1]
    return w


def _install_xvla_rtc():
    import lerobot.policies.xvla.modeling_xvla as MX
    M = [c for c in vars(MX).values() if isinstance(c, type) and hasattr(c, "generate_actions") and c.__module__ == MX.__name__][0]
    if getattr(M, "_rtc_patched", False):
        return
    orig = M.generate_actions

    def generate_actions(self, input_ids, image_input, image_mask, domain_id, proprio, steps, x1=None):
        R = getattr(self, "_rtc", None)
        if R is None:
            return orig(self, input_ids, image_input, image_mask, domain_id, proprio, steps, x1=x1)
        self.eval()
        dt_ = self._get_target_dtype()
        image_input = image_input.to(dtype=dt_); proprio = proprio.to(dtype=dt_)
        enc = self.forward_vlm(input_ids, image_input, image_mask)
        assert self.target_align is None and self.xy_source == "none", "RTC path: target_align / xy_source not supported"
        B = input_ids.shape[0]; A = self.dim_action
        if x1 is None:
            x1 = torch.randn(B, self.chunk_size, A, device=proprio.device, dtype=dt_)
        else:
            x1 = x1.to(device=proprio.device, dtype=dt_)
        Y = R["Y"].to(device=proprio.device, dtype=torch.float32)[None]          # (1, T, A)
        W = R["W"].to(device=proprio.device, dtype=torch.float32)[None]          # (1, T, A)
        action = torch.zeros_like(x1); steps = max(1, int(steps)); gws = []; errs = []
        for i in range(steps, 0, -1):
            tt = i / steps
            t = torch.full((B,), tt, device=proprio.device, dtype=dt_)
            x_t = (x1 * tt + action * (1 - tt)).detach().requires_grad_(True)
            with torch.enable_grad():
                proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
                pred = self.transformer(target_embed=None, target_token=None, domain_id=domain_id, action_with_noise=x_t_m,
                                        proprio=proprio_m, t=t, **enc)
                err = (Y - pred.float()) * W
                corr = torch.autograd.grad(pred.float(), x_t, err.detach())[0].float()
            tau = 1.0 - tt
            if tau <= 0:
                gw = RTC_MAX_GW
            else:
                inv_r2 = ((1 - tau) ** 2 + tau ** 2) / max((1 - tau) ** 2, 1e-12)
                gw = min(((1 - tau) / tau) * inv_r2, RTC_MAX_GW)
            action = (pred.float() + tt * gw * corr).to(dt_).detach()
            gws.append(round(gw, 2)); errs.append(round(float(err.abs().sum() / W.sum().clamp(min=1)), 4))
        self._rtc_log = dict(gw=gws, err=errs)
        return self.action_space.postprocess(action)
    M.generate_actions = generate_actions; M._rtc_patched = True
    print("[infer_core_v4] RTC sampler installed on X-VLA (active only when V4_RTC / /mode rtc=1 and a previous chunk exists)", flush=True)


REMOTE_INFER = os.environ.get("V4_REMOTE_INFER", "")
REMOTE_CKPT = os.environ.get("V4_REMOTE_CKPT", "")
_REMOTE_SESS = None
REMOTE_LAST = {}


def remote_forward(images_b64_by_view, state20, task, noise=None, rtc=None):
    global _REMOTE_SESS
    if _REMOTE_SESS is None:
        import requests as _rq; _REMOTE_SESS = _rq.Session()
    t0 = time.time()
    payload = {"ckpt": REMOTE_CKPT, "dtype": DTYPE, "images": images_b64_by_view, "state": np.asarray(state20, float).tolist(), "task": task,
               "rot180": ["global"] if GLOBAL_ROT180 else [], "mirror": ["global"] if GLOBAL_MIRROR else [],
               "denoise": int(os.environ["V4_DENOISE_STEPS"]) if os.environ.get("V4_DENOISE_STEPS") else None,
               "noise": None if noise is None else np.asarray(noise.detach().float().cpu().numpy())[0].tolist(),
               # [2026-10-06] RTC on the server (xvla_infer_server_rtc.py): previous-chunk prefix Y / weights W
               "rtc": None if rtc is None else {"Y": rtc["Y"].tolist(), "W": rtc["W"].tolist(), "max_gw": RTC_MAX_GW}}
    r = _REMOTE_SESS.post(REMOTE_INFER + "/infer", json=payload, timeout=30).json()
    if "act" not in r:
        raise RuntimeError(f"remote inference failed: {r}")
    REMOTE_LAST.update(rt_ms=round((time.time() - t0) * 1000, 1), server_ms=r["ms"], gpu_ms=r["gpu_ms"], rtc_log=r.get("rtc"))
    return np.asarray(r["act"], np.float64)



_CUROBO_SESS = None


def curobo_solve(q12, pos, quat, timeout=10.0):
    global _CUROBO_SESS
    if _CUROBO_SESS is None:
        import requests as _rq; _CUROBO_SESS = _rq.Session()          # keep-alive: no TCP/HTTP setup per chunk over the tailnet
    r = _CUROBO_SESS.post(CUROBO_URL + "/solve", json={"q0": [float(x) for x in q12], "pos": np.asarray(pos, float).tolist(),
                                             "quat": np.asarray(quat, float).tolist()}, timeout=timeout).json()
    return (np.asarray(r["q"], np.float64), np.asarray(r["pos_err_mm"], np.float64), np.asarray(r["success"], bool),
            dict(ms=r["ms"], rot_err_deg=r["rot_err_deg"]))
YAW_MAX_STEP_DEG = float(os.environ.get("V4_YAW_MAX_STEP_DEG", "0"))   # [2026-10-01] 0 = off; e.g. 8 with V4_PINK_LOCK= (joint5 free, slew-limited)
PINK_LOCK = tuple(x for x in os.environ.get("V4_PINK_LOCK", "").split(",") if x)   # [2026-09-30] pink|pinkdq: joint-name substrings hard-locked at the measured q (e.g. wrist_yaw)
PINK_ORI = os.environ.get("V4_PINK_ORI", "full")          # [2026-09-30] IK_BACKEND=pink|pinkdq: full | yaw (position + base-z rotation; roll/pitch excluded) | pitch (position + approach-axis elevation; roll/yaw free) | pitch_hold (roll/yaw held at the measured pose, pitch from the model) | yaw_hold (roll/pitch from the model, yaw held at the measured pose)
FK_MAX_MM = float(os.environ.get("V4_FK_MAX_MM", "60"))
# mutable so the UI can raise/lower the envelope between presses without a restart: a 3 mm command turned
# out to be smaller than what the servos resolve (2.204 mm commanded -> 0.126 mm measured), so the first
# question "does the arm move the way the model predicts" needs a command big enough to measure.
LIMITS = {"mm": float(os.environ.get("V4_CLAMP_MM", "10")), "deg": float(os.environ.get("V4_CLAMP_DEG", "2"))}
CLAMP_MM = LIMITS["mm"]
CLAMP_DEG = LIMITS["deg"]


def set_limits(mm=None, deg=None):
    if mm is not None:
        LIMITS["mm"] = float(mm)
    if deg is not None:
        LIMITS["deg"] = float(deg)
    return dict(LIMITS)
N_ACTION = int(os.environ.get("V4_N_ACTION", "4"))
# Which horizon point to execute, 1-based. A[1] is one 15 Hz step of ground truth -- about 0.4 mm -- which is
# below this policy's own per-step noise, so executing it integrates the noise instead of the motion. A later
# waypoint is a larger, better-conditioned target; it is NOT a sum of the earlier ones, because every A[k]
# hangs off the same T_now. This changes only which point of the predicted trajectory is sent to IK.
EXEC_K = int(os.environ.get("V4_EXEC_K", "1"))
# [2026-10-01 user] inference-side stabilisers, all OFF unless set. Order inside infer(): chunk -> temporal ensemble -> adaptive k ->
# near-zero hold -> (UI) IK on waypoint k-1. Each cycle executes ONE waypoint (chunk index k-1) and re-infers, so the robot advances k
# dataset steps per cycle; a step counter (sum of the k actually chosen) lines older chunks up with the current one.
# V4_TE=1  ACT-style temporal ensemble over the last V4_TE_N (3) chunks, weights V4_TE_W ("1,0.5,0.25", newest first). Waypoint k of
#          the new chunk = weighted mean, in the BASE frame, of the predictions of the same moment by each stored chunk still inside
#          its 16-step horizon (chunk inferred s steps earlier -> its index k + s). Translation only unless V4_TE_ROT=1 (weighted
#          quaternion mean) / V4_TE_GRIP=1 (width mean). Raw chunks are the history. Cleared on /run, RESET ANCHOR, gap > V4_TE_MAXGAP_S.
# V4_ADAPTIVE_K="4,2,1"  per-cycle k: far -> 4, medium -> 2, near/align -> 1. Distance = max over arms of the (ensembled) predicted
#          translation at the chunk end (index 15) from the measured TCP: > V4_AK_FAR_MM (40) far, > V4_AK_NEAR_MM (15) medium, else
#          near. Forced to near when either arm's predicted gripper crosses V4_GRIP_THRESH within the chunk (grasp/release = align phase).
#          V4_EXEC_K is ignored while this is set.
# V4_DEADBAND_XY_MM / V4_DEADBAND_Z_MM  near-zero hold: per arm, if the xy norm of (target - measured TCP) < eps_xy both x and y are
#          held at the measured value; z separately if |dz| < eps_z. Rotation untouched. Decided on the (ensembled) displacement at
#          chunk step V4_DEADBAND_REF_K (8) -- "the policy predicts < eps of motion over 8 steps" -- and applied to every waypoint.
#          (V4_DEADBAND_MM="x,y,z" = per-axis variant.)
TE_ON = os.environ.get("V4_TE", "0") == "1"
TE_W = np.array([float(x) for x in os.environ.get("V4_TE_W", "1,0.5,0.25").split(",")], np.float64)[:int(os.environ.get("V4_TE_N", "3"))]
TE_ROT = os.environ.get("V4_TE_ROT", "0") == "1"; TE_GRIP = os.environ.get("V4_TE_GRIP", "0") == "1"
TE_MAXGAP_S = float(os.environ.get("V4_TE_MAXGAP_S", "15"))
_ak = os.environ.get("V4_ADAPTIVE_K", "").strip()
ADAPTIVE_K = [int(x) for x in _ak.split(",")] if _ak else None
AK_FAR_MM = float(os.environ.get("V4_AK_FAR_MM", "40")); AK_NEAR_MM = float(os.environ.get("V4_AK_NEAR_MM", "15"))
_db = os.environ.get("V4_DEADBAND_MM", "").strip()
DEADBAND_MM = np.array([float(x) for x in _db.split(",")], np.float64) if _db else None
if DEADBAND_MM is not None:
    assert DEADBAND_MM.shape == (3,), "V4_DEADBAND_MM needs x,y,z"
DB_XY_MM = float(os.environ["V4_DEADBAND_XY_MM"]) if os.environ.get("V4_DEADBAND_XY_MM") else None
DB_Z_MM = float(os.environ["V4_DEADBAND_Z_MM"]) if os.environ.get("V4_DEADBAND_Z_MM") else None
DB_REF_K = int(os.environ.get("V4_DEADBAND_REF_K", "8"))
GRIPPER_MODE = os.environ.get("V4_GRIPPER", "hold")
LO = np.array([-2.8, -3.14, -3.14, -1.87, -1.57, -3.14] * 2) + math.radians(5)
HI = np.array([2.8, 0, 0, 1.57, 1.57, 3.14] * 2) - math.radians(5); HI[[1, 2, 7, 8]] = 0.0

# The dataset's TCP rotation and this FK's TCP rotation are the same frame up to a FIXED R_y(90 deg):
# measured on 31 frames of recorded joints vs the recorded tcp_state, position identical to 0.00 mm, rotation
# 90.00 deg with standard deviation 0.00, axis [0,1,0]. So  R_fk = R_dataset @ Ry(90).
# Labels are unaffected (they are pose-to-pose relatives, where the fixed rotation cancels), but every robot
# command is not: without this the arm tracks the demonstration's position with its wrist 90 deg out.
R_DS_TO_FK = Rot.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()
R_FK_TO_DS = R_DS_TO_FK.T
USE_FRAME_FIX = os.environ.get("V4_FRAME_FIX", "1") != "0"
# [2026-10-06 user] HandUMI TCP rotation = the wrist CAMERA's axes (handumi_camera_tcp_v2: TCP x = camera z), not the fingers: the
# camera is pitched against the jaw, so every human-learned rotation / translation direction is expressed in a frame tilted by a
# fixed angle about the jaw-closing axis (TCP y). V4_UMI_TCP_PITCH_DEG = theta: the robot TCP is used as T @ Ry(-theta) ("HandUMI
# convention") for state, anchors, chunk targets and RTC; IK gets the inverse. Position unchanged. 0 = off.
# 33.6 = robot rest (fingers 2.4 deg below horizontal) placed at the HandUMI start pose vs the human starts' camera-axis tilt 36.0.
UMI_TCP_PITCH_DEG = float(os.environ.get("V4_UMI_TCP_PITCH_DEG", "0"))
R_UMI_OFF = Rot.from_rotvec([0.0, -np.radians(UMI_TCP_PITCH_DEG), 0.0]).as_matrix()
# [2026-10-06 user "9.5 기준점 오프셋도"] V4_UMI_TCP_OFFSET_MM = "x,y,z": the HandUMI TCP POINT expressed in the robot dataset TCP frame
# (hand-eye: robot wrist cam X_tcp_cam @ HandUMI X_cam_tcp -> (-94.8, 2.4, 38.4) mm). HandUMI pose = T_robot @ [R_UMI_OFF | t];
# IK targets get the exact inverse. Empty = 0 (rotation-only behaviour unchanged).
UMI_TCP_OFF_T = np.array([float(x) for x in os.environ.get("V4_UMI_TCP_OFFSET_MM", "0,0,0").split(",")]) / 1000.0
UMI_CONV = bool(UMI_TCP_PITCH_DEG) or bool(np.any(UMI_TCP_OFF_T))

# The GLOBAL feed's orientation is part of the deployment contract, not a detail. B663 trained on two
# camera pools -- the HEAD pool as recorded and the FRONT (R675) pool with its global feed rotated 180 deg
# to match it -- so the DATASET has a single global orientation. Verified 2026-09-25 by putting the live
# frame beside frames from both pools: the dark equipment band sits at the BOTTOM in every training frame
# and at the TOP live, i.e. the live global camera is mounted 180 deg from the training convention.
# RETRACTED 2026-09-25, same day: that reasoning was appearance-only (where the dark equipment band sits)
# plus a magnitude change, and it got the answer backwards. The test that decides it compares against the
# relation measured IN the training data -- over 60 HEAD episodes, a first target on the image LEFT is
# grasped at base y +226 mm and one on the image RIGHT at -157 mm, corr(image x, grasp y) = -0.709. Replaying
# the same frozen scene through the policy:
#     rot180 ON   the right arm always acts and runs from -315 out to -418, away from every target,
#                 corr(image x, predicted y) = +0.944 -- the opposite sign
#     rot180 OFF  the left arm acts for left-side targets and moves +315 -> +260, toward the +226 the data
#                 says is there, corr = -0.263 -- the same sign as training
# So the default is 0. Only the global feed would be rotated; the wrist feeds are mounted on the arms.
# Do not turn this back on from appearance alone: re-measure against the training relation.
GLOBAL_ROT180 = os.environ.get("V4_GLOBAL_ROT180", "0") != "0"
# Mirror is a separate axis, applied AFTER the rotation. rot180 flips both image axes and a mirror flips x,
# so rot180 + mirror is a pure vertical flip -- the four settings are the four ways the feed can be oriented.
# A mirror is not a harmless visual change: it reverses handedness, so a left-handed approach in the picture
# becomes right-handed while the action stays as it was. The converter measured this on the FRONT pool and
# rejected it (residual 7.7 px for rot180, 58.7 px for a mirror, 145.6 px for both), but that was FRONT
# against HEAD, not the live camera against training, so the live case is decided by its own measurement.
GLOBAL_MIRROR = os.environ.get("V4_GLOBAL_MIRROR", "0") != "0"
# [2026-10-07 user "feed 영상을 확대"] right-wrist zoom = focal scale s about the robot C922 principal point (318.3, 224.9 @ 640x480):
# f' = s f, c unchanged (not a centre crop). 1.0 = off. Mutable at runtime (UI /wrist_zoom) through WRIST_ZOOM["s"].
WRIST_ZOOM = {"s": float(os.environ.get("V4_WRIST_ZOOM", "1.0")), "c": (318.3126, 224.9012), "wh": (640, 480)}


def wrist_zoom(bgr, s):
    if abs(s - 1.0) < 1e-6:
        return bgr
    h, w = bgr.shape[:2]; cx = WRIST_ZOOM["c"][0] * w / WRIST_ZOOM["wh"][0]; cy = WRIST_ZOOM["c"][1] * h / WRIST_ZOOM["wh"][1]
    M = np.array([[s, 0.0, cx * (1 - s)], [0.0, s, cy * (1 - s)]], np.float32)     # dst = s (src - c) + c
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# UMI76 deployment. The B663 models are slim20 + chunk 32; the HEAD180 UMI76 twins are 76-D state + chunk 16
# with the observation history taken at a PHYSICAL 50.05 ms, not at the previous frame. Serving 15 Hz frames
# straight in would hand the policy a 66.7 ms history it never trained on, so the history pose is
# interpolated out of a timestamped ring buffer. Contract: ~/umi_bridge/umi76/UMI76_CONTRACT.md.
STATE_MODE = os.environ.get("V4_STATE_MODE", "slim20")        # slim20 | umi76 | umi94 (v2 diagnostic twin B)
# [2026-10-03] ONE-ARM models (HRA_red right-only approach prior, ~/ego_cart20/ego_cart20/right_only.py): V4_ARM_ONLY=right serves the
# model exactly as it was trained -- LEFT state dims = dummy (identity pose9, gL = ARM_ONLY_LEFT_G), LEFT wrist image = black (the
# exporter wrote a blank frame), and the outputs that had NO loss (left arm 0:10, gripper 19 when masked) are never executed: the left
# arm and both jaws hold their measured position.  Unset = previous behaviour, byte for byte.  relcart20 state only.
ARM_ONLY = os.environ.get("V4_ARM_ONLY", "").strip().lower()
ARM_ONLY_LEFT_G = float(os.environ.get("V4_ARM_ONLY_LEFT_G", "0.0"))
ARM_ONLY_GRIP = os.environ.get("V4_ARM_ONLY_GRIP", "hold")     # hold | model (only if the right gripper WAS supervised)
# output guard (one-arm models): a chunk whose right-arm target leaves the training range is REFUSED (never clipped into motion).
# HRA training: 0.8 s right translation p95 ~11 cm. A garbage output otherwise drives DQ_MAX (0.6 rad) per waypoint (measured).
ARM_ONLY_MAX_PRED_MM = float(os.environ.get("V4_ARM_ONLY_MAX_PRED_MM", "250"))
ARM_ONLY_MAX_PRED_DEG = float(os.environ.get("V4_ARM_ONLY_MAX_PRED_DEG", "60"))


def jaw_hold_cmd(raw):
    """[2026-10-03] the /execute_step jaw COMMAND (0..45) that keeps the jaw where /observe reads it (obs = cmd * -6).
    The generic GRIPPER_MODE=hold path re-sends the RAW reading (0..-270) as a command, which the bridge clips to 0 = CLOSED;
    the one-arm paths use this instead."""
    return float(np.clip(unwrap_grip(raw) / GRIP_OBS_PER_CMD, 0.0, GRIP_CMD_MAX))
_IDENTITY9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], np.float32)
if ARM_ONLY not in ("", "right"):
    raise SystemExit(f"V4_ARM_ONLY={ARM_ONLY!r}: only 'right' is supported")
if ARM_ONLY and STATE_MODE != "relcart20":
    raise SystemExit("V4_ARM_ONLY=right needs V4_STATE_MODE=relcart20 (the right-only dataset's state)")


def _black_b64():
    ok, enc = cv2.imencode(".jpg", np.zeros((224, 224, 3), np.uint8))
    return base64.b64encode(enc.tobytes()).decode()


def arm_only_images(imgs):
    """left wrist -> the blank frame the right-only exporter wrote (only with V4_ARM_ONLY=right)"""
    if not ARM_ONLY:
        return imgs
    global _BLACK_B64
    if _BLACK_B64 is None: _BLACK_B64 = _black_b64()
    out = dict(imgs); out["left"] = _BLACK_B64
    return out


_BLACK_B64 = None
CART20_W_OPEN = 270.0 * 0.05 / 118.0                            # raw -270 = fully open, in m (derive_cart20.py)


# [2026-10-07 user "B를 만들어줄래"] V4_FIXED_ANCHOR_JSON = json with "T_base_F" (4x4, robot base, dataset-TCP convention): the RIGHT-arm
# RELCART20 anchor is this fixed frame instead of the first observation, so the state is the absolute pose in F (the HRA_A100
# origin-anchored models: F = the HandUMI fingertip-on-X frame). Unset = unchanged behaviour.
FIXED_ANCHOR_R = (np.array(json.load(open(os.path.expanduser(os.environ["V4_FIXED_ANCHOR_JSON"])))["T_base_F"], np.float64)
                  if os.environ.get("V4_FIXED_ANCHOR_JSON") else None)

def relcart20_state(anchor, cur, w):
    """[2026-09-29] RELCART20 state (derive_relcart20.py layout): per arm T_rel = inv(T_anchor) @ T_t -> [pos3 | rot6d rows], then
    g_L, g_R = width / W_OPEN clipped to [0, 1]. anchor = TCP mats (2,4,4) at the rollout's first observation (training: the
    episode's frame_index-0 row). anchor == cur gives [0,0,0, 1,0,0,0,1,0] per arm."""
    st = np.zeros(20, np.float32)
    for r in (0, 1):
        st[r * 9:r * 9 + 9] = mat_to_pose10d((np.linalg.inv(anchor[r]) @ cur[r])[None])[0]
    st[18:20] = np.clip(np.asarray(w, np.float64) / CART20_W_OPEN, 0.0, 1.0)
    return st


def cart20_state(cur, w):
    """[2026-09-29] CART20 state (derive_cart20.py layout): [L pos3 | L rot6d rows | R pos3 | R rot6d rows | g_L | g_R],
    cur = dataset-frame TCP mats at t (2,4,4), w = follower jaw widths at t (m); g = w / W_OPEN clipped to [0, 1], 1 = OPEN."""
    st = np.zeros(20, np.float32)
    for r in (0, 1):
        st[r * 9:r * 9 + 9] = mat_to_pose10d(cur[r][None])[0]
    st[18:20] = np.clip(np.asarray(w, np.float64) / CART20_W_OPEN, 0.0, 1.0)
    return st
UMI_HISTORY_DT = 3.0 / 59.94                                   # 50.05 ms, upstream's own spacing
OBS_BUF_S = float(os.environ.get("V4_OBS_BUF_S", "1.0"))

GRIP_M_PER_RAW = 0.05 / 118.0                # umi_schema.grip_linear_v1_anchor5cm_raw118, inverted
GRIP_RAW_MIN, GRIP_RAW_MAX = -270.0, 0.0


def unwrap_grip(raw):
    """The follower's gripper encoder wraps: a jaw that is physically CLOSED can read 358 instead of -2.

    Measured on this robot 2026-09-24: left closed = 358.09, right closed = 0.01, and the zero re-latches
    on every power cycle (0 / -100 / +90 / 358 have all been seen). The dataset pipeline has always
    unwrapped this -- `grip_norm_dataset.py` does `g - 360 if g > 180` -- and the deployment never did, so
    the state's width channel was nonsense on a wrapped arm and, worse, a command of 0 asked a jaw sitting
    at 358 to travel 358 counts, which is the "TRAIN START throws the gripper wide open" symptom.
    """
    r = float(raw)
    return r - 360.0 if r > 180.0 else r


def raw_to_width(raw):
    return float(np.clip(abs(float(np.clip(unwrap_grip(raw), GRIP_RAW_MIN, GRIP_RAW_MAX))), 0, None)
                 * GRIP_M_PER_RAW)


def width_to_raw(width_m):
    return float(np.clip(-abs(float(width_m)) / GRIP_M_PER_RAW, GRIP_RAW_MIN, GRIP_RAW_MAX))


# The model predicts a metric jaw WIDTH, /observe reports a raw count, and /execute_step takes a LEADER-frame
# command -- three different scales. Measured on this robot (GRIPPER_COMMAND_SCALE.md, re-checked per session
# because the raw zero re-latches at power-on): cmd * -6 = obs, full travel cmd 0..45, and raw 0 = jaws CLOSED
# with raw -270 = fully OPEN (video-confirmed 2026-09-22). Sending width_to_raw() straight out as the command,
# which the v3 core does, would ask for -270 on a 0..45 scale.
GRIP_OBS_PER_CMD = float(os.environ.get("V4_GRIP_OBS_PER_CMD", "-6.0"))
GRIP_CMD_MAX = float(os.environ.get("V4_GRIP_CMD_MAX", "45"))


def width_to_cmd(width_m):
    """predicted jaw width [m] -> /execute_step leader-frame gripper command."""
    return float(np.clip(width_to_raw(width_m) / GRIP_OBS_PER_CMD, 0.0, GRIP_CMD_MAX))


# [2026-09-28] REL16-v2 binary gripper (V4_GRIPPER=binary). The v2 label is the LEADER's intent, 1 = OPEN and
# 0 = CLOSE (derive_v2.py), not a width: the v1 follower width stalls at cube contact, so commanding it gave no grip
# force. The actuator values live in gripper_contract_v2.json, written by `gripper_smoke_test.py --cube` only when
# the squeeze-on-cube gate passes; until then the 2026-09-19 smoke-test endpoints stand (CLOSED 0, OPEN 42).
def _grip_contract():
    import json as _json
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gripper_contract_v2.json")
    try:
        c = _json.load(open(p))
        if c.get("passed"):
            return float(c["cmd_open"]), float(c["cmd_closed"]), p
    except (OSError, ValueError, KeyError):
        pass
    return 42.0, 0.0, None


GRIP_BIN_OPEN, GRIP_BIN_CLOSED, GRIP_CONTRACT_FILE = _grip_contract()
# [2026-09-29] execution threshold on the model's gripper channel (binary v2 label, or continuous v3 g = clip(leader/45, 0, 1)):
# g >= V4_GRIP_THRESH -> OPEN. Default 0.5 (the v2 behaviour); 0.6 = cmd 27/45, the old binary rule. Frozen per deployment.
GRIP_BIN_THRESH = float(os.environ.get("V4_GRIP_THRESH", "0.5"))
GRIP_CMD_FULL = float(os.environ.get("V4_GRIP_CMD_FULL", "45"))   # cmd 0..45 = full jaw travel (derive_v2 CMD_FULL_OPEN)
GRIP_REL_EPS = float(os.environ.get("V4_GRIP_REL_EPS", "0.05"))   # relative mode deadband on delta-g
GRIP_REL_SCALE = float(os.environ.get("V4_GRIP_REL_SCALE", "1.0"))  # divide g by this first (width-labelled runs: 0.11441 m -> 0..1)
# [2026-09-29] V4_GRIP_REL_MODE=acc: sum of the last N delta-g (per executed waypoint) instead of one delta; OPEN if > ACC_OPEN,
# CLOSE if < ACC_CLOSE, else HOLD; the history is cleared after every OPEN/CLOSE so one burst does not keep re-triggering.
GRIP_REL_MODE = os.environ.get("V4_GRIP_REL_MODE", "delta")
GRIP_ACC_N = int(os.environ.get("V4_GRIP_ACC_N", "4"))
GRIP_ACC_OPEN = float(os.environ.get("V4_GRIP_ACC_OPEN", "0.06"))
GRIP_ACC_CLOSE = float(os.environ.get("V4_GRIP_ACC_CLOSE", "-0.06"))


def grip_to_cmd(v):
    """model gripper channel -> /execute_step command, for the active GRIPPER_MODE (predict = width, binary = intent)."""
    if GRIPPER_MODE == "binary":
        return GRIP_BIN_OPEN if float(v) >= GRIP_BIN_THRESH else GRIP_BIN_CLOSED
    if GRIPPER_MODE in ("continuous", "relative"):   # relative: the executed command comes from _grip_relative; this is the log value
        # [2026-09-29 user] continuous g (label = clip(leader_cmd/45, 0, 1)) -> its inverse, cmd = clip(g, 0, 1) * 45 (full travel)
        return float(np.clip(float(v), 0.0, 1.0) * GRIP_CMD_FULL)
    return width_to_cmd(v)


def clamp_delta(dT, max_mm, max_deg):
    """Shrink one body-frame delta to the safety envelope, keeping its direction."""
    out = np.eye(4); t = dT[:3, 3].copy()
    n = float(np.linalg.norm(t)) * 1000.0
    scale_t = 1.0 if (max_mm <= 0 or n <= max_mm) else max_mm / max(n, 1e-9)
    rv = Rot.from_matrix(dT[:3, :3]).as_rotvec()
    ang = float(np.degrees(np.linalg.norm(rv)))
    scale_r = 1.0 if (max_deg <= 0 or ang <= max_deg) else max_deg / max(ang, 1e-9)
    out[:3, 3] = t * scale_t
    out[:3, :3] = Rot.from_rotvec(rv * scale_r).as_matrix()
    return out, (scale_t < 1.0 or scale_r < 1.0), n, ang


class V4Inferencer:
    obs_horizon = 2                      # prev_rel needs t-1 and t

    def __init__(self, ckpt: str = CKPT, device: str = DEVICE):
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        _install_xvla_rtc(); self._ckpt_dir = ckpt; self._rtc_prev = None
        self.policy = XVLAPolicy.from_pretrained(ckpt)
        self.policy.to(device)
        # fp16 on MPS is 2.50x faster than fp32 (498 -> 199 ms at 5 denoising steps) and, compared at the
        # SAME initial flow sample, moves the predicted chunk by at most 0.365 mm -- against a per-step
        # motion of 4-8 mm and a sampler spread of 18-65 mm. bf16 is no faster and 12x less accurate
        # (4.4 mm), so it is not offered.
        if DTYPE == "fp16":
            self.policy.half()
        elif DTYPE == "bf16":
            self.policy.config.dtype = "bfloat16"; self.policy.to(torch.bfloat16)
        # [2026-10-02 user] flow-matching denoising steps at inference (checkpoint default 10). Mac MPS bf16, A30k, 80 R312c frames,
        # same noise: 10 steps 382 ms (k4 |err vs GT| med 4.0 / p90 36.4 mm), 5 steps 268 ms (3.8 / 25.9 mm); ~19 ms per step + ~115 ms fixed.
        if os.environ.get("V4_DENOISE_STEPS"):
            self.policy.config.num_denoising_steps = int(os.environ["V4_DENOISE_STEPS"])
            print(f"[infer_core_v4] denoising steps {self.policy.config.num_denoising_steps} (V4_DENOISE_STEPS)", flush=True)
        self.policy.eval()
        self.chunk = int(self.policy.config.chunk_size)
        if CHUNK_OVERRIDE:
            if CHUNK_OVERRIDE != self.chunk:
                print(f"[infer_core_v4] chunk {self.chunk} -> {CHUNK_OVERRIDE} by V4_CHUNK; rows beyond the "
                      f"trained horizon are extrapolation", flush=True)
            # the inner model copies chunk_size at construction, so the config alone is not enough
            self.policy.config.chunk_size = CHUNK_OVERRIDE
            self.policy.config.n_action_steps = CHUNK_OVERRIDE
            self.policy.model.chunk_size = CHUNK_OVERRIDE
            self.chunk = CHUNK_OVERRIDE
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": device},
                                    "rename_observations_processor": {"rename_map": RENAME}},
            postprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        # [2026-09-29] the HandUMI solver clipped every joint to ±0.35 rad per solve (eef_kin default, inherited from the
        # offline retarget); on HEAD180 GT 33 % of the executed k16 targets need more, so the arm was asked for less.
        # Default stays N0 = 0.35 (official deployed baseline; 13:20 decision: clip-off N1 is offline diagnostic only, never on
        # the real robot). V4_IK_MAX_JOINT_DELTA=none gives N1 for offline use. DQ_MAX/LO/HI apply either way.
        _mjd = os.environ.get("V4_IK_MAX_JOINT_DELTA", "0.35").lower()
        self.kin = eef_kin.Kin({"max_joint_delta": None if _mjd in ("none", "0", "") else float(_mjd)})
        self.n_action = N_ACTION
        self.task = os.environ.get(
            "V4_TASK",
            "Stack the red cube on the bottom, blue cube in the middle, and purple cube on the top, on the plate.")
        self.calls = 0; self.last = {}
        print(f"[v4] {os.path.basename(str(ckpt).rstrip('/'))} device={device} chunk={self.policy.config.chunk_size} "
              f"state={self.policy.config.max_state_dim} action_mode={self.policy.config.action_mode} "
              f"n_action={self.n_action} clamp={LIMITS['mm']}mm/{LIMITS['deg']}deg gripper={GRIPPER_MODE} "
              f"dq_max={DQ_MAX} ik_max_joint_delta={self.kin.solver.config.max_joint_delta} ik_backend={IK_BACKEND} fk_guard={FK_MAX_MM}mm"
              + (f" ARM_ONLY={ARM_ONLY} (left state dummy, left image black, left arm + jaws hold, grip {ARM_ONLY_GRIP})" if ARM_ONLY else ""), flush=True)

    # ---------------------------------------------------------------- observation
    @staticmethod
    def _img(b64, rot180=False, mirror=False, zoom=1.0):
        a = np.frombuffer(base64.b64decode(b64), np.uint8)
        bgr = cv2.imdecode(a, cv2.IMREAD_COLOR)
        bgr = wrist_zoom(bgr, zoom)                          # [2026-10-07] right wrist only (callers pass WRIST_ZOOM["s"])
        if rot180:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        if mirror:
            bgr = cv2.flip(bgr, 1)
        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.moveaxis(rgb.astype(np.float32) / 255.0, -1, 0)).contiguous()

    def _tcp_mat(self, joints14):
        q = np.asarray([float(x) for x in joints14], np.float64)
        L7, R7 = self.kin.fk_pose7(q[ARM_IDX])
        poses = np.stack([np.concatenate([p7[:3], Rot.from_quat(p7[3:]).as_rotvec()]) for p7 in (L7, R7)])
        mats = pose_to_mat(poses)
        if USE_FRAME_FIX:                                    # FK frame -> dataset frame
            for r in (0, 1):
                mats[r][:3, :3] = mats[r][:3, :3] @ R_FK_TO_DS
        if UMI_CONV:                                         # [10-06] robot TCP -> HandUMI (camera-axis) convention
            for r in (0, 1):
                mats[r][:3, 3] = mats[r][:3, 3] + mats[r][:3, :3] @ UMI_TCP_OFF_T
                mats[r][:3, :3] = mats[r][:3, :3] @ R_UMI_OFF
            poses = np.stack([np.concatenate([mats[r][:3, 3],
                                              Rot.from_matrix(mats[r][:3, :3]).as_rotvec()]) for r in (0, 1)])
        return mats, poses                                   # (2,4,4), (2,6) in the DATASET frame

    def push_obs(self, joints14, t=None):
        """Record one observation for the UMI76 history buffer. Cheap: FK only, no model."""
        if not hasattr(self, "_buf"):
            self._buf = collections.deque()
        t = time.time() if t is None else float(t)
        m, _ = self._tcp_mat(joints14)
        w = np.array([raw_to_width(joints14[GRIP_IDX[r]]) for r in (0, 1)], np.float64)
        self._buf.append((t, m.copy(), w))
        while len(self._buf) > 2 and self._buf[0][0] < t - OBS_BUF_S:
            self._buf.popleft()
        return len(self._buf)

    def _interp_at(self, tq):
        """SE(3) + width at an arbitrary time, from the buffer. Linear position/width, slerp rotation."""
        b = getattr(self, "_buf", None)
        if not b or len(b) < 2:
            raise RuntimeError("umi76: need at least two timestamped observations")
        ts = np.array([e[0] for e in b])
        if tq < ts[0] - 1e-6:
            raise RuntimeError(
                f"umi76: the buffer spans {ts[-1]-ts[0]:.4f}s but the query needs {UMI_HISTORY_DT:.4f}s of "
                f"history. Observations must be timestamped when the robot is READ (_observe sets 't'); "
                f"pushing them at inference time collapses the gap to microseconds.")
        j = int(np.clip(np.searchsorted(ts, tq), 1, len(ts) - 1))
        t0, m0, w0 = b[j - 1]
        t1, m1, w1 = b[j]
        u = 0.0 if t1 <= t0 else float(np.clip((tq - t0) / (t1 - t0), 0.0, 1.0))
        out = np.zeros((2, 4, 4))
        for r in (0, 1):
            key = Rot.from_matrix(np.stack([m0[r][:3, :3], m1[r][:3, :3]]))
            out[r][:3, :3] = Slerp([0.0, 1.0], key)([u]).as_matrix()[0]
            out[r][:3, 3] = m0[r][:3, 3] * (1 - u) + m1[r][:3, 3] * u
            out[r][3, 3] = 1.0
        return out, w0 * (1 - u) + w1 * u

    def build_state_umi76(self):
        """The 76-D upstream-bimanual packing, from the buffer, at [t-50.05ms, t].

        Slice order is alphabetical by upstream key and timestep-major inside each key -- the order
        timm_obs_encoder.py:169 produces after `sorted(low_dim_keys)`. It is NOT the yaml's key order, and
        it must not be re-packed for slim20 compatibility: the widened encoder was trained on this layout.
        """
        b = getattr(self, "_buf", None)
        if not b:
            raise RuntimeError("umi76: no observations buffered")
        t = b[-1][0]
        cur, wc = self._interp_at(t)
        hist, wh = self._interp_at(t - UMI_HISTORY_DT)
        inv_cur = [np.linalg.inv(cur[r]) for r in (0, 1)]
        OFF = {0: dict(pos=0, pos_wrt=6, rot=12, rot_wrt=24, grip=36),
               1: dict(pos=38, pos_wrt=44, rot=50, rot_wrt=62, grip=74)}
        st = np.zeros(76, np.float32)
        for r in (0, 1):
            o = 1 - r
            sh = mat_to_pose10d((inv_cur[r] @ hist[r])[None])[0]
            sc = mat_to_pose10d((inv_cur[r] @ cur[r])[None])[0]
            ch = mat_to_pose10d((inv_cur[o] @ hist[r])[None])[0]
            cc = mat_to_pose10d((inv_cur[o] @ cur[r])[None])[0]
            f = OFF[r]
            st[f["pos"]:f["pos"] + 3] = sh[:3];        st[f["pos"] + 3:f["pos"] + 6] = sc[:3]
            st[f["pos_wrt"]:f["pos_wrt"] + 3] = ch[:3]; st[f["pos_wrt"] + 3:f["pos_wrt"] + 6] = cc[:3]
            st[f["rot"]:f["rot"] + 6] = sh[3:9];        st[f["rot"] + 6:f["rot"] + 12] = sc[3:9]
            st[f["rot_wrt"]:f["rot_wrt"] + 6] = ch[3:9]; st[f["rot_wrt"] + 6:f["rot_wrt"] + 12] = cc[3:9]
            st[f["grip"]] = wh[r];                      st[f["grip"] + 1] = wc[r]
        poses = np.stack([np.concatenate([cur[r][:3, 3], Rot.from_matrix(cur[r][:3, :3]).as_rotvec()])
                          for r in (0, 1)])
        if STATE_MODE == "cart20":
            return cart20_state(cur, wc), cur, poses
        if STATE_MODE == "relcart20":
            # anchor = the first observation after reset_anchor() (the UI calls it on /run): the deploy counterpart of the
            # training episode's frame_index-0 row. Start a rollout from the rest pose, as the recordings did.
            if getattr(self, "_anchor", None) is None:
                self._anchor = [cur[0].copy(), cur[1].copy()]
                if FIXED_ANCHOR_R is not None:            # [2026-10-07] B / origin-anchored models: RIGHT anchor = the fixed frame F (base)
                    self._anchor[1] = FIXED_ANCHOR_R.copy()
                print(f"[relcart20] anchor set: L {np.round(cur[0][:3, 3] * 1000, 1)} R {np.round(cur[1][:3, 3] * 1000, 1)} mm", flush=True)
            st = relcart20_state(self._anchor, cur, wc)
            if ARM_ONLY == "right":                                  # trained with a dummy left arm: never show it the real one
                st[0:9] = _IDENTITY9; st[18] = ARM_ONLY_LEFT_G
            return st, cur, poses
        if STATE_MODE == "umi94":
            # [2026-09-28] twin B: + per-arm base-frame TCP [pos3, rot6d] at t, L then R, in the DATASET frame
            # (cur is already frame-fixed), exactly derive_v2.py's layout. Diagnostic only -- ego has no such pose.
            st = np.concatenate([st, mat_to_pose10d(cur[0][None])[0], mat_to_pose10d(cur[1][None])[0]]).astype(np.float32)
        return st, cur, poses

    def build_state(self, history):
        """per arm [prev_rel pos3 + rot6d, gripper width] -> (20,), exactly the converter's layout."""
        assert len(history) == self.obs_horizon, f"need {self.obs_horizon} observations, got {len(history)}"
        m_prev, _ = self._tcp_mat(history[-2]["joints14"])
        m_now, poses_now = self._tcp_mat(history[-1]["joints14"])
        state = np.zeros(20, np.float32)
        for r in (0, 1):
            prev_rel = np.linalg.inv(m_now[r]) @ m_prev[r]
            state[r * 10:r * 10 + 9] = mat_to_pose10d(prev_rel)
            state[r * 10 + 9] = raw_to_width(history[-1]["joints14"][GRIP_IDX[r]])
        return state, m_now, poses_now

    # ---------------------------------------------------------------- one inference
    def infer_raw(self, images_b64: dict, state20, task=None, noise=None) -> np.ndarray:
        """One forward pass on an observation assembled by the caller: (T,20) in raw units.

        `infer` builds the state from two observations and then goes through IK, clamps and the gripper
        scale. This is the same preprocessing and the same policy with none of that, so an image and a state
        from different sources can be combined -- which is the only way to ask whether a live rollout misses
        because of the pictures or because of the proprioception.
        """
        images_b64 = arm_only_images(images_b64)
        if REMOTE_INFER:
            return remote_forward({v: images_b64[k] for k, v in CAM_MAP.items() if k in images_b64}, state20, task or self.task, noise)
        obs = {f"observation.images.{v}": self._img(images_b64[k], GLOBAL_ROT180 and v == "global", GLOBAL_MIRROR and v == "global", WRIST_ZOOM["s"] if v == "right_wrist" else 1.0)
               for k, v in CAM_MAP.items() if k in images_b64}
        obs["observation.state"] = torch.as_tensor(np.asarray(state20, np.float32))
        obs["task"] = task or self.task
        batch = self.pre(obs)
        with torch.no_grad():
            chunk = (self.policy.predict_action_chunk(batch, noise=noise.to(self.device))
                     if noise is not None else self.policy.predict_action_chunk(batch))
        return self.post(chunk)[0].detach().float().cpu().numpy().astype(np.float64)

    def infer(self, history, noise=None) -> np.ndarray:
        if STATE_MODE in ("umi76", "umi94", "cart20", "relcart20"):
            for h in history:
                self.push_obs(h["joints14"], h.get("t"))
            state, m_now, poses_now = self.build_state_umi76()
        else:
            state, m_now, poses_now = self.build_state(history)
        imgs = arm_only_images(history[-1]["images"])
        if REMOTE_INFER:
            miss = [v for k, v in CAM_MAP.items() if k not in imgs]
            if miss:
                raise ValueError(f"camera missing: {miss} (observation had {list(imgs)})")
            rtc = self._rtc_build(m_now) if RTC_ON else None          # [2026-10-06] RTC now also with remote inference
            act = remote_forward({v: imgs[k] for k, v in CAM_MAP.items()}, state, self.task, noise, rtc=rtc)
            self._rtc_info = None if rtc is None else dict(rtc["info"], **(REMOTE_LAST.get("rtc_log") or {}))
        else:
            rtc = self._rtc_build(m_now) if RTC_ON else None
            self.policy.model._rtc = rtc; self.policy.model._rtc_log = None
            try:
                act = self._local_forward(imgs, state, noise)
            finally:
                self.policy.model._rtc = None
            self._rtc_info = None if rtc is None else dict(rtc["info"], **(self.policy.model._rtc_log or {}))
        if not np.isfinite(act).all():
            raise RuntimeError("model output has NaN/Inf -- inference aborted")
        return self._post_act(act, state, m_now, poses_now, history)

    def _rtc_build(self, m_now):
        """RTC prefix from the previous chunk (see RTC_ON): Y / W in the model's normalised action space, or None."""
        plan = getattr(self, "_rtc_plan_prefix", None)
        if plan is not None and ACTION_MODE == "umi":
            # [2026-10-06 PLAN mode] explicit prefix from the UI's plan table: rows anchor+1 .. as (P (2,3), Q (2,4) xyzw, W (2,)) absolute
            # base-frame targets; new chunk step i <-> plan row anchor+1+i. Hard (W=1) for the d rows that run during this forward
            # (_rtc_delay_hint), then the LINEAR release to the end of the prefix.
            if not hasattr(self, "_act_ms"):
                from safetensors.torch import load_file
                st = load_file(os.path.join(self._ckpt_dir, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
                self._act_ms = (st["action.mean"].float().numpy(), st["action.std"].float().numpy())
            mean, std = self._act_ms; A = int(self.policy.model.dim_action); C = int(self.chunk); L = min(len(plan), C)
            if L < 1:
                return None
            Yr = np.zeros((L, 20))
            for i in range(L):
                Pp, Qq, Ww = plan[i]
                for r in (0, 1):
                    Tp = np.eye(4); Tp[:3, :3] = Rot.from_quat(Qq[r]).as_matrix(); Tp[:3, 3] = Pp[r]
                    Yr[i, r * 10:r * 10 + 9] = mat_to_pose10d((np.linalg.inv(m_now[r]) @ Tp)[None])[0]
                    Yr[i, r * 10 + 9] = Ww[r]
            Y = torch.zeros(C, A); Y[:L, :20] = torch.as_tensor((Yr - mean[:20]) / (std[:20] + 1e-8), dtype=torch.float32)
            hint = getattr(self, "_rtc_delay_hint", None); dly = min(int(hint) if hint is not None else 0, L)
            W = torch.zeros(C, A); W[:, :20] = _rtc_weights(dly, L, C)[:, None]
            return {"Y": Y, "W": W, "info": dict(prev_idx=-1, nearest_mm=None, leftover=L, delay=dly, horizon=L, age_s=0.0, source="plan")}
        prev = getattr(self, "_rtc_prev", None)
        if prev is None or time.time() - prev["t"] > RTC_MAX_AGE_S or ACTION_MODE != "umi":
            return None
        if not hasattr(self, "_act_ms"):
            from safetensors.torch import load_file
            st = load_file(os.path.join(self._ckpt_dir, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
            self._act_ms = (st["action.mean"].float().numpy(), st["action.std"].float().numpy())
        mean, std = self._act_ms
        P, Q, Wd = prev["pos"], prev["quat"], prev["widths"]; T = len(P)
        d = sum(np.linalg.norm(P[:, r] - m_now[r][:3, 3], axis=1) for r in (0, 1))
        js = int(d.argmin()); L = T - 1 - js
        if L < 1:
            return None
        Yr = np.zeros((L, 20))
        for i in range(L):
            j = js + 1 + i
            for r in (0, 1):
                Tp = np.eye(4); Tp[:3, :3] = Rot.from_quat(Q[j, r]).as_matrix(); Tp[:3, 3] = P[j, r]
                Yr[i, r * 10:r * 10 + 9] = mat_to_pose10d((np.linalg.inv(m_now[r]) @ Tp)[None])[0]
                Yr[i, r * 10 + 9] = Wd[j, r]
        A = int(self.policy.model.dim_action); C = int(self.chunk)
        Y = torch.zeros(C, A); Y[:L, :20] = torch.as_tensor((Yr - mean[:20]) / (std[:20] + 1e-8), dtype=torch.float32)
        # [2026-10-06] delay = the waypoints the arm executes WHILE this chunk is inferred (the UI's prefetch sets _rtc_delay_hint =
        # lead + 1); those steps are pinned to the previous chunk (W = 1), then a linear release to the horizon. V4_RTC_HORIZON <= 0 ->
        # the whole leftover of the previous chunk.
        hint = getattr(self, "_rtc_delay_hint", None)
        dly = min(int(hint) if hint is not None else RTC_DELAY, L)
        hor = L if RTC_HORIZON <= 0 else min(max(RTC_HORIZON, dly), L)
        W = torch.zeros(C, A); W[:, :20] = _rtc_weights(dly, hor, C)[:, None]
        return {"Y": Y, "W": W, "info": dict(prev_idx=js, nearest_mm=round(float(d[js]) * 1000 / 2, 1), leftover=L, delay=dly, horizon=hor,
                                             age_s=round(time.time() - prev["t"], 2))}

    def _local_forward(self, imgs, state, noise):
        obs = {f"observation.images.{v}": self._img(imgs[k], GLOBAL_ROT180 and v == "global", GLOBAL_MIRROR and v == "global", WRIST_ZOOM["s"] if v == "right_wrist" else 1.0)
               for k, v in CAM_MAP.items() if k in imgs}
        missing = [f"observation.images.{v}" for v in CAM_MAP.values() if f"observation.images.{v}" not in obs]
        if missing:
            raise ValueError(f"camera missing: {missing} (observation had {list(imgs)})")
        obs["observation.state"] = torch.from_numpy(state)
        obs["task"] = self.task

        batch = self.pre(obs)
        with torch.no_grad():
            chunk = (self.policy.predict_action_chunk(batch, noise=noise.to(self.device))
                     if noise is not None else self.policy.predict_action_chunk(batch))
        return self.post(chunk)[0].detach().float().cpu().numpy().astype(np.float64)      # (T,20) raw units

    def _post_act(self, act, state, m_now, poses_now, history):
        T = len(act)
        if ARM_ONLY == "right":                                      # left outputs had no loss: hold the left TCP (identity delta)
            act = act.copy(); act[:, 0:9] = _IDENTITY9
            mr = pose10d_to_mat(act[:, 10:19])
            mm = float(np.linalg.norm(mr[:, :3, 3], axis=1).max()) * 1000.0
            dg = float(np.degrees(np.max([np.linalg.norm(Rot.from_matrix(m[:3, :3]).as_rotvec()) for m in mr])))
            if (ARM_ONLY_MAX_PRED_MM > 0 and mm > ARM_ONLY_MAX_PRED_MM) or (ARM_ONLY_MAX_PRED_DEG > 0 and dg > ARM_ONLY_MAX_PRED_DEG):   # 0 = off
                raise RuntimeError(f"ARM_ONLY output guard: right-arm chunk predicts {mm:.0f} mm / {dg:.0f} deg "
                                   f"(> {ARM_ONLY_MAX_PRED_MM:.0f} mm / {ARM_ONLY_MAX_PRED_DEG:.0f} deg) -- refused, nothing executed")

        tgt_pos = np.zeros((T, 2, 3)); tgt_quat = np.zeros((T, 2, 4)); widths = np.zeros((T, 2))
        raw_mm = np.zeros((T, 2)); raw_deg = np.zeros((T, 2)); cl_mm = np.zeros((T, 2)); n_clamped = 0
        for r in (0, 1):
            o = r * 10
            mats = pose10d_to_mat(act[:, o:o + 9])
            if ACTION_MODE == "umi":
                # current-anchor: every waypoint hangs off the SAME measured pose, no accumulation
                for k in range(T):
                    a, was_clamped, mm, deg = clamp_delta(mats[k], LIMITS["mm"] * (k + 1),
                                                          LIMITS["deg"] * (k + 1))
                    raw_mm[k, r], raw_deg[k, r] = mm, deg
                    cl_mm[k, r] = float(np.linalg.norm(a[:3, 3])) * 1000.0
                    n_clamped += int(was_clamped)
                    tgt = m_now[r] @ a
                    tgt_pos[k, r] = tgt[:3, 3]
                    tgt_quat[k, r] = Rot.from_matrix(tgt[:3, :3]).as_quat()
            else:
                # sequential body-frame composition, with each step clamped before it is applied
                cur = m_now[r].copy()
                for k in range(T):
                    d, was_clamped, mm, deg = clamp_delta(mats[k], LIMITS["mm"], LIMITS["deg"])
                    raw_mm[k, r], raw_deg[k, r] = mm, deg
                    cl_mm[k, r] = float(np.linalg.norm(d[:3, 3])) * 1000.0
                    n_clamped += int(was_clamped)
                    cur = cur @ d                                    # body frame -> RIGHT multiply
                    tgt_pos[k, r] = cur[:3, 3]
                    tgt_quat[k, r] = Rot.from_matrix(cur[:3, :3]).as_quat()
            widths[:, r] = act[:, o + 9]

        # [2026-10-01] temporal ensemble -> adaptive k -> near-zero hold (see TE_ON / ADAPTIVE_K / DB_* above)
        te_info = db_info = ak_info = None
        if TE_ON:
            now_t = time.time()
            if getattr(self, "_te_hist", None) is None or now_t - getattr(self, "_te_last_t", 0.0) > TE_MAXGAP_S:
                self._te_hist = collections.deque(maxlen=len(TE_W)); self._te_step = 0
            self._te_last_t = now_t
            self._te_hist.appendleft((self._te_step, tgt_pos.copy(), tgt_quat.copy(), widths.copy()))   # [0] = current chunk (raw)
            n_src = np.zeros(T, int); raw = tgt_pos.copy()
            for k in range(T):
                ps, qs, ws, wt = [], [], [], []
                for j, (s0, P, Qq, W) in enumerate(self._te_hist):
                    idx = k + (self._te_step - s0)
                    if idx >= T: continue
                    ps.append(P[idx]); qs.append(Qq[idx]); ws.append(W[idx]); wt.append(TE_W[j])
                wt = np.asarray(wt); n_src[k] = len(wt)
                if len(wt) > 1:
                    wn = wt / wt.sum()
                    tgt_pos[k] = np.einsum("j,jrc->rc", wn, np.asarray(ps))
                    if TE_GRIP: widths[k] = wn @ np.asarray(ws)
                    if TE_ROT:
                        for r in (0, 1):
                            tgt_quat[k, r] = Rot.from_quat(np.asarray(qs)[:, r]).mean(weights=wt).as_quat()
            te_info = dict(n_src=n_src.tolist(), w=TE_W.tolist(), rot=TE_ROT, grip=TE_GRIP, _raw=raw)
        k_exec = max(1, EXEC_K)
        if ADAPTIVE_K is not None:
            dist = max(float(np.linalg.norm(tgt_pos[T - 1, r] - m_now[r][:3, 3])) * 1000.0 for r in (0, 1))
            grip_x = bool(((widths.max(0) > GRIP_BIN_THRESH) & (widths.min(0) < GRIP_BIN_THRESH)).any()) if GRIPPER_MODE != "hold" else False
            if grip_x or dist <= AK_NEAR_MM: k_exec, phase = ADAPTIVE_K[2], ("align(grip)" if grip_x else "near")
            elif dist <= AK_FAR_MM: k_exec, phase = ADAPTIVE_K[1], "medium"
            else: k_exec, phase = ADAPTIVE_K[0], "far"
            ak_info = dict(k=k_exec, phase=phase, dist_end_mm=round(dist, 1), grip_cross=grip_x)
        k_exec = min(k_exec, T); ke = k_exec - 1
        if te_info is not None:
            raw = te_info.pop("_raw"); te_info["n_src_exec"] = int(n_src[ke])
            te_info["shift_mm"] = [[round(float(x) * 1000, 1) for x in (tgt_pos[ke, r] - raw[ke, r])] for r in (0, 1)]
            self._te_step += k_exec                       # the next inference happens k_exec steps later
        if DEADBAND_MM is not None or DB_XY_MM is not None or DB_Z_MM is not None:
            held = {"L": [False] * 3, "R": [False] * 3}
            # the hold is decided ONCE per arm on the displacement at a fixed reference horizon (DB_REF_K), not on the executed
            # waypoint: with adaptive k = 1-2 a real approach moves only a few mm per waypoint and would always be held
            kr = min(T, DB_REF_K) - 1; dref = {}
            for r in (0, 1):
                p0 = m_now[r][:3, 3]; d = (tgt_pos[kr, r] - p0) * 1000.0
                hold = np.zeros(3, bool)
                if DEADBAND_MM is not None: hold |= np.abs(d) < DEADBAND_MM
                if DB_XY_MM is not None and np.linalg.norm(d[:2]) < DB_XY_MM: hold[:2] = True
                if DB_Z_MM is not None and abs(d[2]) < DB_Z_MM: hold[2] = True
                tgt_pos[:, r, hold] = p0[hold]
                held["LR"[r]] = [bool(h) for h in hold]; dref["LR"[r]] = [round(float(x), 1) for x in d]
            db_info = dict(xy_mm=DB_XY_MM, z_mm=DB_Z_MM, axis_mm=None if DEADBAND_MM is None else DEADBAND_MM.tolist(), ref_k=kr + 1,
                           d_ref_mm=dref, held_exec=held)
        self.te_info, self.db_info, self.ak_info, self.exec_k_now = te_info, db_info, ak_info, k_exec

        q_now = np.asarray([float(x) for x in history[-1]["joints14"]], np.float64)
        qa = q_now[ARM_IDX]
        q_curobo = None
        if IK_BACKEND == "curobo":
            q_tgt, fkerr, ok, self._curobo_info = curobo_solve(qa, tgt_pos, tgt_quat)
            q_curobo = q_tgt.copy(); bad = np.flatnonzero(fkerr.max(1) > FK_MAX_MM)
            valid_T = int(bad[0]) if len(bad) else T
            print(f"[v4] curobo chunk {T} wp: {self._curobo_info['ms']} ms server, pos err max {fkerr.max():.1f} mm, valid {valid_T}", flush=True)
            if valid_T < 1:
                raise RuntimeError(f"cuRobo could not reach the first predicted pose (FK error {fkerr[0].max():.0f} mm)")
        elif IK_BACKEND in ("pink", "learned", "policydq", "pinkdq", "diffik"):
            # [2026-09-29] the validity pre-check used PyRoki WITHOUT the frame fix whatever backend executes; with pink/learned
            # each executed waypoint is guarded by its own backend (FK > FK_MAX_MM / NaN -> stop), so no PyRoki pre-check here.
            q_tgt = np.repeat(qa[None, :], T, 0); fkerr = np.zeros((T, 2)); ok = np.ones((T, 2), bool); valid_T = T
        else:
            q_tgt, fkerr, ok = self.kin.solve_chunk(qa, tgt_pos, tgt_quat)
            bad = np.flatnonzero((fkerr.max(1) > FK_MAX_MM) | ~ok.all(1))
            valid_T = int(bad[0]) if len(bad) else T
            if valid_T < 1:
                raise RuntimeError(f"IK could not reach the first predicted pose (FK error {fkerr[0].max():.0f} mm)")
        dq = q_tgt - qa[None, :]
        q_cmd = np.clip(qa[None, :] + np.clip(dq, -DQ_MAX, DQ_MAX), LO, HI)

        if ARM_ONLY == "right":
            q_cmd[:, 0:6] = qa[None, 0:6]                            # left arm joints: exactly where they are
        cmd = np.zeros((T, 14))
        cmd[:, ARM_IDX] = q_cmd
        for r, gi in enumerate(GRIP_IDX):
            cmd[:, gi] = [grip_to_cmd(w) for w in widths[:, r]] if GRIPPER_MODE in ("predict", "binary", "continuous", "relative") \
                else float(q_now[gi])                                # HOLD: re-send what the jaw already is
            if ARM_ONLY and (r == 0 or ARM_ONLY_GRIP == "hold"):
                cmd[:, gi] = jaw_hold_cmd(q_now[gi])                 # one-arm model: unsupervised jaws hold (as a 0..45 COMMAND)
        cmd[:, FLIP_IDX] *= -1.0

        keep = min(valid_T, self.n_action)
        self.last = dict(act=act.copy(), tgt_pos=tgt_pos.copy(), tgt_quat=tgt_quat.copy(), widths=widths.copy(),
                         fkerr_mm=fkerr.copy(), dq=dq.copy(), q_cmd=q_cmd.copy(), q_now=q_now.copy(),
                         valid_step=valid_T, kept=keep, clipped=int((np.abs(dq) > DQ_MAX).sum()),
                         cur_tcp=poses_now.copy(), cur_mat=m_now.copy(), state=state.copy(),
                         pred_mm=raw_mm.copy(), pred_deg=raw_deg.copy(), cmd_mm=cl_mm.copy(),
                         clamped_steps=n_clamped, gripper_mode=GRIPPER_MODE, action_mode=ACTION_MODE,
                         grip_cmd=np.array([[grip_to_cmd(widths[k, r]) for r in (0, 1)] for k in range(T)]))
        # [2026-09-29] IK_BACKEND=policydq / pinkdq: the policy's own dq head (action dims 20:32 = q(t+k) - q(t), follower frame, rad)
        self.last["dq_pol"] = act[:, 20:32].copy() if act.shape[1] >= 32 else None
        self.last["q_arm0"] = qa.copy()
        # [2026-10-03] RTC: this chunk's targets as sent (after TE / deadband) are the next chunk's inpainting prefix
        self._rtc_prev = dict(t=time.time(), pos=tgt_pos.copy(), quat=tgt_quat.copy(), widths=widths.copy())
        self.last["rtc"] = getattr(self, "_rtc_info", None); self._rtc_info = None
        self.last["q_curobo"] = q_curobo
        self.calls += 1
        return cmd[:keep]

    def _learned_waypoint(self, qa, joints14_now, tgt_pos_k):
        """[2026-09-29] IK_BACKEND=learned: LearnedIK-v0 (umi_bridge/learned_ik, LIK0-P12 best.pt) on the measured q and the REL of the
        whole current chunk re-expressed from the MEASURED TCP (dataset frame); returns the dq of the waypoint being executed.
        Guard: NaN or |FK(q_t + dq) - target| > FK_MAX_MM -> None (the caller falls back to numerical IK and logs it)."""
        try:
            if not hasattr(self, "_lik"):
                sys.path.insert(0, os.path.expanduser("~/umi_bridge/learned_ik")); import lik0_common as _LC
                ck = torch.load(os.path.expanduser(os.environ.get("V4_LIK_CKPT", "~/umi_bridge/learned_ik/runs/LIK0-P12/best.pt")), map_location="cpu", weights_only=False)
                self._lik = _LC.LIK0(ck["stats"]); self._lik.load_state_dict(ck["model"]); self._lik.eval()
                print(f"[v4] learned IK loaded ({os.environ.get('V4_LIK_CKPT', 'LIK0-P12/best.pt')}, step {ck['step']})", flush=True)
            L = self.last; m_meas, _ = self._tcp_mat(joints14_now)
            k = int(np.argmin(np.linalg.norm(L["tgt_pos"].reshape(len(L["tgt_pos"]), -1) - np.asarray(tgt_pos_k).reshape(1, -1), axis=1)))
            rel = np.zeros((16, 18))
            for kk in range(16):
                for r in (0, 1):
                    Tk = np.eye(4); Tk[:3, :3] = Rot.from_quat(L["tgt_quat"][kk, r]).as_matrix(); Tk[:3, 3] = L["tgt_pos"][kk, r]
                    A = np.linalg.inv(m_meas[r]) @ Tk
                    rel[kk, r * 9:r * 9 + 3] = A[:3, 3]; rel[kk, r * 9 + 3:r * 9 + 9] = A[:2, :3].reshape(6)
            # input check: REL re-expressed from the measured TCP vs the policy's own REL (identical when the arm is still)
            act = L.get("act")
            if act is not None:
                self._rel_diff_mm = round(float(max(np.abs(rel[:, [0, 1, 2, 9, 10, 11]] - act[:, [0, 1, 2, 10, 11, 12]]).max(), 0) * 1000), 2)
            with torch.no_grad():
                dq = self._lik(torch.tensor(qa, dtype=torch.float32)[None], torch.tensor(rel, dtype=torch.float32)[None])[0, k].double().numpy()
            q_t = qa + dq
            if not np.isfinite(q_t).all():
                self._lik_fallback = "nan"; return None, None, None
            # [2026-09-29 user] no extra stop rules: large dq / joint limits are handled by the common DQ_MAX + LO/HI clip like every backend
            q14 = np.zeros(14); q14[ARM_IDX] = q_t; m_new, _ = self._tcp_mat(q14)
            fe = np.array([[np.linalg.norm(m_new[r][:3, 3] - np.asarray(tgt_pos_k)[r]) * 1000 for r in (0, 1)]])
            self._lik_fallback = None
            return q_t[None], fe, np.ones((1, 2), bool)
        except Exception as e:  # noqa: BLE001
            self._lik_fallback = f"error {e}"; print("[v4] learned IK failed, numerical fallback:", e, flush=True)
            return None, None, None

    def _grip_relative(self, r, g, raw_now):
        """[2026-09-29] V4_GRIPPER=relative: command from the CHANGE of the predicted g between cycles, with a deadband.
        dg > +eps -> OPEN (GRIP_BIN_OPEN), dg < -eps -> CLOSE (GRIP_BIN_CLOSED), else HOLD the last command. The first cycle of a
        rollout has no previous g: it holds the MEASURED jaw state (open if the jaw reads > half travel). Reset on /run."""
        if not hasattr(self, "_g_prev"):
            self._g_prev = [None, None]; self._g_cmd = [None, None]
        g = float(g) / GRIP_REL_SCALE; prev = self._g_prev[r]
        if prev is None:
            act = "init"; cmdv = GRIP_BIN_OPEN if raw_to_width(raw_now) > 0.5 * CART20_W_OPEN else GRIP_BIN_CLOSED
        elif GRIP_REL_MODE == "acc":
            h = self._g_hist[r] if hasattr(self, "_g_hist") else None
            if h is None:
                self._g_hist = getattr(self, "_g_hist", [[], []]); h = self._g_hist[r]
            h.append(g - prev); del h[:-GRIP_ACC_N]; sc = sum(h)
            if sc > GRIP_ACC_OPEN:
                act = "open"; cmdv = GRIP_BIN_OPEN; h.clear()
            elif sc < GRIP_ACC_CLOSE:
                act = "close"; cmdv = GRIP_BIN_CLOSED; h.clear()
            else:
                act = "hold"; cmdv = self._g_cmd[r] if self._g_cmd[r] is not None else GRIP_BIN_CLOSED
            act = f"{act} (acc {sc:+.3f})"
        elif g > prev + GRIP_REL_EPS:
            act = "open"; cmdv = GRIP_BIN_OPEN
        elif g < prev - GRIP_REL_EPS:
            act = "close"; cmdv = GRIP_BIN_CLOSED
        else:
            act = "hold"; cmdv = self._g_cmd[r] if self._g_cmd[r] is not None else GRIP_BIN_CLOSED
        self._g_prev[r] = g; self._g_cmd[r] = cmdv
        self._grip_rel_log = getattr(self, "_grip_rel_log", [None, None]); self._grip_rel_log[r] = dict(g_prev=prev, g=round(g, 3), act=act, cmd=cmdv)
        return cmdv

    def reset_anchor(self):
        """RELCART20: forget the anchor; the next observation becomes the new one (rollout start)."""
        self._anchor = None
        self._te_hist = None                                           # [2026-10-01] temporal ensemble: new rollout
        self._g_prev = [None, None]; self._g_cmd = [None, None]      # relative gripper: new rollout
        self._g_hist = [[], []]

    def solve_waypoint(self, joints14_now, tgt_pos_k, tgt_quat_k, widths_k=None):
        """Re-solve ONE nominal Cartesian target from the MEASURED joints.

        The chunk's targets T1..T4 come from the model and are never recomputed -- only the IK seed changes.
        Solving the whole chunk once (solve_chunk from the pre-execution pose) stacks each step's solution on
        the previous PREDICTED joints, so the FK residual grew 2.1 -> 4.3 -> 6.7 -> 9.2 mm along one chunk as
        the robot fell behind. Seeding from where the arm actually is cuts that accumulation.
        """
        q_now = np.asarray([float(x) for x in joints14_now], np.float64)
        qa = q_now[ARM_IDX]
        if UMI_CONV:                                         # [10-06] HandUMI convention -> robot TCP (dataset frame), for EVERY IK backend
            Rr_ = [Rot.from_quat(q_).as_matrix() @ R_UMI_OFF.T for q_ in np.asarray(tgt_quat_k, np.float64)]
            tgt_quat_k = np.stack([Rot.from_matrix(R_).as_quat() for R_ in Rr_])
            tgt_pos_k = np.stack([np.asarray(p_, np.float64) - R_ @ UMI_TCP_OFF_T for p_, R_ in zip(np.asarray(tgt_pos_k, np.float64), Rr_)])
        tq = np.asarray(tgt_quat_k, np.float64).copy()
        if USE_FRAME_FIX:                                    # dataset frame -> FK frame, for the IK target
            for r in range(tq.shape[0]):
                tq[r] = Rot.from_matrix(Rot.from_quat(tq[r]).as_matrix() @ R_DS_TO_FK).as_quat()
        self._lik_stop = None; self._shadow = None
        if IK_BACKEND == "learned":
            q_tgt, fkerr, ok = self._learned_waypoint(qa, joints14_now, tgt_pos_k)
            if q_tgt is None:
                # [2026-09-29 advisor] no silent hybrid: the pure-learned A/B stops this cycle and records why (no numerical fallback)
                self._lik_stop = self._lik_fallback or "learned IK failed"; self._ik_used = "learned_STOP"
                q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
            else:
                self._ik_used = "learned"
                qs, fs, oks = self.kin.solve_chunk(qa, np.asarray(tgt_pos_k)[None], tq[None])      # shadow numerical: logged, NOT executed
                self._shadow = dict(q=[round(float(x), 4) for x in qs[0]], fk_err_mm=round(float(fs.max()), 2),
                                    dq_diff_vs_learned=[round(float(x), 4) for x in (qs[0] - q_tgt[0])])
        elif IK_BACKEND in ("policydq", "pinkdq"):
            # [2026-09-29] option 5: the policy's own joint head. q_ref = q(inference) + dq_pol[k]  (the same q_t the label used).
            #   policydq: execute q_ref directly (joint-space, no IK).   pinkdq: Pink to the Cartesian target, seeded at q_ref and
            #   pulled toward it (posture cost V4_PINKDQ_POSTURE, default 1e-2) -> demo-like branch + exact FK.
            L = self.last; dqp = L.get("dq_pol")
            if dqp is None:
                raise RuntimeError(f"IK_BACKEND={IK_BACKEND} needs a 32-dim action with the dq12 head (this ckpt has {L['act'].shape[1]})")
            k = int(np.argmin(np.linalg.norm(L["tgt_pos"].reshape(len(L["tgt_pos"]), -1) - np.asarray(tgt_pos_k).reshape(1, -1), axis=1)))
            q_ref = L["q_arm0"] + dqp[k]
            q14 = np.zeros(14); q14[ARM_IDX] = q_ref; m_ref, _ = self._tcp_mat(q14)
            fe_ref = [round(float(np.linalg.norm(m_ref[r][:3, 3] - np.asarray(tgt_pos_k)[r]) * 1000), 2) for r in (0, 1)]
            if IK_BACKEND == "policydq":
                self._ik_used = "policydq"; q_tgt, fkerr, ok = q_ref[None], np.array([fe_ref]), np.ones((1, 2), bool)
                self._shadow = dict(k=k, fk_vs_cart_target_mm=fe_ref)
            else:
                if not hasattr(self, "_pink"):
                    import pink_ik
                    self._pink = pink_ik.PinkIK([self.kin._names[j] for j in self.kin._arm["left"]] + [self.kin._names[j] for j in self.kin._arm["right"]])
                T = []
                for r in range(2):
                    Tk = np.eye(4); Tk[:3, :3] = Rot.from_quat(np.asarray(tgt_quat_k, np.float64)[r]).as_matrix(); Tk[:3, 3] = np.asarray(tgt_pos_k)[r]; T.append(Tk)
                qp, perr, it = self._pink.solve(q_ref, T, posture_q12=q_ref, posture_cost=float(os.environ.get("V4_PINKDQ_POSTURE", "1e-2")), ori=PINK_ORI, hold_q12=qa, lock=PINK_LOCK)
                if not np.isfinite(qp).all():
                    self._lik_stop = "pinkdq returned NaN"; self._ik_used = "pinkdq_STOP"
                    q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
                else:
                    self._ik_used = "pinkdq"; q_tgt, fkerr, ok = qp[None], perr[None], np.ones((1, 2), bool)
                self._shadow = dict(k=k, policy_fk_vs_cart_target_mm=fe_ref, pink_iters=int(it),
                                    dq_diff_pink_vs_policy=[round(float(x), 4) for x in (q_tgt[0] - q_ref)])
        elif IK_BACKEND == "pink":
            # [2026-09-29] Pinocchio/Pink QP IK to convergence (pink_ik.py); targets in the DATASET frame (its TCP == dataset TCP),
            # so tgt_quat_k is used as given (no R_DS_TO_FK). NaN -> stop this cycle (no fallback). Shadow PyRoki logged.
            if not hasattr(self, "_pink"):
                import pink_ik
                self._pink = pink_ik.PinkIK([self.kin._names[j] for j in self.kin._arm["left"]] + [self.kin._names[j] for j in self.kin._arm["right"]])
                print(f"[v4] pink IK ready (pos 1.0 / ori 0.5 / posture {os.environ.get("V4_PINK_POSTURE", "1e-3")} to the seed, quadprog, ori={PINK_ORI})", flush=True)
            T = []
            for r in range(2):
                Tk = np.eye(4); Tk[:3, :3] = Rot.from_quat(np.asarray(tgt_quat_k, np.float64)[r]).as_matrix(); Tk[:3, 3] = np.asarray(tgt_pos_k)[r]; T.append(Tk)
            qp, perr, it = self._pink.solve(qa, T, ori=PINK_ORI, lock=PINK_LOCK, posture_cost=float(os.environ.get("V4_PINK_POSTURE", "1e-3")))   # [2026-09-30] posture prior to the measured q (default unchanged)
            # [2026-10-01 user, option C] joint5 (wrist_yaw) free but slew-limited: if the full-pose solve moves joint5 by more than
            # V4_YAW_MAX_STEP_DEG on an arm, joint5 is pinned at q_now +- cap and the pose is re-solved, so the other joints absorb
            # what they can (instead of clipping joint5 after the fact). Diagnostics per waypoint in self._yaw_diag:
            # target-vs-current TCP rotation (total, base-z part), IK joint5 change, clamped flag.
            J5 = (4, 10)
            import pinocchio as pin
            try:
                Tc = self._pink.fk(qa)
                rot_tot = [round(float(np.degrees(np.linalg.norm(pin.log3(T[r][:3, :3] @ Tc[r][:3, :3].T)))), 2) for r in (0, 1)]
                rot_z = [round(float(np.degrees(pin.log3(T[r][:3, :3] @ Tc[r][:3, :3].T)[2])), 2) for r in (0, 1)]
            except Exception:
                rot_tot = rot_z = None
            dj5 = [float(np.degrees(qp[j] - qa[j])) for j in J5]; clamped = [False, False]
            mode = "none"
            if YAW_MAX_STEP_DEG > 0 and not any("joint5" in x for x in PINK_LOCK) and np.isfinite(qp).all() and max(abs(d) for d in dj5) > YAW_MAX_STEP_DEG:
                # 1) shrink the ORIENTATION step (slerp the target rotation toward the current one by cap/|dj5|) and re-solve with
                #    joint5 free: position stays exact, the wrist turns at most ~cap this waypoint and catches up on the next ones
                T2 = [t_.copy() for t_ in T]
                for r in (0, 1):
                    clamped[r] = abs(dj5[r]) > YAW_MAX_STEP_DEG
                    if clamped[r]:
                        frac = YAW_MAX_STEP_DEG / abs(dj5[r]); Rc = Tc[r][:3, :3]
                        T2[r][:3, :3] = Rc @ pin.exp3(frac * pin.log3(Rc.T @ T[r][:3, :3]))
                q2, p2, it2 = self._pink.solve(qa, T2, ori=PINK_ORI, lock=PINK_LOCK, posture_cost=float(os.environ.get("V4_PINK_POSTURE", "1e-3")))
                mode = "ori_slerp"
                # 2) fallback: still over the cap -> pin joint5 at q_now +- cap on the shrunk target
                if np.isfinite(q2).all() and max(abs(float(np.degrees(q2[j] - qa[j]))) for j in J5) > YAW_MAX_STEP_DEG + 0.5:
                    hold = np.asarray(q2, np.float64).copy(); cap = np.radians(YAW_MAX_STEP_DEG)
                    for j in J5: hold[j] = qa[j] + np.clip(q2[j] - qa[j], -cap, cap)
                    q3, p3, it3 = self._pink.solve(qa, T2, ori=PINK_ORI, lock=tuple(PINK_LOCK) + ("joint5",), hold_q12=hold,
                                                   posture_cost=float(os.environ.get("V4_PINK_POSTURE", "1e-3")))
                    if np.isfinite(q3).all():
                        q2, p2, it2, mode = q3, p3, it2 + it3, "ori_slerp+pin"
                if np.isfinite(q2).all():
                    qp, perr, it = q2, p2, it + it2
            self._yaw_diag = dict(tgt_rot_deg=rot_tot, tgt_rot_z_deg=rot_z, ik_j5_delta_deg=[round(d, 2) for d in dj5],
                                  j5_clamped=clamped, cap_mode=mode, j5_cmd_delta_deg=[round(float(np.degrees(qp[j] - qa[j])), 2) for j in J5], cap_deg=YAW_MAX_STEP_DEG)
            if not np.isfinite(qp).all():
                self._lik_stop = "pink returned NaN"; self._ik_used = "pink_STOP"
                q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
            else:
                self._ik_used = "pink"; q_tgt, fkerr, ok = qp[None], perr[None], np.ones((1, 2), bool)
                qs, fs, _ = self.kin.solve_chunk(qa, np.asarray(tgt_pos_k)[None], tq[None])
                self._shadow = dict(q=[round(float(x), 4) for x in qs[0]], fk_err_mm=round(float(fs.max()), 2), pink_iters=int(it),
                                    dq_diff_vs_pink=[round(float(x), 4) for x in (qs[0] - qp)])
        elif IK_BACKEND == "curobo":
            # chunk path from step() (solved once from the measured q at observation time); a target not in the chunk (stream blend)
            # is solved alone from the measured q
            L = getattr(self, "last", {}) or {}; Qc = L.get("q_curobo"); j = None
            if Qc is not None:
                d = np.abs(np.asarray(L["tgt_pos"])[:len(Qc)] - np.asarray(tgt_pos_k)[None]).reshape(len(Qc), -1).max(1)
                j = int(d.argmin()) if d.min() < 1e-9 else None
            try:
                if j is not None:
                    q_tgt, fkerr, ok = Qc[j][None], np.asarray(L["fkerr_mm"])[j][None], np.ones((1, 2), bool); self._ik_used = f"curobo[k{j + 1}]"
                else:
                    qq, fe, okk, _ = curobo_solve(qa, np.asarray(tgt_pos_k)[None], np.asarray(tgt_quat_k)[None])
                    q_tgt, fkerr, ok = qq, fe, np.ones((1, 2), bool); self._ik_used = "curobo[single]"
                if fkerr.max() > FK_MAX_MM or not np.isfinite(q_tgt).all():
                    raise RuntimeError(f"FK error {fkerr.max():.0f} mm")
            except Exception as e:
                self._lik_stop = f"curobo failed: {e}"; self._ik_used = "curobo_STOP"
                q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
        elif IK_BACKEND == "diffik":
            # [2026-10-01 user] differential IK: ONE damped-least-squares Jacobian step from the MEASURED q toward the same Cartesian
            # target (pink_ik.diff_step), instead of solving to convergence. Same TCP frame/targets as pink; V4_PINK_LOCK columns
            # zeroed; V4_DIFFIK_ORI_W (0.5 = pink's ori/pos cost ratio), V4_DIFFIK_DAMP (1e-2). Shadow: the converged pink solution.
            if not hasattr(self, "_pink"):
                import pink_ik
                self._pink = pink_ik.PinkIK([self.kin._names[j] for j in self.kin._arm["left"]] + [self.kin._names[j] for j in self.kin._arm["right"]])
            T = []
            for r in range(2):
                Tk = np.eye(4); Tk[:3, :3] = Rot.from_quat(np.asarray(tgt_quat_k, np.float64)[r]).as_matrix(); Tk[:3, 3] = np.asarray(tgt_pos_k)[r]; T.append(Tk)
            qp, perr, req = self._pink.diff_step(qa, T, ori_weight=float(os.environ.get("V4_DIFFIK_ORI_W", "0.5")),
                                                 damping=float(os.environ.get("V4_DIFFIK_DAMP", "1e-2")), lock=PINK_LOCK)
            if not np.isfinite(qp).all():
                self._lik_stop = "diffik returned NaN"; self._ik_used = "diffik_STOP"
                q_tgt, fkerr, ok = qa[None].copy(), np.zeros((1, 2)), np.zeros((1, 2), bool)
            else:
                self._ik_used = "diffik"; q_tgt, fkerr, ok = qp[None], perr[None], np.ones((1, 2), bool)
                qc, pc, it = self._pink.solve(qa, T, ori=PINK_ORI, lock=PINK_LOCK)
                self._shadow = dict(requested_mm=[round(float(x), 1) for x in req], diffik_residual_mm=[round(float(x), 1) for x in perr],
                                    pink_residual_mm=[round(float(x), 1) for x in pc], dq_diff_vs_pink=[round(float(x), 4) for x in (qc - qp)])
        else:
            self._ik_used = "numerical"
            q_tgt, fkerr, ok = self.kin.solve_chunk(qa, np.asarray(tgt_pos_k)[None], tq[None])
        dq = q_tgt - qa[None, :]
        q_cmd = np.clip(qa[None, :] + np.clip(dq, -DQ_MAX, DQ_MAX), LO, HI)[0]
        # [2026-09-29] per-waypoint IK log: how often the outer DQ_MAX 0.6 and LO/HI bite once the solver clip is off
        try:
            q_uncl = qa + np.clip(dq[0], -DQ_MAX, DQ_MAX)
            rec = dict(t=round(time.time(), 3), ckpt=os.path.basename(str(os.environ.get("V4_CKPT", "")).rstrip("/")),
                       exec_k=int(os.environ.get("V4_EXEC_K", "0") or 0), state_mode=STATE_MODE, ik_backend_requested=IK_BACKEND, pink_ori=PINK_ORI, pink_lock=list(PINK_LOCK), ik_backend_executed=getattr(self, "_ik_used", IK_BACKEND), lik_stop_reason=getattr(self, "_lik_stop", None), shadow_numerical=getattr(self, "_shadow", None), rel_input_diff_mm=getattr(self, "_rel_diff_mm", None), ik_max_joint_delta=self.kin.solver.config.max_joint_delta,
                       max_abs_dq=round(float(np.abs(dq).max()), 4), abs_dq=[round(float(x), 4) for x in np.abs(dq[0])],
                       dqmax_hits=[int(i) for i in np.flatnonzero(np.abs(dq[0]) > DQ_MAX)],
                       limit_hits=[int(i) for i in np.flatnonzero((q_uncl < LO) | (q_uncl > HI))],
                       limit_excess=[round(float(x), 4) for x in np.maximum(LO - q_uncl, q_uncl - HI).clip(0)],   # rad past LO/HI per joint
                       q_meas=[round(float(x), 4) for x in qa],
                       fk_err_mm=round(float(fkerr.max()), 2), ok=bool(ok.all()))
            self.last_wp = rec
            with open(IK_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001 -- logging must never break a waypoint
            pass
        if ARM_ONLY == "right":                                       # one-arm model: the left arm never moves (all IK backends)
            q_cmd = np.array(q_cmd, np.float64).reshape(-1, 12) if np.ndim(q_cmd) == 2 else np.array(q_cmd, np.float64)
            if q_cmd.ndim == 2: q_cmd[:, 0:6] = qa[None, 0:6]
            else: q_cmd[0:6] = qa[0:6]
        cmd = np.zeros(14)
        cmd[ARM_IDX] = q_cmd
        for r, gi in enumerate(GRIP_IDX):
            cmd[gi] = (self._grip_relative(r, widths_k[r], q_now[gi]) if (GRIPPER_MODE == "relative" and widths_k is not None) else
                       grip_to_cmd(widths_k[r]) if (GRIPPER_MODE in ("predict", "binary", "continuous", "relative") and widths_k is not None)
                       else float(q_now[gi]))
            if ARM_ONLY and (r == 0 or ARM_ONLY_GRIP == "hold"):
                cmd[gi] = jaw_hold_cmd(q_now[gi])                     # one-arm model: unsupervised jaws hold (as a 0..45 COMMAND)
        cmd[FLIP_IDX] *= -1.0
        return cmd, float(fkerr.max()), bool(ok.all()), int((np.abs(dq) > DQ_MAX).sum())

    def tcp_now(self, joints14):
        """FK of an observation, for measuring what the arm actually did."""
        m, poses = self._tcp_mat(joints14)
        return m, poses
