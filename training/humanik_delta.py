"""[HumanIK-280] HUMANIK_DELTA=1 -> X-VLA joint-space training on the frozen HumanIK / R675 target definition.

Target (both datasets, identical model-facing units):
    state  [14] = [q_L6 (rad), grip_L (0/1), q_R6 (rad), grip_R (0/1)]      absolute, at t
    action [30,14]: arm dims  = q_abs[t+LEAD+i] - q_abs[t]   (rad, cumulative delta, LEAD=5)
                    grip dims = grip[t+LEAD+i]               (absolute 0/1, 1 = closed)
Datasets store per-frame ABSOLUTE values; this patch
  1. shifts the action delta indices to [LEAD, LEAD+chunk)  (XVLAConfig.action_delta_indices)
  2. fetches per-frame validity features with the same indices (arm_valid[2], grip_valid[2]) and sample_valid[1] at t,
     re-injecting them past the preprocessor (which drops non-policy keys)
  3. converts the batch in policy.forward: robot deg->rad + gripper raw->0/1 (HUMANIK_ROBOT=1), then abs->cumulative delta
  4. masks the JointActionSpace loss per (sample, step, dim) with arm_valid / grip_valid / sample_valid
Env: HUMANIK_DELTA=1 enables; HUMANIK_LEAD (5); HUMANIK_ROBOT=1 for the R675 LeRobot set (deg, gripper raw);
     HUMANIK_ROBOT_GRIP_STATE_CLOSED (-135: state grip <= this -> closed), HUMANIK_ROBOT_GRIP_CMD_CLOSED (27: cmd >= this -> closed)
"""
import os, math, torch

LEAD = int(os.environ.get("HUMANIK_LEAD", "5"))
ROBOT = os.environ.get("HUMANIK_ROBOT", "0") == "1"
G_STATE_CLOSED = float(os.environ.get("HUMANIK_ROBOT_GRIP_STATE_CLOSED", "-135"))
G_CMD_CLOSED = float(os.environ.get("HUMANIK_ROBOT_GRIP_CMD_CLOSED", "27"))
ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]; GRIP = [6, 13]
MASK_KEYS = ("arm_valid", "grip_valid", "sample_valid", "observation.state_is_pad", "arm_conf", "grip_conf")
# [method-4, 2026-09-16] HUMANIK_RELIABILITY=1: per-frame reliability WEIGHTS (arm_conf[2], grip_conf[2] in [0,1], fetched with the
# action window) multiply the binary validity mask. Because the C (aux) mask and the D (FK) arm mask are both derived from the arm
# columns of this mask, the weighting is exactly  L = w_arm*L_dq + lambda_C*w_geom*L_C + lambda_D*w_geom*L_FK + w_grip*L_grip  with
# w_geom = w_arm. Datasets without the keys (all robot sets) give weights of 1 -> numerically identical to the frozen recipe.
RELIABILITY = os.environ.get("HUMANIK_RELIABILITY", "0") == "1"
_STATS = {"batches": 0}

# ---- [2026-09-08] EE auxiliary supervision (C: relative-TCP xyz targets in spare action dims 14..19; D: FK-consistency loss) ----
EE_AUX = os.environ.get("XVLA_EE_AUX", "0") == "1"
EE_AUX_SCALE = float(os.environ.get("EE_AUX_SCALE", "10"))      # aux target units: 1.0 = 10 cm (balances rad-scale joint targets)
EE_AUX_LAMBDA = float(os.environ.get("EE_AUX_LAMBDA", "1.0"))   # weight on the aux-dim MSE (scaled units)
EE_FK_LAMBDA = float(os.environ.get("EE_FK_LAMBDA", "100"))     # weight on FK-consistency MSE (m^2): 1 cm err -> 1e-4*100 = 1e-2
EE_LOG_EVERY = int(os.environ.get("EE_LOG_EVERY", "500"))
EE_AUX_SOURCE = os.environ.get("EE_AUX_SOURCE", "cmd")
TARGET = os.environ.get("HUMANIK_TARGET", "cmd")   # [2026-09-17 state-native] "cmd": dq target from the action (q_cmd) column | "state": dq target from the FUTURE measured state (q_state[t+LEAD+i] - q_state[t]); FK-consistency and C then use the same state targets
assert TARGET in ("cmd", "state")   # "cmd": inv(T_state_t)@T_cmd_{t+k} (command target) | "state": inv(T_state_t)@T_state_{t+k} (actual motion, HandUMI-shared)
assert EE_AUX_SOURCE in ("cmd", "state")
# [2026-09-25 C-old] EEF_TARGET_SOURCE: "legacy" (default) = the B1 path below, untouched | "measured" = C and D targets from the
# dataset field observation.ee.tcp_tgt (per frame 2 x 4 x 4, robot base frame, rebot_fk_torch TCP convention = measured human
# EEF retargeted by named transforms, NO IK): C = trans(inv(T_tgt[t]) @ T_tgt[t+LEAD+i]), D = FK(q_t + dq_hat) pos <-> T_tgt[t+LEAD+i] pos.
EEF_TARGET_SOURCE = os.environ.get("EEF_TARGET_SOURCE", "legacy"); assert EEF_TARGET_SOURCE in ("legacy", "measured")
MEAS_KEY = "observation.ee.tcp_tgt"
MEAS_SELFTEST = os.environ.get("EEF_MEASURED_SELFTEST", "0") == "1"   # TEST ONLY: synthesize tcp_tgt := FK(state window) in forward
_FK = {"fk": None}
def _fk_for(t):
    import rebot_fk_torch
    fk = _FK["fk"]
    if fk is None: fk = rebot_fk_torch.ReBotFKTorch(device=t.device, dtype=torch.float32)
    elif fk.dev != t.device: fk = fk.to(t.device, torch.float32)
    _FK["fk"] = fk; return fk


def _to_model_units(state, action):
    """robot: deg->rad for arm dims, raw gripper -> 0/1.  human: already rad / 0-1."""
    if not ROBOT:
        return state, action
    s = state.clone(); a = action.clone()
    s[..., ARM] = torch.deg2rad(s[..., ARM]); a[..., ARM] = torch.deg2rad(a[..., ARM])
    s[..., GRIP] = (s[..., GRIP] <= G_STATE_CLOSED).to(s.dtype)      # follower position: 0 open .. -270 closed
    a[..., GRIP] = (a[..., GRIP] >= G_CMD_CLOSED).to(a.dtype)        # command: 0 open .. ~55 closed
    return s, a


def _build_mask(batch, B, K, D, device, dtype):
    m = torch.ones(B, K, D, device=device, dtype=dtype)
    if "arm_valid" in batch:                       # (B,K,2)
        av = batch["arm_valid"].to(device=device, dtype=dtype)
        m[:, :, 0:6] *= av[:, :, 0:1]; m[:, :, 7:13] *= av[:, :, 1:2]
    if "grip_valid" in batch:
        gv = batch["grip_valid"].to(device=device, dtype=dtype)
        m[:, :, 6:7] *= gv[:, :, 0:1]; m[:, :, 13:14] *= gv[:, :, 1:2]
    if "sample_valid" in batch:                    # (B,1) or (B,)
        sv = batch["sample_valid"].to(device=device, dtype=dtype).reshape(B, 1, 1)
        m = m * sv
    if RELIABILITY:                                # method-4: continuous reliability weights on top of the binary validity
        if "arm_conf" in batch:
            ac = batch["arm_conf"].to(device=device, dtype=dtype).clamp(0, 1)
            m[:, :, 0:6] *= ac[:, :, 0:1]; m[:, :, 7:13] *= ac[:, :, 1:2]
        if "grip_conf" in batch:
            gc = batch["grip_conf"].to(device=device, dtype=dtype).clamp(0, 1)
            m[:, :, 6:7] *= gc[:, :, 0:1]; m[:, :, 13:14] *= gc[:, :, 1:2]
    if "action_is_pad" in batch:                   # chunk crossing the episode end
        m = m * (~batch["action_is_pad"].to(device)).to(dtype).unsqueeze(-1)
    return m


def _valid_start_indices(dataset):
    """Frame indices allowed as chunk starts, or None if this is not a LeRobot dataset."""
    try:
        meta = dataset.meta; hf = dataset.hf_dataset
    except Exception:
        return None
    try:
        import numpy as np
        if "sample_valid" in meta.features:
            sv = np.asarray(hf["sample_valid"], dtype=float).reshape(-1)
            return [int(i) for i in np.flatnonzero(sv > 0.5)]
        tail = LEAD + 30 - 1
        allowed = []
        # [2026-09-25 c8probe] POSITIONAL per-episode ranges from the loaded rows, so an episode subset
        # (--dataset.episodes) works: the original used meta.episodes' GLOBAL dataset_from/to_index, which index past a
        # subset dataset (IndexError). On the full dataset both give the identical set (verified: c8probe sampler_equiv).
        ep_col = np.asarray(hf["episode_index"]).reshape(-1)
        cut = np.flatnonzero(np.diff(ep_col) != 0) + 1; bounds = np.r_[0, cut, len(ep_col)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            allowed.extend(range(int(a), max(int(a), int(b) - tail)))
        return allowed
    except Exception as ex:  # noqa
        print(f"[humanik_delta] sampler fallback (uniform): {ex}", flush=True); return None


def install():
    if os.environ.get("HUMANIK_DELTA", "0") != "1":
        return
    import lerobot.policies.xvla.configuration_xvla as cfgmod
    import lerobot.policies.xvla.modeling_xvla as modmod
    import lerobot.policies.xvla.action_hub as hub
    import lerobot.datasets.factory as fac
    import lerobot.policies.factory as pfac
    import lerobot.scripts.lerobot_train as lt

    # 1. action window starts LEAD frames ahead
    cfgmod.XVLAConfig.action_delta_indices = property(lambda self: list(range(LEAD, LEAD + self.chunk_size)))

    # 2. validity features fetched with the action window; sample_valid at t
    orig_resolve = fac.resolve_delta_timestamps
    def resolve(cfg, ds_meta):
        dt = orig_resolve(cfg, ds_meta) or {}
        if "action" in dt:
            for k in ("arm_valid", "grip_valid") + (("arm_conf", "grip_conf") if RELIABILITY else ()):
                if k in ds_meta.features: dt[k] = list(dt["action"])
            if "sample_valid" in ds_meta.features: dt["sample_valid"] = [0.0]
            if (EE_AUX and EE_AUX_SOURCE == "state") or TARGET == "state":   # state at t plus the K future states (same window as the action)
                dt["observation.state"] = [0.0] + list(dt["action"])
            if EE_AUX and EEF_TARGET_SOURCE == "measured" and not MEAS_SELFTEST:
                if MEAS_KEY not in ds_meta.features: raise RuntimeError(f"[humanik_delta] EEF_TARGET_SOURCE=measured but the dataset has no {MEAS_KEY}")
                dt[MEAS_KEY] = [0.0] + list(dt["action"])
        return dt or None
    fac.resolve_delta_timestamps = resolve

    class _Keep:
        def __init__(self, pre): self._pre = pre
        def __call__(self, batch):
            saved = {k: batch[k] for k in (*MASK_KEYS, "action_is_pad", MEAS_KEY, MEAS_KEY + "_is_pad") if isinstance(batch, dict) and k in batch}
            out = self._pre(batch)
            if isinstance(out, dict):
                dev = next((x.device for x in out.values() if torch.is_tensor(x)), None)
                for k, v in saved.items():
                    if k not in out:
                        t = v if torch.is_tensor(v) else torch.as_tensor(v); out[k] = t.to(dev) if dev is not None else t
            return out
        def __getattr__(self, n): return getattr(self._pre, n)
    orig_mk = pfac.make_pre_post_processors
    def mk(*a, **k):
        pre, post = orig_mk(*a, **k); return _Keep(pre), post
    pfac.make_pre_post_processors = mk
    if hasattr(lt, "make_pre_post_processors"): lt.make_pre_post_processors = mk

    # 3. batch conversion in policy.forward
    orig_forward = modmod.XVLAPolicy.forward
    def forward(self, batch, *a, **k):
        if "action" in batch and "observation.state" in batch:
            _st_raw = batch["observation.state"]; _fut_raw = None
            if _st_raw.ndim == 3:                                    # (B, 1+K, 14) when EE_AUX_SOURCE == "state": split current / future
                _fut_raw = _st_raw[:, 1:]; _st_raw = _st_raw[:, 0]
            state, action = _to_model_units(_st_raw, batch["action"])
            fut_state = _to_model_units(_fut_raw, batch["action"])[0] if _fut_raw is not None else None   # (B,K,14) model units
            if TARGET == "state":
                if fut_state is None: raise RuntimeError("[humanik_delta] HUMANIK_TARGET=state needs the future states in the batch (observation.state window)")
                action = fut_state                                            # state-native: supervise future measured follower state, never q_cmd
            if action.ndim == 2: action = action.unsqueeze(1)
            B, K, D = action.shape
            tgt = action.clone()
            tgt[..., ARM] = action[..., ARM] - state[:, None, ARM]           # cumulative delta w.r.t. current state
            batch = dict(batch); batch["observation.state"] = state; batch["action"] = tgt
            g = tgt[..., GRIP]
            if not bool(((g - g.round()).abs() < 1e-4).all() and (g.min() >= -1e-4) and (g.max() <= 1 + 1e-4)):
                raise RuntimeError(f"[humanik_delta] HARD FAIL: grip targets are not binary ({torch.unique(g).tolist()[:8]}). "
                                   "ACTION normalization must be IDENTITY (pass --policy.normalization_mapping with ACTION=IDENTITY); "
                                   "otherwise the preprocessor rescales absolute values before the cumulative-delta conversion.")
            sp = self.model.action_space
            if not getattr(sp, "_hk_masked", False):
                raise RuntimeError("[humanik_delta] HARD FAIL: masked loss not installed on the action space instance")
            sp._hk_mask = _build_mask(batch, B, K, D, tgt.device, tgt.dtype)
            sp._hk_state_arm = state[..., ARM].float()                              # (B,12) rad at t, for FK loss
            if EE_AUX:
              with torch.autocast(device_type="cuda", enabled=False):               # geometry ALWAYS in fp32 (even under bf16 AMP)
                fk = _fk_for(tgt); T_now = fk.tcp(sp._hk_state_arm.float())        # (B,2,4,4) base->TCP at t
                if EE_AUX_SOURCE == "state":
                    if fut_state is None: raise RuntimeError("[humanik_delta] EE_AUX_SOURCE=state but no future states in the batch (observation.state ndim != 3)")
                    q_fut = fut_state[..., ARM].float()                             # (B,K,12) actual future q (rad)
                    T_fut = fk.tcp(q_fut)
                    d = torch.linalg.inv(T_now)[:, None] @ T_fut                    # C_state: actual TCP motion (HandUMI-shared definition)
                    _sp = batch.get("observation.state_is_pad", None)
                    if _sp is not None and _sp.ndim == 2 and _sp.shape[1] == K + 1:
                        sp._hk_state_pad = (~_sp[:, 1:].to(torch.bool)).to(tgt.dtype).to(tgt.device)   # (B,K) 1 = valid future state
                else:
                    q_cmd = (state[:, None, ARM].float() + tgt[..., ARM].float())   # (B,K,12) absolute commanded q (rad)
                    T_cmd = fk.tcp(q_cmd)                                           # (B,K,2,4,4)
                    d = torch.linalg.inv(T_now)[:, None] @ T_cmd                    # C_cmd: relative command TCP in the current TCP frame
                if EEF_TARGET_SOURCE == "measured":                                 # C-old: C and D from the measured EEF, never FK(q)
                    if MEAS_SELFTEST:
                        if fut_state is None: raise RuntimeError("[humanik_delta] selftest needs the future states (EE_AUX_SOURCE=state)")
                        Tm = torch.cat([T_now[:, None], fk.tcp(fut_state[..., ARM].float())], 1)   # (B,1+K,2,4,4) := FK(state window)
                        _mp = None
                    else:
                        _tm = batch[MEAS_KEY].to(tgt.device).float(); Tm = _tm.reshape(B, K + 1, 2, 4, 4)
                        _mp = batch.get(MEAS_KEY + "_is_pad", None)
                    d = torch.linalg.inv(Tm[:, 0])[:, None] @ Tm[:, 1:]            # C: measured relative TCP (t -> t+LEAD+i)
                    sp._hk_meas_pos = Tm[:, 1:, :, :3, 3]                            # (B,K,2,3) D target positions (base frame)
                    sp._hk_meas_pad = (~_mp[:, 1:].to(torch.bool)).to(tgt.dtype).to(tgt.device) if (_mp is not None and _mp.ndim == 2 and _mp.shape[1] == K + 1) else None
                aux = (d[..., :3, 3].reshape(B, K, 6) * EE_AUX_SCALE).to(tgt.dtype) # dims 14..16 = L dxyz, 17..19 = R dxyz
              tgt = torch.cat([tgt, aux], -1); batch["action"] = tgt
              m = sp._hk_mask
              _ma = torch.cat([m[..., 0:1].expand(B, K, 3), m[..., 7:8].expand(B, K, 3)], -1)
              if EE_AUX_SOURCE == "state" and getattr(sp, "_hk_state_pad", None) is not None: _ma = _ma * sp._hk_state_pad[..., None]
              if EEF_TARGET_SOURCE == "measured" and getattr(sp, "_hk_meas_pad", None) is not None: _ma = _ma * sp._hk_meas_pad[..., None]
              sp._hk_mask = torch.cat([m, _ma], -1)   # aux masked by arm validity (+ future-state padding for C_state)
              if _STATS["batches"] == 0:
                  print(f"[humanik_delta][ee] aux targets ON (source={EE_AUX_SOURCE}): |dxyz| p50 {d[..., :3, 3].norm(dim=-1).median()*1000:.1f} mm p95 {torch.quantile(d[..., :3, 3].norm(dim=-1).flatten().float(), 0.95)*1000:.1f} mm (scale {EE_AUX_SCALE}, lambda aux {EE_AUX_LAMBDA}, lambda fk {EE_FK_LAMBDA}) | TCP z at t: L {T_now[:,0,2,3].mean():.3f} R {T_now[:,1,2,3].mean():.3f} m", flush=True)
            if _STATS["batches"] == 0:
                try:
                    _pd = next(self.parameters()).dtype; _ac = torch.is_autocast_enabled()
                    print(f"[humanik_delta][dtype] parameters {_pd} | autocast enabled in forward: {_ac} (dtype {torch.get_autocast_gpu_dtype() if _ac else None}) | state {state.dtype} target {tgt.dtype}", flush=True)
                except Exception as _de: print(f"[humanik_delta][dtype] probe failed: {_de}", flush=True)
            _STATS["batches"] += 1
            if _STATS["batches"] == 1:
                m = sp._hk_mask; print(f"[humanik_delta] first batch: action {tuple(tgt.shape)} arm|dq| p50 {tgt[..., ARM].abs().median():.4f} p95 {torch.quantile(tgt[..., ARM].abs().flatten().float(), 0.95):.4f} rad | grip uniq {torch.unique(tgt[..., GRIP]).tolist()} | mask cov arm {m[..., ARM].mean():.3f} grip {m[..., GRIP].mean():.3f} sample {(m.flatten(1).amax(1) > 0).float().mean():.3f} | robot={ROBOT} lead={LEAD}", flush=True)
        return orig_forward(self, batch, *a, **k)
    modmod.XVLAPolicy.forward = forward

    # 4. masked JointActionSpace loss
    def compute_loss(self, pred, target):
        assert pred.shape == target.shape
        B, K, D = pred.shape
        m = getattr(self, "_hk_mask", None)
        if m is None or m.shape[0] != B or m.shape[1] != K:
            m = torch.ones_like(pred)
        elif m.shape[-1] != D:                                             # action padded to dim_action: keep padded dims like the original loss
            pad = torch.ones(B, K, D - m.shape[-1], device=pred.device, dtype=pred.dtype); m = torch.cat([m.to(pred.dtype), pad], -1)
        m = m.to(pred.dtype)
        gi = list(self.gripper_idx); ji = [i for i in range(D) if i not in set(gi)]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[..., gi], target[..., gi], reduction="none")
        gripper_loss = (bce * m[..., gi]).sum() / m[..., gi].sum().clamp_min(1.0) * self.GRIPPER_SCALE
        se = (pred[..., ji] - target[..., ji]) ** 2
        joints_loss = (se * m[..., ji]).sum() / m[..., ji].sum().clamp_min(1.0) * self.JOINTS_SCALE
        return {"joints_loss": joints_loss, "gripper_loss": gripper_loss}
    hub.JointActionSpace.compute_loss = compute_loss
    # 5. sampler: only valid chunk starts are drawn (sample_valid==1 when the dataset has it; else drop the last
    #    LEAD+chunk-1 frames of every episode).  Invalid future targets inside a chunk are still masked in the loss.
    orig_dl = torch.utils.data.DataLoader
    class HumanIKDataLoader(orig_dl):
        def __init__(self, dataset, *a, **k):
            allowed = _valid_start_indices(dataset)
            if allowed is not None and "batch_sampler" not in k:
                k.pop("shuffle", None); k["sampler"] = torch.utils.data.SubsetRandomSampler(allowed)
                print(f"[humanik_delta] sampler: {len(allowed)} valid chunk starts / {len(dataset)} frames ({len(allowed)/max(1,len(dataset)):.3f})", flush=True)
            super().__init__(dataset, *a, **k)
    torch.utils.data.DataLoader = HumanIKDataLoader

    # 4b. masked loss on the ACTUAL action-space instance (action_mode=auto builds AutoActionSpace, not JointActionSpace).
    #     Same weighting as the original: mean of squared error over all real dims, but only over valid (mask=1) entries;
    #     reported as joints_loss + gripper_loss components (they sum to the total). Grip stays MSE on 0/1 targets so
    #     inference needs no sigmoid (same as the R675 baselines with action_mode=auto).
    def _install_masked_loss(space):
        def masked(pred, target):
            if hasattr(space, "_pad_to_model_dim"):
                pred = space._pad_to_model_dim(pred); target = space._pad_to_model_dim(target)
            D = int(getattr(space, "real_dim", target.shape[-1]))
            p, t = pred[..., :D], target[..., :D]
            B, K, _ = p.shape
            m = getattr(space, "_hk_mask", None)
            if m is None or m.shape[0] != B or m.shape[1] != K:
                m = torch.ones_like(p)
            else:
                m = m.to(p.dtype)
                if m.shape[-1] < D: m = torch.cat([m, torch.ones(B, K, D - m.shape[-1], device=p.device, dtype=p.dtype)], -1)
                m = m[..., :D]
            se = (p - t) ** 2 * m
            denom = m.sum().clamp_min(1.0)
            gi = [i for i in GRIP if i < D]; ai = [i for i in range(D) if i not in gi]
            out = {"joints_loss": se[..., ai].sum() / denom, "gripper_loss": se[..., gi].sum() / denom}
            if not getattr(space, "_hk_dtype_logged", False):
                space._hk_dtype_logged = True
                print(f"[humanik_delta][dtype] compute_loss: pred {pred.dtype} target {target.dtype} main_loss {out['joints_loss'].dtype} | autocast here: {torch.is_autocast_enabled()}", flush=True)
            st = getattr(space, "_hk_state_arm", None)
            if EE_AUX and pred.shape[-1] >= 20 and target.shape[-1] >= 20 and st is not None and st.shape[0] == B:
              with torch.autocast(device_type="cuda", enabled=False):               # C/D geometry in fp32 even under bf16 AMP
                pred = pred.float(); target = target.float(); p = p.float(); t = t.float(); m = m.float()
                mfull = getattr(space, "_hk_mask", None)
                ma = mfull[..., 14:20].to(p.dtype) if (mfull is not None and mfull.shape[-1] >= 20) else torch.ones(B, K, 6, device=p.device, dtype=p.dtype)
                se_aux = (pred[..., 14:20] - target[..., 14:20]) ** 2 * ma                     # C: relative-TCP xyz aux dims
                out["ee_aux_loss"] = (EE_AUX_LAMBDA * se_aux.sum() / ma.sum().clamp_min(1.0)).to(p.dtype)
                fk = _fk_for(st); st32 = st.to(p.device)
                q_pred = st32[:, None] + p[..., ARM].float(); q_gt = st32[:, None] + t[..., ARM].float()   # absolute q (rad)
                P_pred = fk.tcp(q_pred)[..., :3, 3]; P_gt = fk.tcp(q_gt)[..., :3, 3]                    # (B,K,2,3) TCP positions
                marm = torch.stack([m[..., 0], m[..., 7]], -1).float()                                  # (B,K,2) arm validity
                if EEF_TARGET_SOURCE == "measured":                                                    # C-old D: measured EEF target
                    P_gt = space._hk_meas_pos.to(P_pred.device).float()
                    if getattr(space, "_hk_meas_pad", None) is not None: marm = marm * space._hk_meas_pad[..., None].float()
                d2 = ((P_pred - P_gt) ** 2).sum(-1)                                                      # m^2
                out["fk_pos_loss"] = (EE_FK_LAMBDA * (d2 * marm).sum() / marm.sum().clamp_min(1.0)).to(p.dtype)   # D: FK consistency
                if not getattr(space, "_hk_dtype_logged_ee", False):
                    space._hk_dtype_logged_ee = True
                    print(f"[humanik_delta][dtype] C/D block: pred->{p.dtype} FK pos {P_pred.dtype} aux_loss {out['ee_aux_loss'].dtype} fk_loss {out['fk_pos_loss'].dtype} | autocast inside block: {torch.is_autocast_enabled()}", flush=True)
                with torch.no_grad():
                    mm = d2.detach().sqrt() * 1000.0; _STATS.setdefault("fk_mm", []).append(mm.reshape(-1, 2).cpu())
                    if _STATS["batches"] % EE_LOG_EVERY == 0 and _STATS["fk_mm"]:
                        allmm = torch.cat(_STATS["fk_mm"], 0); _STATS["fk_mm"] = []
                        print(f"[humanik_delta][ee] batch {_STATS['batches']}: FK pos err (pred vs cmd, over chunk) L p50 {allmm[:,0].median():.1f} p95 {torch.quantile(allmm[:,0], 0.95):.1f} mm | R p50 {allmm[:,1].median():.1f} p95 {torch.quantile(allmm[:,1], 0.95):.1f} mm | aux_loss {out['ee_aux_loss'].item():.4f} fk_loss {out['fk_pos_loss'].item():.4f}", flush=True)
            return out
        space.compute_loss = masked
        space._hk_masked = True

    orig_init = modmod.XVLAPolicy.__init__
    def __init__(self, config, *a, **k):
        orig_init(self, config, *a, **k)
        nm = getattr(config, "normalization_mapping", None)
        sp = self.model.action_space
        _install_masked_loss(sp)
        print(f"[humanik_delta] policy normalization_mapping = {nm}  (ACTION must be IDENTITY) | action_space={type(sp).__name__} real_dim={getattr(sp, 'real_dim', None)} -> masked loss installed", flush=True)
    modmod.XVLAPolicy.__init__ = __init__

    if RELIABILITY: print("[humanik_delta][reliability] method-4 ON: arm_conf/grip_conf multiply the validity mask when the dataset has them (robot sets: none -> weights 1)", flush=True)
    print(f"[humanik_delta] installed: LEAD={LEAD} ROBOT={ROBOT} EE_AUX={EE_AUX} RELIABILITY={RELIABILITY} EEF_TARGET_SOURCE={EEF_TARGET_SOURCE}{' SELFTEST' if MEAS_SELFTEST else ''} (masked loss, abs->cumulative-delta targets" + (f", aux relative-TCP dims 14..19 + FK-consistency loss lambda={EE_FK_LAMBDA}" if EE_AUX else "") + ")", flush=True)
