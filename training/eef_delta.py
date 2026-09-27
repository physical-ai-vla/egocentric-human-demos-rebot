"""[EEF-delta, 2026-09-17] EEF_DELTA=1 -> X-VLA training in a bimanual END-EFFECTOR DELTA action space (the new D0/D250 main line).

Canonical action, per arm 7 dims -> 14 total (model-facing dim stays 20; the policy pads with zeros as before):
    [dx, dy, dz, drotvec_x, drotvec_y, drotvec_z, grip_bool]     grip: 0 = close/hold, 1 = open
For LEAD=5 and chunk K at every start frame t, from the per-frame TCP state (16 dims: L pos3 quat4 grip1, R the same):
    dp[i]      = p[t+LEAD+i] - p[t]                       world frame (the robot base / table frame)
    dR[i]      = R[t+LEAD+i] R[t]^-1                      WORLD frame by default (EEF_ROT_FRAME=body -> R[t]^-1 R[t+LEAD+i])
    drotvec[i] = LogSO3(dR[i])
World frame is the default because the human and the robot do NOT share a body frame: ego16 maps the human wrist into the robot
base frame (position and orientation), so base-frame deltas are directly comparable, while body-frame deltas differ by the constant
offset between the mapped human wrist frame and the real TCP frame. Execution: R_des = exp(drotvec) R_now (world) -> continuity IK.
Units are shared physical constants (never per-dataset statistics), so human and robot targets stay in one space:
    position / EEF_POS_SCALE (default 0.25 m)      rotvec / EEF_ROT_SCALE (default 1.0 rad)      grip stays 0/1
Loss: masked MSE on the 12 continuous dims + BCE-with-logits on the 2 gripper dims (weight EEF_GRIP_LAMBDA, default 1.0),
exactly the component split X-VLA's own `joint` / `ee6d` spaces use (the model predicts the clean action, not a velocity field).
Robot targets come from the MEASURED follower state via FK (`observation.ee.tcp_state`), never from q_cmd.
Env: EEF_DELTA=1 | EEF_LEAD (5) | EEF_TCP_KEY (observation.ee.tcp_state) | EEF_ROT_FRAME (world|body) | EEF_POS_SCALE | EEF_ROT_SCALE
     EEF_GRIP_LAMBDA | EEF_GRIP_OPEN (tcp01_open: dataset grip01 already 1=open | tcp01_closed: invert) | EEF_LOG_EVERY
"""
import os, torch

LEAD = int(os.environ.get("EEF_LEAD", "5"))
TCP_KEY = os.environ.get("EEF_TCP_KEY", "observation.ee.tcp_state")
ROT_FRAME = os.environ.get("EEF_ROT_FRAME", "world")
POS_SCALE = float(os.environ.get("EEF_POS_SCALE", "0.25"))
ROT_SCALE = float(os.environ.get("EEF_ROT_SCALE", "1.0"))
GRIP_LAMBDA = float(os.environ.get("EEF_GRIP_LAMBDA", "1.0"))
GRIP_CONV = os.environ.get("EEF_GRIP_OPEN", "tcp01_open")
# A3 (2026-09-19). Default OFF so the running A1/A2 path is bit-identical.
INCREMENTAL = os.environ.get("EEF_INCREMENTAL", "0") == "1"
LOG_EVERY = int(os.environ.get("EEF_LOG_EVERY", "500"))
# use_proprio=False would change the pretrained action_encoder input width (73728 -> 53248) and the xvla-base weights no longer load,
# so proprio stays enabled and this decides WHAT is fed: "zero" (default) = a constant, i.e. the model sees no proprioception and the
# human and robot observations carry exactly the same information; "tcp" = the current 16-dim TCP state (domain-shared, IK-free) for a
# later ablation; "raw" = whatever the dataset stores (robot joints; NOT available for the human set).
PROPRIO = os.environ.get("EEF_PROPRIO", "zero")
AUX = os.environ.get("EEF_AUX_JCF", "0") == "1"          # master switch for the training-only joint auxiliary branch
AUX_USE_FK = os.environ.get("EEF_AUX_FK", "0") == "1"    # add the proven FK-consistency term on the auxiliary joints
AUX_USE_C = os.environ.get("EEF_AUX_C", "0") == "1"      # add the proven C auxiliary in the spare action dims   # restore the proven recipe's joint + C + FK supervision as TRAINING-ONLY auxiliaries
AUX_C_LAMBDA = float(os.environ.get("EE_AUX_LAMBDA", "0.5"))     # exactly the proven value
AUX_FK_LAMBDA = float(os.environ.get("EE_FK_LAMBDA", "20"))      # exactly the proven value
AUX_C_SCALE = float(os.environ.get("EE_AUX_SCALE", "10"))        # proven: 1.0 = 10 cm
AUX_ACT_LAMBDA = float(os.environ.get("EE_ACT_LAMBDA", "1.0"))   # the proven recipe's own action MSE, on the auxiliary joint branch
LOSS = os.environ.get("EEF_LOSS", "eef")   # "eef": our masked MSE + gripper BCE | "default": whatever action space the policy built (the proven recipe's plain MSE)
SAMPLE_EVERY = int(os.environ.get("EEF_SAMPLE_CHECK", "0"))   # >0: every N steps, run the full denoising sampler on this batch and report the error vs the GT chunk
assert PROPRIO in ("zero", "tcp", "raw")
REAL_DIM = 14; CONT = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]; GRIP = [6, 13]
MASK_KEYS = (TCP_KEY, f"{TCP_KEY}_is_pad", "arm_valid", "grip_valid", "sample_valid", "action_is_pad")
assert ROT_FRAME in ("world", "body"); assert GRIP_CONV in ("tcp01_open", "tcp01_closed")
_STATS = {"batches": 0}


# ----------------------------------------------------------------- quaternion helpers (xyzw, batched, fp32)
def _qnorm(q): return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _qmul(a, b):
    ax, ay, az, aw = a.unbind(-1); bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], -1)


def _qinv(q): return q * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=q.device, dtype=q.dtype)


def _qlog(q):
    """quaternion -> rotation vector (radians), shortest arc."""
    q = _qnorm(q); q = torch.where(q[..., 3:4] < 0, -q, q)                      # w >= 0 -> angle in [0, pi]
    v = q[..., :3]; w = q[..., 3].clamp(-1.0, 1.0); n = v.norm(dim=-1)
    ang = 2.0 * torch.atan2(n, w)
    scale = torch.where(n > 1e-6, ang / n.clamp_min(1e-9), torch.full_like(n, 2.0))   # small angle: 2*v
    return v * scale[..., None]


def _targets_from_tcp(tcp):
    """tcp (B, 1+K, 16) absolute TCP states -> (B, K, 14) action targets in model units."""
    cur, fut = tcp[:, 0], tcp[:, 1:]
    out = []
    for o in (0, 8):
        p0 = cur[:, None, o:o + 3]; q0 = _qnorm(cur[:, None, o + 3:o + 7].expand(-1, fut.shape[1], -1).contiguous())
        p1 = fut[..., o:o + 3]; q1 = _qnorm(fut[..., o + 3:o + 7])
        if INCREMENTAL:
            # element 0 spans t -> t+LEAD; the rest are adjacent. Integrating from the current pose is exact.
            pprev = torch.cat([p0[:, :1], p1[:, :-1]], dim=1)
            qprev = torch.cat([q0[:, :1], q1[:, :-1]], dim=1)
            dp = (p1 - pprev) / POS_SCALE
            qr = _qmul(q1, _qinv(qprev)) if ROT_FRAME == "world" else _qmul(_qinv(qprev), q1)
        else:
            dp = (p1 - p0) / POS_SCALE
            qr = _qmul(q1, _qinv(q0)) if ROT_FRAME == "world" else _qmul(_qinv(q0), q1)
        drv = _qlog(qr) / ROT_SCALE
        g = fut[..., o + 7:o + 8]
        g = g if GRIP_CONV == "tcp01_open" else 1.0 - g
        out += [dp, drv, g]
    return torch.cat(out, -1)


def _build_mask(batch, B, K, D, device, dtype):
    m = torch.ones(B, K, D, device=device, dtype=dtype)
    if "arm_valid" in batch:
        av = batch["arm_valid"].to(device=device, dtype=dtype)
        m[:, :, 0:6] *= av[:, :, 0:1]; m[:, :, 7:13] *= av[:, :, 1:2]
    if "grip_valid" in batch:
        gv = batch["grip_valid"].to(device=device, dtype=dtype)
        m[:, :, 6:7] *= gv[:, :, 0:1]; m[:, :, 13:14] *= gv[:, :, 1:2]
    if "sample_valid" in batch:
        m = m * batch["sample_valid"].to(device=device, dtype=dtype).reshape(B, 1, 1)
    pad = batch.get(f"{TCP_KEY}_is_pad", batch.get("action_is_pad"))
    if pad is not None:
        pad = pad.to(device)
        if pad.ndim == 2 and pad.shape[1] == K + 1: pad = pad[:, 1:]
        if pad.ndim == 2 and pad.shape[1] == K: m = m * (~pad.to(torch.bool)).to(dtype).unsqueeze(-1)
    return m


def _valid_start_indices(dataset, chunk):
    try:
        meta = dataset.meta; hf = dataset.hf_dataset
        import numpy as np
        if "sample_valid" in meta.features:
            sv = np.asarray(hf["sample_valid"], dtype=float).reshape(-1)
            return [int(i) for i in np.flatnonzero(sv > 0.5)]
        tail = LEAD + chunk - 1; allowed = []
        for i in range(len(meta.episodes)):
            e = meta.episodes[i]; a, b = int(e["dataset_from_index"]), int(e["dataset_to_index"])
            allowed.extend(range(a, max(a, b - tail)))
        return allowed
    except Exception as ex:  # noqa
        print(f"[eef_delta] sampler fallback (uniform): {ex}", flush=True); return None


# ----------------------------------------------------------------- action space
def _register_space():
    import lerobot.policies.xvla.action_hub as hub
    if "bimanual_eef_delta" in hub.ACTION_REGISTRY: return hub.ACTION_REGISTRY["bimanual_eef_delta"]

    @hub.register_action("bimanual_eef_delta")
    class BimanualEEFDelta(hub.BaseActionSpace):
        """14 semantic dims (per arm: dxyz, drotvec, grip_bool) padded to the pretrained 20-dim model action head."""
        dim_action = 20
        real_dim = REAL_DIM
        gripper_idx = (6, 13)
        GRIPPER_SCALE = GRIP_LAMBDA

        def __init__(self):
            super().__init__(); self._eef_mask = None

        def _pad_to_model_dim(self, x):
            if x is None or x.size(-1) == self.dim_action: return x
            if x.size(-1) != self.real_dim: x = x[..., : self.real_dim]
            return torch.cat([x, x.new_zeros(*x.shape[:-1], self.dim_action - x.size(-1))], -1)

        def _trim_to_real_dim(self, x): return x[..., : self.real_dim]

        def compute_loss(self, pred, target):
            pred = self._pad_to_model_dim(pred); target = self._pad_to_model_dim(target)
            p, t = pred[..., :REAL_DIM].float(), target[..., :REAL_DIM].float()
            B, K, _ = p.shape
            m = self._eef_mask
            m = torch.ones_like(p) if (m is None or m.shape[0] != B or m.shape[1] != K) else m.to(p.dtype)[..., :REAL_DIM]
            mc, mg = m[..., CONT], m[..., GRIP]
            eef_loss = (((p[..., CONT] - t[..., CONT]) ** 2) * mc).sum() / mc.sum().clamp_min(1.0)
            c_loss = None
            if AUX and AUX_USE_C and pred.shape[-1] >= 20 and target.shape[-1] >= 20:
                c_loss = AUX_C_LAMBDA * torch.nn.functional.mse_loss(pred[..., 14:20].float(), target[..., 14:20].float()).to(pred.dtype)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(p[..., GRIP], t[..., GRIP], reduction="none")
            grip_loss = (bce * mg).sum() / mg.sum().clamp_min(1.0) * self.GRIPPER_SCALE
            out = {"eef_loss": eef_loss.to(pred.dtype), "gripper_loss": grip_loss.to(pred.dtype)}
            if c_loss is not None: out["c_aux_loss"] = c_loss
            return out

        def preprocess(self, proprio, action, mode="train"):
            """the gripper channels are logits, not continuous state: never show their noised value to the model."""
            a = action.clone(); a[..., self.gripper_idx] = 0.0
            return proprio, a

        def postprocess(self, action):
            if action.size(-1) > max(self.gripper_idx):
                action = action.clone(); action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
            return action[..., :REAL_DIM]

    return hub.ACTION_REGISTRY["bimanual_eef_delta"]


_FK = {"fk": None}
_AUX_HEADS = {}          # id(policy) -> training-only head, deliberately outside the module tree
_AUX_HIDDEN = {}


def _fk_for(t):
    import rebot_fk_torch
    fk = _FK["fk"]
    if fk is None: fk = rebot_fk_torch.ReBotFKTorch(device=t.device, dtype=torch.float32)
    elif fk.dev != t.device: fk = fk.to(t.device, torch.float32)
    _FK["fk"] = fk; return fk


ARM_J = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
G_STATE_CLOSED = float(os.environ.get("HUMANIK_ROBOT_GRIP_STATE_CLOSED", "-135"))
G_CMD_CLOSED = float(os.environ.get("HUMANIK_ROBOT_GRIP_CMD_CLOSED", "27"))


def _old_action_target(state_raw, action_raw):
    """The proven recipe's own 14D robot target, rebuilt verbatim: deg->rad on the arms, raw gripper -> 0/1, then the arm dims
    become a cumulative delta w.r.t. the current state. Returns (state_model_units, target)."""
    s = state_raw.clone(); a = action_raw.clone()
    s[..., ARM_J] = torch.deg2rad(s[..., ARM_J]); a[..., ARM_J] = torch.deg2rad(a[..., ARM_J])
    s[..., GRIP] = (s[..., GRIP] <= G_STATE_CLOSED).to(s.dtype)
    a[..., GRIP] = (a[..., GRIP] >= G_CMD_CLOSED).to(a.dtype)
    tgt = a.clone(); tgt[..., ARM_J] = a[..., ARM_J] - s[:, None, ARM_J]
    return s, tgt


# ----------------------------------------------------------------- install
def install():
    if os.environ.get("EEF_DELTA", "0") != "1": return
    import lerobot.policies.xvla.configuration_xvla as cfgmod
    import lerobot.policies.xvla.modeling_xvla as modmod
    import lerobot.datasets.factory as fac
    import lerobot.policies.factory as pfac
    import lerobot.scripts.lerobot_train as lt
    _register_space()

    # 1. the action window starts LEAD frames ahead (the TCP window uses the same indices, plus the current frame)
    cfgmod.XVLAConfig.action_delta_indices = property(lambda self: list(range(LEAD, LEAD + self.chunk_size)))

    orig_resolve = fac.resolve_delta_timestamps
    def resolve(cfg, ds_meta):
        dt = orig_resolve(cfg, ds_meta) or {}
        if "action" in dt:
            if TCP_KEY not in ds_meta.features:
                raise RuntimeError(f"[eef_delta] dataset has no '{TCP_KEY}' feature (needed for the EEF targets); features: {list(ds_meta.features)[:12]}")
            dt[TCP_KEY] = [0.0] + list(dt["action"])
            for k in ("arm_valid", "grip_valid"):
                if k in ds_meta.features: dt[k] = list(dt["action"])
            if "sample_valid" in ds_meta.features: dt["sample_valid"] = [0.0]
        return dt or None
    fac.resolve_delta_timestamps = resolve

    # 2. keep the extra keys past the preprocessor (it drops everything that is not a policy feature)
    class _Keep:
        def __init__(self, pre): self._pre = pre
        def __call__(self, batch):
            saved = {k: batch[k] for k in MASK_KEYS if isinstance(batch, dict) and k in batch}
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

    # 3. build the EEF targets in policy.forward
    orig_forward = modmod.XVLAPolicy.forward
    def forward(self, batch, *a, **k):
        if AUX and id(self) not in _AUX_HEADS: _attach_aux_head(self)
        _raw_state = batch.get("observation.state"); _raw_action = batch.get("action")
        if _raw_state is not None and _raw_state.ndim == 3: _raw_state = _raw_state[:, 0]
        if TCP_KEY in batch:
            tcp = batch[TCP_KEY]
            if tcp.ndim != 3: raise RuntimeError(f"[eef_delta] expected {TCP_KEY} window (B,1+K,16), got {tuple(tcp.shape)}")
            with torch.autocast(device_type="cuda", enabled=False):
                tgt = _targets_from_tcp(tcp.float())
            B, K, D = tgt.shape
            sp = self.model.action_space
            if not getattr(sp, "_eef_installed", False):
                raise RuntimeError("[eef_delta] masked EEF loss is not installed on the action space instance")
            if LOSS != "default": sp._eef_mask = _build_mask(batch, B, K, D, tgt.device, tgt.dtype)
            g = tgt[..., GRIP]
            if not bool(((g - g.round()).abs() < 1e-4).all()):
                raise RuntimeError(f"[eef_delta] HARD FAIL: gripper targets are not binary ({torch.unique(g).tolist()[:8]}) -- check EEF_GRIP_OPEN and the dataset grip column")
            batch = dict(batch); batch["action"] = tgt.to(batch["action"].dtype) if "action" in batch else tgt
            if PROPRIO != "raw" and "observation.state" in batch:
                st = batch["observation.state"]
                st = st[:, 0] if st.ndim == 3 else st
                batch["observation.state"] = torch.zeros_like(st) if PROPRIO == "zero" else tcp[:, 0].to(st.dtype)
            if AUX:
                st_raw = _raw_state; act_raw = _raw_action
                if st_raw is None or act_raw is None:
                    raise RuntimeError("[eef_delta] EEF_AUX_JCF=1 needs the robot state/action window in the batch")
                st_m, old_tgt = _old_action_target(st_raw.float(), act_raw.float())
                with torch.autocast(device_type="cuda", enabled=False):
                    fk = _fk_for(tgt); q_now = st_m[..., ARM_J].float()
                    T_now = fk.tcp(q_now)                                        # (B,2,4,4) base->TCP at t
                    q_cmd = q_now[:, None] + old_tgt[..., ARM_J].float()         # absolute commanded q, the proven C_cmd source
                    d = torch.linalg.inv(T_now)[:, None] @ fk.tcp(q_cmd)
                    c_tgt = (d[..., :3, 3].reshape(B, K, 6) * AUX_C_SCALE)       # body-frame L/R dxyz x10, exactly the proven definition
                sp._aux = dict(old_tgt=old_tgt.to(tgt.dtype), c_tgt=c_tgt.to(tgt.dtype), q_now=q_now)
                if AUX_USE_C:
                    batch["action"] = torch.cat([tgt, c_tgt.to(tgt.dtype)], -1)  # C rides in the spare model dims 14..19, as before
            _STATS["batches"] += 1
            if _STATS["batches"] == 1 or _STATS["batches"] % LOG_EVERY == 0:
                dp = tgt[..., [0, 1, 2]].norm(dim=-1) * POS_SCALE; dr = tgt[..., [3, 4, 5]].norm(dim=-1) * ROT_SCALE
                m = getattr(sp, "_eef_mask", None)
                if m is None: m = torch.ones_like(tgt)
                print(f"[eef_delta] batch {_STATS['batches']}: action {tuple(tgt.shape)} | L |dp| p50/p95 {dp.median()*1000:.0f}/{torch.quantile(dp.flatten().float(),0.95)*1000:.0f} mm | "
                      f"L |drot| p50/p95 {torch.rad2deg(dr.median()):.1f}/{torch.rad2deg(torch.quantile(dr.flatten().float(),0.95)):.1f} deg | grip open frac {g.mean():.3f} | "
                      f"mask cont {m[..., CONT].mean():.3f} grip {m[..., GRIP].mean():.3f} | frame={ROT_FRAME} pos/{POS_SCALE} rot/{ROT_SCALE}", flush=True)
            if SAMPLE_EVERY and _STATS["batches"] % SAMPLE_EVERY == 0:
                was = self.training
                with torch.no_grad():
                    self.eval()
                    act = self.model.generate_actions(**self._build_model_inputs(batch), steps=getattr(self.config, "num_denoising_steps", 10))
                    self.train(was)
                pr = act[..., :REAL_DIM].float(); t2 = tgt.float()          # never name this `a`: it would shadow the *a of forward()
                dp = (pr[..., [0, 1, 2]] - t2[..., [0, 1, 2]]).norm(dim=-1) * POS_SCALE * 1000
                dpr = (pr[..., [7, 8, 9]] - t2[..., [7, 8, 9]]).norm(dim=-1) * POS_SCALE * 1000
                dr = torch.rad2deg((pr[..., [3, 4, 5]] - t2[..., [3, 4, 5]]).norm(dim=-1) * ROT_SCALE)
                gp = (pr[..., GRIP] > 0.5).float() if pr[..., GRIP].max() <= 1.0 and pr[..., GRIP].min() >= 0.0 else (pr[..., GRIP] > 0).float()
                acc = (gp == t2[..., GRIP]).float().mean()
                gt_dp = t2[..., [0, 1, 2]].norm(dim=-1) * POS_SCALE * 1000
                print(f"[eef_delta][sample] batch {_STATS['batches']}: sampled-vs-GT chunk err L |dp| p50/p95 {dp.median():.1f}/{torch.quantile(dp.flatten(),0.95):.1f} mm | "
                      f"R |dp| p50 {dpr.median():.1f} mm | L |drot| p50 {dr.median():.2f} deg | grip acc {acc:.3f} | GT |dp| p50/p95 {gt_dp.median():.1f}/{torch.quantile(gt_dp.flatten(),0.95):.1f} mm", flush=True)
        loss, log = orig_forward(self, batch, *a, **k)
        if AUX:
            if id(self) not in _AUX_HEADS: _attach_aux_head(self)
            extra = {}
            _aux_losses(self, extra)
            for kk, vv in extra.items(): loss = loss + vv; log[kk] = float(vv.detach())
            log["loss"] = float(loss.detach())
        return loss, log
    modmod.XVLAPolicy.forward = forward

    # 3b. training-only auxiliary branch: a small head on the per-action-step hidden state predicts the proven recipe's own
    #     14D robot action target, and that prediction feeds the proven action MSE and the proven FK-consistency loss.
    #     It is never used at inference; the EEF main head remains the policy output.
    def _attach_aux_head(policy):
        tr = policy.model.transformer
        dec = tr.action_decoder
        hid = getattr(dec, "input_size", None) or getattr(dec, "in_features", None)   # DomainAwareLinear exposes input_size
        if hid is None: raise RuntimeError("[eef_delta] could not determine the action-decoder input width for the auxiliary head")
        head = torch.nn.Linear(hid, 14).to(next(policy.parameters()).device, next(policy.parameters()).dtype)
        torch.nn.init.zeros_(head.bias); torch.nn.init.normal_(head.weight, std=0.01)
        _AUX_HEADS[id(policy)] = head
        _AUX_HEADS[("opt", id(policy))] = torch.optim.AdamW(head.parameters(), lr=1e-4)   # the trainer never sees this head, so it gets its own step        # training-only: outside the module tree, so no state-dict key is ever created
        def pre_hook(mod, args, kwargs=None):
            x = args[0] if args else None
            if x is not None: _AUX_HIDDEN[id(policy)] = x
            return None
        dec.register_forward_pre_hook(pre_hook, with_kwargs=True)
        print(f"[eef_delta] auxiliary joint head attached ({hid} -> 14), joint {AUX_ACT_LAMBDA} | FK {AUX_FK_LAMBDA if AUX_USE_FK else "off"} | C {AUX_C_LAMBDA if AUX_USE_C else "off"}", flush=True)

    def _aux_losses(policy, out):
        sp = policy.model.action_space; a = getattr(sp, "_aux", None); h = _AUX_HIDDEN.get(id(policy))
        if not AUX or a is None or h is None: return
        head = _AUX_HEADS.get(id(policy))
        if head is None: return
        pred = head(h.to(head.weight.dtype))
        if pred.ndim == 3 and pred.shape[1] != a["old_tgt"].shape[1]: pred = pred[:, : a["old_tgt"].shape[1]]
        t = a["old_tgt"].to(pred.dtype)
        out["aux_action_loss"] = AUX_ACT_LAMBDA * torch.nn.functional.mse_loss(pred, t)
        if not AUX_USE_FK:
            _AUX_HEADS[("opt", id(policy))].zero_grad(set_to_none=True); return
        with torch.autocast(device_type="cuda", enabled=False):
            fk = _fk_for(pred); qn = a["q_now"].float()
            P_pred = fk.tcp(qn[:, None] + pred[..., ARM_J].float())[..., :3, 3]
            P_gt = fk.tcp(qn[:, None] + t[..., ARM_J].float())[..., :3, 3]
            out["aux_fk_loss"] = (AUX_FK_LAMBDA * ((P_pred - P_gt) ** 2).sum(-1).mean()).to(pred.dtype)
        _AUX_HEADS[("opt", id(policy))].zero_grad(set_to_none=True)   # stepped separately; the trunk still receives gradient through `out`

    # 4. mark the instance (the loss lives on the registered class; the mask is per batch)
    orig_init = modmod.XVLAPolicy.__init__
    def __init__(self, config, *a, **kk):
        orig_init(self, config, *a, **kk)
        sp = self.model.action_space
        if type(sp).__name__ == "BimanualEEFDelta":
            sp._eef_installed = True; sp._eef_mask = None
        elif LOSS == "default":
            sp._eef_installed = True        # EEF_LOSS=default: keep the target change only and let the stock action space own the loss
        else:
            raise RuntimeError(f"[eef_delta] policy built action space {type(sp).__name__}; pass --policy.action_mode=bimanual_eef_delta")
        print(f"[eef_delta] action space {type(sp).__name__} real_dim={sp.real_dim} model_dim={sp.dim_action} | normalization={getattr(config,'normalization_mapping',None)} (ACTION must be IDENTITY)", flush=True)
    modmod.XVLAPolicy.__init__ = __init__

    # 5. sample only valid chunk starts
    orig_dl = torch.utils.data.DataLoader
    class EEFDataLoader(orig_dl):
        def __init__(self, dataset, *a, **kk):
            allowed = _valid_start_indices(dataset, int(os.environ.get("EEF_CHUNK", "30")))
            n_of = int(os.environ.get("EEF_OVERFIT_N", "0"))
            if allowed is not None and n_of > 0:
                allowed = allowed[:: max(1, len(allowed) // n_of)][:n_of]        # tiny-overfit mode: a fixed handful of chunk starts
                print(f"[eef_delta] OVERFIT MODE: {len(allowed)} fixed chunk starts", flush=True)
            if allowed is not None and "batch_sampler" not in kk:
                kk.pop("shuffle", None); kk["sampler"] = torch.utils.data.SubsetRandomSampler(allowed)
                print(f"[eef_delta] sampler: {len(allowed)} valid chunk starts / {len(dataset)} frames ({len(allowed)/max(1,len(dataset)):.3f})", flush=True)
            super().__init__(dataset, *a, **kk)
    torch.utils.data.DataLoader = EEFDataLoader

    print(f"[eef_delta] installed: target={'incremental' if INCREMENTAL else 'cumulative'} LEAD={LEAD} TCP={TCP_KEY} frame={ROT_FRAME} pos_scale={POS_SCALE} m rot_scale={ROT_SCALE} rad grip_lambda={GRIP_LAMBDA} grip={GRIP_CONV} proprio={PROPRIO} loss={LOSS}", flush=True)
