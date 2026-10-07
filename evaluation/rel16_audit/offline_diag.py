#!/usr/bin/env python3
"""[2026-09-29] Two offline diagnostics on one checkpoint (logging only; model, data and deploy contract untouched).

T1 grip-state copying: on dataset frames, set the jaw state of BOTH arms (history + current) to a counterfactual and predict with
   the SAME noise. If predicted g follows the injected jaw state rather than the scene, the policy copies its own gripper state.
   strata (own arm = the arm with the larger GT A16 motion; fixed before running):
     approach_open  own jaw open (> 80 mm), GT g stays open over 16 steps (min g > 0.9)
     grip_onset     own jaw open (> 80 mm), GT g closes within 16 steps (g16 < 0.3)
     holding        own jaw 20..80 mm (object between the fingers), GT g stays closed (max g < 0.3)
   conditions: real | empty (0 mm) | hold (50 mm) | open (107 mm)
T2 language path: same observation + same noise, the 6 order prompts. Pairwise relative distance between prompts at
   tokens -> VLM encoder output (image / text positions) -> action-decoder input hidden (last flow step) -> A16 (k8, k16 pos)
   and, for A16, the distance between two noise draws with the SAME prompt (prompt/draw ratio as in the v2 gate).
usage: V4_CKPT=<ckpt> offline_diag.py <out.json> [--state cart20]
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, torch

OUT = sys.argv[1]; _ST = sys.argv[sys.argv.index("--state") + 1] if "--state" in sys.argv else "umi76"
CART20 = _ST in ("cart20", "relcart20")                  # both use the 20-D layout with openness at 18:20
os.environ.setdefault("V4_STATE_MODE", _ST if CART20 else "umi76"); os.environ["V4_ACTION_MODE"] = "umi"
sys.path.insert(0, os.path.expanduser("~/holobrain-mac-model"))
import infer_core_v4 as IC
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = os.path.expanduser("~/holobrain-data/lerobot/r180_umi76_rel16_v3d")
FEED_ROOT = os.path.expanduser(f"~/holobrain-data/lerobot/r180_{_ST}_rel16_v3d")
col = lambda t, c: (lambda x: x.storage if isinstance(x, pa.ExtensionArray) else x)(t.column(c).combine_chunks())
def load(root, cols):
    ts = [pq.read_table(f, columns=cols) for f in sorted(glob.glob(f"{root}/data/chunk-000/*.parquet"))]
    return {c: np.concatenate([np.asarray(col(t, c).to_pylist() if c in ("observation.state", "action") else col(t, c).to_numpy()) for t in ts]) for c in cols}
D = load(ROOT, ["observation.state", "action", "index", "task_index", "episode_index"])
S76, A = D["observation.state"].astype(np.float64), D["action"].astype(np.float64)
S = load(FEED_ROOT, ["observation.state"])["observation.state"].astype(np.float64) if CART20 else S76
tasks = pd.read_parquet(f"{ROOT}/meta/tasks.parquet"); TASK = {int(v): k for k, v in tasks["task_index"].items()}

W = S76[:, [37, 75]] * 1000                                                      # current jaw width mm (L, R)
own = np.argmax(np.stack([np.linalg.norm(A[:, 15, r * 10:r * 10 + 3], axis=1) for r in (0, 1)], 1), 1)
gw = W[np.arange(len(W)), own]; G = np.stack([A[np.arange(len(A)), :, own * 10 + 9]], 0)[0]      # own-arm GT g (n, 16)
strata = {"approach_open": (gw > 80) & (G.min(1) > 0.9), "grip_onset": (gw > 80) & (G[:, 15] < 0.3),
          "holding": (gw > 20) & (gw < 80) & (G.max(1) < 0.3)}
rng = np.random.default_rng(0); N = 20
pick = {k: np.sort(rng.choice(np.flatnonzero(m), min(N, m.sum()), replace=False)) for k, m in strata.items()}
print({k: (int(m.sum()), len(pick[k])) for k, m in strata.items()}, flush=True)

INF = IC.V4Inferencer(os.environ["V4_CKPT"])
ds = LeRobotDataset(f"rebot/{os.path.basename(ROOT)}", root=ROOT, video_backend="pyav")
dim = INF.policy.model.dim_action


def set_jaw(st, mm):
    st = st.copy()
    if CART20:
        st[18:20] = np.clip(mm / 1000 / IC.CART20_W_OPEN, 0, 1)
    else:
        st[[36, 37, 74, 75]] = mm / 1000
    return st


def run(idx, st, task, seed, hooks=None):
    x = ds[int(idx)]
    obs = {f"observation.images.{k}": x[f"observation.images.{k}"] for k in ("global", "left_wrist", "right_wrist")}
    obs["observation.state"] = torch.as_tensor(st, dtype=torch.float32); obs["task"] = task
    b = INF.pre(obs)
    g = torch.Generator().manual_seed(seed); noise = torch.randn(1, INF.chunk, dim, generator=g)
    with torch.no_grad():
        ch = INF.policy.predict_action_chunk(b, noise=noise.to(INF.device))
    return INF.post(ch)[0].detach().float().cpu().numpy(), b


res = {"ckpt": os.environ["V4_CKPT"], "state": _ST}
# ---------------- T1
t0 = time.time(); T1 = {}
SKIP_T1 = os.environ.get("SKIP_T1") == "1"          # [2026-09-29] rerun T2 only
for sname, ids in ({} if SKIP_T1 else pick).items():
    rows = []
    for n, i in enumerate(ids):
        o = own[i]; out = {}
        for cname, mm in (("real", None), ("empty", 0.0), ("hold", 50.0), ("open", 107.0)):
            st = S[i] if mm is None else set_jaw(S[i], mm)
            a, _ = run(i, st, TASK[int(D["task_index"][i])], 1000 + n)
            out[cname] = a[:, o * 10 + 9].tolist()
        rows.append(dict(idx=int(i), own=int(o), jaw_mm=float(gw[i]), gt_g=G[i].tolist(), **out))
    T1[sname] = rows
    print(f"[T1 {sname}] n {len(rows)} | own-arm g_pred k1/k8/k16 (median):", flush=True)
    for c in ("real", "empty", "hold", "open"):
        g = np.array([r[c] for r in rows])
        print(f"    {c:6s} {np.median(g[:, 0]):5.2f} {np.median(g[:, 7]):5.2f} {np.median(g[:, 15]):5.2f}   | frac g8 < 0.6 (CLOSE at the deploy threshold) {np.mean(g[:, 7] < 0.6):.2f}", flush=True)
    gt = np.array([r["gt_g"] for r in rows]); print(f"    GT     {np.median(gt[:, 0]):5.2f} {np.median(gt[:, 7]):5.2f} {np.median(gt[:, 15]):5.2f}   | frac {np.mean(gt[:, 7] < 0.6):.2f}  ({time.time() - t0:.0f}s)", flush=True)
res["T1"] = T1

# ---------------- T2
cap = {}
enc = INF.policy.model.vlm.language_model.model.encoder
h1 = enc.register_forward_hook(lambda m, i, o: cap.__setitem__("enc", (o[0] if isinstance(o, tuple) else o.last_hidden_state).detach().float().cpu()))
h2 = INF.policy.model.transformer.action_decoder.register_forward_pre_hook(lambda m, i: cap.__setitem__("dec_in", i[0].detach().float().cpu()))
ids2 = np.sort(np.concatenate([pick["approach_open"][:4], pick["grip_onset"][:3], pick["holding"][:3]]))
if os.environ.get("T2_STARTS") == "1":
    # [2026-09-29 advisor] ambiguous states: episode start (frame 0) of 12 episodes, 2 per order -- the first pick target differs by prompt
    fi = load(ROOT, ["frame_index"])["frame_index"]
    # [2026-09-29] only episodes with a valid initial-cube table (cube_table.py, rule fixed before model outputs), 2 per order
    CT = json.load(open(os.path.expanduser("~/umi_bridge/rel16_audit/cube_table.json")))["episodes"]
    starts = np.array([i for i in np.flatnonzero(fi == 0) if str(int(D["episode_index"][i])) in CT]); st_order = D["task_index"][starts]
    ids2 = np.sort(np.concatenate([starts[st_order == o][:2] for o in range(6)]))
    QT = np.concatenate([np.asarray(col(t, "aux.q_t").to_pylist()) for t in [pq.read_table(f, columns=["aux.q_t"]) for f in sorted(glob.glob(f"{ROOT}/data/chunk-000/*.parquet"))]])
rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9))
T2 = []
for n, i in enumerate(ids2):
    per = {}
    for p in range(6):
        a, b = run(i, S[i], TASK[p], 5000 + n)
        tok = b["observation.language.tokens"][0].cpu().numpy()
        tpv = INF.policy.model._tpv
        per[p] = dict(tok=tok, img=cap["enc"][0, :tpv].numpy(), txt=cap["enc"][0, tpv:].numpy(), dec=cap["dec_in"][0].numpy(), a=a)
    a_d, _ = run(i, S[i], TASK[int(D["task_index"][i])], 9000 + n)                                    # same prompt, other noise
    own_p = int(D["task_index"][i]); o = own[i] * 10
    pairs = [(x, y) for x in range(6) for y in range(x + 1, 6)]
    rec = dict(idx=int(i), token_diff=float(np.mean([np.mean(per[x]["tok"] != per[y]["tok"]) for x, y in pairs])),
               enc_img=float(np.mean([rel(per[x]["img"], per[y]["img"]) for x, y in pairs])),
               enc_txt=float(np.mean([rel(per[x]["txt"], per[y]["txt"]) for x, y in pairs])),
               dec_in=float(np.mean([rel(per[x]["dec"], per[y]["dec"]) for x, y in pairs])))
    for k in (7, 15):
        dp = np.mean([np.linalg.norm(per[x]["a"][k, o:o + 3] - per[y]["a"][k, o:o + 3]) for x, y in pairs]) * 1000
        dd = np.linalg.norm(per[own_p]["a"][k, o:o + 3] - a_d[k, o:o + 3]) * 1000
        rec[f"A{k + 1}_prompt_mm"] = float(dp); rec[f"A{k + 1}_draw_mm"] = float(dd)
    # first-move direction per prompt (own arm = larger |A16|, base frame not needed: compare in the T_t frame)
    dirs = np.stack([per[p]["a"][15, o:o + 3] / max(np.linalg.norm(per[p]["a"][15, o:o + 3]), 1e-9) for p in range(6)])
    rec["A16_dir_cos_min"] = float((dirs @ dirs.T).min())
    if os.environ.get("T2_STARTS") == "1":
        # prompt -> correct first cube: per prompt p, the bottom colour of p is the target; base-frame predicted motion of the
        # moving arm (larger |A16|) vs the direction from its TCP to each of the 3 initial cubes (cube_table.json)
        import re as _re
        sys.path.insert(0, os.path.expanduser("~/c8/c8old")); import rebot_fk_torch as _fkt
        _fk = globals().setdefault("_FKT", _fkt.ReBotFKTorch(dtype=torch.float64))
        with torch.no_grad():
            Tt = _fk.tcp(torch.tensor(np.asarray(QT[i], np.float64))[None]).numpy()[0]              # (2,4,4)
        ct = CT[str(int(D["episode_index"][i]))]; cubes = {c: np.array(ct[c]["xyz_mm"]) for c in ct["_order"]}
        cos_t, top1, arm_ok = [], [], []
        for p in range(6):
            tgt = _re.findall(r"(\w+) cube", TASK[p])[0]
            mv = [np.linalg.norm(per[p]["a"][15, r * 10:r * 10 + 3]) for r in (0, 1)]; am = int(np.argmax(mv))
            v = Tt[am, :3, :3] @ per[p]["a"][15, am * 10:am * 10 + 3] * 1000
            tcp = Tt[am, :3, 3] * 1000
            cs = {c: float(v @ (x - tcp) / max(np.linalg.norm(v) * np.linalg.norm(x - tcp), 1e-9)) for c, x in cubes.items()}
            cos_t.append(cs[tgt]); top1.append(max(cs, key=cs.get) == tgt); arm_ok.append(am == ct[tgt]["arm"])
        rec["tgt_cos_mean"] = float(np.mean(cos_t)); rec["tgt_top1"] = float(np.mean(top1)); rec["tgt_arm_ok"] = float(np.mean(arm_ok))
    T2.append(rec)
    print(f"[T2 {n + 1}/{len(ids2)}] tokens differ {rec['token_diff']:.2f} | enc img {rec['enc_img']:.4f} txt {rec['enc_txt']:.4f} | dec_in {rec['dec_in']:.4f} | "
          f"A8 prompt {rec['A8_prompt_mm']:.1f} vs draw {rec['A8_draw_mm']:.1f} mm | A16 prompt {rec['A16_prompt_mm']:.1f} vs draw {rec['A16_draw_mm']:.1f} mm", flush=True)
h1.remove(); h2.remove()
res["T2"] = T2
m = {k: float(np.median([r[k] for r in T2])) for k in T2[0] if k != "idx"}
print("[T2 median]", {k: round(v, 4) for k, v in m.items()}, flush=True)
if "tgt_top1" in T2[0]:
    import csv
    row = dict(ckpt=os.path.basename(os.environ["V4_CKPT"].rstrip("/")), state=_ST, n=len(T2),
               D_prompt_A16=m["A16_prompt_mm"], D_noise_A16=m["A16_draw_mm"], dec_hidden=m["dec_in"], dir_cos_min=m["A16_dir_cos_min"],
               tgt_cos_mean=float(np.mean([r["tgt_cos_mean"] for r in T2])), tgt_top1=float(np.mean([r["tgt_top1"] for r in T2])),
               tgt_arm_ok=float(np.mean([r["tgt_arm_ok"] for r in T2])))
    print("[T2 target] prompt -> correct first cube:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, "(chance top1 = 0.33)", flush=True)
    f = os.path.expanduser("~/umi_bridge/rel16_audit/lang_target.csv"); new = not os.path.exists(f)
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()})
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
