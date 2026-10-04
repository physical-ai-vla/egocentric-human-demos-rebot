"""[2026-10-02] READ-ONLY provenance audit of X-VLA domain slots (D6 vs D20) across the base lineage.
Lineage: xvla-base (HF snapshot cdb7964e) -> widen_v3.py -> xvla_base_rel16v3 -> widen_cart20.py -> xvla_base_cart20v3
         -> xvla_base_cart20v3_d6 / _d20 (hard links + processor json). Opens safetensors read-only; writes only to OUT."""
import hashlib, json, os, sys, csv
import torch
from safetensors import safe_open
OUT = sys.argv[1]
H = "${REMOTE_HOME}"
P = {"orig": f"{H}/.cache/huggingface/hub/models--lerobot--xvla-base/snapshots/cdb7964e4fe842935d671bfab5a5ebe00a96648c",
     "v3": f"{H}/xvla_base_rel16v3", "cart20": f"{H}/xvla_base_cart20v3", "d6": f"{H}/xvla_base_cart20v3_d6", "d20": f"{H}/xvla_base_cart20v3_d20"}
K = {"enc_fc": "model.transformer.action_encoder.fc.weight", "enc_b": "model.transformer.action_encoder.bias.weight",
     "dec_fc": "model.transformer.action_decoder.fc.weight", "dec_b": "model.transformer.action_decoder.bias.weight",
     "sp": "model.transformer.soft_prompt_hub.weight"}
HID, T = 1024, 32
# input-row layout of the encoder per checkpoint: name -> (action, proprio, time) widths; decoder out width
LAY = {"orig": (20, 20, 32, 20), "v3": (32, 76, 32, 32), "cart20": (32, 20, 32, 32), "d6": (32, 20, 32, 32), "d20": (32, 20, 32, 32)}
rep = {"files": {}, "dims": {}}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()

W = {}
for n, d in P.items():
    mp = os.path.realpath(f"{d}/model.safetensors"); st = os.stat(mp)
    rep["files"][n] = {"dir": d, "model_realpath": mp, "inode": st.st_ino, "nlink": st.st_nlink, "size": st.st_size, "sha256": sha(mp),
                       "files": sorted(os.listdir(d))}
    cfg = json.load(open(f"{d}/config.json"))
    pp = json.load(open(f"{d}/policy_preprocessor.json"))
    dom = [s["config"]["domain_id"] for s in pp["steps"] if s.get("registry_name") == "xvla_add_domain_id"]
    with safe_open(mp, "pt") as f:
        W[n] = {k: f.get_tensor(v).float() for k, v in K.items()}
        allkeys = list(f.keys())
    rep["dims"][n] = {"num_domains": cfg.get("num_domains"), "hidden_size": cfg.get("hidden_size"), "len_soft_prompts": cfg.get("len_soft_prompts"),
                      "dim_time": cfg.get("dim_time"), "max_state_dim": cfg.get("max_state_dim"), "max_action_dim": cfg.get("max_action_dim"),
                      "dim_action/dim_proprio (from encoder layout)": LAY[n][:2], "processor_domain_id": dom,
                      "shapes": {K[k]: list(W[n][k].shape) for k in K}, "n_keys": len(allkeys)}
    print(n, rep["files"][n]["sha256"][:16], rep["files"][n]["inode"], dom, {k: tuple(W[n][k].shape) for k in K}, flush=True)

def blocks(n):
    """Named sub-blocks per slot, in TRUE input-major layout. Each value: tensor [nd, ...]."""
    a, p, t, o = LAY[n]; nd = W[n]["enc_fc"].shape[0]
    E = W[n]["enc_fc"].view(nd, a + p + t, HID); D = W[n]["dec_fc"].view(nd, HID, o)
    b = {"enc.action_old0:20": E[:, :20], "enc.proprio": E[:, a:a + p], "enc.time": E[:, a + p:],
         "enc.bias": W[n]["enc_b"], "dec.out_old0:20": D[..., :20], "dec.bias_old0:20": W[n]["dec_b"][:, :20], "soft_prompt": W[n]["sp"]}
    if a > 20: b["enc.action_new20:32"] = E[:, 20:a]
    if o > 20: b["dec.out_new20:32"] = D[..., 20:]; b["dec.bias_new20:32"] = W[n]["dec_b"][:, 20:]
    return b

def cmp(x, y):
    x = x.reshape(-1).double(); y = y.reshape(-1).double()
    if x.shape != y.shape: return {"exact": False, "note": f"shape {tuple(x.shape)} vs {tuple(y.shape)}"}
    d = x - y; nx, ny = x.norm().item(), y.norm().item()
    cos = (x @ y).item() / (nx * ny) if nx > 0 and ny > 0 else float("nan")
    return {"exact": bool(torch.equal(x, y)), "max_abs": d.abs().max().item(), "l2": d.norm().item(),
            "rel_l2": d.norm().item() / max(ny, 1e-30), "cos": cos}

B = {n: blocks(n) for n in P}
nd = 30
# 1) sanity: d6 / d20 vs cart20 file + every tensor
rep["sanity"] = {"d6_d20_cart20_same_inode": len({rep["files"][n]["inode"] for n in ("cart20", "d6", "d20")}) == 1,
                 "d6_d20_cart20_same_sha256": len({rep["files"][n]["sha256"] for n in ("cart20", "d6", "d20")}) == 1}
# 2) slot-vs-slot similarity within the d6/d20 file (identical file; D6 = slot 6 rows, D20 = slot 20 rows)
rows = []
for name, X in B["d20"].items():
    for ref in (6, 20):
        for s in range(nd):
            c = cmp(X[ref], X[s]); rows.append({"group": name, "ref_slot": ref, "slot": s, **{k: c.get(k) for k in ("exact", "max_abs", "l2", "rel_l2", "cos")}})
with open(f"{OUT}/domain_similarity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# 3) per-slot fingerprint stats (d20 file + original)
fp = {}
for tag in ("orig", "d20"):
    fp[tag] = {}
    for name, X in B[tag].items():
        fp[tag][name] = [{"slot": s, "mean": X[s].mean().item(), "std": X[s].std().item(), "min": X[s].min().item(), "max": X[s].max().item(),
                          "all_zero": bool((X[s] == 0).all())} for s in range(nd)]
rep["fingerprint"] = fp
# xavier_uniform expected std for the original DomainAwareLinear fc (Embedding(nd, in*out)): bound = sqrt(6/(nd + in*out)), std = bound/sqrt(3)
def xav(nd_, n): return (6 / (nd_ + n)) ** 0.5 / 3 ** 0.5
rep["init_expected_std"] = {"enc_fc_orig (xavier, 30 x 72*1024)": xav(30, 72 * 1024), "dec_fc_orig (xavier, 30 x 1024*20)": xav(30, 1024 * 20),
                            "soft_prompt (normal)": 0.02, "biases": 0.0}
# 4) lineage per slot (6, 20, plus a trained reference 15 and untrained 0): where does each sub-block first change
lin = {}
order = ["orig", "v3", "cart20", "d6", "d20"]
for s in (0, 6, 15, 20):
    lin[s] = {}
    for name in B["d20"]:
        steps = []
        for a_, b_ in zip(order, order[1:]):
            if name in B[a_] and name in B[b_]:
                c = cmp(B[a_][name][s], B[b_][name][s]); steps.append({"from": a_, "to": b_, **c})
            else:
                steps.append({"from": a_, "to": b_, "note": f"block absent in {a_ if name not in B[a_] else b_} (new in widening)"})
        lin[s][name] = steps
rep["lineage"] = lin
json.dump(rep, open(f"{OUT}/lineage_comparison.json", "w"), indent=1, default=float)
print("WROTE", OUT)
