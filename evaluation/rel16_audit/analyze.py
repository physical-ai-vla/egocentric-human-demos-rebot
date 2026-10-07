"""[2026-09-28] REL16 audit analysis of step2_preds.npz / step2b_grip.npz. Usage: analyze.py <npz> [...]"""
import sys, numpy as np

KS = [0, 7, 15]                                        # k = 1, 8, 16


def tr(a, r):                                           # (..., 16, 20) -> (..., 16, 3) translation of arm r, mm
    return a[..., r * 10:r * 10 + 3] * 1000.0


def cos(a, b):
    return np.sum(a * b, -1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)


for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    n = int(d["done"]); st = d["stratum"][:n]; gt = d["gt"][:n]
    P = {k[5:]: d[k][:n] for k in d.files if k.startswith("pred_")}
    print(f"\n==== {path}  n={n}  ckpt={str(d['ckpt']).split('/')[-1]}")
    # active arm per sample = the arm with the larger GT |A_16|
    act = np.argmax(np.stack([np.linalg.norm(tr(gt, r)[:, 15], axis=-1) for r in (0, 1)], 1), 1)
    sel = lambda a: np.stack([a[i, ..., act[i] * 10:act[i] * 10 + 10] for i in range(n)])   # (n,...,16,10)
    g, pr = sel(gt), {c: sel(v) for c, v in P.items()}
    gT = g[..., :3] * 1000; pT = {c: v[..., :3] * 1000 for c, v in pr.items()}           # pred (n,K,16,3)

    print("\n[2] motion magnitude, active arm, real state (mean over K draws)")
    print(f"{'stratum':10s} {'k':>3s} {'|GT| med':>9s} {'|pred| med':>10s} {'ratio med':>9s} {'ratio IQR':>13s} {'cos med':>7s} {'gain':>6s}")
    for s in dict.fromkeys(st):
        m = st == s
        for k in KS:
            G = np.linalg.norm(gT[m, k], axis=-1); Pm = pT["real"][m][:, :, k].mean(1); Pn = np.linalg.norm(Pm, axis=-1)
            ratio = Pn / np.maximum(G, 1e-6); c = cos(Pm, gT[m, k])
            # gain over the zero action: 1 - |pred-GT| / |GT| (1 = perfect, 0 = no better than standing still)
            gain = 1 - np.linalg.norm(Pm - gT[m, k], axis=-1) / np.maximum(G, 1e-6)
            q = np.percentile(ratio, [25, 75])
            print(f"{s:10s} {k+1:3d} {np.median(G):9.1f} {np.median(Pn):10.1f} {np.median(ratio):9.2f} "
                  f"[{q[0]:5.2f},{q[1]:5.2f}] {np.median(c):7.2f} {np.median(gain):6.2f}")
    sd = np.linalg.norm(pT["real"][:, 0, 15] - pT["real"][:, 1, 15], axis=-1) if pT["real"].shape[1] > 1 else None
    if sd is not None:
        print(f"    draw-to-draw |A16| spread med {np.median(sd):.1f} mm")

    print("\n[2g] gripper width (active arm, mm): current, GT k16, pred k16, pred min over chunk")
    cur = g[:, 0, 9] * 0 + np.nan
    stt = d["state"][:n]
    OFFG = {0: 36, 1: 74}
    cur = np.array([stt[i, OFFG[act[i]] + 1] for i in range(n)]) * 1000
    for s in dict.fromkeys(st):
        m = st == s
        pw = pr["real"][m][..., 9].mean(1) * 1000
        print(f"{s:10s} cur {np.median(cur[m]):6.1f}  GT16 {np.median(g[m, 15, 9]*1000):6.1f}  pred16 {np.median(pw[:, 15]):6.1f}  "
              f"pred_min {np.median(pw.min(1)):6.1f} | GT closes {np.median(cur[m]-g[m,15,9]*1000):5.1f} pred closes {np.median(cur[m]-pw[:,15]):5.1f}")
    # both arms for grip onset: use the arm that actually closes
    if "grip_go" in set(st):
        ga = np.argmax(np.stack([stt[:, OFFG[r] + 1] - gt[:, 15, r * 10 + 9] for r in (0, 1)], 1), 1)
        for c in ("real", "ident", "rev"):
            pw = np.stack([P[c][i, :, :, ga[i] * 10 + 9].mean(0) for i in range(n)]) * 1000
            cw = np.array([stt[i, OFFG[ga[i]] + 1] for i in range(n)]) * 1000
            gw = np.array([gt[i, :, ga[i] * 10 + 9] for i in range(n)]) * 1000
            frac = (cw - pw[:, 15]) / np.maximum(cw - gw[:, 15], 1e-6)
            print(f"  grip_go closing arm [{c:5s}] cur {np.median(cw):5.1f} GT16 {np.median(gw[:,15]):5.1f} pred16 {np.median(pw[:,15]):5.1f} "
                  f"closing fraction pred/GT med {np.median(frac):.2f} IQR {np.percentile(frac,25):.2f}..{np.percentile(frac,75):.2f}")

    print("\n[3] history ablation, active arm, k=8 / k=16 (pred vs real-state pred)")
    for s in dict.fromkeys(st):
        m = st == s
        for c in ("ident", "rev"):
            if c not in pT:
                continue
            out = []
            for k in (7, 15):
                R = pT["real"][m][:, :, k].mean(1); X = pT[c][m][:, :, k].mean(1)
                out.append(f"k{k+1}: |x|/|real| {np.median(np.linalg.norm(X,axis=-1)/np.maximum(np.linalg.norm(R,axis=-1),1e-6)):5.2f} "
                           f"cos(x,real) {np.median(cos(X,R)):5.2f} cos(x,GT) {np.median(cos(X,gT[m,k])):5.2f}")
            print(f"{s:10s} {c:5s} " + " | ".join(out))

    print("\n[4] prompt counterfactual, real state, active arm A_16 (mm)")
    for s in dict.fromkeys(st):
        m = np.flatnonzero(st == s); dists, own_vs = [], []
        for i in m:
            ends = [pT[f"p{j}"][i, :, 15].mean(0) for j in range(6) if not np.isnan(pT[f"p{j}"][i]).any()]
            ends.append(pT["real"][i, :, 15].mean(0)); E = np.stack(ends)
            dd = np.linalg.norm(E[:, None] - E[None], axis=-1); dists.append(dd[np.triu_indices(len(E), 1)].mean())
            own_vs.append(np.linalg.norm(pT["real"][i, 0, 15] - pT["real"][i, 1, 15]) if pT["real"].shape[1] > 1 else np.nan)
        print(f"{s:10s} mean pairwise A16 distance across 6 prompts {np.median(dists):6.1f} mm  (same prompt, other draw: {np.nanmedian(own_vs):5.1f} mm)")

    # [2026-09-28] v2 gate (binary gripper datasets): prompt ratio and close-intent onset
    if gt[..., [9, 19]].min() >= 0 and gt[..., [9, 19]].max() <= 1 and np.isin(gt[..., [9, 19]], [0.0, 1.0]).mean() > 0.5:   # v2 binary or v3 continuous
        print("\n[v2 gate]")
        for s in dict.fromkeys(st):
            m = np.flatnonzero(st == s); rr = []
            for i in m:
                E = np.stack([pT[f"p{j}"][i, :, 15].mean(0) for j in range(6) if not np.isnan(pT[f"p{j}"][i]).any()] + [pT["real"][i, :, 15].mean(0)])
                pw = np.linalg.norm(E[:, None] - E[None], axis=-1)[np.triu_indices(len(E), 1)].mean()
                dr = np.linalg.norm(pT["real"][i, 0, 15] - pT["real"][i, 1, 15]) if pT["real"].shape[1] > 1 else np.nan
                rr.append(pw / max(dr, 1e-6))
            print(f"  {s:10s} prompt/draw ratio med {np.median(rr):5.2f}  (v1 REL16 moving 0.27, C-old 8.3; gate > 1)")
            # [2026-09-28] split by pool: FRONT200 order is 62.6 % predictable from the colored layout alone (HEAD 30.3 %),
            # so a model can look prompt-insensitive in FRONT while still reading the prompt in HEAD, or vice versa
            epn = d["ep"][:n][m]; rr = np.array(rr)
            for pl, mk in (("HEAD", epn < 180), ("FRONT", epn >= 180)):
                if mk.any():
                    print(f"      {pl:5s} n {mk.sum():3d} ratio med {np.median(rr[mk]):5.2f}")
        if "grip_go" in set(st):
            ga = np.argmax(np.stack([gt[:, 0, r * 10 + 9] - gt[:, 15, r * 10 + 9] for r in (0, 1)], 1), 1)
            p16 = np.array([P["real"][i, :, 15, ga[i] * 10 + 9].mean() for i in range(n)])
            p0 = np.array([P["real"][i, :, 0, ga[i] * 10 + 9].mean() for i in range(n)])
            print(f"  grip_go: pred k1 {np.median(p0):.2f} (GT 1) -> k16 {np.median(p16):.2f} (GT 0); close predicted (<0.5) at k16 in {100*np.mean(p16<0.5):.0f} % of samples")

    # [2026-09-28] FROZEN Stage-1 verdict (thresholds fixed before any v2 checkpoint exists; do not tune on results)
    if gt[..., [9, 19]].min() >= 0 and gt[..., [9, 19]].max() <= 1 and np.isin(gt[..., [9, 19]], [0.0, 1.0]).mean() > 0.5:
        def k16(s_):
            m_ = st == s_
            if not m_.any():
                return None
            Gm = np.linalg.norm(gT[m_, 15], axis=-1); Pm = pT["real"][m_][:, :, 15].mean(1)
            return float(np.median(np.linalg.norm(Pm, axis=-1) / np.maximum(Gm, 1e-6))), float(np.median(cos(Pm, gT[m_, 15])))
        head_rr = []
        for i in np.flatnonzero(np.isin(st, ["moving", "rest_go"]) & (d["ep"][:n] < 180)):
            E = np.stack([pT[f"p{j}"][i, :, 15].mean(0) for j in range(6) if not np.isnan(pT[f"p{j}"][i]).any()] + [pT["real"][i, :, 15].mean(0)])
            pw = np.linalg.norm(E[:, None] - E[None], axis=-1)[np.triu_indices(len(E), 1)].mean()
            head_rr.append(pw / max(np.linalg.norm(pT["real"][i, 0, 15] - pT["real"][i, 1, 15]), 1e-6))
        mv, rg = k16("moving"), k16("rest_go")
        gg = None
        if "grip_go" in set(st):
            gm = st == "grip_go"
            ga = np.argmax(np.stack([gt[:, 0, r * 10 + 9] - gt[:, 15, r * 10 + 9] for r in (0, 1)], 1), 1)
            # onset, not level: still OPEN at k1 AND CLOSE by k16 (a constant-CLOSE predictor must not pass; found in the
            # 2026-09-28 dry-run, where v1's width output ~0.02 scored "100 % CLOSE")
            gg = float(np.mean([(P["real"][i, :, 0, ga[i] * 10 + 9].mean() >= 0.5) and (P["real"][i, :, 15, ga[i] * 10 + 9].mean() < 0.5)
                                for i in np.flatnonzero(gm)]))
        checks = {"HEAD prompt/draw > 1.0": (len(head_rr) > 0 and np.median(head_rr) > 1.0, f"{np.median(head_rr) if head_rr else float('nan'):.2f} (n {len(head_rr)})"),
                  "moving k16 ratio 0.8-1.25, cos >= 0.9": (mv is not None and 0.8 <= mv[0] <= 1.25 and mv[1] >= 0.9, str(None if mv is None else tuple(round(x, 2) for x in mv))),
                  "rest_go k16 ratio 0.7-1.3, cos >= 0.9": (rg is not None and 0.7 <= rg[0] <= 1.3 and rg[1] >= 0.9, str(None if rg is None else tuple(round(x, 2) for x in rg))),
                  "grip_go onset OPEN@k1 -> CLOSE@k16 >= 50 %": (gg is not None and gg >= 0.5, f"{None if gg is None else round(100 * gg)} %")}
        print("\n[STAGE-1 VERDICT, frozen thresholds]")
        for k_, (ok, v) in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {k_:40s} {v}")
        allok = all(ok for ok, _ in checks.values())
        print("  => " + ("PRELIM PASS: re-check with POOL=HEAD, n >= 30 before any robot run" if allok and len(head_rr) < 30
                          else "PASS (HEAD n >= 30)" if allok else "FAIL -- not a robot candidate"))
