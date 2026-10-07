#!/usr/bin/env python3
"""[2026-09-25] C-old checkpoint evaluator on the Mac (MPS, groot-infer-env, pyav) for the overnight chain.
Same inference path as the B1 serving / probe_eval: state = arms rad + grips 1[raw <= -135], pre -> predict_action_chunk -> post;
pred[:, arm] = dq (rad) for t + 5 + i (i = 0..29); torch.manual_seed(n) before every call.
  ego   : c8old_val (57 segments, 26 source episodes), starts every 5 frames in 0..30; own target dq_pseudo = action q (deg) -
          state q; geometry target = measured T_store (observation.ee.tcp_tgt) positions at t+5+i (the D target)
  r150  : R150 probe val-10 episodes (NOTE: included in the R150 fine-tuning data), stride 20; own target dq_cmd;
          geometry target = FK(q_cmd)
Per horizon k in (1, 4, 8, 16, 30): joint MAE (deg) and its zero-action baseline, FK position error p50 / p90 (mm) vs the
geometry target, direction cosine (|disp| > 5 mm), pred norm / own-target norm; collapse = std(pred) / std(target) at k30.
Selection metric (fixed before any result): geo_score = mean over k of FK position error p50 (lower is better).
[2026-09-27] r150 mode: + r120_groups (seen8 / heldout2 of val-10 w.r.t. the frozen R120). [2026-09-26] + joint-space action cosine (|target dq| > 0.5 deg) and motion-gated metrics (added after the 60k result, reporting only; geo_score unchanged): a (sample, arm) is
MOVING when its OWN target at k30 moves >= 3 deg (L2 over the 6 arm joints, "motion_joint") or its geometry target TCP moves
>= 20 mm ("motion_tcp"); per k the same metrics on the moving subset + n. Raw per-(sample, arm, k) arrays -> <out>.npz.
Usage: c8old_mac_eval.py <ego|r150> <pretrained_model dir> <out json>
"""
import json, os, pathlib, sys, time
import numpy as np, torch
H = pathlib.Path.home(); C8 = H / "c8"
os.environ.setdefault("REBOT_URDF", str(H / "umi_bridge/track_c/v3/recipe_full/rebot_ee/reBot_B601_DM_dualarm.urdf")); sys.path.insert(0, str(C8 / "c8old")); sys.path.insert(0, str(H / "umi_bridge/track_c/v3/recipe_full/rebot_ee"))
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import rebot_fk_torch
MODE, CK, OUTJ = sys.argv[1], sys.argv[2], sys.argv[3]
LEAD, K = 5, 30; KS = (1, 4, 8, 16, 30); ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]; GRIP = [6, 13]; DEV = "mps"
RENAME = {"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2", "observation.images.right_wrist": "observation.images.image3"}
FK = rebot_fk_torch.ReBotFKTorch(device="cpu", dtype=torch.float32)
if MODE == "ego":
    ds = LeRobotDataset("local/c8old_val", root=str(C8 / "c8old_data/c8old_val"), video_backend="pyav")
else:
    val = json.load(open(C8 / "probe_r150_split.json"))["val"]
    ds = LeRobotDataset("rebot/rebot_3stack_R150_headview", root=str(C8 / "r150_ds"), episodes=val, video_backend="pyav")
hf = ds.hf_dataset; ST = np.stack(hf["observation.state"]); AC = np.stack(hf["action"]); EP = np.asarray(hf["episode_index"])
TT = np.stack(hf["observation.ee.tcp_tgt"]).reshape(-1, 2, 4, 4) if MODE == "ego" else None
starts = []
for e in np.unique(EP):
    ix = np.flatnonzero(EP == e)
    starts += list(ix[0] + np.arange(0, 31, 5)) if MODE == "ego" else list(range(ix[0], ix[-1] + 1 - (LEAD + K - 1), 20))
CACHE = C8 / f"c8old_runs/eval_cache_{MODE}.npz"                               # verified frame cache (c8old_eval_cache.py)
CI = None
if CACHE.exists():
    _c = np.load(CACHE); assert np.array_equal(_c["starts"], np.array(starts)), "eval cache samples differ"; CI = _c["images"]
IMK = ("observation.images.global", "observation.images.left_wrist", "observation.images.right_wrist")
TASKS = {int(i): t for i, t in zip(ds.meta.tasks["task_index"], ds.meta.tasks.index)} if hasattr(ds.meta.tasks, "index") else None
t0 = time.time(); p = XVLAPolicy.from_pretrained(CK).float().to(DEV).eval()
pre, post = make_pre_post_processors(p.config, pretrained_path=CK, preprocessor_overrides={"device_processor": {"device": DEV}, "rename_observations_processor": {"rename_map": RENAME}},
                                     postprocessor_overrides={"device_processor": {"device": DEV}})
rec = {k: dict(mae=[], mae0=[], fk=[], cos=[], rn=[]) for k in KS}; P30, G30 = [], []
RAW = {k: dict(mae=[], mae0=[], fk=[], cos=[], jcos=[], mj=[], mt=[], ep=[]) for k in KS}; GJ, GT = np.deg2rad(3.0), 0.020
for n, t in enumerate(starts):
    q_now = np.deg2rad(ST[t][ARM]); s14 = np.zeros(14, np.float32); s14[ARM] = q_now; s14[GRIP] = (ST[t][GRIP] <= -135)
    if CI is not None:
        obs = {k: torch.from_numpy(CI[n, j]).permute(2, 0, 1).float() / 255 for j, k in enumerate(IMK)}; obs["task"] = TASKS[int(hf[int(t)]["task_index"])]
    else:
        it = ds[int(t)]; obs = {k: it[k] for k in IMK}; obs["task"] = it["task"]
    obs["observation.state"] = torch.tensor(s14); torch.manual_seed(n)
    with torch.no_grad(): ch = p.predict_action_chunk(pre(obs))
    pred = torch.stack([post(ch[:, i, :]).squeeze(0) for i in range(ch.shape[1])]).float().cpu().numpy()[:, :14]
    fut = np.arange(t + LEAD, t + LEAD + K); dq_p = pred[:, ARM]; dq_g = np.deg2rad(AC[fut][:, ARM]) - q_now[None]
    Pp = FK.tcp(torch.tensor(q_now[None] + dq_p, dtype=torch.float32))[..., :3, 3].numpy()
    Pg = TT[fut][..., :3, 3] if MODE == "ego" else FK.tcp(torch.tensor(q_now[None] + dq_g, dtype=torch.float32))[..., :3, 3].numpy()
    P0 = FK.tcp(torch.tensor(q_now[None], dtype=torch.float32))[..., :3, 3].numpy()[0]
    for k in KS:
        i = k - 1
        for a in (0, 1):
            sl = slice(6 * a, 6 * a + 6); r = rec[k]
            r["mae"].append(np.degrees(np.abs(dq_p[i, sl] - dq_g[i, sl]).mean())); r["mae0"].append(np.degrees(np.abs(dq_g[i, sl]).mean()))
            r["fk"].append(np.linalg.norm(Pp[i, a] - Pg[i, a]) * 1e3); ng = np.linalg.norm(dq_g[i, sl])
            if ng > 1e-6: r["rn"].append(np.linalg.norm(dq_p[i, sl]) / ng)
            dg, dp = Pg[i, a] - P0[a], Pp[i, a] - P0[a]
            if np.linalg.norm(dg) > 0.005: r["cos"].append(float(dg @ dp / (np.linalg.norm(dg) * np.linalg.norm(dp) + 1e-12)))
            w = RAW[k]; w["mae"].append(r["mae"][-1]); w["mae0"].append(r["mae0"][-1]); w["fk"].append(r["fk"][-1])
            w["cos"].append(float(dg @ dp / (np.linalg.norm(dg) * np.linalg.norm(dp) + 1e-12)) if np.linalg.norm(dg) > 0.005 else np.nan)
            w["jcos"].append(float(dq_p[i, sl] @ dq_g[i, sl] / (np.linalg.norm(dq_p[i, sl]) * ng + 1e-12)) if ng > np.deg2rad(0.5) else np.nan)   # joint-space action cosine
            w["ep"].append(int(EP[t]))
            w["mj"].append(bool(np.linalg.norm(dq_g[-1, sl]) >= GJ)); w["mt"].append(bool(np.linalg.norm(Pg[-1, a] - P0[a]) >= GT))
    P30.append(dq_p[-1]); G30.append(dq_g[-1])
res = dict(mode=MODE, checkpoint=CK, samples=len(starts), frame_cache=CI is not None, seconds=round(time.time() - t0, 1),
           collapse_std_ratio_k30=float(np.median(np.array(P30).std(0) / np.maximum(np.array(G30).std(0), 1e-9))))
for k in KS:
    r = rec[k]; res[f"k{k}"] = dict(joint_mae_deg=float(np.median(r["mae"])), zero_action_mae_deg=float(np.median(r["mae0"])), fk_mm_p50=float(np.median(r["fk"])),
                                    fk_mm_p90=float(np.percentile(r["fk"], 90)), dir_cos_p50=float(np.median(r["cos"])) if r["cos"] else float("nan"),
                                    norm_ratio_p50=float(np.median(r["rn"])) if r["rn"] else float("nan"))
res["geo_score"] = float(np.mean([res[f"k{k}"]["fk_mm_p50"] for k in KS]))
def _gated(mask_key):
    out = {}
    for k in KS:
        w = {a: np.array(b) for a, b in RAW[k].items()}; m = w[mask_key]; c = w["cos"][m]; c = c[np.isfinite(c)]
        out[f"k{k}"] = dict(n=int(m.sum()), frac=float(m.mean()), joint_mae_deg=float(np.median(w["mae"][m])) if m.any() else float("nan"),
                            zero_action_mae_deg=float(np.median(w["mae0"][m])) if m.any() else float("nan"),
                            fk_mm_p50=float(np.median(w["fk"][m])) if m.any() else float("nan"), fk_mm_p90=float(np.percentile(w["fk"][m], 90)) if m.any() else float("nan"),
                            dir_cos_p50=float(np.median(c)) if len(c) else float("nan"),
                            action_cos_p50=float(np.nanmedian(w["jcos"][m])) if np.isfinite(w["jcos"][m]).any() else float("nan"))
    out["geo_score"] = float(np.mean([out[f"k{k}"]["fk_mm_p50"] for k in KS])); return out
for k in KS: RAW[k]["all"] = [True] * len(RAW[k]["mae"])
if MODE == "r150":   # [2026-09-27] R120 split of probe val-10: seen8 (in R120 train) / heldout2 (in R150, not in R120); all10 = the aggregate
    _r120 = set(json.load(open(C8 / "r150_nested_subset_v1.json"))["ladder"]["R120"]["episodes"])
    for k in KS:
        RAW[k]["seen8"] = [e in _r120 for e in RAW[k]["ep"]]; RAW[k]["held2"] = [e not in _r120 for e in RAW[k]["ep"]]
        RAW[k]["seen8_mt"] = [a and b for a, b in zip(RAW[k]["seen8"], RAW[k]["mt"])]; RAW[k]["held2_mt"] = [a and b for a, b in zip(RAW[k]["held2"], RAW[k]["mt"])]
    res["r120_groups"] = dict(seen8_episodes=sorted({e for e in RAW[1]["ep"] if e in _r120}), heldout2_episodes=sorted({e for e in RAW[1]["ep"] if e not in _r120}),
                              seen8=_gated("seen8"), heldout2=_gated("held2"), seen8_motion_tcp=_gated("seen8_mt"), heldout2_motion_tcp=_gated("held2_mt"))
if MODE == "r150":   # [2026-09-28] R60 split of probe val-10: seen7 (in R60 train) / heldout3 (eps 10, 66, 77)
    _r60 = set(json.load(open(C8 / "r150_nested_subset_v1.json"))["ladder"]["R60"]["episodes"])
    for k in KS:
        RAW[k]["s60"] = [e in _r60 for e in RAW[k]["ep"]]; RAW[k]["h60"] = [e not in _r60 for e in RAW[k]["ep"]]
        RAW[k]["s60_mt"] = [a and b for a, b in zip(RAW[k]["s60"], RAW[k]["mt"])]; RAW[k]["h60_mt"] = [a and b for a, b in zip(RAW[k]["h60"], RAW[k]["mt"])]
    res["r60_groups"] = dict(seen7_episodes=sorted({e for e in RAW[1]["ep"] if e in _r60}), heldout3_episodes=sorted({e for e in RAW[1]["ep"] if e not in _r60}),
                             seen7=_gated("s60"), heldout3=_gated("h60"), seen7_motion_tcp=_gated("s60_mt"), heldout3_motion_tcp=_gated("h60_mt"))
if MODE == "r150":   # [2026-09-29] R30 split of probe val-10: seen4 (91, 98, 103, 107) / heldout6
    _r30 = set(json.load(open(H / "umi_bridge/track_c/v2k/r30_episodes.json"))["episodes"])
    for k in KS:
        RAW[k]["s30"] = [e in _r30 for e in RAW[k]["ep"]]; RAW[k]["h30"] = [e not in _r30 for e in RAW[k]["ep"]]
        RAW[k]["s30_mt"] = [a and b for a, b in zip(RAW[k]["s30"], RAW[k]["mt"])]; RAW[k]["h30_mt"] = [a and b for a, b in zip(RAW[k]["h30"], RAW[k]["mt"])]
    res["r30_groups"] = dict(seen4_episodes=sorted({e for e in RAW[1]["ep"] if e in _r30}), heldout6_episodes=sorted({e for e in RAW[1]["ep"] if e not in _r30}),
                             seen4=_gated("s30"), heldout6=_gated("h30"), seen4_motion_tcp=_gated("s30_mt"), heldout6_motion_tcp=_gated("h30_mt"))
res["all_samples"] = _gated("all"); res["motion_joint"] = _gated("mj"); res["motion_tcp"] = _gated("mt")
np.savez(str(OUTJ).rsplit(".", 1)[0] + ".npz", **{f"k{k}_{a}": np.array(b) for k in KS for a, b in RAW[k].items()})
json.dump(res, open(OUTJ, "w"), indent=1); print(json.dumps({k: res[k] for k in ("mode", "samples", "geo_score", "collapse_std_ratio_k30", "seconds")}))
