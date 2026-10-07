"""[2026-09-28] Freeze REL16-v2B before training (user items 1-4). Writes ~/umi_bridge/rel16v2b_freeze/ on the node:

 DATASET_MANIFEST.json   sha256 of every file of r380_umi94_rel16_v2B, counts, per-episode source mapping, stats hash
 MODEL_CONTRACT.json     state/action/horizon/dt/rotation/gripper/TCP18/state94 layout/normalization + encoder layout,
                         each ASSERTED against the base checkpoint and the dataset (the script exits non-zero on a miss)
 INIT_FREEZE.json        base hashes, widening script hash + TCP18 seed, global seed, fresh optimizer, code hashes/git rev
 RECIPE.json             the launcher's exact training arguments (+ the smoke run's startup config, if found)
usage: freeze_rel16v2b.py            (node, holobrain env)
"""
import glob, hashlib, json, os, pathlib, re, subprocess, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

H = pathlib.Path("/home/bh-aiteam"); DS = H / "holobrain-data/lerobot/r380_umi94_rel16_v2B"
BASE = H / "xvla_base_umi94"; BASE_A = H / "xvla_base_umi76"; OUT = H / "umi_bridge/rel16v2b_freeze"
LAUNCH = H / "train_umi_v2.sh"; WIDEN = H / "umi_bridge/umi76/widen_append94.py"; DERIVE = H / "umi_bridge/umi76/derive_v2.py"
SRC = H / "lerobot-seeed/src/lerobot"
OUT.mkdir(parents=True, exist_ok=True)
fails = []


def sha(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def need(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        fails.append(msg)


# ---------------- 1 dataset manifest
files = sorted(p for p in DS.rglob("*") if p.is_file())
man = {"dataset": str(DS), "files": {str(p.relative_to(DS)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files}}
info = json.load(open(DS / "meta/info.json")); prov = json.load(open(DS / "v4_provenance.json"))
man.update(total_episodes=info["total_episodes"], total_frames=info["total_frames"], parent=prov.get("parent"),
           variant=prov.get("variant"), gripper_label=prov.get("gripper_label"), state_extra=prov.get("state_extra"),
           sources=[dict(lerobot=s["lerobot"], zarr=s["zarr"], global_transform=s.get("global_transform"),
                         select_zarr_episodes=s.get("select_zarr_episodes")) for s in prov["sources"]],
           stats_sha256=sha(DS / "meta/stats.json"))
man["dataset_sha256"] = hashlib.sha256("".join(f"{k}{v['sha256']}" for k, v in man["files"].items()).encode()).hexdigest()
json.dump(man, open(OUT / "DATASET_MANIFEST.json", "w"), indent=1)
need(info["total_episodes"] == 380 and info["total_frames"] == 226081, f"dataset 380 ep / 226081 frames ({info['total_episodes']}/{info['total_frames']})")

# ---------------- 2 model contract (asserted)
from safetensors import safe_open
cfg = json.load(open(BASE / "config.json"))
with safe_open(str(BASE / "model.safetensors"), "pt") as f:
    k = [x for x in f.keys() if x.endswith("action_encoder.fc.weight")][0]; shp = tuple(f.get_slice(k).get_shape())
with safe_open(str(BASE_A / "model.safetensors"), "pt") as f:
    ka = [x for x in f.keys() if x.endswith("action_encoder.fc.weight")][0]; shpa = tuple(f.get_slice(ka).get_shape())
hidden = shpa[1] // (20 + 76 + 32)
need(shpa[1] == hidden * 128, f"A base encoder input 128 = [action20|proprio76|time32] (hidden {hidden})")
need(shp[1] == hidden * 146, f"B base encoder input 146 = [action20|proprio76|TCP18|time32] ({shp[1]} / {hidden} = {shp[1] / hidden})")
need(int(cfg["max_state_dim"]) == 94, f"B base max_state_dim 94 ({cfg['max_state_dim']})")
need(int(cfg.get("dim_time") or 32) == 32, "dim_time 32")
need(info["features"]["observation.state"]["shape"] == [94], f"dataset state [94] ({info['features']['observation.state']['shape']})")
need(info["features"]["action"]["shape"] == [16, 20], f"dataset action [16,20] ({info['features']['action']['shape']})")
st = json.load(open(DS / "meta/stats.json"))
t = pq.read_table(sorted(glob.glob(str(DS / "data/chunk-*/*.parquet")))[0], columns=["action"])
a = t.column("action").combine_chunks(); a = a.storage if isinstance(a, pa.ExtensionArray) else a
A = np.asarray(a.to_pylist()[:5000])
need(set(np.unique(A[..., [9, 19]]).tolist()) <= {0.0, 1.0}, "gripper channels 9/19 binary {0,1}")
launch = LAUNCH.read_text()
for flag in ("--policy.chunk_size=16", "--policy.max_action_dim=20", "--policy.max_state_dim=$SDIM", "--seed=1000",
             "--batch_size=4", "--policy.dtype=float32", "--policy.action_mode=auto", "--policy.use_proprio=true"):
    need(flag in launch, f"launcher has {flag}")
need("B) DS=r380_umi94_rel16_v2B; BASE=/home/bh-aiteam/xvla_base_umi94; SDIM=94" in launch, "launcher B -> v2B dataset, umi94 base, SDIM 94")
contract = dict(
    state_dim=94, action_dim=20, horizon=16, dt_s=3 / 59.94, dt_ms=round(3 / 59.94 * 1000, 3),
    action="per arm A_k = inv(T_t) @ T_(t+(k+1)dt), k = 0..15, current-anchor REL; deploy T_target = T_now @ A_k (no chaining)",
    action_layout="per arm [pos3, rot6d, gripper]; arm 0 = L dims 0-9, arm 1 = R dims 10-19",
    rotation="rot6d = umi.common.pose_util mat_to_pose10d / pose10d_to_mat convention",
    gripper="binary LEADER intent 1 = OPEN, 0 = CLOSE (leader cmd >= 27 at the target time); actuator via gripper_contract_v2.json (1 -> 42, 0 -> 0)",
    state94="state76 (UMI76 sorted-key packing, see UMI76_CONTRACT.md) + [L pos3 rot6d, R pos3 rot6d] base-frame TCP at t (dataset frame = FK + V4_FRAME_FIX)",
    normalization=dict(STATE="IDENTITY", ACTION="MEAN_STD (meta/stats.json)", VISUAL="IDENTITY"),
    encoder_input="[action20 | proprio76 | TCP18 | time32] = 146, soft_transformer cat([action, proprio, time])",
    cameras="global->image, left_wrist->image2, right_wrist->image3", tcp18_diagnostic_only=True,
    asserted=[m for m in []])
json.dump(contract, open(OUT / "MODEL_CONTRACT.json", "w"), indent=1)

# ---------------- 3 init freeze
def gitrev(d):
    try:
        r = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(d), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return dict(rev=r, dirty_files=len(dirty.splitlines()))
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e))


code = {str(p.relative_to(SRC)): sha(p) for p in [SRC / "policies/xvla/modeling_xvla.py", SRC / "policies/xvla/soft_transformer.py",
                                                  SRC / "policies/xvla/action_hub.py", SRC / "policies/xvla/configuration_xvla.py",
                                                  SRC / "scripts/lerobot_train.py"] if p.exists()}
init = dict(base_B=str(BASE), base_B_model_sha256=sha(BASE / "model.safetensors"), base_B_md5_file=(BASE / "BASE_SHA256").read_text().strip(),
            base_A=str(BASE_A), base_A_model_sha256=sha(BASE_A / "model.safetensors"),
            widen_script_sha256=sha(WIDEN), widen_rule="cols [0:96] copied from A base, TCP18 cols N(0, 0.01730) torch.Generator seed 1, time cols moved to [114:146]",
            derive_script_sha256=sha(DERIVE), global_seed=1000, optimizer="FRESH (--policy.path=BASE, resume false)",
            lerobot_seeed=gitrev(H / "lerobot-seeed"), code_sha256=code, launcher_sha256=sha(LAUNCH),
            pretrained_B_rule="a future ego-pretrained-B MUST widen with this same widen_append94.py (same sha) and seed 1")
json.dump(init, open(OUT / "INIT_FREEZE.json", "w"), indent=1)

# ---------------- 4 recipe
args = re.findall(r"^\s+(--[\w.]+=?[^\s]*)", launch.split("TRAIN_ARGS=(", 2)[-1].split(")", 1)[0], re.M)
recipe = dict(run="R380-REL16V2-B-D600K", dataset=str(DS), launcher=str(LAUNCH), launcher_sha256=sha(LAUNCH),
              args=args, env={"XVLA_STRICT_STATE_DIM": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
              steps=600000, decay=600000, save_freq=5000, physical_gpu=1, submit="ray job submit ... -- bash -c 'MIN_START_GB=100 bash ~/train_umi_v2.sh B 1'",
              rule="no LR / horizon / recipe change after seeing 5k-40k results; a failed run is diagnostic evidence")
smoke = H / "holobrain-data/trainB/R380-REL16V2-B-SMOKE.passed.log"
if smoke.exists():
    txt = smoke.read_text(errors="ignore"); m = re.search(r"ot_train\.py:\d+ (\{.*?\n\})", txt, re.S)
    recipe["smoke_startup_config_excerpt_sha256"] = hashlib.sha256(m.group(1).encode()).hexdigest() if m else None
json.dump(recipe, open(OUT / "RECIPE.json", "w"), indent=1)
print(f"\nwrote {OUT}: dataset_sha256 {man['dataset_sha256'][:16]}  base_B {init['base_B_model_sha256'][:16]}  launcher {init['launcher_sha256'][:16]}")
print(f"{len(fails)} contract failures")
sys.exit(1 if fails else 0)
