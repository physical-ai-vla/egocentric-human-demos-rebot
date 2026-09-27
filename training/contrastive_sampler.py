"""Same-scene contrastive batch sampler for lerobot training (2026-08-27).
Each batch = (batch_size // GROUP) groups; a group = GROUP frames from GROUP distinct episodes of the SAME layout set
(different prompts, same cube placement). With prob EARLY_FRAC the group uses one shared early frame offset t~U[0,EARLY_N)
(scenes are still identical before the arms move), else independent random frames.
Requires <dataset.root>/set_map.json = {"episode_sets": [set_id per episode_index]} (see make_set_map.py).
env: CONTRASTIVE=1 CONTRASTIVE_GROUP=4 CONTRASTIVE_EARLY_FRAC=0.5 CONTRASTIVE_EARLY_N=60
install() monkeypatches torch.utils.data.DataLoader so lerobot_train's dataloader uses this batch_sampler."""
import os, json, math, random, collections, torch

class ContrastiveBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, group=4, early_frac=0.5, early_n=60, seed=0):
        meta = dataset.meta; ep_from = list(meta.episodes["dataset_from_index"]); ep_to = list(meta.episodes["dataset_to_index"])
        root = str(dataset.root); sm = json.load(open(os.path.join(root, "set_map.json")))
        sets = sm["episode_sets"]; assert len(sets) == len(ep_from), f"set_map has {len(sets)} eps, dataset {len(ep_from)}"
        tasks = None
        try:
            tasks = [str(t) for t in meta.episodes["tasks"]]
        except Exception: pass
        self.ep_from, self.ep_to, self.tasks = ep_from, ep_to, tasks
        self.by_set = collections.defaultdict(list)
        for e, s in enumerate(sets): self.by_set[s].append(e)
        self.multi = [s for s, eps in self.by_set.items() if len(eps) >= 2]
        self.weights = [len(self.by_set[s]) for s in self.multi]
        self.n_frames = ep_to[-1]; self.bs = batch_size; self.G = max(1, min(group, batch_size)); self.early_frac = early_frac; self.early_n = early_n
        self.rng = random.Random(seed); self.n_batches = math.ceil(self.n_frames / batch_size)
        print(f"[contrastive] episodes={len(ep_from)} sets={len(self.by_set)} multi-episode sets={len(self.multi)} group={self.G} early_frac={early_frac} early_n={early_n} batches/epoch={self.n_batches}", flush=True)
    def _pick_group(self):
        s = self.rng.choices(self.multi, weights=self.weights)[0]; eps = self.by_set[s][:]
        self.rng.shuffle(eps)
        if self.tasks:  # prefer distinct prompts within the group
            seen, chosen, rest = set(), [], []
            for e in eps:
                (chosen if self.tasks[e] not in seen else rest).append(e); seen.add(self.tasks[e])
            eps = chosen + rest
        eps = eps[: self.G]
        while len(eps) < self.G: eps.append(self.rng.randrange(len(self.ep_from)))
        if self.rng.random() < self.early_frac:
            t = self.rng.randrange(self.early_n)
            return [self.ep_from[e] + min(t, self.ep_to[e] - self.ep_from[e] - 1) for e in eps]
        return [self.rng.randrange(self.ep_from[e], self.ep_to[e]) for e in eps]
    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            while len(batch) < self.bs: batch.extend(self._pick_group())
            yield batch[: self.bs]
    def __len__(self): return self.n_batches

def install_first_seg():
    """[success-xy] FIRST_SEG=1: sample only frames with frame_index < max_frame[episode] (first-target segment) from <root>/target_labels.json."""
    if os.environ.get("FIRST_SEG", "0") != "1": return
    orig = torch.utils.data.DataLoader
    class FirstSegDataLoader(orig):
        def __init__(self, dataset, *a, **k):
            if hasattr(dataset, "meta") and hasattr(dataset, "root") and os.path.exists(os.path.join(str(dataset.root), "target_labels.json")) and "batch_size" in k:
                import numpy as np
                L = json.load(open(os.path.join(str(dataset.root), "target_labels.json"))); MF = np.array(L["max_frame"])
                ep = np.array(dataset.hf_dataset["episode_index"]); fr = np.array(dataset.hf_dataset["frame_index"])
                allowed = np.where(fr < MF[ep])[0].tolist()
                k.pop("shuffle", None); k["sampler"] = torch.utils.data.SubsetRandomSampler(allowed)
                print(f"[first_seg] sampling {len(allowed)} / {len(fr)} frames (frame_index < max_frame)", flush=True)
            super().__init__(dataset, *a, **k)
    torch.utils.data.DataLoader = FirstSegDataLoader

def install():
    install_first_seg()
    if os.environ.get("CONTRASTIVE", "0") != "1": return
    orig = torch.utils.data.DataLoader
    G = int(os.environ.get("CONTRASTIVE_GROUP", "4")); EF = float(os.environ.get("CONTRASTIVE_EARLY_FRAC", "0.5")); EN = int(os.environ.get("CONTRASTIVE_EARLY_N", "60"))
    class ContrastiveDataLoader(orig):  # subclass (not a function) so isinstance(dl, DataLoader) checks in accelerate keep working
        def __init__(self, dataset, *a, **k):
            if hasattr(dataset, "meta") and hasattr(dataset, "root") and os.path.exists(os.path.join(str(dataset.root), "set_map.json")) and "batch_size" in k:
                bs = k.pop("batch_size"); k.pop("shuffle", None); k.pop("sampler", None); k.pop("drop_last", None)
                k["batch_sampler"] = ContrastiveBatchSampler(dataset, bs, group=G, early_frac=EF, early_n=EN, seed=int(os.environ.get("SEED", "0")))
                print("[contrastive] DataLoader patched with ContrastiveBatchSampler", flush=True)
            super().__init__(dataset, *a, **k)
    torch.utils.data.DataLoader = ContrastiveDataLoader
