"""[2026-10-03 user] ego + robot CO-TRAINING on top of the B8 REL-only recipe (code_8bit wrapper -> train_rel16_relonly.py).
--dataset.repo_id/--dataset.root = the ROBOT dataset (R312c): its meta/stats/features drive the policy + normalizer (deploy target).
COTRAIN_EGO_ROOT = the ego dataset (ego_cart20_v2b_robot100_train), same CART20 contract, same domain 20, same loss (REL 0:20).
Sampling: every batch is exactly COTRAIN_EGO_PER_BATCH ego + rest robot rows (default 4 + 4 at batch 8), each source drawn
uniformly without replacement (reshuffled when exhausted) -> dataset-level 50/50, NOT frame-count proportional.
Items are cut to the keys both datasets share (+ task); ego aux.q_t is NaN but unused by the REL-only loss."""
import copy, os, random, runpy, sys
import torch
import lerobot.scripts.lerobot_train as LT

EGO_ROOT = os.environ["COTRAIN_EGO_ROOT"]; EGO_N = int(os.environ.get("COTRAIN_EGO_PER_BATCH", "4"))
_orig_make = LT.make_dataset


class CoTrainDataset(torch.utils.data.Dataset):
    def __init__(self, robot, ego):
        self.robot, self.ego, self.meta = robot, ego, robot.meta
        self.episodes = None; self.num_frames = robot.num_frames + ego.num_frames
        self.num_episodes = robot.num_episodes + ego.num_episodes
        self.keys = (set(robot[0].keys()) & set(ego[0].keys())) | {"task"}
        print(f"[cotrain] robot {robot.num_frames} fr / {robot.num_episodes} ep + ego {ego.num_frames} fr / {ego.num_episodes} ep; "
              f"shared keys {sorted(self.keys)}", flush=True)

    def __len__(self): return self.num_frames

    def __getitem__(self, i):  # i >= 0 robot row, i < 0 ego row ~i
        x = self.robot[i] if i >= 0 else self.ego[~i]
        return {k: v for k, v in x.items() if k in self.keys}


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, ds, bs, seed=1000):
        self.nr, self.ne, self.bs, self.rng = len(ds.robot), len(ds.ego), bs, random.Random(seed)
        assert 0 < EGO_N < bs; self.pr, self.pe = [], []

    def _take(self, pool, n, k):
        out = []
        while len(out) < k:
            if not pool: pool.extend(self.rng.sample(range(n), n))
            out.append(pool.pop())
        return out

    def __len__(self): return (self.nr + self.ne) // self.bs

    def __iter__(self):
        for _ in range(len(self)):
            b = self._take(self.pr, self.nr, self.bs - EGO_N) + [~j for j in self._take(self.pe, self.ne, EGO_N)]
            self.rng.shuffle(b); yield b


def make_dataset(cfg):
    robot = _orig_make(cfg)
    c2 = copy.deepcopy(cfg); c2.dataset.repo_id = "rebot/" + os.path.basename(EGO_ROOT.rstrip("/")); c2.dataset.root = EGO_ROOT
    return CoTrainDataset(robot, _orig_make(c2))


_DL = torch.utils.data.DataLoader


class DataLoader(_DL):
    def __init__(self, dataset, *a, **kw):
        if isinstance(dataset, CoTrainDataset):
            bs = kw.pop("batch_size"); kw.pop("shuffle", None); kw.pop("sampler", None); kw.pop("drop_last", None)
            kw["batch_sampler"] = BalancedBatchSampler(dataset, bs)
            print(f"[cotrain] batch {bs} = {EGO_N} ego + {bs - EGO_N} robot (exact, every batch)", flush=True)
        super().__init__(dataset, *a, **kw)


LT.make_dataset = make_dataset
LT.torch.utils.data.DataLoader = DataLoader
sys.argv[0] = "/srv/data/johann/relonly/code_8bit/train_rel16_relonly.py"
runpy.run_path(sys.argv[0], run_name="__main__")
