"""Right-only task (HRA_red) contract: left = dummy + loss-masked, right = the v2 REL16 contract, gripper masked iff static.
Run: ~/xvla-mac/bin/python tests/test_right_only.py"""
import json, pathlib, sys, tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ego_cart20.config import UMI_DT
from ego_cart20.geometry.rotation6d import matrix_to_quaternion
from ego_cart20.io.raw_episode_loader import save_raw_episode
from ego_cart20.right_only import IDENTITY9, LEFT_DUMMY_G, TASK_ID, convert_right_only, convert_tree, loss_mask20

V = 0.05   # m/s along x


def make_raw(root, eid, grip=lambda t: np.full_like(t, 0.7), t_end=6.0):
    t = np.arange(0, t_end, 1 / 30); n = len(t); T = np.tile(np.eye(4), (n, 1, 1)); T[:, 0, 3] = V * t
    t_ns = (1_000_000_000 + t * 1e9).astype(np.int64)
    arrays = {"right_t_ns": t_ns, "right_position": T[:, :3, 3], "right_quaternion": matrix_to_quaternion(T[:, :3, :3]),
              "right_valid": np.ones(n, bool), "right_gripper": grip(t), "cam_head_t_ns": t_ns, "cam_head_frame": np.arange(n),
              "cam_right_wrist_t_ns": t_ns, "cam_right_wrist_frame": np.arange(n)}
    meta = dict(episode_id=eid, task="approach", task_id=TASK_ID, instruction="Approach to the red cube", stack_order="none",
                videos={"head": "/x/head.mp4", "right_wrist": "/x/rw.mp4"})
    save_raw_episode(pathlib.Path(root) / eid, meta, arrays); return pathlib.Path(root) / eid


def test_left_dummy_right_real_and_mask():
    with tempfile.TemporaryDirectory() as d:
        ep = convert_right_only(make_raw(d, "e1"))
        a, s, lm = ep["action"], ep["state"], ep["loss_mask"]
        assert a.shape[1:] == (16, 32) and len(a) > 50 and np.count_nonzero(a[..., 20:]) == 0
        # right: current-anchor REL16 at UMI_DT, translation along x only
        k = np.arange(1, 17); assert np.allclose(a[0, :, 10], V * k * UMI_DT, atol=1e-9) and np.allclose(a[0, :, 13:19], IDENTITY9[3:], atol=1e-9)
        # left: dummy identity + LEFT_DUMMY_G, and never a target
        assert np.allclose(a[..., 0:9], IDENTITY9) and np.allclose(a[..., 9], LEFT_DUMMY_G)
        assert np.allclose(s[ep["train_rows"]][:, 0:9], IDENTITY9) and np.allclose(s[ep["train_rows"]][:, 18], LEFT_DUMMY_G)
        # state right = task anchor relative: x grows with time
        r = ep["train_rows"]; assert np.allclose(s[r, 9], V * (ep["timestamps"][r] - ep["timestamps"][0]), atol=1e-9)
        # constant gripper -> masked
        assert np.array_equal(lm[0], loss_mask20(False)) and lm[0, :10].sum() == 0 and lm[0, 10:19].sum() == 9 and lm[0, 19] == 0
        assert not ep["metadata"]["loss_mask"]["right_gripper_supervised"]


def test_moving_gripper_is_supervised():
    with tempfile.TemporaryDirectory() as d:
        ep = convert_right_only(make_raw(d, "e2", grip=lambda t: 0.4 + 0.3 * np.sin(t)))
        assert ep["loss_mask"][0, 19] == 1 and ep["metadata"]["loss_mask"]["right_gripper_supervised"]


def test_tree_split_and_files():
    with tempfile.TemporaryDirectory() as d:
        raw = pathlib.Path(d) / "raw"; [make_raw(raw, f"S_{i:03d}") for i in range(12)]
        m = convert_tree(raw, pathlib.Path(d) / "proc", val_frac=0.3)
        assert m["status"] == {"ok": 12} and m["counts"]["train"]["episodes"] + m["counts"]["val"]["episodes"] == 12
        e = next((pathlib.Path(d) / "proc" / "episodes").iterdir())
        assert np.load(e / "loss_mask.npy").shape[1] == 20 and json.load(open(e / "metadata.json"))["task_id"] == TASK_ID


if __name__ == "__main__":
    for f in (test_left_dummy_right_real_and_mask, test_moving_gripper_is_supervised, test_tree_split_and_files): f(); print("PASS", f.__name__)
