#!/usr/bin/env python3
"""reBot B601-DM dual-arm forward kinematics (numpy, URDF-parsed, no deps).

Input convention = LeRobot dataset `observation.state` / follower-convention action:
  14-D [L: pan, lift, elbow, wrist_flex, wrist_yaw, wrist_roll, grip | R: same]
  arm joints in RADIANS here (call with deg→rad if needed). Grippers ignored.
Joint i of each arm maps to URDF {side}_joint{i+1} directly (sign/offset assumed
identity; validated empirically by set_cubes.py consistency check).
Returns EEF = {side}_gripper_link origin (+ optional tip offset) in `base` frame.
"""
import math, os
import xml.etree.ElementTree as ET
import numpy as np

URDF = os.environ.get("REBOT_URDF", os.path.expanduser("~/robot-cockpit/urdf/reBot_B601_DM_dualarm.urdf"))
ARM_IDX = {"left": [0, 1, 2, 3, 4, 5], "right": [7, 8, 9, 10, 11, 12]}
GRIP_IDX = {"left": 6, "right": 13}
# gripper_link origin = palm. Empirically (80 teleop grasps) the gripper +x axis points DOWN at grasp and TIP_X=+0.07 puts the
# implied cube centre at z≈+0.04 m (table-top), while -0.07 would float it at +0.17 m → fingers/tip extend along +x. (verified 2026-08-25)
TIP_OFFSET = np.array([float(os.environ.get("REBOT_TIP_X", "0.07")), 0.0, 0.0])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _T(xyz, rpy):
    T = np.eye(4); T[:3, :3] = _rpy(*rpy); T[:3, 3] = xyz; return T


def _rot_axis(axis, th):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    T = np.eye(4); T[:3, :3] = R; return T


class ReBotFK:
    def __init__(self, urdf=URDF):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin"); a = j.find("axis")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            axis = [float(v) for v in a.get("xyz").split()] if a is not None else [0, 0, 1]
            self.joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                             child=j.find("child").get("link"), T=_T(xyz, rpy), axis=axis)
        self.chains = {}
        for side in ("left", "right"):
            self.chains[side] = [f"{side}_mount"] + [f"{side}_joint{i}" for i in range(1, 7)] + [f"{side}_gripper_joint"]
            for n in self.chains[side]:
                assert n in self.joints, n

    def side_pose(self, q6, side):
        """q6: 6 arm joint angles (rad). returns 4x4 pose of {side}_gripper_link in base frame."""
        T = np.eye(4); k = 0
        for n in self.chains[side]:
            j = self.joints[n]
            T = T @ j["T"]
            if j["type"] == "revolute":
                T = T @ _rot_axis(j["axis"], q6[k]); k += 1
        return T

    def eef(self, q14_rad, tip=True):
        """q14 in dataset order (arms rad). returns {'left': xyz, 'right': xyz} (m, base frame)."""
        out = {}
        for side, idx in ARM_IDX.items():
            T = self.side_pose(np.asarray(q14_rad, float)[idx], side)
            p = T @ np.append(TIP_OFFSET if tip else np.zeros(3), 1.0)
            out[side] = p[:3]
        return out


if __name__ == "__main__":
    fk = ReBotFK()
    q = np.zeros(14)
    print("zero pose:", {k: np.round(v, 3) for k, v in fk.eef(q).items()})
    q[[0, 1, 2, 3, 4, 5]] = [-0.398, -0.010, -0.260, 0.583, -0.243, -0.601]
    q[[7, 8, 9, 10, 11, 12]] = [-0.265, -0.103, -0.149, 0.534, -0.007, -0.220]
    print("ep028 frame0:", {k: np.round(v, 3) for k, v in fk.eef(q).items()})
