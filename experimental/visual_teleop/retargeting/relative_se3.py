"""Relative bimanual SE(3) retargeting (2026-10-14, spec 5-6, 25).

    anchor:  T_W_H0 <- human virtual TCP now,  T_RB_R0 <- robot TCP of the LAST EXECUTED command (not the observation)
    target:  T_RB_R(t) = T_RB_R0 . Scale(Adapter(inv(T_W_H0) . T_W_H(t)))

Absolute human position never reaches the robot. Clutch (spec 25): while engaged, the human anchor follows the human every
tick and the robot anchor follows the executed command, so on release the target equals the executed pose (zero jump by
construction). The same re-anchor is used after deadman re-press and HOLD recovery."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ego_teleop.transforms.se3 import inv_T
from .frame_adapter import FrameAdapter
from .motion_scaling import MotionScaling


@dataclass
class SideAnchor:
    T_W_H0: np.ndarray | None = None
    T_RB_R0: np.ndarray | None = None

    @property
    def engaged(self) -> bool: return self.T_W_H0 is not None and self.T_RB_R0 is not None


@dataclass
class RelativeRetargeter:
    adapters: dict
    scaling: MotionScaling
    anchors: dict = field(default_factory=lambda: {"left": SideAnchor(), "right": SideAnchor()})

    def anchor(self, side: str, T_W_H_now: np.ndarray, T_RB_R_exec: np.ndarray) -> None:
        a = self.anchors[side]; a.T_W_H0 = np.asarray(T_W_H_now, np.float64).copy(); a.T_RB_R0 = np.asarray(T_RB_R_exec, np.float64).copy()

    def disengage(self, side: str | None = None) -> None:
        for s in ([side] if side else list(self.anchors)): self.anchors[s] = SideAnchor()

    def delta_H(self, side: str, T_W_H: np.ndarray) -> np.ndarray:
        a = self.anchors[side]
        if not a.engaged: raise RuntimeError(f"{side}: not anchored")
        return inv_T(a.T_W_H0) @ np.asarray(T_W_H, np.float64)

    orientation_mode: str = "full"     # lock (position only) | yaw (rotation about W vertical only) | full  (user 10-15 staging)
    translation_frame: str = "body"    # body (UMI-relative, robot TCP axes) | heading (world-aligned: INIT-push forward -> robot +x)
    # 10-15 (user: "손목을 꺾으면 그리퍼 끝이 아니라 wrist 전부분이 비틀고 움직여"): rotation pivot. Translation is matched between
    # two points that sit BEHIND the TCPs along their tool axis (z): the human wrist (pivot_human_m behind the UMI jaw tip) and the
    # robot wrist pitch axis (pivot_robot_m behind the robot TCP). A wrist bend then rotates the robot about ITS wrist and the tip
    # draws the arc, even while the tip position is held. 0 / 0 = rotate about the TCP (previous behaviour).
    pivot_human_m: float = 0.0
    pivot_robot_m: float = 0.0
    # 10-15 (live: 23 HOLDs "anchor UMI points too vertically"): the heading is fixed at INIT and KEPT through clutch / deadman /
    # loss re-anchors (re-deriving it from the UMI at every re-anchor refused anchors taken with the jaws pointing down).
    heading: dict = field(default_factory=dict)
    # 10-15 (live: "end effector를 따라간다는 느낌이 없어"): world = the hand's rotation since the anchor is taken in W, re-expressed in
    # the robot base through the INIT heading frame and LEFT-multiplied onto the robot's anchor orientation (a yaw of the hand is a
    # yaw of the gripper about RB z, a forward tilt is a forward tilt) | body = the old TCP-frame delta (hand-own axes -> gripper-own axes).
    rotation_frame: str = "body"
    # 10-16: per-axis gain on the hand rotation in the robot base (rotation_frame world): (roll about RB x, pitch about RB y, yaw about RB z).
    # (1, 1, 1) = unchanged; laundry suggestion (1.0 yaw, 0.7 pitch, 0.5 roll) -> (0.5, 0.7, 1.0)
    rotation_axis_scale: tuple = (1.0, 1.0, 1.0)
    # 10-16: base = hand heading -> RB x (old) | robot = hand heading -> the robot anchor gripper's horizontal pointing (falls back to base when it points vertically)
    rotation_heading: str = "base"

    def reset_heading(self, side: str | None = None) -> None:
        for s in ([side] if side else list(self.anchors)): self.heading.pop(s, None)

    def target(self, side: str, T_W_H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (T_RB_R_target, Delta_H). Raises on non-finite input (caller => HOLD)."""
        T = np.asarray(T_W_H, np.float64)
        if not np.all(np.isfinite(T)): raise ValueError(f"{side}: non-finite human pose")
        D_H = self.delta_H(side, T)
        if self.orientation_mode != "full":
            R0 = self.anchors[side].T_W_H0[:3, :3]
            if self.orientation_mode == "lock": D_H[:3, :3] = np.eye(3)
            elif self.orientation_mode == "yaw":
                from scipy.spatial.transform import Rotation
                R_w = T[:3, :3] @ R0.T                                      # world-frame rotation since the anchor
                x = R_w[:, 0] if abs(R_w[2, 0]) < 0.9 else R_w[:, 1]
                yaw = float(np.arctan2(x[1], x[0]) - (0.0 if abs(R_w[2, 0]) < 0.9 else np.pi / 2))
                D_H[:3, :3] = R0.T @ Rotation.from_rotvec([0, 0, yaw]).as_matrix() @ R0      # yaw about W z, in the anchor body frame
            else: raise ValueError(f"orientation_mode {self.orientation_mode!r}")
        if self.translation_frame == "heading":
            # 10-15: world-aligned translation. Teleop frame T: x = horizontal heading of the anchor UMI z (the INIT push direction),
            # z = up, y = z x x  ->  robot base RB (x forward, y left, z up). Intuitive whatever the robot's starting gripper
            # orientation is (the body-frame mapping sent "human up" to "robot down/sideways" from the folded rest pose).
            a = self.anchors[side]
            if side not in self.heading:
                zH = a.T_W_H0[:3, 2]; xT = np.array([zH[0], zH[1], 0.0])
                if np.linalg.norm(xT) < 0.3: raise ValueError(f"{side}: anchor UMI points too vertically to define a heading")
                self.heading[side] = xT / np.linalg.norm(xT)
            xT = self.heading[side]; zT = np.array([0, 0, 1.0]); R_W_T = np.column_stack([xT, np.cross(zT, xT), zT])
            ph = lambda X: X[:3, 3] - self.pivot_human_m * X[:3, 2]                # human pivot (behind the jaw tip)
            dp_T = R_W_T.T @ (ph(T) - ph(a.T_W_H0))
            D_R = self.scaling.apply(self.adapters[side].map_delta(D_H))           # rotation part (lock/yaw/full) in the TCP frame
            if self.rotation_frame == "world":
                R0h = a.T_W_H0[:3, :3]; dR_W = R0h @ D_R[:3, :3] @ R0h.T                # same rotation, expressed in W
                dR_T = R_W_T.T @ dR_W @ R_W_T                                           # in the human heading frame (x = INIT push, z = up)
                if self.rotation_axis_scale != (1.0, 1.0, 1.0):                        # 10-16 VR/mocap-style attenuation: [roll x, pitch y, yaw z] about the pointing heading
                    from scipy.spatial.transform import Rotation as _R
                    dR_T = _R.from_rotvec(_R.from_matrix(dR_T).as_rotvec() * np.asarray(self.rotation_axis_scale)).as_matrix()
                # 10-16 (user: "wrist 위를 향하거나 아래로 코박는 동작은 못해"; ep 143849: robot anchor TCP z pointed at azimuth ~76 deg = RB +y,
                # so a hand nose-dive about T y became a ROLL about the gripper's own axis, tilt gain 0.19). rotation_heading robot: the
                # human heading frame is mapped onto the ROBOT ANCHOR gripper's horizontal pointing direction, so pitch stays pitch.
                R_RB_Tr = np.eye(3)
                if self.rotation_heading == "robot":
                    zr = a.T_RB_R0[:3, 2]; xr = np.array([zr[0], zr[1], 0.0])
                    if np.linalg.norm(xr) > 0.3: xr /= np.linalg.norm(xr); R_RB_Tr = np.column_stack([xr, np.cross([0, 0, 1.0], xr), [0, 0, 1.0]])
                dR_RB = R_RB_Tr @ dR_T @ R_RB_Tr.T
                out = np.eye(4); out[:3, :3] = dR_RB @ a.T_RB_R0[:3, :3]
            else:
                out = a.T_RB_R0 @ D_R
            pivot_R0 = a.T_RB_R0[:3, 3] - self.pivot_robot_m * a.T_RB_R0[:3, 2]
            out[:3, 3] = pivot_R0 + np.asarray(self.scaling.translation) * dp_T + self.pivot_robot_m * out[:3, 2]
            return out, D_H
        return self.anchors[side].T_RB_R0 @ self.scaling.apply(self.adapters[side].map_delta(D_H)), D_H
