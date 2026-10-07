"""Typed configuration for ego_teleop, loaded from configs/ego_teleop/*.yaml (nothing tunable lives in code)."""
from __future__ import annotations
import dataclasses as dc
import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints
import yaml
from handumi_collector.config import REPO_ROOT
from .transforms.frames import FrameMapperConfig
from .tracking.wrist_pose_provider import TrackingSupervisorConfig
from .tracking.vio_stable import VioStableConfig
from .hand3d.palm_pose import PalmPoseConfig
from .retarget.hand_features import HandSupervisorConfig
from .retarget.aero_retarget import ChannelMap
from .robot.safety import SafetyConfig, WorkspaceBox
from .robot.coordinator import CoordinatorConfig
from .hand3d.depth import DepthSamplerConfig
from .hand3d.identity import IdentityConfig
from .hand3d.supervisor import HandPoseSupervisorConfig
from .retarget.aero_backends import DexPilotConfig, AeroLimiterConfig
from .tracking.fused_wrist import FusedWristConfig

CONFIG_DIR = REPO_ROOT / "configs" / "ego_teleop"


@dataclass
class RebotCfg:
    base_url: str = "http://127.0.0.1:8020"   # robot-cockpit robot_service
    side: str = "right"                     # arm carrying the Aero Hand
    ik_backend: str = "pyroki"              # pyroki (handumi-sw venv) | mock
    max_ik_pos_err_m: float = 0.01
    max_ik_rot_err_deg: float = 5.0
    joint_max_step_rad: float = 0.05
    joint_limits_rad: dict | None = None    # {lower_rad: [6], upper_rad: [6]} follower frame; None => rebot_client C30 defaults


@dataclass
class AeroCfg:
    port: str | None = None                 # /dev/cu.usbmodem* on macOS (SDK autodetect is Linux-only)
    baudrate: int = 921600
    side: str = "right"
    homing_timeout_s: float = 175.0
    telemetry_every_n: int = 10
    channels: list[dict] = field(default_factory=list)   # ChannelMap overrides per canonical channel
    relaxed_normalized: list[float] | None = None


@dataclass
class HeadRgbdGates:
    """A1 acceptance gate (spec section 28). Thresholds are starting points: re-fit them on real depth measurements."""
    min_rate_hz: float = 30.0
    min_metric_landmark_fraction: float = 0.9
    max_identity_swaps: int = 0
    max_stationary_jitter_mm_p95: float = 10.0
    max_palm_local_drift_mm_p95: float = 10.0


@dataclass
class HeadRgbdCfg:
    """The RGB-D hand branch. The head camera stays observation-only for the ARM; this is the hand branch alone."""
    enabled: bool = True
    provider: str = "head_rgbd"              # head_rgbd (B1) | mediapipe_mono (B0 official baseline)
    side: str = "right"
    selfie_mirrored: bool = False
    num_hands: int = 2
    fill_missing: bool = True
    max_fill: int = 6
    depth: DepthSamplerConfig = field(default_factory=DepthSamplerConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    hand_pose: HandPoseSupervisorConfig = field(default_factory=HandPoseSupervisorConfig)
    backend: str = "dexpilot"                # dexpilot | semantic7d
    dexpilot: DexPilotConfig = field(default_factory=DexPilotConfig)
    aero_limits: AeroLimiterConfig = field(default_factory=AeroLimiterConfig)
    gates: HeadRgbdGates = field(default_factory=HeadRgbdGates)

    def provider_config(self):
        from .hand3d.providers import HandPoseProviderConfig
        return HandPoseProviderConfig(side=self.side, selfie_mirrored=self.selfie_mirrored, num_hands=self.num_hands,
                                      fill_missing=self.fill_missing, max_fill=self.max_fill,
                                      depth=self.depth, identity=self.identity)

    def build_provider(self, **kw):
        from .hand3d.providers import MediaPipeHandPoseProvider, RgbdHandPoseProvider
        cls = {"head_rgbd": RgbdHandPoseProvider, "mediapipe_mono": MediaPipeHandPoseProvider}[self.provider]
        return cls(self.provider_config(), **kw)

    def build_backend(self):
        from .retarget.aero_backends import DexPilotAeroRetargeter, Semantic7DAeroRetargeter
        if self.backend == "dexpilot":
            return DexPilotAeroRetargeter(dc.replace(self.dexpilot, side=self.side))
        if self.backend == "semantic7d":
            return Semantic7DAeroRetargeter(side=self.side)
        raise ValueError(f"unknown Aero retargeting backend {self.backend!r}")


@dataclass
class RgbdArmPocGates:
    """Gate A / Gate B thresholds for the RGB-D arm POC. Pre-hardware starting points: re-fit them on the first
    measured run before a PASS is treated as evidence (same rule as the A1 gate)."""
    min_rate_hz: float = 25.0
    min_arm_pose_ok_fraction: float = 0.90         # frames with a usable palm origin, over frames with a hand
    max_stationary_jitter_mm_p95: float = 8.0      # palm origin scatter, hand and camera still
    max_stationary_orientation_jitter_deg_p95: float = 5.0   # measured even in 3-DoF runs (POC section 7)
    max_return_to_origin_mm: float = 25.0          # hand leaves and comes back: residual position error
    max_ik_failure_fraction: float = 0.02          # Gate B, only when an IK solver is available
    max_workspace_clamp_fraction: float = 0.05     # Gate B: a target clamped this often means the box or scale is wrong


@dataclass
class RgbdArmPocCfg:
    """TEMPORARY RGB-D-only arm teleoperation POC (docs/ego_teleop/RGBD_ARM_POC.md).

    A second *pose source* for the existing relative-SE(3) arm stack while the wrist IMU/VIO hardware is not built.
    It does not replace the production architecture (wrist fisheye + IMU -> OpenVINS -> WristPoseProvider) and has its
    own frame mapping, supervisor and safety numbers — the production ones stay untouched in teleop.yaml/frames.yaml."""
    enabled: bool = False
    side: str = "right"                    # the human hand driving the arm (and the arm side on the robot)
    camera_mode: str = "fixed"             # INVARIANT for this POC: a moving (head-mounted) camera turns head motion
                                           # into robot motion. Only "fixed" is accepted by the tooling.
    rate_hz: float = 30.0
    palm: PalmPoseConfig = field(default_factory=PalmPoseConfig)
    tracking: TrackingSupervisorConfig = field(default_factory=TrackingSupervisorConfig)   # POC-specific, not teleop.tracking
    stable: VioStableConfig = field(default_factory=VioStableConfig)
    frames: FrameMapperConfig = field(default_factory=FrameMapperConfig)                   # POC-specific mapping + scale
    safety: SafetyConfig = field(default_factory=SafetyConfig)                             # POC-specific low speed / clamp mode
    workspace_from_anchor: bool = True     # POC section 8: the first real runs move inside a small box around the
                                           # ENGAGE TCP. The box is built from the anchor and handed to the EXISTING
                                           # workspace clamp — no new safety path, just a tighter box.
    workspace_half_extent_m: tuple = (0.10, 0.10, 0.10)
    gates: RgbdArmPocGates = field(default_factory=RgbdArmPocGates)

    def __post_init__(self) -> None:
        if self.camera_mode != "fixed":
            raise ValueError("rgbd_arm_poc.camera_mode must be 'fixed': a head-mounted RGB-D camera makes head "
                             "motion look like hand motion and the robot would follow the operator's head")

    def anchor_workspace(self, T_anchor) -> "WorkspaceBox":
        """The small Cartesian box around the ENGAGE TCP (robot base frame), intersected with the configured box so
        the POC can only ever be MORE restrictive than the standing workspace limits."""
        import numpy as np
        p = np.asarray(T_anchor, np.float64)[:3, 3]
        h = np.asarray(self.workspace_half_extent_m, np.float64)
        lo = np.maximum(p - h, np.asarray(self.safety.workspace.min_xyz, np.float64))
        hi = np.minimum(p + h, np.asarray(self.safety.workspace.max_xyz, np.float64))
        if np.any(lo > hi):
            raise RuntimeError(f"ENGAGE TCP {p.round(3).tolist()} is outside rgbd_arm_poc.safety.workspace "
                               f"{self.safety.workspace.min_xyz}..{self.safety.workspace.max_xyz}")
        return WorkspaceBox(tuple(lo), tuple(hi))

    def build_pose_provider(self, *, hand_pose_cfg=None, hand_provider=None, head_rgbd: "HeadRgbdCfg | None" = None):
        """-> `WristPoseProvider` (RGB-D palm). Reuses the existing RGB-D hand front-end unchanged."""
        from .tracking.rgbd_hand_pose import build_rgbd_hand_pose_provider
        if hand_provider is None:
            hr = head_rgbd or HeadRgbdCfg()
            hand_provider = dc.replace(hr, provider="head_rgbd", side=self.side).build_provider()
            if hand_pose_cfg is None: hand_pose_cfg = hr.hand_pose
        return build_rgbd_hand_pose_provider(hand_provider=hand_provider, palm=dc.replace(self.palm, side=self.side),
                                             tracking=self.tracking, hand_pose_cfg=hand_pose_cfg)


@dataclass
class FusedWristGates:
    """Acceptance thresholds for the multi-sensor evaluation protocol (multi-sensor spec section 20). Pre-hardware
    starting points, like every other gate block here: re-fit them on the first measured A/B/C run before a PASS is
    read as evidence."""
    min_fusion_rate_hz: float = 45.0
    min_tracking_ok_fraction: float = 0.95         # valid duty over the take
    max_stationary_jitter_mm_p95: float = 8.0
    max_stationary_rotation_jitter_deg_p95: float = 2.0
    max_return_to_start_mm: float = 30.0
    max_rotation_translation_leak_mm: float = 20.0  # translation seen during the rotation-only window (lever arm!)
    max_rgbd_vi_discrepancy_mm_p95: float = 40.0    # |anchor − VI prediction| once aligned
    max_catastrophic_jumps: int = 0
    max_command_discontinuity_mm: float = 10.0      # per-tick step of the commanded TCP
    max_end_to_end_latency_ms: float = 80.0


@dataclass
class FusedWristCfg:
    """Chest RGB-D + wrist Arducam + wrist IMU -> one fused wrist pose (docs/ego_teleop/MULTISENSOR_FUSION.md).

    The three sensors are not alternatives: RGB-D supplies metric absolute XYZ, the Arducam supplies visual motion and
    the map, the IMU supplies fast rotation and dynamics. This block configures the branch that combines them and
    hands the result to the UNCHANGED downstream (HumanRobotFrameMapper -> RelativeSE3Retargeter -> ArmSafetyPipeline
    -> reBot). `rgbd_arm_poc` (RGB-D alone) and the production wrist-VIO path stay exactly as they are; ablation A/B/C
    of spec section 21 runs through `mode` here, so the three arms of the comparison share one code path."""
    enabled: bool = False
    side: str = "right"
    mode: str = "fused"                     # fused | vi_only (ablation B) | rgbd_only (ablation A)
    rate_hz: float = 50.0                   # fusion output / robot command rate; sensors keep their own rates
    vi_backend: str = "mast3r_live"         # spec section 2: the live wrist frontend is MASt3R, not OpenVINS
    vi_backend_options: dict = field(default_factory=dict)
    vi_tracking: TrackingSupervisorConfig = field(default_factory=TrackingSupervisorConfig)
    palm: PalmPoseConfig = field(default_factory=PalmPoseConfig)                 # the RGB-D anchor read-out
    rgbd_tracking: TrackingSupervisorConfig = field(default_factory=TrackingSupervisorConfig)
    provider: FusedWristConfig = field(default_factory=FusedWristConfig)         # pairing + alignment + C(t) + state
    stable: VioStableConfig = field(default_factory=VioStableConfig)
    frames: FrameMapperConfig = field(default_factory=FrameMapperConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    workspace_from_anchor: bool = True
    workspace_half_extent_m: tuple = (0.10, 0.10, 0.10)
    gates: FusedWristGates = field(default_factory=FusedWristGates)

    def provider_config(self) -> FusedWristConfig:
        return dc.replace(self.provider, side=self.side, mode=self.mode)

    def anchor_workspace(self, T_anchor) -> "WorkspaceBox":
        """The small Cartesian box around the ENGAGE TCP, intersected with the configured box — same rule as the
        RGB-D POC: this branch may only ever be MORE restrictive than the standing workspace limits."""
        import numpy as np
        p = np.asarray(T_anchor, np.float64)[:3, 3]
        h = np.asarray(self.workspace_half_extent_m, np.float64)
        lo = np.maximum(p - h, np.asarray(self.safety.workspace.min_xyz, np.float64))
        hi = np.minimum(p + h, np.asarray(self.safety.workspace.max_xyz, np.float64))
        if not self.safety.workspace.contains(p):
            # NOT a clampable situation: the workspace clamp would move the very first command to the nearest face of
            # the box, which on the arm is an unannounced jump of however far outside the TCP was standing.
            raise RuntimeError(f"ENGAGE TCP {p.round(3).tolist()} is outside fused_wrist.safety.workspace "
                               f"{list(self.safety.workspace.min_xyz)}..{list(self.safety.workspace.max_xyz)}: fix "
                               f"the box or move the arm before ARM. Clamping here would jump the TCP on engage.")
        if np.any(lo > hi): raise RuntimeError(f"empty anchor workspace around {p.round(3).tolist()}")
        return WorkspaceBox(tuple(lo), tuple(hi))

    def build_rgbd_provider(self, *, hand_provider=None, hand_pose_cfg=None, head_rgbd: "HeadRgbdCfg | None" = None):
        """The chest RGB-D anchor branch: the existing metric hand front-end + palm read-out, unchanged."""
        from .tracking.rgbd_hand_pose import build_rgbd_hand_pose_provider
        if hand_provider is None:
            hr = head_rgbd or HeadRgbdCfg()
            hand_provider = dc.replace(hr, provider="head_rgbd", side=self.side).build_provider()
            if hand_pose_cfg is None: hand_pose_cfg = hr.hand_pose
        return build_rgbd_hand_pose_provider(hand_provider=hand_provider, palm=dc.replace(self.palm, side=self.side),
                                             tracking=self.rgbd_tracking, hand_pose_cfg=hand_pose_cfg)

    def build_vi_provider(self, *, estimator=None, T_H_C=None, camera_imu_offset_ns: int = 0, initialize: bool = False,
                          downscale: int = 1):
        """The wrist Arducam + IMU branch. `estimator` is any `PoseEstimator`; omitted, it is built from
        `vi_backend` / `vi_backend_options` through the normal registry (BackendUnavailable is not caught here —
        a missing VIO must be reported, never substituted).

        `initialize=True` also calls `PoseEstimator.initialize()` with the versioned wrist calibration. A replayed
        trajectory needs none of that, but a LIVE backend does: OpenVINS writes its filter config from the camera
        model and mast3r_live builds its rectification and its handshake from it. Skipping it leaves a live backend
        running on defaults that have nothing to do with this camera, which is a systematic error nothing downstream
        can see."""
        import numpy as np
        from handumi_collector.pose.backends import make_backend
        from .tracking.wrist_pose_provider import EstimatorWristPoseProvider, TrackingSupervisor
        import ego_teleop.tracking.backends as _reg  # noqa: F401  (registers openvins, mast3r_live)
        est = estimator if estimator is not None else make_backend(self.vi_backend, **dict(self.vi_backend_options))
        if T_H_C is None:
            raise ValueError("build_vi_provider needs T_H_C (wrist_camera_<side> calibration): the Arducam sits on a "
                             "lever arm from the wrist rotation centre and an identity here fabricates translation")
        if initialize:
            self.initialize_estimator(est, downscale=downscale)
        return EstimatorWristPoseProvider(est, T_H_C=np.asarray(T_H_C, np.float64), supervisor=TrackingSupervisor(self.vi_tracking),
                                          camera_imu_offset_ns=int(camera_imu_offset_ns))

    def initialize_estimator(self, est, *, downscale: int = 1) -> dict:
        """Hand a live backend the camera model, T_camera_imu and the IMU noise, all from `configs/calibration`.

        The raw Kannala-Brandt model is passed through unchanged, distortion and all: a fisheye-native backend
        (OpenVINS) wants exactly that, and a pinhole-only one (mast3r_live -> MASt3R-Fusion) rectifies it itself with
        `transforms/fisheye.py`. Deciding here which backend gets which would put a camera-model assumption in the
        config layer, where nothing can check it."""
        import numpy as np
        from handumi_collector.pose.calibration import load_versioned
        from .transforms.fisheye import load_wrist_fisheye
        c = load_wrist_fisheye(self.side)
        _, ci = load_versioned(f"camera_imu_{self.side}")
        # CONTRACT: K, D and `image_size` are always the FULL-RESOLUTION calibration, exactly as the file has them,
        # and `downscale` says what the frames will actually arrive at. Scaling them here as well as passing the
        # factor would apply it twice — which silently puts the principal point outside the image.
        d = max(int(downscale), 1)
        intr = dict(model=c["model"], K=c["K"].tolist(), D=c["D"].tolist(),
                    image_size=[int(c["image_size"][0]), int(c["image_size"][1])],
                    width=int(c["image_size"][0]) // d, height=int(c["image_size"][1]) // d,
                    downscale=d, calibration_version=c["version"])
        T_C_I = np.asarray(ci["T_camera_imu"], float) if ci and "T_camera_imu" in ci else None
        est.initialize(intrinsics=intr, T_camera_imu=T_C_I, imu_noise=(ci or {}).get("imu_noise"),
                       image_size=(intr["width"], intr["height"]))
        return intr

    def build(self, *, vi_provider=None, rgbd_provider=None, head_rgbd: "HeadRgbdCfg | None" = None, **kw):
        """-> `FusedWristPoseProvider` (a `WristPoseProvider`). Branches may be passed in (live threads, replay)."""
        from .tracking.fused_wrist import FusedWristPoseProvider
        if rgbd_provider is None and self.mode != "vi_only":
            rgbd_provider = self.build_rgbd_provider(head_rgbd=head_rgbd, **kw)
        return FusedWristPoseProvider(self.provider_config(), vi_provider=vi_provider, rgbd_provider=rgbd_provider)


@dataclass
class TeleopCfg:
    wrists: list[str] = field(default_factory=lambda: ["left", "right"])   # both wrist units are tracked + logged (independent VIO worlds)
    side: str = "right"                      # the wrist/arm pair that drives reBot + Aero in v1 (single-arm teleop)
    finger_source: str = "unverified"        # unverified | wrist_fisheye | dedicated_camera | head_c922 | head_rgbd
                                             # -> retarget.hand_features.FingerSource. head_rgbd stays OFF until the A1 gate passes
    head_camera_role: str = "observation_only"   # fixed: the head C922 never has a tracking/control role
    rate_hz: float = 30.0
    pose_backend: str = "orbslam3"           # handumi_collector.pose backend name (must be online_capable + metric for live teleop)
    pose_backend_options: dict = field(default_factory=dict)
    tracking: TrackingSupervisorConfig = field(default_factory=TrackingSupervisorConfig)
    hand: HandSupervisorConfig = field(default_factory=HandSupervisorConfig)
    hand_rectify_fov_deg: float = 100.0      # used only when finger_source == wrist_fisheye
    degraded_speed_factor: float = 0.3
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    frames: FrameMapperConfig = field(default_factory=FrameMapperConfig)
    rebot: RebotCfg = field(default_factory=RebotCfg)
    aero: AeroCfg = field(default_factory=AeroCfg)
    head_rgbd: HeadRgbdCfg = field(default_factory=HeadRgbdCfg)
    rgbd_arm_poc: RgbdArmPocCfg = field(default_factory=RgbdArmPocCfg)
    fused_wrist: FusedWristCfg = field(default_factory=FusedWristCfg)
    episode_root: str = "datasets/ego_teleop_raw"


@functools.lru_cache(maxsize=None)
def _nested(cls) -> dict[str, type]:
    """Nested config dataclasses of `cls`, resolved from the declared types. Resolving by type rather than by field
    name matters once two configs own a differently-typed field of the same name (`gates`, `safety`, `tracking`, ...)."""
    hints = get_type_hints(cls)
    return {f.name: hints[f.name] for f in dc.fields(cls) if dc.is_dataclass(hints.get(f.name))}


def _from_dict(cls, d: dict[str, Any] | None):
    d = dict(d or {}); names = {f.name for f in dc.fields(cls)}; nested = _nested(cls)
    kw = {}
    for k, v in d.items():
        if k not in names: raise KeyError(f"{cls.__name__}: unknown key {k!r}")
        sub = nested.get(k)
        if sub is FrameMapperConfig: kw[k] = FrameMapperConfig.from_dict(v)
        elif sub is not None and isinstance(v, dict): kw[k] = _from_dict(sub, v)
        else: kw[k] = v
    return cls(**kw)


def _load_yaml(p: Path) -> dict:
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def load_teleop_cfg(config_dir: str | Path = CONFIG_DIR) -> TeleopCfg:
    cd = Path(config_dir)
    d = _load_yaml(cd / "teleop.yaml")
    d["frames"] = _load_yaml(cd / "frames.yaml") or d.get("frames", {})
    d["rebot"] = {**_load_yaml(cd / "rebot.yaml"), **d.get("rebot", {})}
    d["aero"] = {**_load_yaml(cd / "aero.yaml"), **d.get("aero", {})}
    d["head_rgbd"] = {**_load_yaml(cd / "head_rgbd.yaml"), **d.get("head_rgbd", {})}
    d["rgbd_arm_poc"] = {**_load_yaml(cd / "rgbd_arm_poc.yaml"), **d.get("rgbd_arm_poc", {})}
    d["fused_wrist"] = {**_load_yaml(cd / "fused_wrist.yaml"), **d.get("fused_wrist", {})}
    return _from_dict(TeleopCfg, d)


def aero_channels_from_cfg(a: AeroCfg) -> list[ChannelMap] | None:
    from .tracking.interfaces import AERO_CHANNELS
    from ego_collector.hands3d.aero import COMPACT_UPPER_DEG
    if not a.channels: return None
    by = {c["name"]: c for c in a.channels}
    return [ChannelMap(**{"name": n, "robot_max_deg": COMPACT_UPPER_DEG[i], **{k: v for k, v in by.get(n, {}).items() if k != "name"}}) for i, n in enumerate(AERO_CHANNELS)]
