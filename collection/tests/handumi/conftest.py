import dataclasses
import pytest
from handumi_collector.config import AppConfig, CameraCfg, CollectorCfg, GripperCfg, HardwareCfg, ImuCfg, TasksCfg, DEFAULT_CONFIG_DIR


@pytest.fixture
def mock_cfg(tmp_path) -> AppConfig:
    hw = HardwareCfg(cameras=[CameraCfg("head", "policy_obs", backend="mock", width=64, height=48, fps=30, codec="libx264"),
                              CameraCfg("left_wrist", "pose_estimation", backend="mock", width=64, height=48, fps=30, codec="libx264"),
                              CameraCfg("right_wrist", "pose_estimation", backend="mock", width=64, height=48, fps=30, codec="libx264"),
                              CameraCfg("head_depth", "aux_depth", backend="mock", width=32, height=24, fps=30, codec="libx264", required=False)],
                     imus=[ImuCfg("left", backend="mock", rate_hz=400), ImuCfg("right", backend="mock", rate_hz=400)],
                     grippers=[GripperCfg("left", backend="mock", ticks_closed=1000, ticks_open=2000),
                               GripperCfg("right", backend="mock", ticks_closed=3000, ticks_open=2000)])
    return AppConfig(hardware=hw, tasks=TasksCfg(target_per_order=2),
                     collector=CollectorCfg(dataset_root=str(tmp_path / "raw"), session_prefix="Htest", min_episode_s=0.1),
                     config_dir=DEFAULT_CONFIG_DIR)
