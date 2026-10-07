from handumi_collector.config import load_config, DEFAULT_CONFIG_DIR


def test_repo_configs_load_and_mock_override():
    cfg = load_config(DEFAULT_CONFIG_DIR)
    names = [c.name for c in cfg.hardware.cameras]
    # 2026-09-15: the head is a C922 again, reversing the 2026-09-11 switch to the Orbbec and for the same reason it
    # was made — matching the human head to the reBot head, which is a C922. Depth was never the V1 policy input, and
    # the Orbbec cannot share a host with the wrist cameras: opening it breaks a UVC camera that is already streaming,
    # and opening it first stops the next UVC camera from opening at all. 0/3 thirty-second takes survived either
    # ordering with all three cameras on separate host controllers; without it the same rig is 3/3.
    assert names == ["left_wrist", "right_wrist", "head"]
    assert not [c for c in cfg.hardware.cameras if c.backend == "orbbec"], "production is RGB only"
    assert "head_depth" not in names
    head = [c for c in cfg.hardware.cameras if c.name == "head"][0]
    assert (head.width, head.height, head.fps, head.role, head.required) == (640, 480, 30, "policy_obs", True)
    assert head.match_name == "C922"
    assert cfg.tasks.orders == ["RBP", "RPB", "BRP", "BPR", "PRB", "PBR"] and cfg.tasks.target_per_order == 20
    assert cfg.tasks.instruction("RBP").startswith("Stack the red cube on the bottom, blue cube in the middle, and purple")
    m = load_config(DEFAULT_CONFIG_DIR, mock=True)
    assert {c.backend for c in m.hardware.cameras} == {"mock"} and {i.backend for i in m.hardware.imus} == {"mock"}


def test_profiles_of_the_same_rig_agree_about_which_board_is_which_hand():
    """Two profiles describing one physical rig must not disagree about device identity.

    On 2026-09-15 the IMU side assignment was corrected in hardware_temporal_sync_calib after a bench tap test and
    not carried across to hardware_handumi_v1, so the pilot recorded five episodes with the wrist IMUs labelled
    backwards: the working hand's board arrived as imu_left. Nothing failed, nothing was flagged, and the data looked
    perfectly healthy -- it was only visible as a hand that gripped without moving."""
    from pathlib import Path
    from handumi_collector.config import load_hardware
    prof = {n: load_hardware(Path(f"configs/handumi/hardware_{n}.yaml"))
            for n in ("handumi_v1", "temporal_sync_calib")}
    serials = {n: {i.side: i.serial_number for i in h.imus if i.backend == "teensy"} for n, h in prof.items()}
    a, b = serials["handumi_v1"], serials["temporal_sync_calib"]
    shared = set(a) & set(b)
    assert shared, "neither profile names a teensy IMU side"
    for side in shared:
        assert a[side] == b[side], (
            f"imu {side}: handumi_v1 says {a[side]}, temporal_sync_calib says {b[side]} — one of them records the "
            f"wrong hand, and the episode metadata will look healthy either way")
    cams = {n: {c.name: c.match_name for c in h.cameras if c.backend == "uvc"} for n, h in prof.items()}
    for name in set(cams["handumi_v1"]) & set(cams["temporal_sync_calib"]):
        assert cams["handumi_v1"][name] == cams["temporal_sync_calib"][name], f"camera {name} differs between profiles"
