"""Sensor devices. Every device exposes the same minimal surface (see base.Device) and appends timestamped samples to a
SampleBuffer; nothing model- or task-specific lives here. Real drivers: camera (UVC/Orbbec), teensy_imu, feetech_gripper.
Mocks with identical behaviour: mock.py (used when hardware is absent and by the tests)."""
