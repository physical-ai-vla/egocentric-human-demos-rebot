"""HandUMI egocentric data collector (model-independent raw recording + offline processing).

Package layout: devices/ (sensor drivers + mocks), collector/ (timeline, recorder, episode manager, session),
ui/ (PySide6 desktop UI), processing/ (offline sync/TCP/QA, M2+), exporters/ (canonical / LeRobot, M5).
Reuses ego_collector.camera.capture / tracking.transforms; see docs/handumi_collector/PLAN.md.
"""
__version__ = "0.1.0"
