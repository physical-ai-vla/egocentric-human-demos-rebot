# reBot B601-DM bimanual assets

`rebot_b601_bimanual.urdf` = `~/robot-cockpit/urdf/reBot_B601_DM_dualarm.urdf`
(two single-arm B601-DM URDFs, 63 cm baseline, left at y=+0.315) plus fixed
`left_tcp` / `right_tcp` links: 70 mm along `{side}_gripper_link` +x (finger
direction; validated on 2026-08-25 teleop grasps), rotated so TCP +Z is the
approach axis and TCP +Y the jaw-opening axis.

Mesh files are referenced as `package://rebotarm_bringup/description/meshes_b601*`
and are NOT vendored (upstream repo EclipseaHime017/reBotArmController_ROS2). IK /
conversion load the URDF with `load_meshes=False`; for `handumi replay` visuals copy
the mesh folders into `assets/rebot_b601/rebotarm_bringup/description/`.
