"""[2026-10-06] nearest FEASIBLE robot pose to the human wrist-cam start view: grid over TCP position/pitch (human cam heading),
cost = |cam pos err| mm + 3 mm/deg * |cam pitch err|; IK OK = pos<2 mm, rot<1 deg, joint margin>8 deg."""
import json, sys, pathlib, numpy as np
exec(open("match_view.py").read().split("# feasible search")[0].split("Tt = Tc @")[0])
hc = Tc[:3, 2] - (Tc[:3, 2] @ up) * up; hc /= np.linalg.norm(hc); res = []
for p in range(15, 46, 5):
    pr = np.radians(p); a = np.cos(pr) * hc - np.sin(pr) * up; y = np.cross(up, a); y /= np.linalg.norm(y)
    if y @ T0[:3, 1] < 0: y = -y
    R = np.stack([a, y, np.cross(a, y)], 1)
    for dx in range(-120, 1, 20):
        for dz in range(0, 121, 20):
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = Tc[:3, 3] - R @ Xr[:3, 3] + np.array([dx, 0, dz]) / 1000
            if T[2, 3] < -0.054 + 0.05: continue
            pe, re, mg, qd = reach(T)
            if pe < 2 and re < 1 and mg > 8:
                c = T @ Xr; cp = desc(c)
                e_pos = np.linalg.norm(c[:3, 3] - Tc[:3, 3]) * 1000; e_p = cp["pitch"] - 51.0
                res.append((e_pos + 3 * abs(e_p), p, dx, dz, e_pos, e_p, T, qd, mg, (c[:3, 3] - Tc[:3, 3]) * 1000))
res.sort(key=lambda r: r[0])
for r in res[:8]:
    print(f"cost {r[0]:6.1f}  TCP pitch {r[1]}  TCP mm {np.round(r[6][:3,3]*1000)} (table+{r[6][2,3]*1000+54.3:.0f})  cam pos err {r[4]:5.1f} mm {np.round(r[9])}  cam pitch err {r[5]:+5.1f}  margin {r[8]:.0f}  q {np.round(r[7],1)}")
b = res[0]
json.dump(dict(T_base_tcp=b[6].tolist(), tcp_mm=np.round(b[6][:3, 3] * 1000, 1).tolist(), tcp_pitch=b[1], cam_pos_err_mm=b[4], cam_pos_err_vec_mm=b[9].tolist(),
               cam_pitch_err_deg=b[5], q_deg=b[7].tolist(), note="2026-10-06 nearest feasible robot pose to the median human HRA wrist-cam start view; offline IK only"),
          open("matched_view_feasible.json", "w"), indent=1)
