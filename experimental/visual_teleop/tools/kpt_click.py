"""Click the UMI body keypoints on a few teacher frames (2026-10-15, tagless teacher: kpt+depth position, IMU orientation).
Serves http://127.0.0.1:8742: 8 well-tracked, slow frames of the Orbbec recording, cropped around the jaw tip and shown 2x.
For each image click the named points in order (S = skip a point that is not visible). Saves <episode>/kpt_clicks.json with
full-frame pixel coordinates; scripts/umi_body_kpts.py turns them into 3D points in the UMI (H/TCP) frame.
    .venv/bin/python -m visual_teleop.tools.kpt_click datasets/VisualTeleop_v1/episode_20261015_174210"""
from __future__ import annotations
import argparse, json, subprocess, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np

POINTS = [("jaw_tip", "집게 끝 (두 집게 가운데)"), ("jaw_root", "집게 뿌리 (집게가 몸체에서 나오는 곳)"), ("cam_lens", "손목 카메라 렌즈"),
          ("body_top_back", "몸체 위쪽 뒤 모서리"), ("handle_bottom", "손잡이 아래 끝")]
CROP_W, CROP_H, SCALE = 440, 320, 2


def pick_frames(E: Path, n: int = 8) -> list[dict]:
    d = np.load(E / "det_log.npz"); tk = np.load(E / "ticks.npz", allow_pickle=True); tc = np.load(E / "frames/rgbd_color_t_ns.npy")
    st = tk["hand_status_right"]; T = tk["human_T_right"]; tt = tk["t_ns"]
    good = []
    for j in np.flatnonzero(d["accepted"]):
        i = np.searchsorted(tt, d["t_ns"][j]); i = min(max(i, 1), len(tt) - 1)
        if st[i] != "TRACKING" or not np.all(np.isfinite(T[i])): continue
        Ra, Rb = T[i - 1][:3, :3], T[i][:3, :3]; w = np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))) / max(1e-3, (tt[i] - tt[i - 1]) / 1e9)
        if w < 30: good.append((j, int(np.argmin(np.abs(tc - d["t_ns"][j])))))
    sel = [good[int(k)] for k in np.linspace(0, len(good) - 1, n)] if len(good) >= n else good
    out = []
    for j, k in sel:
        u, v = d["uv"][j]; x0, y0 = int(max(0, u - CROP_W / 2)), int(max(0, v - CROP_H / 2)); out.append(dict(frame=k, t_ns=int(tc[k]), x0=x0, y0=y0, tip_uv=[float(u), float(v)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--port", type=int, default=8742); a = ap.parse_args()
    E = Path(a.episode); frames = pick_frames(E); cap = cv2.VideoCapture(str(E / "frames/rgbd_color.mp4")); imgs = []
    for f in frames:
        cap.set(1, f["frame"]); ok, im = cap.read(); crop = im[f["y0"]:f["y0"] + CROP_H, f["x0"]:f["x0"] + CROP_W]
        imgs.append(cv2.imencode(".jpg", cv2.resize(crop, (crop.shape[1] * SCALE, crop.shape[0] * SCALE), interpolation=cv2.INTER_CUBIC))[1].tobytes())
    out_path = E / "kpt_clicks.json"; done = threading.Event()
    page = """<!doctype html><html><head><meta charset=utf-8><title>UMI keypoints</title><style>body{font-family:-apple-system,sans-serif;background:#222;color:#eee;margin:16px}
img{cursor:crosshair;border:2px solid #555}#q{font-size:24px;margin:8px 0}button{font-size:18px;margin-right:8px}</style></head><body>
<div id=q></div><div><button onclick=skip()>이 점은 안 보임 (S)</button><button onclick=undo()>되돌리기</button></div><div style=position:relative;display:inline-block><img id=im><canvas id=cv style=position:absolute;left:2px;top:2px;pointer-events:none></canvas></div>
<script>const P=%s, N=%d, S=%d; let i=0,p=0,res=[];for(let k=0;k<N;k++)res.push({});
function show(){if(i>=N){q.textContent='끝났습니다. 저장 중...';fetch('/save',{method:'POST',body:JSON.stringify(res)}).then(()=>q.textContent='저장했습니다. 이 창을 닫아도 됩니다.');im.style.display='none';return}
q.textContent=`사진 ${i+1}/${N} — 클릭: ${P[p][1]}`;im.src='/img/'+i;draw()}
function draw(){cv.width=im.width;cv.height=im.height;const c=cv.getContext('2d');c.fillStyle='#0f0';c.font='16px sans-serif';for(const [k,v] of Object.entries(res[i]||{})){if(!v)continue;c.beginPath();c.arc(v[2],v[3],5,0,7);c.fill();c.fillText(k,v[2]+7,v[3]-7)}}
im.onload=draw;im.onclick=e=>{const r=im.getBoundingClientRect();const x=e.clientX-r.left,y=e.clientY-r.top;res[i][P[p][0]]=[x/S,y/S,x,y];next()};
function next(){p++;if(p>=P.length){p=0;i++}show()} function skip(){res[i][P[p][0]]=null;next()}
function undo(){if(p>0)p--;else if(i>0){i--;p=P.length-1}delete res[i][P[p][0]];show()}
document.onkeydown=e=>{if(e.key==='s'||e.key==='S')skip()};show()</script></body></html>""" % (json.dumps(POINTS, ensure_ascii=False), len(imgs), SCALE)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/img/"):
                b = imgs[int(self.path.split("/")[-1])]; self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers(); self.wfile.write(b); return
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(page.encode())
        def do_POST(self):
            res = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            clicks = []
            for f, r in zip(frames, res):
                clicks.append(dict(f, points={k: (None if v is None else [f["x0"] + v[0], f["y0"] + v[1]]) for k, v in r.items()}))
            out_path.write_text(json.dumps(dict(points=POINTS, frames=clicks), indent=1, ensure_ascii=False)); print(f"[kpt_click] saved {out_path}", flush=True)
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); done.set()
        def log_message(self, *x): pass
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[kpt_click] http://127.0.0.1:{a.port}  ({len(imgs)} images x {len(POINTS)} points)", flush=True); subprocess.Popen(["open", f"http://127.0.0.1:{a.port}"])
    done.wait(); srv.shutdown()


if __name__ == "__main__":
    main()
