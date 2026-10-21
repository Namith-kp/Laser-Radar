"""
Balloon tracker: follows ONE balloon with your trained YOLO model (balloon.pt). No laser, no hunting - tracking only.

    python balloon_track.py --source http://192.168.1.114:81/stream --port COM7 --invert-pan --invert-tilt
    python balloon_track.py --source serial:COM5 --port COM7 --latency 0.08
    python balloon_track.py --source 0 --no-turret          # just test the detector on a webcam, no motors

How it works
    * YOLO runs on every picture (GPU if PyTorch sees your NVIDIA card).
    * It locks ONE balloon: the one you click, or (default) the one nearest the middle when nothing is locked.
    * Identity: each frame it picks the detection closest to where the locked balloon should be (Kalman prediction
      in absolute angles), inside a gate. Other balloons are ignored. If it disappears (out of frame, hidden) the head
      coasts on the prediction, then waits at the last place; when a balloon shows up near the predicted spot it
      is re-locked, and after --reacq-wait seconds with no match it accepts any balloon (or keeps waiting with --strict).
    * Aiming = same lag-compensated Kalman controller as hunter.py (learns the camera delay by itself).

Keys / mouse:  click a balloon = lock it | n = next balloon | u = unlock | h = go home | p = pause | q = quit
"""
import argparse, time
import cv2
import numpy as np
from turret_common import open_camera, Kalman2D, clamp

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="webcam index, stream URL, or 'serial:COM5'")
ap.add_argument("--port", help="Uno serial port, e.g. COM7 (not needed with --no-turret)")
ap.add_argument("--no-turret", action="store_true", help="only show detections/lock, do not move motors")
ap.add_argument("--model", default="balloon.pt")
ap.add_argument("--classes", nargs="*", default=["balloon"])
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--device", default="auto")
ap.add_argument("--fov-h", type=float, default=52.0)
ap.add_argument("--fov-v", type=float, default=39.0)
ap.add_argument("--invert-pan", action="store_true")
ap.add_argument("--invert-tilt", action="store_true")
ap.add_argument("--kp", type=float, default=5.0)
ap.add_argument("--max-rate", type=float, default=220.0)
ap.add_argument("--still", type=float, default=12.0)
ap.add_argument("--hold", type=float, default=1.0)
ap.add_argument("--latency", type=float, default=0.12)
ap.add_argument("--head-lag", type=float, default=0.06)
ap.add_argument("--lead", type=float, default=0.05)
ap.add_argument("--no-auto-delay", action="store_true")
ap.add_argument("--gate", type=float, default=10.0, help="deg: a detection this close to the predicted spot is the same balloon")
ap.add_argument("--coast", type=float, default=0.6, help="s: keep moving on the prediction this long after the balloon vanishes")
ap.add_argument("--reacq-wait", type=float, default=2.0, help="s without a match before any balloon may be taken")
ap.add_argument("--strict", action="store_true", help="never switch to a different balloon; wait for the locked one")
ap.add_argument("--cam-baud", type=int, default=2000000)
ap.add_argument("--no-cam-setup", action="store_true")
ap.add_argument("--cam-set")
ap.add_argument("--cam-port", type=int, default=80)
args = ap.parse_args()
if not args.no_turret and not args.port:
    raise SystemExit("Give --port COM7 (or --no-turret to test without motors)")

PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX = -90, 90, -30, 45      # keep equal to the Arduino limits


# ------------------------------------------------------------------ aiming (lag-compensated, Kalman-filtered)
class Aim:
    def __init__(s):
        s.hist, s.meas = [], []
        s.kal = Kalman2D(q_pos=0.05, q_vel=60, r=3.0)
        s.delay = args.latency + args.head_lag
        s.last_fit, s.printed = 0.0, 0.0

    def log(s, t, pan, tilt):
        s.hist.append((t, pan, tilt))
        if len(s.hist) > 200: del s.hist[:100]

    def at(s, t):
        for ht, hp, htl in reversed(s.hist):
            if ht <= t: return hp, htl
        return (s.hist[0][1], s.hist[0][2]) if s.hist else (0.0, 0.0)

    def world(s, ep, et, now):
        """image error (deg) -> absolute angle of the object: where the head pointed when the picture was taken + the error."""
        p0, t0 = s.at(now - s.delay)
        return p0 + ep, t0 + et

    def fit(s, now):
        pts = [m for m in s.meas if m[0] >= now - 1.2]
        if len(pts) < 20: return None
        ts = np.array([m[0] for m in pts]); ep = np.array([m[1] for m in pts]); et = np.array([m[2] for m in pts])
        hp_ = np.array([h[1] for h in s.hist if h[0] >= now - 1.8])
        if len(hp_) < 10 or (hp_.max() - hp_.min()) < 5.0: return None
        x = ts - ts.mean(); cands = np.arange(0.04, 0.42, 0.02); scores = []
        for D in cands:
            sc = 0.0
            for e_, k in ((ep, 0), (et, 1)):
                y = np.array([s.at(t - D)[k] for t in ts]) + e_
                sc += float(np.mean((y - np.polyval(np.polyfit(x, y, 3), x)) ** 2))
            scores.append(sc)
        scores = np.array(scores); i = int(np.argmin(scores))
        cur = int(np.argmin(np.abs(cands - s.delay)))
        return float(cands[i]) if scores[i] < 0.8 * scores[cur] else None

    def step(s, ep, et, seen, now, dt, pan, tilt):
        """One control step toward the target. ep/et = error in the picture (deg). Returns (pan, tilt)."""
        s.meas.append((now, ep, et))
        if len(s.meas) > 200: del s.meas[:100]
        if not args.no_auto_delay and now - s.last_fit > 0.3:
            s.last_fit = now
            d = s.fit(now)
            if d is not None:
                s.delay = 0.5 * s.delay + 0.5 * d
                if abs(s.delay - s.printed) > 0.03:
                    s.printed = s.delay
                    print(f"Camera delay learned: {s.delay:.2f} s  (use --latency {max(0.0, s.delay - args.head_lag):.2f})")
        ap_m, at_m = s.world(ep, et, now)
        if not s.kal.ready: s.kal.reset(ap_m, at_m)
        ap_e, at_e, vp, vt = s.kal.step((ap_m, at_m) if seen else None, dt)
        vp, vt = clamp(vp, -150, 150), clamp(vt, -150, 150)
        vp = float(np.sign(vp)) * max(0.0, abs(vp) - args.still)
        vt = float(np.sign(vt)) * max(0.0, abs(vt) - args.still)
        hz = min(0.3, s.delay + args.lead)
        cp, ct = ap_e + clamp(vp * hz, -40, 40), at_e + clamp(vt * hz, -40, 40)
        lim = args.max_rate * dt
        dp, dtl = cp - pan, ct - tilt
        dp = float(np.sign(dp)) * max(0.0, abs(dp) - args.hold)
        dtl = float(np.sign(dtl)) * max(0.0, abs(dtl) - args.hold)
        if dp != 0.0 or vp != 0.0: pan = clamp(pan + clamp(args.kp * dp * dt + vp * dt, -lim, lim), PAN_MIN, PAN_MAX)
        if dtl != 0.0 or vt != 0.0: tilt = clamp(tilt + clamp(args.kp * dtl * dt + vt * dt, -lim, lim), TILT_MIN, TILT_MAX)
        return pan, tilt

    def predicted(s):
        if not s.kal.ready: return None
        a = np.asarray(s.kal.kf.statePost, np.float64).reshape(-1)
        return float(a[0]), float(a[1])


# ------------------------------------------------------------------ detector
try:
    from ultralytics import YOLO
except Exception:
    raise SystemExit("Install the detector library first:  pip install ultralytics")
model = YOLO(args.model)
names = model.names
try:
    import torch
    _cuda = torch.cuda.is_available()
except Exception:
    torch, _cuda = None, False
dev = (0 if _cuda else "cpu") if args.device == "auto" else (int(args.device) if str(args.device).isdigit() else args.device)
use_half = dev != "cpu"
print("Detector on", torch.cuda.get_device_name(dev) if dev != "cpu" else "the CPU (install CUDA PyTorch for the GPU)")
want = set(args.classes)

def detect(frame):
    out = []
    res = model(frame, conf=args.conf, imgsz=args.imgsz, device=dev, half=use_half, verbose=False)[0]
    for b in res.boxes:
        if names[int(b.cls)] in want:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(b.conf)))
    return out


# ------------------------------------------------------------------ main
cap = open_camera(args.source, args.cam_baud, args.cam_set, args.cam_port, not args.no_cam_setup)
ser = None
if not args.no_turret:
    import serial
    ser = serial.Serial(args.port, 115200, timeout=0.1)
    time.sleep(2.0); ser.reset_input_buffer()

def send(p, t):
    if ser: ser.write(f"P{p:.2f},T{t:.2f},F0\n".encode())       # F0: the laser is never switched on here

pan = tilt = 0.0
aim = Aim()
sent, last_send = (None, None), 0.0
locked = False
lost_since = None
click = [None]
paused = False
last_t = time.time()
skip_ids = 0

def on_mouse(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN: click[0] = (x, y)
cv2.namedWindow("balloon_track"); cv2.setMouseCallback("balloon_track", on_mouse)
print("Tracking. Click a balloon to lock it (or it takes the one nearest the middle). q quits.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.003); continue
        now = time.time(); dt = min(0.25, now - last_t); last_t = now
        H, W = frame.shape[:2]
        dets = detect(frame)

        def img_err(d):
            ep = (d[0] - W / 2) / W * args.fov_h
            et = (H / 2 - d[1]) / H * args.fov_v
            return (-ep if args.invert_pan else ep), (-et if args.invert_tilt else et)
        world = [aim.world(*img_err(d), now) for d in dets]

        target = None
        # --- user click: lock the detection under / nearest the click
        if click[0] is not None and dets:
            cx, cy = click[0]
            i = int(np.argmin([np.hypot(d[0] - cx, d[1] - cy) for d in dets]))
            if np.hypot(dets[i][0] - cx, dets[i][1] - cy) < max(40, dets[i][2]):
                aim.kal = Kalman2D(q_pos=0.05, q_vel=60, r=3.0); aim.meas.clear()
                target = i; locked = True; lost_since = None
        click[0] = None

        if target is None and locked and dets:
            pred = aim.predicted()
            if pred is not None:
                dd = [np.hypot(w[0] - pred[0], w[1] - pred[1]) for w in world]
                i = int(np.argmin(dd))
                lost = 0.0 if lost_since is None else now - lost_since
                gate = min(40.0, args.gate + 8.0 * lost)                     # the longer it was gone, the bigger the search area
                if dd[i] < gate:
                    # keep identity: a second balloon inside the gate must be clearly farther than the best one
                    others = sorted(dd)[1:2]
                    if not others or others[0] > 1.5 * dd[i] or dd[i] < 3.0:
                        target = i
                elif not args.strict and lost > args.reacq_wait:
                    target = int(np.argmin([abs(img_err(d)[0]) + abs(img_err(d)[1]) for d in dets]))
                    aim.kal = Kalman2D(q_pos=0.05, q_vel=60, r=3.0); aim.meas.clear()
        if target is None and not locked and dets and not paused:
            target = int(np.argmin([abs(img_err(d)[0]) + abs(img_err(d)[1]) for d in dets]))    # nearest the middle
            locked = True; lost_since = None; aim.kal = Kalman2D(q_pos=0.05, q_vel=60, r=3.0); aim.meas.clear()

        seen = target is not None
        if seen:
            lost_since = None
            ep, et = img_err(dets[target])
        else:
            if locked and lost_since is None: lost_since = now
            ep = et = 0.0

        if locked and not paused:
            if seen:
                pan, tilt = aim.step(ep, et, True, now, dt, pan, tilt)
            elif lost_since is not None and now - lost_since < args.coast and aim.kal.ready:
                pan, tilt = aim.step(0.0, 0.0, False, now, dt, pan, tilt)   # coast on the Kalman prediction
            elif lost_since is not None and now - lost_since < 1.5 and aim.kal.ready:
                aim.kal.step(None, dt)                                      # head waits, but the prediction keeps moving
            # else: stay where we are and wait for it to come back
        if args.no_turret:
            pan = tilt = 0.0

        changed = sent[0] is None or abs(pan - sent[0]) >= 0.1 or abs(tilt - sent[1]) >= 0.1
        if changed or now - last_send > 0.3:
            if changed or sent[0] is None: sent = (pan, tilt)
            send(*sent); last_send = now
            if ser: ser.reset_input_buffer()
        aim.log(now, pan, tilt)

        # --- display
        for i, d in enumerate(dets):
            mine = (i == target)
            col = (0, 255, 0) if mine else (150, 150, 150)
            cv2.rectangle(frame, (int(d[0] - d[2] / 2), int(d[1] - d[3] / 2)), (int(d[0] + d[2] / 2), int(d[1] + d[3] / 2)), col, 2 if mine else 1)
            cv2.putText(frame, f"{d[4]:.2f}", (int(d[0] - d[2] / 2), int(d[1] - d[3] / 2) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.line(frame, (W // 2, 0), (W // 2, H), (255, 255, 255), 1); cv2.line(frame, (0, H // 2), (W, H // 2), (255, 255, 255), 1)
        status = "PAUSED" if paused else ("LOCKED" if seen else ("LOST - waiting" if locked else "searching"))
        cv2.putText(frame, f"{status}  balloons {len(dets)}  pan {pan:.0f} tilt {tilt:.0f}  delay {aim.delay:.2f}s",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("balloon_track", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("u"): locked = False; lost_since = None; aim.kal = Kalman2D(q_pos=0.05, q_vel=60, r=3.0); aim.meas.clear()
        if k == ord("p"): paused = not paused
        if k == ord("h"): locked = False; pan = tilt = 0.0
        if k == ord("n") and len(dets) > 1 and target is not None:
            order = sorted(range(len(dets)), key=lambda j: dets[j][0])
            nxt = order[(order.index(target) + 1) % len(order)]
            click[0] = (dets[nxt][0], dets[nxt][1])
finally:
    try:
        if ser:
            ser.write(b"P0,T0,F0\n"); time.sleep(0.05); ser.write(b"H\n"); ser.close()
    except Exception:
        pass
    cap.release(); cv2.destroyAllWindows()