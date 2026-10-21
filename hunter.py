"""
Balloon hunter: finds balloons with YOUR trained detector, tracks them fast, and fires the laser only at a detected balloon.

    python hunter.py --source http://192.168.1.114:81/stream --port COM7 --model balloon.pt --invert-pan --invert-tilt
    python hunter.py --source serial:COM5 --port COM7 --model balloon.pt --latency 0.08 --arm --count 3

Modes
    hunt (default): scans the room once, lists every balloon it saw, then visits them NEAREST-FIRST, locks on, fires,
                    confirms the pop, and goes to the next. Stops after --count balloons (0 = all of them), then goes home.
    free:           keeps sweeping the room and pops whatever it finds until you press q.

Train the detector first:  collect_balloons.py  ->  train_balloon.py  (gives balloon.pt)

LASER SAFETY (all of it is active by default)
    * Nothing fires unless you pass --arm or press 'a'. Without it the hunt runs as a DRY RUN (no laser).
    * The laser only fires at a balloon the detector sees, locked on the middle of the picture, and stops the instant
      that detection disappears. If this program or the camera stalls, the Uno switches the laser off after 0.2 s.
    * PERSON GUARD: a second detector looks for people; while one is visible the laser is forced off.
      (It only sees what the camera sees. Keep people out of the room / behind the turret.)
    * No-fire zone: nothing fires with the head pointing below --fire-tilt-min or outside +-(--fire-pan).
    * Max burn per balloon (--dwell), then it gives up on it after 2 tries. The Uno enforces its own burn limit too.
    * Wear laser goggles rated for your laser's wavelength (OD 4+). Keep a physical kill switch in the laser's power line.

Keys:  q quit (laser off, go home) | a arm / disarm | h go home | p pause | r forget all balloons and start over
"""
import argparse, time
import cv2, serial
import numpy as np
from turret_common import open_camera, Kalman2D, clamp

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="webcam index, stream URL, or 'serial:COM5' (ESP32-CAM on a USB-TTL cable)")
ap.add_argument("--port", required=True, help="Uno serial port, e.g. COM7")
ap.add_argument("--model", default="balloon.pt", help="your trained balloon detector (from train_balloon.py)")
ap.add_argument("--classes", nargs="*", default=["balloon"])
ap.add_argument("--conf", type=float, default=0.4, help="detector confidence")
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--mode", choices=["hunt", "free"], default="hunt")
ap.add_argument("--count", type=int, default=0, help="hunt: stop after this many balloons (0 = all that were found)")
ap.add_argument("--arm", action="store_true", help="allow the laser to fire (otherwise DRY RUN)")
ap.add_argument("--fov-h", type=float, default=52.0)
ap.add_argument("--fov-v", type=float, default=39.0)
ap.add_argument("--invert-pan", action="store_true")
ap.add_argument("--invert-tilt", action="store_true")
# scanning
ap.add_argument("--scan-range", type=float, default=75.0, help="degrees left/right of centre to scan")
ap.add_argument("--scan-tilts", default="0,25", help="tilt rows to scan, e.g. '0,25'")
ap.add_argument("--scan-speed", type=float, default=30.0, help="deg/s while scanning")
# aiming (same controller as tracker.py --track)
ap.add_argument("--kp", type=float, default=5.0)
ap.add_argument("--max-rate", type=float, default=220.0)
ap.add_argument("--still", type=float, default=12.0)
ap.add_argument("--hold", type=float, default=1.0)
ap.add_argument("--latency", type=float, default=0.12, help="camera delay, s (WiFi ~0.12-0.25, USB-TTL ~0.08). It also self-calibrates")
ap.add_argument("--head-lag", type=float, default=0.06)
ap.add_argument("--lead", type=float, default=0.05)
ap.add_argument("--no-auto-delay", action="store_true")
# firing
ap.add_argument("--lock-tol", type=float, default=2.0, help="degrees: balloon must be this close to the centre to fire")
ap.add_argument("--lock-frames", type=int, default=4, help="frames in a row inside --lock-tol before firing")
ap.add_argument("--dwell", type=float, default=2.5, help="max seconds of laser per attempt")
ap.add_argument("--pop-confirm", type=float, default=0.5, help="seconds the balloon must be gone while firing to count as popped")
ap.add_argument("--fire-pan", type=float, default=80.0, help="never fire outside +-this pan angle")
ap.add_argument("--fire-tilt-min", type=float, default=-5.0, help="never fire with the head pointing lower than this")
ap.add_argument("--fire-tilt-max", type=float, default=45.0)
ap.add_argument("--no-person-guard", action="store_true", help="turn OFF the person guard (not recommended)")
ap.add_argument("--device", default="auto", help="'auto' = NVIDIA GPU if PyTorch can see one, else CPU. Or 0 / cpu")
ap.add_argument("--guard-model", default="yolov8n.pt", help="general detector used to spot people")
ap.add_argument("--cam-baud", type=int, default=2000000)
ap.add_argument("--no-cam-setup", action="store_true")
ap.add_argument("--cam-set")
ap.add_argument("--cam-port", type=int, default=80)
args = ap.parse_args()

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


def slew(pan, tilt, tp, tt, speed, dt):
    """Move the head toward (tp, tt) at most `speed` deg/s. Returns (pan, tilt, arrived)."""
    m = speed * dt
    dp, dtl = tp - pan, tt - tilt
    pan = clamp(pan + clamp(dp, -m, m), PAN_MIN, PAN_MAX)
    tilt = clamp(tilt + clamp(dtl, -m, m), TILT_MIN, TILT_MAX)
    return pan, tilt, abs(tp - pan) < 1.0 and abs(tt - tilt) < 1.0


# ------------------------------------------------------------------ balloon list
class Balloon:
    _n = 0
    def __init__(s, pan, tilt, now):
        Balloon._n += 1
        s.id, s.pan, s.tilt, s.seen, s.hits = Balloon._n, pan, tilt, now, 1
        s.popped = s.skip = False
        s.tries = 0

registry = []

def observe(pan, tilt, now):
    best, bd = None, 8.0
    for b in registry:
        d = float(np.hypot(b.pan - pan, b.tilt - tilt))
        if d < bd: best, bd = b, d
    if best is None:
        registry.append(Balloon(pan, tilt, now)); return
    if best.popped: best.seen = now; return          # remains of a popped balloon: ignore
    best.pan += 0.5 * (pan - best.pan); best.tilt += 0.5 * (tilt - best.tilt)
    best.seen, best.hits = now, best.hits + 1

def valid(b): return b.hits >= 3 and not b.popped and not b.skip

def nearest(pan, tilt):
    c = [b for b in registry if valid(b)]
    return min(c, key=lambda b: np.hypot(b.pan - pan, b.tilt - tilt)) if c else None


# ------------------------------------------------------------------ detectors
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
if args.device == "auto":
    dev = 0 if _cuda else "cpu"
else:
    dev = int(args.device) if str(args.device).isdigit() else args.device
use_half = dev != "cpu"
if dev != "cpu":
    print(f"Detector runs on the GPU: {torch.cuda.get_device_name(dev)}")
else:
    print("Detector runs on the CPU  (to use your NVIDIA GPU install the CUDA build of PyTorch - see the top of train_balloon.py)")
want = set(args.classes)
guard = None
if not args.no_person_guard:
    guard = YOLO(args.guard_model)
    print("Person guard ON (the laser is forced off while a person is visible).")
else:
    print("WARNING: person guard is OFF.")

def detect(frame):
    out = []
    res = model(frame, conf=args.conf, imgsz=args.imgsz, device=dev, half=use_half, verbose=False)[0]
    for b in res.boxes:
        if names[int(b.cls)] in want:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(b.conf)))
    return out

def person_visible(frame):
    res = guard(frame, conf=0.35, imgsz=args.imgsz, classes=[0], device=dev, half=use_half, verbose=False)[0]
    return len(res.boxes) > 0


# ------------------------------------------------------------------ main
cap = open_camera(args.source, args.cam_baud, args.cam_set, args.cam_port, not args.no_cam_setup)
ser = serial.Serial(args.port, 115200, timeout=0.1)
time.sleep(2.0)
ser.reset_input_buffer()

pan = tilt = 0.0
armed = args.arm
aim = Aim()
sent, last_send = (None, None, None), 0.0
tilts = [float(v) for v in args.scan_tilts.split(",") if v.strip()] or [0.0]

def send(p, t, f):
    ser.write(f"P{p:.2f},T{t:.2f},F{int(f)}\n".encode())

def scan_waypoints():
    R, pts, flip = args.scan_range, [], 1
    for tl in tilts:
        pts += [(flip * R, tl), (-flip * R, tl)]; flip = -flip
    pts.append((0.0, 0.0))
    return pts

state, wp, wp_i = "SCAN", scan_waypoints(), 0
target, goto_t0, track_miss_t0, lock_n = None, 0.0, None, 0
fire_t0, cool_until, fire_ok_t0 = None, 0.0, None
n_popped, paused = 0, False
person_until, frame_i, last_t = 0.0, 0, time.time()
dry_lock_t0 = None
draw_cand = None
prev_pan = prev_tilt = 0.0

def start_scan():
    global state, wp, wp_i
    state, wp, wp_i = "SCAN", scan_waypoints(), 0
    print("Scanning the room...")

print("Hunt starts. ARMED" if armed else "Hunt starts as a DRY RUN (add --arm to allow the laser).")
cv2.namedWindow("hunter")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005); continue           # no new picture -> nothing is sent -> the Uno's 0.2 s failsafe switches the laser off
        now = time.time(); dt = min(0.25, now - last_t); last_t = now
        H, W = frame.shape[:2]
        frame_i += 1

        dets = detect(frame)
        if guard is not None and frame_i % 4 == 0 and person_visible(frame):
            person_until = now + 1.0
        person_block = now < person_until

        def img_err(d):
            ep = (d[0] - W / 2) / W * args.fov_h
            et = (H / 2 - d[1]) / H * args.fov_v
            return (-ep if args.invert_pan else ep), (-et if args.invert_tilt else et)

        # every detection updates the list of known balloons (in absolute angles) - but not while the head is
        # whipping around fast: then the camera delay makes the angles unreliable and one balloon would be listed twice
        head_speed = float(np.hypot(pan - prev_pan, tilt - prev_tilt)) / max(dt, 1e-3)
        prev_pan, prev_tilt = pan, tilt
        if head_speed < 45.0:
            for d in dets:
                ep, et = img_err(d)
                ap_, at_ = aim.world(ep, et, now)
                observe(ap_, at_, now)

        fire = False
        draw_cand = None
        if paused:
            pass
        elif state == "SCAN":
            tp, tt = wp[wp_i]
            pan, tilt, arrived = slew(pan, tilt, tp, tt, args.scan_speed, dt)
            if arrived:
                wp_i += 1
                if wp_i >= len(wp):
                    found = [b for b in registry if valid(b)]
                    print(f"Scan done: {len(found)} balloon(s) found" +
                          (": " + ", ".join(f"#{b.id} ({b.pan:.0f},{b.tilt:.0f})" for b in found) if found else "."))
                    state = "PLAN"
        elif state == "PLAN":
            if args.count and n_popped >= args.count:
                state = "DONE"
            else:
                target = nearest(pan, tilt)
                if target is None:
                    if args.mode == "free" or not registry or n_popped == 0:
                        if args.mode == "hunt" and registry == [] :
                            print("No balloons found - scanning again.")
                        start_scan()
                    else:
                        state = "DONE"
                else:
                    state, goto_t0, lock_n, fire_t0 = "GOTO", now, 0, None
                    aim.kal.ready = False
                    print(f"Going to balloon #{target.id} at pan {target.pan:.0f}, tilt {target.tilt:.0f}")
        elif state == "GOTO":
            pan, tilt, _ = slew(pan, tilt, target.pan, target.tilt, 90.0, dt)
            cand = None
            for d in dets:                                          # is a detection where we expect the target?
                ep, et = img_err(d)
                ap_, at_ = aim.world(ep, et, now)
                if np.hypot(ap_ - target.pan, at_ - target.tilt) < 8.0:
                    cand = d; break
            if cand is not None:
                state, track_miss_t0, lock_n, fire_t0, fire_ok_t0, dry_lock_t0 = "TRACK", None, 0, None, None, None
                aim.kal.ready = False
                print(f"Locked on balloon #{target.id}")
            elif now - goto_t0 > 6.0:
                print(f"Balloon #{target.id} not seen again - dropping it.")
                target.skip = True; state = "PLAN"
        elif state == "TRACK":
            pred = aim.predicted()
            cand, cd = None, 1e9
            for d in dets:
                ep, et = img_err(d)
                ap_, at_ = aim.world(ep, et, now)
                ref = pred if pred is not None else (target.pan, target.tilt)
                dist = float(np.hypot(ap_ - ref[0], at_ - ref[1]))
                if dist < cd: cand, cd = d, dist
            if cand is not None and cd < 8.0:
                draw_cand = cand
                track_miss_t0 = None
                ep, et = img_err(cand)
                pan, tilt = aim.step(ep, et, True, now, dt, pan, tilt)
                target.pan, target.tilt = aim.world(ep, et, now)
                locked = abs(ep) < args.lock_tol and abs(et) < args.lock_tol
                lock_n = lock_n + 1 if locked else 0
                in_zone = abs(pan) <= args.fire_pan and args.fire_tilt_min <= tilt <= args.fire_tilt_max
                can = lock_n >= args.lock_frames and in_zone and not person_block and now >= cool_until
                if can and armed:
                    if fire_t0 is None: fire_t0 = now
                    fire = True
                    if now - fire_t0 > args.dwell:
                        target.tries += 1; fire = False; fire_t0 = None; cool_until = now + 1.8
                        print(f"No pop after {args.dwell:.1f} s on balloon #{target.id} (try {target.tries})")
                        if target.tries >= 2:
                            print(f"Giving up on balloon #{target.id}."); target.skip = True; state = "PLAN"
                elif can and not armed:                              # DRY RUN: pretend after 1 s of lock
                    dry_lock_t0 = dry_lock_t0 or now
                    if now - dry_lock_t0 > 1.0:
                        print(f"(dry run) would fire at balloon #{target.id} - counting it as popped")
                        target.popped = True; n_popped += 1; state = "PLAN"
                else:
                    fire_t0 = None; dry_lock_t0 = None
            else:
                # lost sight of the balloon: laser OFF immediately
                if track_miss_t0 is None: track_miss_t0 = now
                gone = now - track_miss_t0
                # it vanished while we were burning it (for at least 0.3 s before it vanished) -> popped
                if fire_t0 is not None and (now - fire_t0 - gone) >= 0.3 and gone >= args.pop_confirm:
                    n_popped += 1; target.popped = True
                    print(f"POP! balloon #{target.id} ({n_popped} popped)")
                    fire_t0 = None; state = "PLAN"
                elif gone > 2.0:
                    print(f"Lost balloon #{target.id}.")
                    state = "GOTO" if target.hits >= 3 else "PLAN"; goto_t0 = now
                    fire_t0 = None; lock_n = 0
        elif state == "DONE":
            pan, tilt, _ = slew(pan, tilt, 0.0, 0.0, 60.0, dt)

        fire = bool(fire and armed and not person_block and not paused)

        # ---- send: only on change, plus keep-alive (every 0.08 s while firing so the Uno's 0.2 s failsafe never trips)
        changed = (sent[0] is None or abs(pan - sent[0]) >= 0.1 or abs(tilt - sent[1]) >= 0.1 or fire != sent[2])
        if changed or now - last_send > (0.08 if fire else 0.3):
            if changed or sent[0] is None: sent = (pan, tilt, fire)
            send(*sent); last_send = now; ser.reset_input_buffer()
        aim.log(now, pan, tilt)

        # ---- display
        for d in dets:
            cv2.rectangle(frame, (int(d[0] - d[2] / 2), int(d[1] - d[3] / 2)), (int(d[0] + d[2] / 2), int(d[1] + d[3] / 2)), (150, 150, 150), 1)
        if draw_cand is not None:
            col = (0, 0, 255) if fire else (0, 255, 0)
            cv2.rectangle(frame, (int(draw_cand[0] - draw_cand[2] / 2), int(draw_cand[1] - draw_cand[3] / 2)),
                          (int(draw_cand[0] + draw_cand[2] / 2), int(draw_cand[1] + draw_cand[3] / 2)), col, 2)
        cv2.line(frame, (W // 2, 0), (W // 2, H), (255, 255, 255), 1); cv2.line(frame, (0, H // 2), (W, H // 2), (255, 255, 255), 1)
        nvalid = len([b for b in registry if valid(b)])
        cv2.putText(frame, f"{state}  found {nvalid}  popped {n_popped}  pan {pan:.0f} tilt {tilt:.0f}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, ("ARMED" if armed else "DRY RUN") + ("  FIRING" if fire else "") + ("  PAUSED" if paused else ""),
                    (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if armed else (0, 255, 255), 1)
        if person_block:
            cv2.putText(frame, "PERSON SEEN - LASER INHIBITED", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if state == "DONE":
            cv2.putText(frame, f"DONE - {n_popped} popped", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("hunter", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("a"): armed = not armed; print("ARMED" if armed else "disarmed (dry run)")
        if k == ord("h"): pan = tilt = 0.0; state = "DONE"
        if k == ord("p"): paused = not paused
        if k == ord("r"):
            registry.clear(); n_popped = 0; fire_t0 = None; start_scan()
finally:
    try:
        ser.write(b"P0,T0,F0\n"); time.sleep(0.05); ser.write(b"H\n"); ser.close()   # laser off first, then home
    except Exception:
        pass
    cap.release(); cv2.destroyAllWindows()