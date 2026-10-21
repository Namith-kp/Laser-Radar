"""
Turret tracker (PC side): camera -> colour / YOLO detection -> serial commands to the Uno.

Install:     pip install "opencv-contrib-python==4.11.0.86" pyserial numpy   (+ ultralytics only for YOLO mode; for --track you need the contrib build:
             pip uninstall opencv-python opencv-python-headless  then  pip install opencv-contrib-python)
Recommended (colour, click to choose ONE target, ESP32-CAM stream):
    python tracker.py --source http://<cam-ip>:81/stream --port COM7 --color blue --select --invert-tilt --show-mask --fov-h 52 --fov-v 39
No colour at all - drag a box around ANY object, it is followed by appearance + motion prediction:
    python tracker.py --source http://<cam-ip>:81/stream --port COM7 --track csrt --invert-pan --invert-tilt
Camera on a USB-TTL cable instead of WiFi (flash esp32_cam_serial.ino first):
    python tracker.py --source serial:COM5 --port COM7 --color blue --select --invert-tilt
Laptop webcam test:
    python tracker.py --source 0 --port COM7 --color blue --select --show-mask
YOLO instead of colour (e.g. people):
    python tracker.py --source 0 --port COM7 --classes person
Own trained model:
    python tracker.py --model best.pt --classes balloon ...

The laser is NEVER commanded on unless you pass --arm or press 'a'.

Keys in the video window:
    left-click = select target (with --select)   c = clear target   h = go home
    a = toggle arm (laser allowed)               q = quit

Tuning:  --kp / --max-rate (speed) | --deadband (shake at rest) | --smooth (jitter)
         --min-area / --close / --hsv (colour detection) | --invert-pan / --invert-tilt (direction)
"""
import argparse, os, time, threading
import urllib.request, urllib.parse
# Ask OpenCV's FFmpeg reader NOT to buffer/probe the network stream (set before cv2 is imported)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0")
import cv2, serial
import numpy as np

FIX_CV2 = ("\nOpenCV is installed incorrectly (several OpenCV packages are mixed up). Fix, in a terminal:\n"
           "  pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless\n"
           "  pip install \"opencv-contrib-python==4.11.0.86\"\n"
           "If this message still appears, delete the leftover folder  ...\\site-packages\\cv2  and run the last line again.")
if not hasattr(cv2, "imdecode") or not hasattr(cv2, "KalmanFilter"):
    raise SystemExit(FIX_CV2)

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="webcam index, stream URL, or 'serial:COM5' for an ESP32-CAM on a USB-TTL cable")
ap.add_argument("--cam-baud", type=int, default=2000000, help="serial camera: baud rate (must match the ESP32-CAM sketch)")
ap.add_argument("--port", required=True, help="Uno serial port, e.g. COM3 or /dev/ttyACM0")
ap.add_argument("--model", default="yolov8n.pt")
ap.add_argument("--classes", nargs="*", default=["sports ball"],
                help="class names to hunt (pretrained COCO: 'sports ball'; balloons need your own model)")
ap.add_argument("--conf", type=float, default=0.25, help="detection confidence; lower finds more but with more false hits")
ap.add_argument("--imgsz", type=int, default=320, help="YOLO input size; smaller = faster (320 or 416 is good on CPU)")
ap.add_argument("--color", choices=["blue", "red", "green", "yellow"], help="track by colour instead of YOLO (no model needed)")
ap.add_argument("--hsv", help="custom colour range 'h1,s1,v1,h2,s2,v2' (OpenCV HSV: H 0-179). Overrides --color")
ap.add_argument("--min-area", type=int, default=150, help="colour mode: ignore blobs smaller than this many pixels")
ap.add_argument("--show-mask", action="store_true", help="colour mode: show the black/white colour mask window (for tuning)")
ap.add_argument("--select", action="store_true",
                help="do NOT auto-target: click the object in the video window to choose it, then only that object is followed")
ap.add_argument("--max-jump", type=int, default=200, help="pixels; once locked, ignore candidates farther than this from the last position")
ap.add_argument("--patience", type=int, default=30, help="frames the locked target may vanish before the lock is released")
ap.add_argument("--close", type=int, default=15, help="colour mode: size of the gap-filling kernel; bigger merges a patterned object into one blob")
ap.add_argument("--no-cam-setup", action="store_true", help="do NOT send the default ESP32-CAM settings at start-up")
ap.add_argument("--cam-set", help="extra/overriding ESP32-CAM settings, e.g. 'framesize=8,quality=10,saturation=2'")
ap.add_argument("--cam-port", type=int, default=80, help="port of the ESP32-CAM web page (control commands go here)")
ap.add_argument("--track", choices=["csrt", "kcf", "mosse"],
                help="NO colour needed: drag a box around ANY object and follow it by appearance (csrt = most accurate, kcf = faster, mosse = fastest)")
ap.add_argument("--latency", type=float, default=0.12, help="--track: camera delay in seconds (picture is this old). WiFi ~0.15-0.3, USB-TTL cable ~0.08. Used to aim at where the target REALLY is")
ap.add_argument("--head-lag", type=float, default=0.06, help="--track: seconds the head takes to follow a command (serial + motor acceleration). Raise if it overshoots on fast moves")
ap.add_argument("--still", type=float, default=12.0, help="--track: target speeds below this (deg/s) count as 'standing still' (ignores hand tremor / jitter). Raise it if the head wobbles on a held object")
ap.add_argument("--hold", type=float, default=1.0, help="--track: the head ignores aiming errors smaller than this many degrees (stops wobble on a still target). Lower = more precise, higher = calmer")
ap.add_argument("--no-auto-delay", action="store_true", help="--track: do NOT self-calibrate the camera delay while running (it normally learns it from your own head movements)")
ap.add_argument("--lead", type=float, default=0.05, help="--track: extra seconds to aim ahead of a moving target (on top of --latency). 0 = off")
ap.add_argument("--reacq", type=float, default=0.7, help="--track: how sure (0-1) it must be that something is YOUR target before it locks on again after losing it. Higher = stricter (never grabs look-alikes), lower = finds it more eagerly")
ap.add_argument("--home-after", type=float, default=10.0, help="--track: seconds of not finding the target before the head returns home (it keeps searching meanwhile)")
ap.add_argument("--verify", type=float, default=0.4, help="--track: if what the tracker follows stops looking like the target (0-1 similarity), call it lost. 0=off. Lower for blurry/JPEG cameras like ESP32-CAM")
ap.add_argument("--box", type=int, default=60, help="--track: size in pixels of the box made when you just click instead of dragging")
ap.add_argument("--show-all", action="store_true", help="draw EVERY object the model sees, with its name (use this to debug detection)")
ap.add_argument("--fov-h", type=float, default=60, help="camera horizontal field of view, degrees")
ap.add_argument("--fov-v", type=float, default=45, help="camera vertical field of view, degrees")
ap.add_argument("--gain", type=float, default=0.6, help="fraction of the error corrected per command (lower = steadier)")
ap.add_argument("--invert-pan", action="store_true")
ap.add_argument("--invert-tilt", action="store_true")
ap.add_argument("--deadband", type=float, default=1.5, help="degrees; ignore smaller errors")
ap.add_argument("--smooth", type=float, default=0.5, help="0-1, weight of the newest detection (lower = smoother)")
ap.add_argument("--min-hits", type=int, default=3, help="consecutive detections before the turret reacts (ignores flicker)")
ap.add_argument("--settle", type=float, default=0.0, help="stop-and-go mode: seconds to wait after each head move (0 = smooth continuous tracking, recommended)")
ap.add_argument("--kp", type=float, default=None, help="head speed = kp x error (deg/s per degree). Default 1.8 (colour/YOLO) or 5 (--track). Higher = faster reaction, too high = wobble")
ap.add_argument("--max-rate", type=float, default=None, help="top head speed in degrees per second. Default 60 (colour/YOLO) or 220 (--track)")
ap.add_argument("--max-step", type=float, default=10.0, help="max degrees the head may move per correction (prevents big overshoot)")
ap.add_argument("--min-step", type=float, default=0.1, help="degrees; commands smaller than this are not sent")
ap.add_argument("--lock-frames", type=int, default=5, help="centered frames needed before 'fire'")
ap.add_argument("--release-on-exit", action="store_true", help="on quit, cut motor power (head may droop). Default: go home and keep holding")
ap.add_argument("--arm", action="store_true", help="allow laser output (start armed)")
args = ap.parse_args()
if args.kp is None: args.kp = 5.0 if args.track else 1.8
if args.max_rate is None: args.max_rate = 220.0 if args.track else 60.0

PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX = -90, 90, -30, 45   # keep equal to the Arduino limits
clamp = lambda v, lo, hi: max(lo, min(hi, v))

class LatestFrame:
    """Reads the camera in a background thread and keeps ONLY the newest frame.
    Without this, OpenCV queues old frames while YOLO is busy, which shows up as huge lag."""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        self.ok, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.02); continue
            with self.lock:
                self.ok, self.frame = True, f
    def read(self):
        with self.lock:
            if self.frame is None: return False, None
            f, self.frame = self.frame, None      # hand each frame out once
            return True, f
    def release(self):
        self.running = False; time.sleep(0.05); self.cap.release()

class SerialCam:
    """ESP32-CAM connected by a USB-TTL cable. The board sends: 4 magic bytes + 4-byte length + JPEG.
    Keeps only the newest frame (same interface as LatestFrame)."""
    MAGIC = b"\xA5\x5A\xC3\x3C"
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.ser.reset_input_buffer()
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.n = 0
        class _C:                                   # tiny stand-in for cv2.VideoCapture.isOpened()
            isOpened = staticmethod(lambda: True)
        self.cap = _C
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        buf = bytearray()
        while self.running:
            try:
                chunk = self.ser.read(max(1, self.ser.in_waiting))
            except Exception:
                time.sleep(0.05); continue
            if chunk: buf += chunk
            while True:
                i = buf.find(self.MAGIC)
                if i < 0:
                    del buf[:-3]; break                  # keep a possible partial magic
                if i > 0: del buf[:i]
                if len(buf) < 8: break
                n = int.from_bytes(buf[4:8], "little")
                if n < 500 or n > 200000:                # nonsense length -> false magic, skip it
                    del buf[:1]; continue
                if len(buf) < 8 + n: break               # wait for the rest of the picture
                jpg = bytes(buf[8:8 + n])
                if jpg[:2] == b"\xff\xd8" and jpg[-2:] == b"\xff\xd9":
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        with self.lock:
                            self.frame = img; self.n += 1
                    del buf[:8 + n]
                else:
                    del buf[:1]                          # corrupted: look for the next magic
    def read(self):
        with self.lock:
            if self.frame is None: return False, None
            f, self.frame = self.frame, None
            return True, f
    def release(self):
        self.running = False; time.sleep(0.1); self.ser.close()


# ---- ESP32-CAM default settings, applied automatically every time the tracker starts ----
# (the stock CameraWebServer forgets web-page changes whenever the board restarts)
CAM_DEFAULTS = {
    "framesize": 5,        # 5 = QVGA 320x240 (fast).  8 = VGA 640x480
    "quality": 12,         # 4-63, lower = better quality but slower
    "brightness": 0, "contrast": 0, "saturation": 1, "special_effect": 0,
    "awb": 1, "awb_gain": 1, "wb_mode": 0,
    "aec": 1, "aec2": 1, "ae_level": 0,
    "agc": 1, "gainceiling": 2,   # 2 = 8x
    "bpc": 1, "wpc": 1, "raw_gma": 1, "lenc": 1,
    "hmirror": 1, "vflip": 0, "colorbar": 0,
}

def apply_cam_settings(source):
    if args.no_cam_setup or not str(source).startswith("http"):
        return
    host = urllib.parse.urlparse(source).hostname
    settings = dict(CAM_DEFAULTS)
    if args.cam_set:
        for pair in args.cam_set.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                settings[k.strip()] = int(v)
    ok_n = 0
    for k, v in settings.items():
        try:
            urllib.request.urlopen(f"http://{host}:{args.cam_port}/control?var={k}&val={v}", timeout=2).read()
            ok_n += 1
        except Exception:
            pass
    print(f"Camera settings applied: {ok_n}/{len(settings)} (host {host}:{args.cam_port})")
    time.sleep(0.5)

src = int(args.source) if args.source.isdigit() else args.source
apply_cam_settings(src)
if str(src).lower().startswith("serial:"):
    cap = SerialCam(str(src)[7:], args.cam_baud)
    print("Serial camera on", str(src)[7:], "at", args.cam_baud, "baud (camera settings come from the sketch)")
else:
    cap = LatestFrame(src)
if not cap.cap.isOpened():
    raise SystemExit("Cannot open camera source")

ser = serial.Serial(args.port, 115200, timeout=0.1)
time.sleep(2.0)                      # Uno resets when the port opens
ser.reset_input_buffer()

COLORS = {  # OpenCV HSV ranges (H 0-179, S 0-255, V 0-255); lists because red wraps around
    "blue":   [((95, 110, 50), (130, 255, 255))],
    "green":  [((40, 80, 50), (85, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
}
ranges = None
if args.hsv:
    v = [int(x) for x in args.hsv.split(",")]
    ranges = [((v[0], v[1], v[2]), (v[3], v[4], v[5]))]
elif args.color:
    ranges = COLORS[args.color]

if args.track:
    model = None
    print("Tracker mode:", args.track.upper(), "- drag a box around the target in the video window")
elif ranges is None:
    from ultralytics import YOLO
    model = YOLO(args.model)
    names = model.names
    want = set(args.classes)
else:
    model = None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.close, args.close))
    print("Colour mode:", args.hsv or args.color)

pan = tilt = 0.0
armed = args.arm
locked_count = 0
last_seen = 0.0
last_send = 0.0
hits = 0
sm_cx = sm_cy = None
sent = (None, None, None)
settle_until = 0.0
last_ctrl = time.time()

def send(p, t, f):
    ser.write(f"P{p:.2f},T{t:.2f},F{int(f)}\n".encode())

def go_home():
    global pan, tilt
    pan = tilt = 0.0
    ser.write(b"H\n")

# ---- appearance tracker + motion prediction (--track) ----
def make_tracker(name):
    wanted = {"csrt": "TrackerCSRT_create", "kcf": "TrackerKCF_create", "mosse": "TrackerMOSSE_create"}[name]
    for mod in (cv2, getattr(cv2, "legacy", None)):
        f = getattr(mod, wanted, None) if mod is not None else None
        if f is not None:
            return f()
    raise SystemExit("This OpenCV has no '%s' tracker (OpenCV 5.x dropped them)." % name + FIX_CV2)

class Kalman2D:
    """Constant-velocity filter on the target's pixel position: smooths jitter and gives its speed (px/s)."""
    def __init__(self, q_pos=2, q_vel=400, r=6):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 50
        self.ready = False
    def reset(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.ready = True
    def step(self, meas, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.transitionMatrix = F
        self.kf.predict()
        if meas is not None:
            self.kf.correct(np.array([[meas[0]], [meas[1]]], np.float32))
        s_ = self.kf.statePost if meas is not None else self.kf.statePre
        a_ = np.asarray(s_, np.float64).reshape(-1)
        return float(a_[0]), float(a_[1]), float(a_[2]), float(a_[3])

def patch_hist(img, x, y, w, h):
    """Colour fingerprint of the middle of a box (used only to notice the tracker drifting onto the background)."""
    H_, W_ = img.shape[:2]
    mx, my = int(w * 0.15), int(h * 0.15)
    x0, y0 = int(clamp(x + mx, 0, W_ - 2)), int(clamp(y + my, 0, H_ - 2))
    x1, y1 = int(clamp(x + w - mx, x0 + 1, W_)), int(clamp(y + h - my, y0 + 1, H_))
    hs = cv2.calcHist([img[y0:y1, x0:x1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hs, hs).flatten()

trk_ref = None; trk_bad = 0
trk = None; trk_area = 0.0; trk_lost = 0; trk_last = None; roi_req = None; drag = None
trk_reinit_t = 0.0     # time of last CSRT re-init (to prevent rapid lose/find thrash)
kal = Kalman2D(); kal_t = time.time(); trk_size = (60, 60)
fps_ema = 0.0; t_prev = time.time()
tmpl_t = 0.0
kal_a = Kalman2D(q_pos=0.05, q_vel=60, r=3.0)    # filter on the target's ABSOLUTE angle (degrees, deg/s)
hist = []                                     # (time, pan, tilt) the head was commanded to -> to undo camera delay

def angles_at(t):
    for ht, hp, htl in reversed(hist):
        if ht <= t:
            return hp, htl
    return (hist[0][1], hist[0][2]) if hist else (0.0, 0.0)

meas_log = []                                # (time, image error pan, image error tilt) of the last seconds
delay_est = None                              # camera delay + head lag, learned while running
last_delay_fit = 0.0
printed_delay = 0.0

def estimate_delay(now):
    """Learn the total delay (camera + head) from the head's OWN movements.
    With the right delay, 'where the head pointed when the picture was taken + error in the picture' gives a smooth
    path for the target; with a wrong delay the head's movement leaks into it and the path becomes rough."""
    pts = [m for m in meas_log if m[0] >= now - 1.2]
    if len(pts) < 20: return None
    ts = np.array([m[0] for m in pts]); ep = np.array([m[1] for m in pts]); et = np.array([m[2] for m in pts])
    hp_ = np.array([h[1] for h in hist if h[0] >= now - 1.8])
    if len(hp_) < 10 or (hp_.max() - hp_.min()) < 5.0: return None        # head barely moved: nothing to learn from
    x = ts - ts.mean()
    scores = []
    cands = np.arange(0.04, 0.42, 0.02)
    for D in cands:
        sc = 0.0
        for e_, k in ((ep, 1), (et, 2)):
            y = np.array([angles_at(t - D)[k - 1] for t in ts]) + e_
            sc += float(np.mean((y - np.polyval(np.polyfit(x, y, 3), x)) ** 2))
        scores.append(sc)
    scores = np.array(scores)
    i = int(np.argmin(scores))
    cur = int(np.argmin(np.abs(cands - (delay_est if delay_est else 0.18))))
    if scores[i] < 0.8 * scores[cur]:                                       # clearly better than what we use now
        return float(cands[i])
    return None

gallery = []                                  # several pictures of YOUR target (different moments / angles)
lost_t = None                                 # when the target was lost (None = not lost)
last_coast = time.time()

def add_template(img, x, y, w, h, keep_first=True):
    H_, W_ = img.shape[:2]
    x0, y0, x1, y1 = int(max(0, x)), int(max(0, y)), int(min(W_, x + w)), int(min(H_, y + h))
    if x1 - x0 < 10 or y1 - y0 < 10: return
    crop = img[y0:y1, x0:x1].copy()
    if len(gallery) >= 5:
        del gallery[1 if keep_first else 0]
    gallery.append(crop)

def box_ncc(img, x, y, w, h):
    """How much does the picture inside this box look like my saved pictures of the target? (-1..1)"""
    H_, W_ = img.shape[:2]
    x0, y0, x1, y1 = int(max(0, x)), int(max(0, y)), int(min(W_, x + w)), int(min(H_, y + h))
    if x1 - x0 < 10 or y1 - y0 < 10 or not gallery: return 0.0
    patch = img[y0:y1, x0:x1]
    best = -1.0
    for tm in gallery:
        r = cv2.matchTemplate(patch, cv2.resize(tm, (patch.shape[1], patch.shape[0])), cv2.TM_CCOEFF_NORMED)
        v = float(np.nan_to_num(r[0, 0]))
        best = max(best, v)
    return best

def find_target(img, size, last_pos, lost_age, kal_spd=0.0):
    """Search the WHOLE picture for MY target. A candidate must match my saved pictures AND the target's
    colour fingerprint, must not have a look-alike right next to it that scores as well, and (right after
    losing it) must be somewhere it could realistically have gone."""
    if not gallery: return None
    ds = 0.5
    small = cv2.resize(img, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    hs_, ws_ = small.shape[:2]
    cands = []
    for tm in gallery:
        # Wider scale range: 0.4-1.6x to handle target at 2-3m appearing much smaller
        for sc in (0.4, 0.55, 0.75, 1.0, 1.3, 1.6):
            w_, h_ = int(size[0] * sc * ds), int(size[1] * sc * ds)
            if w_ < 5 or h_ < 5 or w_ >= ws_ - 1 or h_ >= hs_ - 1: continue
            res = np.nan_to_num(cv2.matchTemplate(small, cv2.resize(tm, (w_, h_), interpolation=cv2.INTER_AREA),
                                                  cv2.TM_CCOEFF_NORMED))
            _, mx, _, loc = cv2.minMaxLoc(res)
            r2 = res.copy()
            r2[max(0, loc[1] - h_):loc[1] + h_ + 1, max(0, loc[0] - w_):loc[0] + w_ + 1] = -1
            _, mx2, _, _ = cv2.minMaxLoc(r2)
            cands.append((mx, mx2, loc[0] / ds, loc[1] / ds, w_ / ds, h_ / ds))
    best, best_score = None, -9.0
    for mx, mx2, x, y, w, h in cands:
        cx_, cy_ = x + w / 2, y + h / 2
        dist = np.hypot(cx_ - last_pos[0], cy_ - last_pos[1]) if last_pos is not None else 1e9
        # Use kal_spd to dynamically expand the "near" radius for fast targets
        speed_allowance = kal_spd * lost_age  # px expected to travel
        near = dist <= 2.0 * max(size) + 200 * lost_age + speed_allowance * 1.5   # much wider for fast/distant drops
        if lost_age < 2.0 and not near and dist > 3.0 * max(size) + 400 * lost_age + speed_allowance * 2.0:
            continue                                               # too far to be the same object so soon
        hsim = cv2.compareHist(trk_ref, patch_hist(img, x, y, w, h), cv2.HISTCMP_CORREL) if trk_ref is not None else 1.0
        
        # Loosen thresholds heavily if the target is known to be moving fast (motion blur)
        blur_allowance = 0.15 if kal_spd > 50 else 0.0
        
        if near:                                                   # right where it vanished: continuity is strong evidence
            if mx < max(0.30, args.reacq - 0.30 - blur_allowance) or hsim < max(0.2, args.verify - 0.20 - blur_allowance): continue
        else:                                                      # somewhere else: must be CLEARLY my target
            if mx < args.reacq - blur_allowance or hsim < args.verify - blur_allowance: continue
            if mx2 > mx - 0.06 and mx < 0.80: continue             # a look-alike scores the same: not sure -> don't guess
        score = mx - dist / 600.0 + (0.1 if near else 0.0)
        if score > best_score:
            best, best_score = (x, y, w, h), score
    return best

# ---- target selection / lock ----
lock_pos = None          # last known centre of the locked target
lock_miss = 0
selected = False         # True once the user clicked (used with --select)
click = None

def on_mouse(event, x, y, flags, param):
    global click, drag, roi_req
    if args.track:                                   # drag a box (or just click = default-size box)
        if event == cv2.EVENT_LBUTTONDOWN:
            drag = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and drag is not None:
            drag[2], drag[3] = x, y
        elif event == cv2.EVENT_LBUTTONUP and drag is not None:
            x0, y0, x1, y1 = drag; drag = None
            if abs(x1 - x0) < 8 or abs(y1 - y0) < 8:
                hb = args.box // 2
                roi_req = (x - hb, y - hb, 2 * hb, 2 * hb)
            else:
                roi_req = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        click = (x, y)

def learn_range(hsv_img, x, y):
    """Build a narrow colour range from the pixels around the clicked point."""
    h_, w_ = hsv_img.shape[:2]
    patch = hsv_img[max(0, y-4):min(h_, y+5), max(0, x-4):min(w_, x+5)].reshape(-1, 3)
    h, s, v = [int(t) for t in np.median(patch, axis=0)]
    s_lo, v_lo = max(50, s - 70), max(40, v - 90)
    lo_h, hi_h = h - 12, h + 12
    if lo_h < 0:
        return [((0, s_lo, v_lo), (hi_h, 255, 255)), ((180 + lo_h, s_lo, v_lo), (179, 255, 255))]
    if hi_h > 179:
        return [((lo_h, s_lo, v_lo), (179, 255, 255)), ((0, s_lo, v_lo), (hi_h - 180, 255, 255))]
    return [((lo_h, s_lo, v_lo), (hi_h, 255, 255))]

def centre(cd):
    return ((cd[1] + cd[3]) / 2, (cd[2] + cd[4]) / 2)

lock_area = None

def choose(cands, click_pt=None):
    """Pick ONE candidate and stay on it.
    - A click picks the biggest candidate box that contains the click (the whole object, not a fragment).
    - Once locked, only candidates of a similar size near the last position are accepted,
      so the lock cannot jump to a small fragment or to a big blue wall."""
    global lock_pos, lock_miss, selected, lock_area
    if click_pt is not None and cands:
        px, py = click_pt
        inside = [cd for cd in cands if cd[1] <= px <= cd[3] and cd[2] <= py <= cd[4]]
        pool = inside if inside else cands
        pick = max(inside, key=lambda cd: cd[0]) if inside else min(
            pool, key=lambda cd: np.hypot(centre(cd)[0] - px, centre(cd)[1] - py))
        lock_pos, lock_miss, lock_area, selected = centre(pick), 0, pick[0], True
        return pick
    if args.select and not selected:
        return None
    if lock_pos is not None:
        ok = [cd for cd in cands if lock_area is None or 0.35 * lock_area <= cd[0] <= 3.0 * lock_area]
        if ok:
            near = min(ok, key=lambda cd: np.hypot(centre(cd)[0] - lock_pos[0], centre(cd)[1] - lock_pos[1]))
            cx_, cy_ = centre(near)
            if np.hypot(cx_ - lock_pos[0], cy_ - lock_pos[1]) <= args.max_jump:
                lock_pos, lock_miss = (cx_, cy_), 0
                lock_area = near[0] if lock_area is None else 0.7 * lock_area + 0.3 * near[0]
                return near
        lock_miss += 1
        if lock_miss > args.patience:                       # target really gone
            lock_pos, lock_miss, lock_area = None, 0, None
            if args.select:
                selected = False                           # wait for a new click
        return None
    if not cands:
        return None
    pick = max(cands, key=lambda cd: cd[0])                 # first lock: largest candidate
    lock_pos, lock_miss, lock_area = centre(pick), 0, pick[0]
    return pick

go_home()
cv2.namedWindow("turret")
cv2.setMouseCallback("turret", on_mouse)
print("Running. q=quit  h=home  a=toggle arm  c=clear target. Armed:", armed)
if args.select:
    print("Click the target in the video window to select it.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005); continue
        H, W = frame.shape[:2]

        cands = []
        track_best = None
        if args.track:
            now_k = time.time(); dt_k = min(0.2, now_k - kal_t); kal_t = now_k
            if roi_req is not None:                                  # new target chosen by the user
                rx, ry, rw, rh = roi_req; roi_req = None
                rx, ry = clamp(rx, 0, W - 10), clamp(ry, 0, H - 10)
                rw, rh = clamp(rw, 10, W - rx), clamp(rh, 10, H - ry)
                trk = make_tracker(args.track)
                trk.init(frame, (int(rx), int(ry), int(rw), int(rh)))
                trk_area, trk_lost, trk_size = float(rw * rh), 0, (rw, rh)
                trk_ref, trk_bad = patch_hist(frame, rx, ry, rw, rh), 0
                trk_last = (rx + rw / 2, ry + rh / 2)
                kal.reset(*trk_last); last_ctrl = time.time(); sm_cx = sm_cy = None
                del gallery[:]; add_template(frame, rx, ry, rw, rh); tmpl_t = time.time(); lost_t = None
                kal_a.ready = False; trk_reinit_t = 0.0
                track_best = (trk_area, rx, ry, rx + rw, ry + rh, 1.0)
                print("Tracking started on box", (int(rx), int(ry), int(rw), int(rh)))
            else:
                found = None
                sim = ncc_v = 0.0
                if trk is not None:
                    good, bx = trk.update(frame)
                    if good:
                        bx_, by_, bw_, bh_ = bx
                        c_ = (bx_ + bw_ / 2, by_ + bh_ / 2)
                        area_ = float(bw_ * bh_)
                        # Speed from Kalman (lags on sudden moves) + raw displacement this frame for instant response
                        spd_kal = float(np.hypot(*np.asarray(kal.kf.statePost, np.float64).reshape(-1)[2:4]))
                        raw_disp = np.hypot(c_[0] - trk_last[0], c_[1] - trk_last[1]) / max(dt_k, 0.01)
                        spd = max(spd_kal, raw_disp * 0.5)   # take the larger: immediate or smoothed
                        # Gate: generous for fast targets — CSRT can report correct large jumps
                        gate = 3.5 * max(trk_size) + 8.0 * spd * dt_k + 50
                        # Allow 0.10x-8x area change to handle full depth range
                        good = (bw_ > 4 and bh_ > 4 and 0.10 * trk_area <= area_ <= 8.0 * trk_area
                                and np.hypot(c_[0] - trk_last[0], c_[1] - trk_last[1]) <= gate)
                        if good and args.verify > 0:                  # is it STILL my object?
                            sim = cv2.compareHist(trk_ref, patch_hist(frame, bx_, by_, bw_, bh_), cv2.HISTCMP_CORREL)
                            # Only use NCC when we have 2+ templates (1 template = not reliable enough)
                            ncc_v = box_ncc(frame, bx_, by_, bw_, bh_) if len(gallery) >= 2 else 1.0
                            # Need 4 consecutive bad frames to drop lock (ESP32-CAM JPEG is lossy/blurry)
                            trk_bad = trk_bad + 1 if (sim < args.verify and ncc_v < 0.25) else max(0, trk_bad - 1)
                            if trk_bad >= 4:
                                good = False
                            elif sim > 0.75:
                                trk_ref = 0.92 * trk_ref + 0.08 * patch_hist(frame, bx_, by_, bw_, bh_)
                    if good:
                        found = (bx_, by_, bw_, bh_)
                if found is None and gallery:                        # lost / wrong object: search everywhere for MY target
                    if lost_t is None: lost_t = now_k
                    # Get Kalman velocity to decide search position and cooldown
                    kal_state = np.asarray(kal.kf.statePost, np.float64).reshape(-1) if kal.ready else np.zeros(4)
                    kal_spd = float(np.hypot(kal_state[2], kal_state[3]))   # px/s
                    # Predict where target is NOW (not where it was): search at predicted position
                    pred_dt = min(0.15, now_k - lost_t) if lost_t is not None else 0.0
                    pred_pos = (trk_last[0] + kal_state[2] * pred_dt,
                                trk_last[1] + kal_state[3] * pred_dt) if trk_last is not None else trk_last
                    # Cooldown: 0.4s when slow (prevent thrash), 0.0s when fast (snap back instantly)
                    cooldown = 0.0 if kal_spd > 50 else 0.4
                    if now_k - trk_reinit_t >= cooldown:
                        r_ = find_target(frame, trk_size, pred_pos, now_k - lost_t, kal_spd)
                        if r_ is not None:
                            found = r_
                            trk = make_tracker(args.track)
                            trk.init(frame, tuple(int(v) for v in found))
                            trk_bad = 0; trk_reinit_t = now_k
                            print("Target found again.")
                if found is not None:
                    bx_, by_, bw_, bh_ = found
                    c_ = (bx_ + bw_ / 2, by_ + bh_ / 2)
                    area_ = float(bw_ * bh_)
                    trk_lost, trk_last, lost_t = 0, c_, None
                    # Smooth size tracking so the area gate adapts as target moves to/from camera
                    trk_area = 0.85 * trk_area + 0.15 * area_
                    trk_size = (0.85 * trk_size[0] + 0.15 * bw_, 0.85 * trk_size[1] + 0.15 * bh_)
                    kal.step(c_, dt_k)
                    track_best = (area_, bx_, by_, bx_ + bw_, by_ + bh_, 1.0)
                    # Save a fresh template more often (every 1s) and also when target is small (far away)
                    if (sim > 0.80 and ncc_v > 0.55 and now_k - tmpl_t > 1.0) or \
                       (ncc_v > 0.4 and now_k - tmpl_t > 2.5):   # distance template even at lower confidence
                        add_template(frame, bx_, by_, bw_, bh_); tmpl_t = now_k       # keep a fresh picture of the target
                elif gallery:
                    trk_lost += 1
                    px_, py_, _, _ = kal.step(None, dt_k)            # coast on the prediction for a few frames
                    if trk_lost <= 20:                                # extended coast: keep predicted box visible longer
                        track_best = (trk_area, px_ - trk_size[0] / 2, py_ - trk_size[1] / 2,
                                      px_ + trk_size[0] / 2, py_ + trk_size[1] / 2, 0.3)
                    if trk_lost == args.patience + 1:
                        print("Target not visible - still searching for it. Drag a new box (or press c) to choose another.")
            if drag is not None:
                cv2.rectangle(frame, (drag[0], drag[1]), (drag[2], drag[3]), (0, 255, 255), 1)
        elif ranges is None:
            res = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            for b in res.boxes:
                if args.show_all:
                    ax1, ay1, ax2, ay2 = [int(v) for v in b.xyxy[0].tolist()]
                    cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (255, 128, 0), 1)
                    cv2.putText(frame, f"{names[int(b.cls)]} {float(b.conf):.2f}", (ax1, max(12, ay1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
                if names[int(b.cls)] in want:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cands.append(((x2 - x1) * (y2 - y1), x1, y1, x2, y2, float(b.conf)))
        else:
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
            if click is not None:                                  # learn the colour of what you clicked
                ranges = learn_range(hsv, click[0], click[1])
                print("Learned colour range from click:", ranges)
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, lo, hi)
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            if args.show_mask:
                cv2.imshow("colour mask", mask)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < args.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                fill = area / float(w * h)          # round blobs fill ~0.75 of their box
                if not args.select and (fill < 0.45 or not (0.4 < w / float(h) < 2.5)):
                    continue                          # auto mode: skip thin / odd shapes
                cands.append((area, x, y, x + w, y + h, min(1.0, fill)))

        if args.track:
            best = track_best
            if trk is None:
                cv2.putText(frame, "DRAG A BOX AROUND THE TARGET", (10, H - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            click_pt, click = click, None
            if click_pt is not None:
                sm_cx = sm_cy = None
            best = choose(cands, click_pt)
            if click_pt is not None and best is None:
                print("No object found where you clicked - click on the middle of the coloured object.")
        for cd in cands:                               # everything that is NOT the chosen target: thin grey box
            if cd is not best:
                cv2.rectangle(frame, (int(cd[1]), int(cd[2])), (int(cd[3]), int(cd[4])), (150, 150, 150), 1)
        if args.select and not selected and not args.track:
            cv2.putText(frame, "CLICK THE TARGET TO SELECT IT", (10, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        fire = False
        hits = hits + 1 if best else 0
        if best and hits >= args.min_hits:
            _, x1, y1, x2, y2, c = best
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            settling = time.time() < settle_until
            if args.track:
                pass                                    # raw box centre; filtering is done on absolute angles below
            elif not settling:
                if sm_cx is None:
                    sm_cx, sm_cy = cx, cy
                else:                                   # smooth out detector jitter
                    sm_cx += args.smooth * (cx - sm_cx)
                    sm_cy += args.smooth * (cy - sm_cy)
                cx, cy = sm_cx, sm_cy
            err_pan = (cx - W / 2) / W * args.fov_h        # +ve = target right of centre
            err_tilt = (H / 2 - cy) / H * args.fov_v       # +ve = target above centre
            if args.invert_pan:  err_pan = -err_pan
            if args.invert_tilt: err_tilt = -err_tilt
            last_seen = time.time()

            if not settling:
                now_c = time.time()
                dt = min(0.25, now_c - last_ctrl)
                last_ctrl = now_c
                moved = False
                if args.track:
                    # 1) where is the target in the WORLD? = where the head pointed when this picture was taken + the
                    #    error seen in the picture. That removes the camera delay, so a high gain is stable.
                    if delay_est is None:
                        delay_est = args.latency + args.head_lag
                    meas_log.append((now_c, err_pan, err_tilt))
                    if len(meas_log) > 200: del meas_log[:100]
                    if not args.no_auto_delay and now_c - last_delay_fit > 0.3:
                        last_delay_fit = now_c
                        d_new = estimate_delay(now_c)
                        if d_new is not None:
                            delay_est = 0.5 * delay_est + 0.5 * d_new       # change fairly quickly, but not in one jump
                            if abs(delay_est - printed_delay) > 0.03:
                                printed_delay = delay_est
                                print(f"Camera delay learned: {delay_est:.2f} s  (use --latency {max(0.0, delay_est - args.head_lag):.2f} to start with it)")
                    t_cap = now_c - delay_est   # the head's real angle then = what we commanded a little earlier
                    p0, t0 = angles_at(t_cap)
                    ap_m, at_m = p0 + err_pan, t0 + err_tilt
                    if not kal_a.ready:
                        kal_a.reset(ap_m, at_m)
                    ap_e, at_e, vp_e, vt_e = kal_a.step((ap_m, at_m) if c > 0.5 else None, dt)
                    vp_e, vt_e = clamp(vp_e, -150, 150), clamp(vt_e, -150, 150)
                    # ignore small speeds: hand tremor / detector jitter would otherwise be amplified into wobble
                    vp_e = float(np.sign(vp_e)) * max(0.0, abs(vp_e) - args.still)
                    vt_e = float(np.sign(vt_e)) * max(0.0, abs(vt_e) - args.still)
                    if c <= 0.5: vp_e, vt_e = 0.5 * vp_e, 0.5 * vt_e
                    # 2) aim where it will be when the head arrives
                    hz = min(0.3, delay_est + args.lead)
                    cmd_pan = ap_e + clamp(vp_e * hz, -40, 40)
                    cmd_tilt = at_e + clamp(vt_e * hz, -40, 40)
                    # 3) move there: proportional pull + the target's own speed as feed-forward.
                    #    Errors smaller than --hold are ignored entirely, so a still target means a still head.
                    lim = args.max_rate * dt
                    dpan, dtilt = cmd_pan - pan, cmd_tilt - tilt
                    dpan = float(np.sign(dpan)) * max(0.0, abs(dpan) - args.hold)
                    dtilt = float(np.sign(dtilt)) * max(0.0, abs(dtilt) - args.hold)
                    if dpan != 0.0 or vp_e != 0.0:
                        pan = clamp(pan + clamp(args.kp * dpan * dt + vp_e * dt, -lim, lim), PAN_MIN, PAN_MAX)
                    if dtilt != 0.0 or vt_e != 0.0:
                        tilt = clamp(tilt + clamp(args.kp * dtilt * dt + vt_e * dt, -lim, lim), TILT_MIN, TILT_MAX)
                    # cyan dot = where the head is aiming, drawn in the picture
                    sp = -1 if args.invert_pan else 1
                    st = -1 if args.invert_tilt else 1
                    dot_x = W / 2 + sp * (cmd_pan - pan) / args.fov_h * W
                    dot_y = H / 2 - st * (cmd_tilt - tilt) / args.fov_v * H
                    cv2.circle(frame, (int(clamp(dot_x, 0, W - 1)), int(clamp(dot_y, 0, H - 1))), 5, (255, 255, 0), -1)
                elif args.settle > 0:                       # old stop-and-go mode
                    if abs(err_pan) > args.deadband:
                        pan = clamp(pan + clamp(args.gain * err_pan, -args.max_step, args.max_step), PAN_MIN, PAN_MAX); moved = True
                    if abs(err_tilt) > args.deadband:
                        tilt = clamp(tilt + clamp(args.gain * err_tilt, -args.max_step, args.max_step), TILT_MIN, TILT_MAX); moved = True
                    if moved:
                        settle_until = now_c + args.settle
                        sm_cx = sm_cy = None
                else:                                     # smooth mode: continuous small steps, speed ~ error
                    lim = args.max_rate * dt
                    if abs(err_pan) > args.deadband:
                        pan = clamp(pan + clamp(args.kp * err_pan * dt, -lim, lim), PAN_MIN, PAN_MAX)
                    if abs(err_tilt) > args.deadband:
                        tilt = clamp(tilt + clamp(args.kp * err_tilt * dt, -lim, lim), TILT_MIN, TILT_MAX)

            centered = abs(err_pan) < 2 * args.deadband and abs(err_tilt) < 2 * args.deadband
            locked_count = locked_count + 1 if centered else 0
            fire = armed and locked_count >= args.lock_frames and not settling

            color = (0, 255, 0) if locked_count >= args.lock_frames else (0, 200, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"LOCKED {c:.2f}", (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            locked_count = 0
            if args.track and gallery and lost_t is not None:
                age_l = time.time() - lost_t
                dtc = min(0.1, time.time() - last_coast)
                if age_l < 0.8 and kal_a.ready:        # it just left: keep swinging the way it was going, fading out
                    vp_c, vt_c = [float(v) for v in np.asarray(kal_a.kf.statePost, np.float64).reshape(-1)[2:4]]
                    fade = 1.0 - age_l / 0.8
                    pan = clamp(pan + clamp(vp_c, -150, 150) * fade * dtc, PAN_MIN, PAN_MAX)
                    tilt = clamp(tilt + clamp(vt_c, -150, 150) * fade * dtc, TILT_MIN, TILT_MAX)
                elif age_l > 1.5:
                    kal_a.ready = False
                cv2.putText(frame, "SEARCHING FOR YOUR TARGET...", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
                if trk_last is not None:
                    cv2.circle(frame, (int(trk_last[0]), int(trk_last[1])), 14, (0, 200, 255), 1)   # where it was last seen
            elif not args.track:
                kal_a.ready = False
            last_coast = time.time()
            last_ctrl = time.time()          # no target: restart the control clock so the first step is small
            if not best:
                sm_cx = sm_cy = None
            if last_seen and time.time() - last_seen > (args.home_after if (args.track and gallery) else 3.0):    # lost for a while -> return home
                go_home(); last_seen = 0.0

        hist.append((time.time(), pan, tilt))
        if len(hist) > 200: del hist[:100]

        # Re-send at ~10 Hz. The Uno turns the laser off if it hears nothing for 0.7 s.
        # Only send when something really changed, plus a keep-alive every 0.3 s.
        # (Sending identical commands 10x/s is harmless, but tiny jitter moves are what shake the motors.)
        now = time.time()
        changed = (sent[0] is None or abs(pan - sent[0]) >= args.min_step
                   or abs(tilt - sent[1]) >= args.min_step or fire != sent[2])
        if changed or now - last_send > 0.3:
            if changed or sent[0] is None:
                sent = (pan, tilt, fire)
            send(sent[0], sent[1], sent[2])
            last_send = now
            ser.reset_input_buffer()

        cv2.line(frame, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)
        cv2.line(frame, (0, H // 2), (W, H // 2), (255, 255, 255), 1)
        cv2.putText(frame, "settling" if time.time() < settle_until else "", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, f"pan {pan:.1f}  tilt {tilt:.1f}  {'ARMED' if armed else 'SAFE'}  {'FIRE' if fire else ''}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if armed else (0, 255, 0), 2)
        if args.track and delay_est is not None:
            cv2.putText(frame, f"delay {delay_est:.2f}s", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1) if trk is not None else None
        t_now = time.time(); fps_ema = 0.9 * fps_ema + 0.1 / max(1e-3, t_now - t_prev); t_prev = t_now
        cv2.putText(frame, f"{fps_ema:.0f} fps", (W - 80, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("turret", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("h"): go_home()
        if k == ord("c"):
            lock_pos, lock_miss, selected, sm_cx, sm_cy = None, 0, False, None, None
            trk = None; del gallery[:]; lost_t = None; trk_reinit_t = 0.0
            print("Target cleared")
        if k == ord("a"):
            armed = not armed
            print("Armed:", armed)
finally:
    try:
        ser.write(b"X\n" if args.release_on_exit else b"H\n")   # default: laser off, go home, keep holding
        ser.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()