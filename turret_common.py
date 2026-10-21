"""Shared building blocks for collect_balloons.py and hunter.py (camera sources, Kalman filter, trackers)."""
import os, time, threading, urllib.request, urllib.parse
# Ask OpenCV's FFmpeg reader NOT to buffer/probe the network stream (set before cv2 is imported)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0")
import cv2, serial
import numpy as np

clamp = lambda v, lo, hi: max(lo, min(hi, v))

FIX_CV2 = ("\nOpenCV is installed incorrectly (several OpenCV packages are mixed up). Fix, in a terminal:\n"
           "  pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless\n"
           "  pip install \"opencv-contrib-python==4.11.0.86\"\n"
           "If this message still appears, delete the leftover folder  ...\\site-packages\\cv2  and run the last line again.")
if not hasattr(cv2, "imdecode") or not hasattr(cv2, "KalmanFilter"):
    raise SystemExit(FIX_CV2)

# ---------------------------------------------------------------- camera sources
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


# ESP32-CAM default settings, applied every time a program starts (WiFi stream only)
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

def apply_cam_settings(source, cam_set=None, cam_port=80, enabled=True):
    if not enabled or not str(source).startswith("http"):
        return
    host = urllib.parse.urlparse(source).hostname
    settings = dict(CAM_DEFAULTS)
    if cam_set:
        for pair in cam_set.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                settings[k.strip()] = int(v)
    ok_n = 0
    for k, v in settings.items():
        try:
            urllib.request.urlopen(f"http://{host}:{cam_port}/control?var={k}&val={v}", timeout=2).read()
            ok_n += 1
        except Exception:
            pass
    print(f"Camera settings applied: {ok_n}/{len(settings)} (host {host}:{cam_port})")
    time.sleep(0.5)

def open_camera(source, cam_baud=2000000, cam_set=None, cam_port=80, cam_setup=True):
    """source: webcam index ('0'), stream URL, or 'serial:COM5' (ESP32-CAM on a USB-TTL cable)."""
    src = int(source) if str(source).isdigit() else source
    if str(src).lower().startswith("serial:"):
        cap = SerialCam(str(src)[7:], cam_baud)
        print("Serial camera on", str(src)[7:], "at", cam_baud, "baud (camera settings come from the sketch)")
    else:
        apply_cam_settings(src, cam_set, cam_port, cam_setup)
        cap = LatestFrame(src)
    if not cap.cap.isOpened():
        raise SystemExit("Cannot open camera source")
    return cap

# ---------------------------------------------------------------- tracking helpers
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

def make_tracker(name):
    if name == "csrt":
        try:                                       # less eager to give up, wider search area (survives blur / quick moves)
            prm = cv2.TrackerCSRT_Params() if hasattr(cv2, "TrackerCSRT_Params") else cv2.legacy.TrackerCSRT_Params()
            prm.psr_threshold = 0.02
            prm.padding = 3.5
            return (cv2.TrackerCSRT_create if hasattr(cv2, "TrackerCSRT_create") else cv2.legacy.TrackerCSRT_create)(prm)
        except Exception:
            pass
    wanted = {"csrt": "TrackerCSRT_create", "kcf": "TrackerKCF_create", "mosse": "TrackerMOSSE_create"}[name]
    for mod in (cv2, getattr(cv2, "legacy", None)):
        f = getattr(mod, wanted, None) if mod is not None else None
        if f is not None:
            return f()
    raise SystemExit("This OpenCV has no '%s' tracker (OpenCV 5.x dropped them)." % name + FIX_CV2)

def patch_hist(img, x, y, w, h):
    """Colour fingerprint of the middle of a box (used to notice a tracker drifting onto something else)."""
    H_, W_ = img.shape[:2]
    mx, my = int(w * 0.15), int(h * 0.15)
    x0, y0 = int(clamp(x + mx, 0, W_ - 2)), int(clamp(y + my, 0, H_ - 2))
    x1, y1 = int(clamp(x + w - mx, x0 + 1, W_)), int(clamp(y + h - my, y0 + 1, H_))
    hs = cv2.calcHist([img[y0:y1, x0:x1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hs, hs).flatten()
