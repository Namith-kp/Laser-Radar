"""
Voice / chat controlled turret.  Tell it what to aim at; it finds it, points at it, tracks it, and (only if armed and allowed) fires.

    python voice_turret.py --source http://192.168.1.114:81/stream --port COM7 --invert-pan --invert-tilt
    python voice_turret.py --source serial:COM5 --port COM7 --latency 0.08 --voice --llm claude
    python voice_turret.py --source 0 --no-turret            # test the brain with a webcam and no motors

What you can say or type (the terminal is the chat box; press V in the video window to talk):
    "scan the room" / "what do you see"         -> sweeps, builds a map of every object it recognised, lists them
    "aim at the chair" / "point at the table on the left" / "look at the red bottle" / "follow the biggest cup"
    "fire" / "fire at the chair"                -> laser pulse, ONLY if armed (key A) and the target is allowed
    "stop" / "home" / "unlock"

Pieces
    detector   YOLOv8 (80 everyday classes, GPU)  - or --detector world: YOLO-World, any word you say ("fire extinguisher")
               + your balloon.pt (if present) so balloons are found too
    map        every object is remembered as an absolute angle, so "aim at the chair" works even when it is out of view
    brain      rules first (instant, offline); if the sentence isn't understood the LLM (--llm claude | ollama) turns it into
               a JSON command. The LLM can never arm the laser or skip any safety check - the code validates everything.
    voice      --voice: faster-whisper on the GPU (push-to-talk with V), optional spoken answers --speak
    aiming     same lag-compensated Kalman controller as hunter.py, with --laser-off-pan/--laser-off-tilt for the
               distance between the camera and the laser

LASER SAFETY (all on by default)
    * Nothing fires unless armed: pass --arm or press A in the window. Voice/LLM can NOT arm it.
    * It only fires at classes you list in --fire-classes (default: balloon only). People and animals can never be fired at.
      Shiny/glass things (tv, laptop, phone, bottle, mirror, window, clock ...) are refused too - a 250 mW beam reflects.
    * Fires only when locked on the target (inside --lock-tol for --lock-frames frames), only a short pulse (--pulse),
      then a cool-down. Laser is forced off the instant a person/animal is visible, the target is lost, or you say stop.
    * No-fire zone: pan beyond --fire-pan, tilt below --fire-tilt-min. The Uno still enforces 0.2 s failsafe + max burn.
    * Furniture, curtains, paper and fabric can SCORCH or ignite under a 250 mW beam. Wear OD 4+ goggles. Keep a kill switch.
"""
import argparse, time, re, json, queue, threading, sys, urllib.request
import cv2
import numpy as np
from turret_common import open_camera, Kalman2D, clamp

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0")
ap.add_argument("--port")
ap.add_argument("--no-turret", action="store_true")
ap.add_argument("--detector", choices=["coco", "world"], default="coco")
ap.add_argument("--coco-model", default="yolov8s.pt")
ap.add_argument("--world-model", default="yolov8s-worldv2.pt")
ap.add_argument("--balloon-model", default="balloon.pt")
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--imgsz", type=int, default=480)
ap.add_argument("--device", default="auto")
ap.add_argument("--llm", choices=["none", "claude", "ollama"], default="none")
ap.add_argument("--llm-model", default="")
ap.add_argument("--voice", action="store_true")
ap.add_argument("--whisper", default="base.en")
ap.add_argument("--speak", action="store_true")
ap.add_argument("--arm", action="store_true")
ap.add_argument("--fire-classes", nargs="*", default=["balloon"])
ap.add_argument("--pulse", type=float, default=1.0, help="max seconds of laser per fire command")
ap.add_argument("--lock-tol", type=float, default=1.5)
ap.add_argument("--lock-frames", type=int, default=4)
ap.add_argument("--fire-pan", type=float, default=80.0)
ap.add_argument("--fire-tilt-min", type=float, default=-5.0)
ap.add_argument("--fire-tilt-max", type=float, default=45.0)
ap.add_argument("--laser-off-pan", type=float, default=0.0, help="deg: where the laser dot lands relative to the picture centre (+ = right)")
ap.add_argument("--laser-off-tilt", type=float, default=0.0, help="deg (+ = up)")
ap.add_argument("--fov-h", type=float, default=52.0)
ap.add_argument("--fov-v", type=float, default=39.0)
ap.add_argument("--invert-pan", action="store_true")
ap.add_argument("--invert-tilt", action="store_true")
ap.add_argument("--scan-range", type=float, default=75.0)
ap.add_argument("--scan-tilts", default="0,25")
ap.add_argument("--scan-speed", type=float, default=30.0)
ap.add_argument("--kp", type=float, default=5.0)
ap.add_argument("--max-rate", type=float, default=220.0)
ap.add_argument("--still", type=float, default=12.0)
ap.add_argument("--hold", type=float, default=1.0)
ap.add_argument("--latency", type=float, default=0.12)
ap.add_argument("--head-lag", type=float, default=0.06)
ap.add_argument("--lead", type=float, default=0.05)
ap.add_argument("--no-auto-delay", action="store_true")
ap.add_argument("--cam-baud", type=int, default=2000000)
ap.add_argument("--no-cam-setup", action="store_true")
ap.add_argument("--cam-set")
ap.add_argument("--cam-port", type=int, default=80)
args = ap.parse_args()
if not args.no_turret and not args.port:
    raise SystemExit("Give --port COM7 (or --no-turret to test without motors)")

PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX = -90, 90, -30, 45      # keep equal to the Arduino limits

LIVING = {"person", "cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "teddy bear"}
SHINY = {"tv", "laptop", "cell phone", "mirror", "window", "clock", "microwave", "oven", "refrigerator", "bottle",
         "wine glass", "vase", "knife", "scissors", "car", "monitor", "screen", "glass", "toaster", "sink", "remote"}
NEVER_FIRE = LIVING | SHINY
SYN = {"table": "dining table", "desk": "dining table", "sofa": "couch", "television": "tv", "monitor": "tv", "screen": "tv",
       "phone": "cell phone", "mobile": "cell phone", "smartphone": "cell phone", "plant": "potted plant", "bike": "bicycle",
       "mug": "cup", "computer": "laptop", "notebook": "laptop", "ball": "sports ball", "human": "person", "people": "person",
       "man": "person", "woman": "person", "kid": "person", "child": "person", "doggy": "dog", "kitty": "cat", "teddy": "teddy bear",
       "balloons": "balloon", "bin": "trash can", "trashcan": "trash can", "bag": "backpack", "lamp": "lamp", "stool": "chair"}
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "gray"]
QUALS = {"left": "left", "right": "right", "top": "top", "upper": "top", "highest": "top", "bottom": "bottom", "lower": "bottom",
         "lowest": "bottom", "nearest": "nearest", "closest": "nearest", "biggest": "biggest", "largest": "biggest",
         "smallest": "smallest", "tiniest": "smallest", "middle": "center", "center": "center", "centre": "center"}


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
    m = speed * dt
    pan = clamp(pan + clamp(tp - pan, -m, m), PAN_MIN, PAN_MAX)
    tilt = clamp(tilt + clamp(tt - tilt, -m, m), TILT_MIN, TILT_MAX)
    return pan, tilt, abs(tp - pan) < 1.0 and abs(tt - tilt) < 1.0

def new_kal(): return Kalman2D(q_pos=0.05, q_vel=60, r=3.0)


# ------------------------------------------------------------------ detector (everyday objects + balloons)
try:
    from ultralytics import YOLO
except Exception:
    raise SystemExit("Install the detector library first:  pip install ultralytics")
try:
    import torch
    _cuda = torch.cuda.is_available()
except Exception:
    torch, _cuda = None, False
dev = (0 if _cuda else "cpu") if args.device == "auto" else (int(args.device) if str(args.device).isdigit() else args.device)
use_half = dev != "cpu"
print("Detector on", torch.cuda.get_device_name(dev) if dev != "cpu" else "the CPU (install CUDA PyTorch for the GPU)")

DEFAULT_VOCAB = ["person", "chair", "table", "couch", "bed", "tv", "laptop", "keyboard", "mouse", "monitor", "bottle", "cup", "book",
                 "clock", "vase", "potted plant", "backpack", "umbrella", "door", "window", "lamp", "fan", "box", "bag", "shoe",
                 "toy", "pillow", "remote", "cell phone", "cat", "dog", "bird", "fire extinguisher", "trash can", "shelf", "curtain"]

class Detector:
    def __init__(s):
        s.world = args.detector == "world"
        s.main = YOLO(args.world_model if s.world else args.coco_model)
        s.vocab = list(DEFAULT_VOCAB)
        if s.world: s.main.set_classes(s.vocab)
        s.names = s.vocab if s.world else list(s.main.names.values())
        s.balloon = None
        try:
            import os
            if os.path.exists(args.balloon_model):
                s.balloon = YOLO(args.balloon_model); print("Balloon model loaded.")
                if "balloon" not in s.names: s.names = list(s.names) + ["balloon"]
        except Exception as e:
            print("Balloon model not loaded:", e)
    def add_class(s, label):
        if s.world and label not in s.vocab:
            s.vocab.append(label); s.main.set_classes(s.vocab); s.names = s.vocab + (["balloon"] if s.balloon is not None and "balloon" not in s.vocab else [])
    def run(s, model, frame, imgsz, conf, rename=None):
        out = []
        res = model(frame, conf=conf, imgsz=imgsz, device=dev, half=use_half, verbose=False)[0]
        nm = model.names if not s.world or model is s.balloon else {i: v for i, v in enumerate(s.vocab)}
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            lab = rename or nm[int(b.cls)]
            out.append({"label": lab, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "w": x2 - x1, "h": y2 - y1, "conf": float(b.conf)})
        return out
    def detect(s, frame):
        d = s.run(s.main, frame, args.imgsz, args.conf)
        if s.balloon is not None:
            d = [x for x in d if x["label"] != "sports ball"] + s.run(s.balloon, frame, 320, 0.4, rename="balloon")
        return d

def colour_name(frame, d):
    H, W = frame.shape[:2]
    x1, x2 = int(max(0, d["cx"] - d["w"] / 4)), int(min(W, d["cx"] + d["w"] / 4))
    y1, y2 = int(max(0, d["cy"] - d["h"] / 4)), int(min(H, d["cy"] + d["h"] / 4))
    if x2 - x1 < 2 or y2 - y1 < 2: return ""
    h, sat, v = [float(np.median(c)) for c in cv2.split(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV))]
    if v < 55: return "black"
    if sat < 40: return "white" if v > 170 else "gray"
    if h < 8 or h > 165: return "red" if sat > 90 or h < 8 else "pink"
    if h < 20: return "orange"
    if h < 35: return "yellow"
    if h < 85: return "green"
    if h < 130: return "blue"
    if h < 150: return "purple"
    return "pink"


# ------------------------------------------------------------------ the room map (absolute angles)
class Obj:
    _n = 0
    def __init__(s, label, pan, tilt, w, h, col, now):
        Obj._n += 1
        s.id, s.label, s.pan, s.tilt, s.w, s.h, s.col, s.seen, s.hits, s.miss = Obj._n, label, pan, tilt, w, h, col, now, 1, 0

registry = []

def observe(label, pan, tilt, w, h, col, now):
    best, bd = None, max(8.0, 0.35 * max(w, h))
    for o in registry:
        if o.label != label: continue
        d = float(np.hypot(o.pan - pan, o.tilt - tilt))
        if d < bd: best, bd = o, d
    if best is None:
        o = Obj(label, pan, tilt, w, h, col, now); registry.append(o); return o
    best.pan += 0.5 * (pan - best.pan); best.tilt += 0.5 * (tilt - best.tilt)
    best.w, best.h, best.col, best.seen, best.hits, best.miss = w, h, col or best.col, now, best.hits + 1, 0
    return best

def side(pan): return pan * (-1.0 if args.invert_pan else 1.0)       # + = right in the picture
def up(tilt): return tilt * (-1.0 if args.invert_tilt else 1.0)      # + = up in the picture

def choose(items, qual, col, pan, tilt, getp):
    """items -> one item according to 'left/right/top/bottom/nearest/biggest/smallest/center' and colour."""
    if not items: return None
    if col:
        c = [i for i in items if getp(i)[4] == col]
        if c: items = c
    if qual == "left":     return min(items, key=lambda i: side(getp(i)[0]))
    if qual == "right":    return max(items, key=lambda i: side(getp(i)[0]))
    if qual == "top":      return max(items, key=lambda i: up(getp(i)[1]))
    if qual == "bottom":   return min(items, key=lambda i: up(getp(i)[1]))
    if qual == "biggest":  return max(items, key=lambda i: getp(i)[2] * getp(i)[3])
    if qual == "smallest": return min(items, key=lambda i: getp(i)[2] * getp(i)[3])
    return min(items, key=lambda i: np.hypot(getp(i)[0] - pan, getp(i)[1] - tilt))     # nearest / center / default


# ------------------------------------------------------------------ the brain: sentence -> command
det = Detector()
def known_labels():
    return sorted(set(det.names) | {"balloon"} | {o.label for o in registry})

def norm_label(t):
    t = t.strip().lower()
    if t in SYN: return SYN[t]
    if t.endswith("s") and t[:-1] in SYN: return SYN[t[:-1]]
    if t.endswith("es") and t[:-2] in SYN: return SYN[t[:-2]]
    for L in known_labels():
        if t == L or t == L + "s" or t == L + "es": return L
    return t

STOP_WORDS = set("""please can could would you the a an at to on toward towards that this those these it my me now and then just
 aim point look turn go target track follow find face fire shoot zap pop burn laser hit it's in of for with is there""".split())

def parse_rule(text):
    t = text.lower().strip(" .!?,")
    if not t: return None
    if re.search(r"\b(stop|cease|abort|cancel|laser off|hold fire|stand down)\b", t): return {"act": "stop"}
    if re.search(r"\b(scan|look around|map the room|search the room|search around|survey)\b", t): return {"act": "scan"}
    if re.search(r"\b(what do you see|what can you see|what.s (here|around)|list|inventory|what objects)\b", t): return {"act": "list"}
    if re.search(r"\b(unlock|release|forget (it|that)|drop (it|that))\b", t): return {"act": "unlock"}
    if re.search(r"\b(home|go back|reset|recenter|re-centre|re-center)\b", t) and not re.search(r"\b(at|to) the\b", t): return {"act": "home"}
    fire = bool(re.search(r"\b(fire|shoot|zap|pop|burn|hit)\b", t))
    aim = bool(re.search(r"\b(aim|point|look|turn|go|target|track|follow|find|face|show)\b", t))
    qual = next((v for k, v in QUALS.items() if re.search(rf"\b{k}\b", t)), None)
    col = next((c for c in COLOURS if re.search(rf"\b{c}\b", t)), None)
    words = [w for w in re.findall(r"[a-z']+", t)]
    # target: longest known label / synonym in the sentence
    cands = sorted(set(known_labels()) | set(SYN), key=len, reverse=True)
    label = None
    for L in cands:
        if re.search(rf"\b{re.escape(L)}(s|es)?\b", t): label = norm_label(L); break
    if label is None:
        rest = [w for w in words if w not in STOP_WORDS and w not in QUALS and w not in COLOURS]
        if rest and (aim or fire): label = norm_label(" ".join(rest[-2:]) if len(rest) > 1 else rest[0])
    if fire and label is None: return {"act": "fire"}
    if (aim or fire) and label: return {"act": "aim", "target": label, "qual": qual, "color": col, "then_fire": fire}
    if label and len(words) <= 4: return {"act": "aim", "target": label, "qual": qual, "color": col, "then_fire": False}
    return None

LLM_SYS = """You control a laser-pointer turret that has a camera. Turn the user's sentence into ONE JSON object, nothing else.
Schema: {"act": "aim|fire|stop|home|scan|list|unlock|say", "target": string|null, "qual": "left|right|top|bottom|nearest|biggest|smallest|center"|null,
"color": string|null, "then_fire": boolean, "reply": string}
- aim: point at an object. If the user also says to fire/shoot/pop/burn it, set then_fire true.  fire: fire at what is already locked.
- target = the object's common name in singular (chair, dining table, couch, tv, cup, balloon ...).
- stop = laser off / cancel. home = return to centre. scan = look around the room. list = say what has been seen.
- If the sentence is a question or chit-chat, use act "say" and put a short answer in reply. Never invent objects.
Objects recognised so far: {labels}"""

def llm_parse(text):
    labels = ", ".join(known_labels()[:80])
    sys_p = LLM_SYS.replace("{labels}", labels)
    try:
        if args.llm == "claude":
            import anthropic
            r = anthropic.Anthropic().messages.create(model=args.llm_model or "claude-haiku-4-5-20251001", max_tokens=200,
                                                      system=sys_p, messages=[{"role": "user", "content": text}])
            out = r.content[0].text
        else:
            body = json.dumps({"model": args.llm_model or "llama3.2", "stream": False, "format": "json",
                               "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": text}]}).encode()
            rq = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(rq, timeout=20).read())["message"]["content"]
        m = re.search(r"\{.*\}", out, re.S)
        c = json.loads(m.group(0))
    except Exception as e:
        print("(LLM unavailable:", str(e)[:80] + ")"); return None
    if c.get("act") not in ("aim", "fire", "stop", "home", "scan", "list", "unlock", "say"): return None   # validated here, not trusted
    c["target"] = norm_label(str(c["target"])) if c.get("target") else None
    c["qual"] = c.get("qual") if c.get("qual") in QUALS.values() else None
    c["color"] = c.get("color") if c.get("color") in COLOURS else None
    c["then_fire"] = bool(c.get("then_fire"))
    return c


# ------------------------------------------------------------------ talking: output, chat, voice
say_q = queue.Queue()
last_say, last_heard = ["Ready. Say or type a command."], [""]
def say(msg):
    print("TURRET:", msg); last_say[0] = msg
    if args.speak: say_q.put(msg)

if args.speak:
    def _tts():
        try:
            import pyttsx3
            eng = pyttsx3.init()
            while True:
                m = say_q.get(); eng.say(m); eng.runAndWait()
        except Exception as e:
            print("(speech output unavailable:", e, ")")
    threading.Thread(target=_tts, daemon=True).start()

text_q, cmd_q = queue.Queue(), queue.Queue()

def brain():
    while True:
        text = text_q.get(); last_heard[0] = text
        c = parse_rule(text)
        if c is None and args.llm != "none": c = llm_parse(text)
        if c is None: say("I didn't understand that. Try: aim at the chair, scan the room, fire, stop.")
        else: cmd_q.put(c)
threading.Thread(target=brain, daemon=True).start()

def stdin_loop():
    for line in sys.stdin:
        if line.strip(): text_q.put(line.strip())
threading.Thread(target=stdin_loop, daemon=True).start()

listen_evt = threading.Event()
if args.voice:
    def voice_loop():
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
        except Exception:
            print("Voice needs:  pip install faster-whisper sounddevice"); return
        try:
            wm = WhisperModel(args.whisper, device="cuda" if _cuda else "cpu", compute_type="float16" if _cuda else "int8")
        except Exception:
            wm = WhisperModel(args.whisper, device="cpu", compute_type="int8")
        prompt = "Aim at the chair. Point at the table. Fire. Stop. Scan the room. Go home. " + ", ".join(known_labels()[:40])
        print("Voice ready: press V in the video window, then speak.")
        while True:
            listen_evt.wait(); listen_evt.clear()
            say("Listening...")
            chunks, started, quiet, t0, pre = [], False, 0.0, time.time(), []
            with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=1600) as st:
                while time.time() - t0 < 10:
                    data, _ = st.read(1600); e = float(np.sqrt(np.mean(data ** 2)))
                    if e > 0.015: started, quiet = True, 0.0
                    elif started: quiet += 0.1
                    if started: chunks.extend(pre); pre = []; chunks.append(data.copy())
                    else: pre = (pre + [data.copy()])[-3:]
                    if started and quiet >= 0.8: break
                    if not started and time.time() - t0 > 4: break
            if not chunks: say("I heard nothing."); continue
            audio = np.concatenate(chunks)[:, 0]
            segs, _ = wm.transcribe(audio, language="en", beam_size=1, initial_prompt=prompt, vad_filter=True)
            txt = " ".join(s.text for s in segs).strip()
            if txt: print("HEARD:", txt); text_q.put(txt)
            else: say("I couldn't make that out.")
    threading.Thread(target=voice_loop, daemon=True).start()


# ------------------------------------------------------------------ main
cap = open_camera(args.source, args.cam_baud, args.cam_set, args.cam_port, not args.no_cam_setup)
ser = None
if not args.no_turret:
    import serial
    ser = serial.Serial(args.port, 115200, timeout=0.1)
    time.sleep(2.0); ser.reset_input_buffer()

def send(p, t, f):
    if ser: ser.write(f"P{p:.2f},T{t:.2f},F{int(f)}\n".encode())

pan = tilt = 0.0
prev_pan = prev_tilt = 0.0
armed = args.arm
aim = Aim()
sent, last_send = (None, None, None), 0.0
tilts = [float(v) for v in args.scan_tilts.split(",") if v.strip()] or [0.0]

mode = "IDLE"                      # IDLE | SCAN | SLEW (goto a point) | GOTO (goto a remembered object) | TRACK
spec = {}                          # what we are aiming at: target / qual / color
goto_pt, goto_obj, goto_t0 = (0.0, 0.0), None, 0.0
wp, wp_i = [], 0
pending = None                     # command to retry after an automatic scan
scanned_for = False
lost_since = None
lock_n = 0
fire_req_until, fire_t0, cool_until = 0.0, None, 0.0
person_until = 0.0
last_dets, last_frame = [], None
last_t, frame_i = time.time(), 0

def start_scan():
    global mode, wp, wp_i
    wp = [(0.0, tilts[0])]
    for i, t in enumerate(tilts):
        xs = [-args.scan_range, args.scan_range] if i % 2 == 0 else [args.scan_range, -args.scan_range]
        wp += [(x, t) for x in xs]
    wp += [(0.0, 0.0)]; wp_i = 0; mode = "SCAN"

def describe(o):
    where = ("right" if side(o.pan) > 10 else "left" if side(o.pan) < -10 else "centre")
    return f"{(o.col + ' ') if o.col else ''}{o.label} ({where}, pan {o.pan:.0f}, tilt {o.tilt:.0f})"

def valid_objs(label=None):
    return [o for o in registry if o.hits >= 3 and (label is None or o.label == label)]

def start_aim(c, retry=True):
    global mode, spec, goto_obj, goto_t0, pending, scanned_for, lost_since, lock_n, fire_req_until
    label = c.get("target")
    if not label: say("Aim at what?"); return
    if det.world: det.add_class(label)
    elif label not in det.names:
        say(f"This detector can't recognise '{label}'. Use --detector world, or one of: " + ", ".join(det.names[:25]) + " ...")
        return
    spec = {"target": label, "qual": c.get("qual"), "color": c.get("color"), "then_fire": bool(c.get("then_fire"))}
    live = [d for d in last_dets if d["label"] == label]
    reg = valid_objs(label)
    if live:
        for d in live: d["col"] = colour_name(last_frame, d)
        d = choose(live, spec["qual"], spec["color"], pan, tilt, lambda i: (i["wp"], i["wt"], i["w"], i["h"], i["col"]))
        aim.kal = new_kal(); aim.meas.clear(); aim.kal.reset(d["wp"], d["wt"])
        mode, lost_since, lock_n = "TRACK", None, 0
        fire_req_until = time.time() + 8.0 if spec["then_fire"] else 0.0
        say(f"Aiming at the {(d['col'] + ' ') if spec['color'] else ''}{label}.")
    elif reg:
        o = choose(reg, spec["qual"], spec["color"], pan, tilt, lambda i: (i.pan, i.tilt, i.w, i.h, i.col))
        goto_obj, goto_t0, mode = o, time.time(), "GOTO"
        say(f"Turning to the {label} I saw at pan {o.pan:.0f}.")
    elif retry:
        pending = c; start_scan(); say(f"I haven't seen a {label} yet - scanning the room.")
    else:
        say(f"I can't find a {label}.")

def execute(c):
    global mode, fire_req_until, pending, goto_pt, lost_since
    act = c.get("act")
    if act == "say": say(c.get("reply") or "OK."); return
    if act == "stop":
        fire_req_until = 0.0; pending = None; mode = "IDLE"; say("Stopped. Laser off."); return
    if act == "home":
        fire_req_until = 0.0; goto_pt = (0.0, 0.0); mode = "SLEW"; say("Going home."); return
    if act == "unlock":
        fire_req_until = 0.0; mode = "IDLE"; say("Target released."); return
    if act == "scan": start_scan(); say("Scanning the room."); return
    if act == "list":
        v = valid_objs()
        say("I have seen: " + "; ".join(describe(o) for o in v) if v else "Nothing mapped yet. Say 'scan the room'.")
        return
    if act == "fire":
        if mode != "TRACK": say("Nothing is locked. Tell me what to aim at first."); return
        if spec.get("target") in LIVING: say("I never fire at people or animals."); return
        fire_req_until = time.time() + 4.0; say("Firing as soon as I'm locked." if armed else "Dry run: not armed (press A in the video window).")
        return
    if act == "aim":
        if c.get("then_fire") and c.get("target") in LIVING:
            say("I will point at that, but I never fire at people or animals."); c = dict(c, then_fire=False)
        start_aim(c)

def img_err(d, off=True):
    ep = (d["cx"] - W / 2) / W * args.fov_h - (args.laser_off_pan if off else 0.0)
    et = (H / 2 - d["cy"]) / H * args.fov_v - (args.laser_off_tilt if off else 0.0)
    return (-ep if args.invert_pan else ep), (-et if args.invert_tilt else et)

print("Ready. Type a command here, or press V in the video window (with --voice).  q = quit, a = arm, s = scan, h = home, x = stop.")
print("ARMED" if armed else "Not armed: 'fire' is a dry run until you press A in the window.")
cv2.namedWindow("voice_turret")
W = H = 1
try:
    while True:
        while not cmd_q.empty():
            execute(cmd_q.get())
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.003); continue
        now = time.time(); dt = min(0.25, now - last_t); last_t = now
        H, W = frame.shape[:2]; frame_i += 1
        dets = det.detect(frame)
        for d in dets:
            d["ep"], d["et"] = img_err(d)                       # aiming error (includes the laser offset)
            d["wp"], d["wt"] = aim.world(*img_err(d, off=False), now)
            d["wdeg"], d["hdeg"] = d["w"] / W * args.fov_h, d["h"] / H * args.fov_v
        last_dets, last_frame = dets, frame
        if any(d["label"] in LIVING and d["conf"] >= 0.35 for d in dets): person_until = now + 1.0
        person_block = now < person_until

        # ---- map the room (not while the head whips around: the camera delay makes angles unreliable)
        head_speed = float(np.hypot(pan - prev_pan, tilt - prev_tilt)) / max(dt, 1e-3)
        prev_pan, prev_tilt = pan, tilt
        if head_speed < 45.0:
            seen_ids = set()
            for d in dets:
                col = colour_name(frame, d) if d["label"] not in LIVING else ""
                seen_ids.add(observe(d["label"], d["wp"], d["wt"], d["wdeg"], d["hdeg"], col, now).id)
            if head_speed < 15.0:                                # forget things that moved away from where we are looking
                for o in registry[:]:
                    if o.id in seen_ids: continue
                    if abs(o.pan - pan) < args.fov_h / 2 - 6 and abs(o.tilt - tilt) < args.fov_v / 2 - 6:
                        o.miss += 1
                        if o.miss >= 25: registry.remove(o)

        fire, tgt = False, None
        if mode == "SCAN":
            pan, tilt, arrived = slew(pan, tilt, wp[wp_i][0], wp[wp_i][1], args.scan_speed, dt)
            if arrived:
                wp_i += 1
                if wp_i >= len(wp):
                    mode = "IDLE"; v = valid_objs()
                    say(f"Scan done. I can see {len(v)} thing(s): " + "; ".join(describe(o) for o in v) if v else "Scan done - nothing recognised.")
                    if pending is not None:
                        c, pending = pending, None; start_aim(c, retry=False)
        elif mode == "SLEW":
            pan, tilt, arrived = slew(pan, tilt, goto_pt[0], goto_pt[1], 60.0, dt)
            if arrived: mode = "IDLE"
        elif mode == "GOTO":
            pan, tilt, _ = slew(pan, tilt, goto_obj.pan, goto_obj.tilt, 90.0, dt)
            cand = [d for d in dets if d["label"] == spec["target"] and np.hypot(d["wp"] - goto_obj.pan, d["wt"] - goto_obj.tilt) < 8.0 + 0.3 * d["wdeg"]]
            if cand:
                d = min(cand, key=lambda d: np.hypot(d["wp"] - goto_obj.pan, d["wt"] - goto_obj.tilt))
                aim.kal = new_kal(); aim.meas.clear(); aim.kal.reset(d["wp"], d["wt"])
                mode, lost_since, lock_n = "TRACK", None, 0
                fire_req_until = now + 8.0 if spec["then_fire"] else 0.0
                say(f"Locked on the {spec['target']}.")
            elif now - goto_t0 > 6.0:
                if goto_obj in registry: registry.remove(goto_obj)
                mode = "IDLE"; say(f"The {spec['target']} isn't there any more.")
        elif mode == "TRACK":
            pred = aim.predicted()
            cands = [d for d in dets if d["label"] == spec["target"]]
            cand = None
            if cands and pred is not None:
                lost = 0.0 if lost_since is None else now - lost_since
                gate = min(40.0, 10.0 + 0.3 * max(d["wdeg"] for d in cands) + 8.0 * lost)
                dd = [float(np.hypot(d["wp"] - pred[0], d["wt"] - pred[1])) for d in cands]
                i = int(np.argmin(dd))
                others = sorted(dd)[1:2]
                if dd[i] < gate and (not others or others[0] > 1.5 * dd[i] or dd[i] < 3.0): cand = cands[i]
            if cand is not None:
                tgt = cand; lost_since = None
                pan, tilt = aim.step(cand["ep"], cand["et"], True, now, dt, pan, tilt)
                locked = abs(cand["ep"]) < args.lock_tol and abs(cand["et"]) < args.lock_tol
                lock_n = lock_n + 1 if locked else 0
                want = now < fire_req_until
                label_ok = spec["target"] in args.fire_classes and spec["target"] not in NEVER_FIRE
                in_zone = abs(pan) <= args.fire_pan and args.fire_tilt_min <= tilt <= args.fire_tilt_max
                if want and lock_n >= args.lock_frames:
                    why = None
                    if not label_ok: why = (f"'{spec['target']}' is not in --fire-classes" if spec["target"] not in NEVER_FIRE
                                            else f"I won't fire at a {spec['target']} (shiny/living)")
                    elif not in_zone: why = "outside the allowed firing zone"
                    elif person_block: why = "a person/animal is in view"
                    elif now < cool_until: why = None
                    if why:
                        say(f"Not firing: {why}."); fire_req_until = 0.0
                    elif now >= cool_until:
                        if not armed:
                            say(f"(dry run) would fire at the {spec['target']} now."); fire_req_until = 0.0
                        else:
                            if fire_t0 is None: fire_t0 = now
                            fire = True
                            if now - fire_t0 > args.pulse:
                                fire = False; fire_t0 = None; fire_req_until = 0.0; cool_until = now + 1.5
                                say("Pulse finished.")
                else:
                    fire_t0 = None if not want else fire_t0
            else:
                if lost_since is None: lost_since = now
                gone = now - lost_since
                if fire_t0 is not None and spec["target"] == "balloon" and now - fire_t0 - gone >= 0.3 and gone >= 0.5:
                    say("Pop!"); fire_t0 = None; fire_req_until = 0.0; mode = "IDLE"
                elif gone < 0.6 and aim.kal.ready:
                    pan, tilt = aim.step(0.0, 0.0, False, now, dt, pan, tilt)        # coast on the prediction
                elif gone < 1.5 and aim.kal.ready:
                    aim.kal.step(None, dt)
                elif gone > 2.0:
                    fire_t0, fire_req_until = None, 0.0
                    reg = valid_objs(spec["target"])
                    if reg:
                        goto_obj = choose(reg, spec["qual"], spec["color"], pan, tilt, lambda i: (i.pan, i.tilt, i.w, i.h, i.col))
                        goto_t0, mode = now, "GOTO"; say(f"Lost the {spec['target']}; going back to where I last saw one.")
                    else:
                        mode = "IDLE"; say(f"Lost the {spec['target']}.")
        if not (mode == "TRACK" and now < fire_req_until): fire_t0 = None     # keep it only while a fire request is live (pop detection)
        fire = bool(fire and armed and not person_block)

        changed = sent[0] is None or abs(pan - sent[0]) >= 0.1 or abs(tilt - sent[1]) >= 0.1 or fire != sent[2]
        if changed or now - last_send > (0.08 if fire else 0.3):
            if changed or sent[0] is None: sent = (pan, tilt, fire)
            send(*sent); last_send = now
            if ser: ser.reset_input_buffer()
        aim.log(now, pan, tilt)

        # ---- display
        for d in dets:
            mine = d is tgt
            col = (0, 0, 255) if (mine and fire) else (0, 255, 0) if mine else (170, 170, 170)
            x1, y1 = int(d["cx"] - d["w"] / 2), int(d["cy"] - d["h"] / 2)
            cv2.rectangle(frame, (x1, y1), (int(d["cx"] + d["w"] / 2), int(d["cy"] + d["h"] / 2)), col, 2 if mine else 1)
            cv2.putText(frame, f"{d['label']} {d['conf']:.2f}", (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        ox = int(W / 2 + args.laser_off_pan / args.fov_h * W); oy = int(H / 2 - args.laser_off_tilt / args.fov_v * H)
        cv2.drawMarker(frame, (ox, oy), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
        cv2.putText(frame, f"{mode} {spec.get('target', '') if mode in ('TRACK', 'GOTO') else ''}  pan {pan:.0f} tilt {tilt:.0f}  map {len(valid_objs())}",
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, ("ARMED" if armed else "DRY RUN") + ("  FIRING" if fire else "") + ("  PERSON/ANIMAL - LASER INHIBITED" if person_block else ""),
                    (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if armed else (0, 255, 255), 1)
        cv2.putText(frame, "you: " + last_heard[0][-60:], (8, H - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "bot: " + last_say[0][-60:], (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow("voice_turret", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("a"): armed = not armed; say("ARMED - the laser can fire." if armed else "Disarmed (dry run).")
        if k == ord("v"): listen_evt.set()
        if k == ord("s"): execute({"act": "scan"})
        if k == ord("h"): execute({"act": "home"})
        if k == ord("x"): execute({"act": "stop"})
finally:
    try:
        if ser:
            ser.write(b"P0,T0,F0\n"); time.sleep(0.05); ser.write(b"H\n"); ser.close()   # laser off first, then home
    except Exception:
        pass
    cap.release(); cv2.destroyAllWindows()
