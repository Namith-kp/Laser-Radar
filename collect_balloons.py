"""
Collect a training set for YOUR balloon detector -- labels itself.

You drag a box around a balloon once; the tracker follows it and saves every few frames together with the
box as a YOLO label. Hold the balloon (or hang it) and move it around: near/far, left/right, different
backgrounds and lighting. About 5-10 minutes gives 500-1500 labelled pictures.

    python collect_balloons.py --source http://192.168.1.114:81/stream
    python collect_balloons.py --source serial:COM5

Keys in the window:
    drag a box around a balloon = start following + labelling it (drag again for a 2nd balloon in view)
    s = pause / resume saving      c = forget all balloons      u = forget the last one
    n = save the current picture as a 'no balloon' example (use when NO balloon is in view; ~10-20% of the set)
    q = quit

IMPORTANT: every balloon that is visible must have a box, otherwise the model learns that balloons are background.
Easiest: keep ONE balloon in view while collecting, or drag a box on each balloon.
Then train:  python train_balloon.py
"""
import argparse, os, random, time
import cv2, numpy as np
from turret_common import open_camera, make_tracker, patch_hist, clamp

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="0", help="webcam index, stream URL, or 'serial:COM5'")
ap.add_argument("--out", default="balloon_data", help="dataset folder (created; new sessions are added to it)")
ap.add_argument("--name", default="balloon", help="class name")
ap.add_argument("--track", choices=["csrt", "kcf", "mosse"], default="csrt")
ap.add_argument("--every", type=int, default=3, help="save every Nth picture while following")
ap.add_argument("--max", type=int, default=3000, help="stop after this many saved pictures")
ap.add_argument("--val", type=float, default=0.15, help="share of the pictures kept for validation")
ap.add_argument("--verify", type=float, default=0.35, help="drop a balloon if what is followed stops looking like it (0 = off)")
ap.add_argument("--no-tighten", action="store_true", help="do NOT shrink the followed box onto the balloon's real outline (tight boxes train a better detector)")
ap.add_argument("--cam-baud", type=int, default=2000000)
ap.add_argument("--no-cam-setup", action="store_true")
ap.add_argument("--cam-set")
ap.add_argument("--cam-port", type=int, default=80)
args = ap.parse_args()

for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(os.path.join(args.out, sub), exist_ok=True)
with open(os.path.join(args.out, "data.yaml"), "w") as f:
    f.write(f"path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: {args.name}\n")

cap = open_camera(args.source, args.cam_baud, args.cam_set, args.cam_port, not args.no_cam_setup)
session = time.strftime("%m%d_%H%M%S")
tracks, roi_req, drag = [], None, None
saving, n_saved, n_train, n_val, frame_i = True, 0, 0, 0, 0

def on_mouse(event, x, y, flags, param):
    global drag, roi_req
    if event == cv2.EVENT_LBUTTONDOWN: drag = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE and drag is not None: drag[2], drag[3] = x, y
    elif event == cv2.EVENT_LBUTTONUP and drag is not None:
        x0, y0, x1, y1 = drag; drag = None
        if abs(x1 - x0) >= 10 and abs(y1 - y0) >= 10:
            roi_req = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

def tighten(img, box):
    """Shrink the tracker's (loose) box onto the object's outline with GrabCut. Falls back to the original box if it looks wrong."""
    H, W = img.shape[:2]
    x, y, w, h = box
    px, py = int(0.25 * w), int(0.25 * h)
    X0, Y0, X1, Y1 = int(max(0, x - px)), int(max(0, y - py)), int(min(W, x + w + px)), int(min(H, y + h + py))
    crop = img[Y0:Y1, X0:X1]
    if crop.shape[0] < 16 or crop.shape[1] < 16: return box
    rect = (int(x - X0), int(y - Y0), int(w), int(h))
    if rect[0] < 0 or rect[1] < 0 or rect[2] < 8 or rect[3] < 8 or rect[0] + rect[2] > crop.shape[1] or rect[1] + rect[3] > crop.shape[0]:
        return box
    try:
        mask = np.zeros(crop.shape[:2], np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        if n < 2: return box
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh = [int(v) for v in st[k, :4]]
        if not (0.35 <= (bw * bh) / float(w * h) <= 1.15): return box          # implausible -> keep the tracker's box
        return (X0 + bx, Y0 + by, bw, bh)
    except Exception:
        return box

def save(raw, boxes):
    """boxes: list of (x, y, w, h) pixels. Empty list = a 'no balloon' picture."""
    global n_saved, n_train, n_val
    H, W = raw.shape[:2]
    split = "val" if random.Random(n_saved // 40).random() < args.val else "train"   # whole 40-picture blocks go to one side
    stem = f"{session}_{n_saved:05d}"
    cv2.imwrite(os.path.join(args.out, "images", split, stem + ".jpg"), raw, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = []
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = clamp(x, 0, W), clamp(y, 0, H), clamp(x + w, 0, W), clamp(y + h, 0, H)
        if x1 - x0 < 6 or y1 - y0 < 6: continue
        lines.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    n_saved += 1
    if split == "val": n_val += 1
    else: n_train += 1

cv2.namedWindow("collect")
cv2.setMouseCallback("collect", on_mouse)
print("Drag a box around a balloon. s=pause saving  n=save 'no balloon'  c=clear  u=undo  q=quit")
try:
    while n_saved < args.max:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005); continue
        raw = frame.copy()
        H, W = frame.shape[:2]
        if roi_req is not None:
            x, y, w, h = roi_req; roi_req = None
            x, y = int(clamp(x, 0, W - 10)), int(clamp(y, 0, H - 10))
            w, h = int(clamp(w, 10, W - x)), int(clamp(h, 10, H - y))
            t = make_tracker(args.track); t.init(raw, (x, y, w, h))
            tracks.append({"t": t, "ref": patch_hist(raw, x, y, w, h), "bad": 0, "box": (x, y, w, h)})
            print(f"Following balloon #{len(tracks)}")
        boxes = []
        for tr in list(tracks):
            good, bx = tr["t"].update(raw)
            if good:
                x, y, w, h = bx
                inside = x >= -2 and y >= -2 and x + w <= W + 2 and y + h <= H + 2 and w > 8 and h > 8
                if args.verify > 0:
                    sim = cv2.compareHist(tr["ref"], patch_hist(raw, x, y, w, h), cv2.HISTCMP_CORREL)
                    tr["bad"] = tr["bad"] + 1 if sim < args.verify else 0
                    if tr["bad"] >= 3: good = False
                    elif sim > 0.85: tr["ref"] = 0.95 * tr["ref"] + 0.05 * patch_hist(raw, x, y, w, h)
            if not good:
                tracks.remove(tr); print("A balloon was lost - drag a box on it again if it is still in view."); continue
            tr["box"] = tuple(float(v) for v in bx)
            if inside: boxes.append(tr["box"])
        frame_i += 1
        if saving and boxes and len(boxes) == len(tracks) and frame_i % args.every == 0:
            save(raw, boxes if args.no_tighten else [tighten(raw, b) for b in boxes])
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        if drag is not None:
            cv2.rectangle(frame, (drag[0], drag[1]), (drag[2], drag[3]), (0, 255, 255), 1)
        msg = f"{'SAVING' if saving else 'PAUSED'}  saved {n_saved} (train {n_train} / val {n_val})  balloons followed: {len(tracks)}"
        cv2.putText(frame, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if saving else (0, 200, 255), 1)
        if not tracks:
            cv2.putText(frame, "DRAG A BOX AROUND A BALLOON", (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("collect", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        if k == ord("s"): saving = not saving
        if k == ord("c"): tracks.clear()
        if k == ord("u") and tracks: tracks.pop()
        if k == ord("n"):
            if tracks: print("Clear the balloons first (c) - 'no balloon' pictures must really contain none.")
            else: save(raw, []); print("Saved a 'no balloon' picture.")
finally:
    cap.release(); cv2.destroyAllWindows()
    print(f"Done: {n_saved} pictures in '{args.out}' (train {n_train}, val {n_val}).  Next:  python train_balloon.py --data {os.path.join(args.out, 'data.yaml')}")
