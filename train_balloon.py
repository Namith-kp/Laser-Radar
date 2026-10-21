"""
Train your balloon detector (YOLOv8-nano) on the pictures made by collect_balloons.py -- on your NVIDIA GPU.

    pip install ultralytics
    python train_balloon.py --data balloon_data/data.yaml

It uses the GPU automatically and refuses to silently fall back to the (very slow) CPU.
The result is copied to  balloon.pt  -- use it with hunter.py:  --model balloon.pt

If it says "no GPU": pip often installs the CPU-only PyTorch on Windows. Fix, in a terminal:
    pip uninstall -y torch torchvision
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
(use cu121 instead of cu124 if your NVIDIA driver is older; "nvidia-smi" shows the driver version)
"""
import argparse, shutil, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="balloon_data/data.yaml")
    ap.add_argument("--base", default="yolov8n.pt", help="starting model (downloaded automatically). yolov8s.pt = bigger/more accurate, slower in the turret")
    ap.add_argument("--imgsz", type=int, default=320, help="must match what hunter.py uses (--imgsz)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32, help="lower to 16 or 8 if you get 'out of memory'; -1 = pick automatically")
    ap.add_argument("--workers", type=int, default=4, help="data loader processes (set 0 if Windows complains)")
    ap.add_argument("--device", default="auto", help="'auto' = first NVIDIA GPU, or 0 / 1 / cpu")
    ap.add_argument("--out", default="balloon.pt")
    args = ap.parse_args()

    try:
        import torch
    except Exception:
        raise SystemExit("PyTorch is missing:  pip install ultralytics   (then see the note at the top of this file for the GPU version)")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = 0
        else:
            raise SystemExit(
                "\nNo NVIDIA GPU is visible to PyTorch (torch %s, CUDA build: %s).\n"
                "Most likely the CPU-only PyTorch is installed. Fix:\n"
                "    pip uninstall -y torch torchvision\n"
                "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
                "(use cu121 if your driver is older). Check with:  python -c \"import torch; print(torch.cuda.is_available())\"\n"
                "To train on the CPU anyway (slow), add:  --device cpu" % (torch.__version__, torch.version.cuda))
    else:
        device = int(args.device) if str(args.device).isdigit() else args.device

    if device != "cpu":
        p = torch.cuda.get_device_properties(device)
        print(f"Training on GPU: {p.name}  ({p.total_memory / 1024**3:.1f} GB)")
        torch.backends.cudnn.benchmark = True
    else:
        print("Training on the CPU (slow).")

    from ultralytics import YOLO
    model = YOLO(args.base)
    res = model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, workers=args.workers,
                      device=device, amp=(device != "cpu"), cache=True,
                      patience=25, degrees=10, fliplr=0.5, hsv_h=0.02, hsv_s=0.5, hsv_v=0.4, mosaic=1.0,
                      project="runs_balloon", name="train", exist_ok=True)
    save_dir = getattr(res, "save_dir", None) or model.trainer.save_dir
    best = os.path.join(str(save_dir), "weights", "best.pt")
    shutil.copy(best, args.out)
    print(f"\nDone. Detector saved as {args.out}")
    print(f"Try it:  python hunter.py --model {args.out} --imgsz {args.imgsz} --source <camera> --port COM7")


# Windows starts the data-loader workers by re-importing this file, so everything must sit behind this guard.
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()