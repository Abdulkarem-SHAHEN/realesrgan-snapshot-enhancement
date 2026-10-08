# enhance.py
# Enhance low-quality snapshots with Real-ESRGAN (x4) and report, for every image,
# the processing time and the peak memory used by the process.
#
# Run from the repository root:
#     python src/enhance.py samples
#     python src/enhance.py samples --face-enhance --side-by-side

import argparse
import csv
import threading
import time
from pathlib import Path

import cv2
import psutil
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
MODEL_NAME = "RealESRGAN_x4plus"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v1.0.0/RealESRGAN_x4plus.pth"
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MB = 1024 * 1024


class PeakMemory:
    """Samples the resident memory (RSS) of this process in a background thread and keeps the maximum."""

    def __init__(self, interval=0.02):
        self.process = psutil.Process()
        self.interval = interval
        self.peak = 0

    def __enter__(self):
        self.peak = self.process.memory_info().rss
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while self._running:
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._running = False
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def weights_path(file_name, url):
    """Return the local path of a weights file, downloading it into weights/ on first use."""
    path = WEIGHTS_DIR / file_name
    if not path.is_file():
        load_file_from_url(url=url, model_dir=str(WEIGHTS_DIR), progress=True, file_name=file_name)
    return str(path)


def build_upsampler(tile, face_enhance, outscale):
    """Load Real-ESRGAN x4plus, and GFPGAN on top of it when face enhancement is requested."""
    use_gpu = torch.cuda.is_available()
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=weights_path(MODEL_NAME + ".pth", MODEL_URL),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_gpu,  # fp16 on GPU, fp32 on CPU
    )
    face_enhancer = None
    if face_enhance:
        from gfpgan import GFPGANer

        face_enhancer = GFPGANer(
            model_path=weights_path("GFPGANv1.3.pth", GFPGAN_URL),
            upscale=outscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )
    return upsampler, face_enhancer


def side_by_side(original, enhanced):
    """Input enlarged with bicubic interpolation (left) next to the model output (right)."""
    h, w = enhanced.shape[:2]
    return cv2.hconcat([cv2.resize(original, (w, h), interpolation=cv2.INTER_CUBIC), enhanced])


def main():
    parser = argparse.ArgumentParser(description="Enhance snapshots with Real-ESRGAN and measure time and memory.")
    parser.add_argument("input", help="image file or folder of images")
    parser.add_argument("-o", "--output", default=str(BASE_DIR / "results"), help="output folder")
    parser.add_argument("-s", "--outscale", type=float, default=4, help="final upscaling factor (default: 4)")
    parser.add_argument("--face-enhance", action="store_true", help="restore faces with GFPGAN")
    parser.add_argument("--tile", type=int, default=0, help="tile size, 0 = whole image; lower it to reduce memory")
    parser.add_argument("--side-by-side", action="store_true", help="also save a before/after comparison image")
    args = parser.parse_args()

    source = Path(args.input)
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_TYPES) if source.is_dir() else [source]
    if not images:
        raise SystemExit(f"No images found in {source}")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    before_load = process.memory_info().rss
    upsampler, face_enhancer = build_upsampler(args.tile, args.face_enhance, args.outscale)
    model_mb = (process.memory_info().rss - before_load) / MB
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} | model loaded: +{model_mb:.0f} MB\n")

    rows = []
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipped (not readable): {path.name}")
            continue

        with PeakMemory() as memory:
            start = time.perf_counter()
            if face_enhancer is not None:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
            seconds = time.perf_counter() - start

        cv2.imwrite(str(out_dir / f"{path.stem}_out.png"), output)
        if args.side_by_side:
            cv2.imwrite(str(out_dir / f"{path.stem}_compare.jpg"), side_by_side(img, output))

        rows.append({
            "image": path.name,
            "input": f"{img.shape[1]}x{img.shape[0]}",
            "output": f"{output.shape[1]}x{output.shape[0]}",
            "seconds": round(seconds, 2),
            "peak_ram_mb": round(memory.peak / MB),
        })
        print("{image:28} {input:>10} -> {output:>10} {seconds:>8.2f} s {peak_ram_mb:>7} MB peak".format(**rows[-1]))

    if rows:
        with open(out_dir / "benchmark.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} image(s) and benchmark.csv to {out_dir}")


if __name__ == "__main__":
    main()
