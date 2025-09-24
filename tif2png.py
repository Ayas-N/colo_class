#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageSequence, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # be forgiving with slightly broken files

def is_tiff(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".tif", ".tiff"}

def save_png(img: Image.Image, out_path: Path, *, optimize=True, compress_level=6):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = {"optimize": optimize, "compress_level": compress_level}
    # Convert modes that PNG won’t like or that you probably want as RGB
    if img.mode in ("CMYK", "P", "LA"):
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        # drop alpha for consistency (or keep by removing this branch)
        img = img.convert("RGB")
    # 'I;16' (16-bit gray) is OK in PNG, leave as is; others like 'I' -> cast to 16-bit or 8-bit
    elif img.mode == "I":
        img = img.point(lambda x: max(0, min(x, 65535))).convert("I;16")
    img.save(out_path, format="PNG", **params)

def convert_one(in_path: Path, in_root: Path, out_root: Path, overwrite=False, all_pages=False):
    rel = in_path.relative_to(in_root)
    out_png = (out_root / rel).with_suffix(".png")
    if not all_pages:
        if out_png.exists() and not overwrite:
            return ("skip", in_path)
        try:
            with Image.open(in_path) as im:
                save_png(im, out_png)
            return ("ok", in_path)
        except Exception as e:
            return ("err", in_path, str(e))
    else:
        # Save each frame as *_p000.png, *_p001.png, ...
        status = "ok"
        try:
            with Image.open(in_path) as im:
                num = 0
                for i, frame in enumerate(ImageSequence.Iterator(im)):
                    out_i = out_png.with_name(out_png.stem + ".png")
                    if out_i.exists() and not overwrite:
                        continue
                    save_png(frame, out_i)
                    num += 1
            return (status, in_path, f"{num} pages")
        except Exception as e:
            return ("err", in_path, str(e))

def main():
    ap = argparse.ArgumentParser(description="Recursively convert TIFF to PNG, preserving folder structure.")
    ap.add_argument("input_dir", type=Path, help="Root folder containing .tif/.tiff files")
    ap.add_argument("output_dir", type=Path, help="Destination root for mirrored PNGs")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="Parallel workers (threads)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs")
    ap.add_argument("--all-pages", action="store_true", help="Export all pages of multipage TIFFs as *_p###.png")
    args = ap.parse_args()

    if not args.input_dir.exists():
        print(f"Input not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    tiffs = [p for p in args.input_dir.rglob("*") if is_tiff(p)]
    if not tiffs:
        print("No .tif/.tiff files found.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(tiffs)} TIFFs. Converting with {args.workers} workers...")
    ok = skipped = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(convert_one, p, args.input_dir, args.output_dir, args.overwrite, args.all_pages)
            for p in tiffs
        ]
        for f in as_completed(futures):
            res = f.result()
            tag = res[0]
            if tag == "ok":
                ok += 1
            elif tag == "skip":
                skipped += 1
            else:
                err += 1
                # Show short error info
                print(f"[ERR] {res[1]} -> {res[2]}", file=sys.stderr)

    print(f"Done. ok={ok}, skipped={skipped}, errors={err}")

if __name__ == "__main__":
    main()
