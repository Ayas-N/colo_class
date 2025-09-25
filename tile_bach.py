#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path
from PIL import Image

# DN-friendly name pattern:
# {row:03d}_{col:03d}_{y_anchor:04d}_{x_anchor:04d}_{parent}.png
def save_tiles(img_path: Path, out_dir: Path, patch: int, stride: int,
               to_png: bool, manifest_writer, reset_manifest_cols=False):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size  # PIL: (width, height)

    # Safety: enforce multiples if you want. BACH is 2048x1536 so this should pass for patch=512.
    if (H - patch) % stride != 0 or (W - patch) % stride != 0:
        print(f"[WARN] {img_path.name}: size {W}x{H} not aligned to patch={patch}, stride={stride}. "
              f"Edges will be dropped.", file=sys.stderr)

    parent = img_path.stem  # e.g., img001
    rows = 1 + max(0, (H - patch) // stride)
    cols = 1 + max(0, (W - patch) // stride)

    # Optional: keep per-parent subfolder (helps batch DN per image)
    parent_dir = out_dir / parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Write header if requested & not yet written
    if reset_manifest_cols and manifest_writer is not None:
        manifest_writer.writerow(["split","label","parent","row","col","y_anchor","x_anchor",
                                  "patch_h","patch_w","filename","src_path","width","height"])

    count = 0
    for r in range(rows):
        for c in range(cols):
            y = r * stride
            x = c * stride
            if y + patch > H or x + patch > W:
                continue  # drop ragged edge (BACH shouldn't hit this)
            crop = img.crop((x, y, x + patch, y + patch))  # (left, top, right, bottom)

            fname = f"{r:03d}_{c:03d}_{y:04d}_{x:04d}_{parent}.png" if to_png \
                    else f"{r:03d}_{c:03d}_{y:04d}_{x:04d}_{parent}{img_path.suffix.lower()}"
            out_path = parent_dir / fname
            crop.save(out_path)  # PNG by default is lossless

            if manifest_writer is not None:
                manifest_writer.writerow([
                    out_dir.parent.name,     # split (train/test) if you keep that structure
                    out_dir.parent.parent.name if out_dir.parent.parent != out_dir.parent else "",  # label (best-effort)
                    parent,
                    r, c, y, x, patch, patch,
                    str(out_path.relative_to(out_dir.parent.parent)) if out_dir.parent.parent in out_path.parents else str(out_path),
                    str(img_path), W, H
                ])
            count += 1
    return count

def is_image(p: Path):
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def main():
    ap = argparse.ArgumentParser(description="Tile BACH images into DN-friendly patches with grid anchors in filenames.")
    ap.add_argument("--in_root",  required=True, help="Input root (e.g., /data/BACH)")
    ap.add_argument("--out_root", required=True, help="Output root for tiles")
    ap.add_argument("--patch",    type=int, default=512, help="Patch size (square). Default 512.")
    ap.add_argument("--stride",   type=int, default=512, help="Stride (default = patch for non-overlap).")
    ap.add_argument("--to-png",   action="store_true",   help="Force PNG output (recommended).")
    ap.add_argument("--make-manifest", action="store_true", help="Write tiles_manifest.csv at out_root.")
    args = ap.parse_args()

    in_root  = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_fp = None
    writer = None
    if args.make_manifest:
        manifest_fp = open(out_root / "tiles_manifest.csv", "w", newline="")
        writer = csv.writer(manifest_fp)

    total = 0
    # Walk splits and labels if they exist; otherwise just tile all images under in_root.
    # Expected: BACH/<split>/<label>/*.*
    splits = []
    if any((in_root / s).is_dir() for s in ("train","test","val","validation")):
        for s in ("train","test","val","validation"):
            d = in_root / s
            if d.is_dir():
                splits.append(d)
    else:
        splits = [in_root]

    for split_dir in splits:
        # labels: Benign, Insitu, Invasive, Normal (case-insensitive)
        label_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not label_dirs:
            label_dirs = [split_dir]  # no label subfolders; treat images directly

        for label_dir in label_dirs:
            # Output mirrors input structure: out_root/<split>/<label>/
            rel = label_dir.relative_to(in_root)
            out_dir = out_root / rel
            out_dir.mkdir(parents=True, exist_ok=True)

            # Iterate images
            imgs = sorted([p for p in label_dir.rglob("*") if p.is_file() and is_image(p)])
            if not imgs:
                continue

            # Reset header for each label group if manifest
            reset_header = True
            for img_path in imgs:
                cnt = save_tiles(img_path, out_dir, args.patch, args.stride, args.to_png,
                                 writer, reset_manifest_cols=reset_header)
                reset_header = False
                total += cnt
            print(f"[OK] {label_dir}: wrote {total} patches so far.")

    if manifest_fp:
        manifest_fp.close()
    print(f"Done. Total patches: {total}. Output at: {out_root}")

if __name__ == "__main__":
    main()
