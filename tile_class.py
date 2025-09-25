#!/usr/bin/env python3
# Tiles images into class-preserving folders; flat inside each class (optional subdir).
import argparse, csv, sys
from pathlib import Path
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def dn_name(parent: str, r: int, c: int, y: int, x: int, force_png: bool, ext: str) -> str:
    # DN-friendly: row_col_yAnchor_xAnchor_parent.(png|orig_ext)
    return f"{r:03d}_{c:03d}_{y:04d}_{x:04d}_{parent}{'.png' if force_png else ext}"

def save_tiles(img_path: Path, out_dir: Path, patch: int, stride: int, to_png: bool,
               writer=None, split="", label="") -> int:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    parent = img_path.stem

    rows = 1 + max(0, (H - patch) // stride)
    cols = 1 + max(0, (W - patch) // stride)

    # Warn if not aligned; we drop ragged edges.
    if (H - patch) % stride != 0 or (W - patch) % stride != 0:
        print(f"[WARN] {img_path.name}: {W}x{H} not aligned to patch={patch}, stride={stride}; "
              f"dropping ragged edges.", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for r in range(rows):
        for c in range(cols):
            y = r * stride
            x = c * stride
            if y + patch > H or x + patch > W:
                continue
            crop = img.crop((x, y, x + patch, y + patch))
            fname = dn_name(parent, r, c, y, x, to_png, img_path.suffix.lower())
            out_path = out_dir / fname
            crop.save(out_path)

            if writer:
                writer.writerow({
                    "split": split, "label": label, "parent": parent,
                    "row": r, "col": c, "y_anchor": y, "x_anchor": x,
                    "patch_h": patch, "patch_w": patch,
                    "out_path": str(out_path),
                    "src_path": str(img_path),
                    "src_w": W, "src_h": H
                })
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser(description="Tile images while preserving class structure; tiles are flat inside each class.")
    ap.add_argument("--in_root",  required=True, help="Input root, e.g. /data/BACH")
    ap.add_argument("--out_root", required=True, help="Output root for tiles")
    ap.add_argument("--patch",    type=int, default=512, help="Patch size (square).")
    ap.add_argument("--stride",   type=int, default=512, help="Stride (default = patch).")
    ap.add_argument("--to-png",   action="store_true",   help="Save tiles as PNG (recommended).")
    ap.add_argument("--class-subdir", default="", help="Optional subfolder name under each class (e.g., 'Patch').")
    ap.add_argument("--make_manifest", action="store_true", help="Write CSV manifest(s) per split.")
    args = ap.parse_args()

    in_root  = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)


    split_dirs = [d for d in in_root.iterdir() if d.is_dir() and d.name.lower() in
                  {"train_folder", "testing_folder", "val", "validation"}]
    print(split_dirs)
    if not split_dirs:
        split_dirs = [in_root]

    total = 0
    for split_dir in split_dirs:
        split_name = split_dir.name if split_dir != in_root else "all"
        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            class_dirs = [split_dir]  # no class subfolders

        writer = None
        mf_fp = None
        if args.make_manifest:
            mf_path = out_root / f"{split_name}_tiles_manifest.csv"
            mf_fp = open(mf_path, "w", newline="")
            writer = csv.DictWriter(mf_fp, fieldnames=[
                "split","label","parent","row","col","y_anchor","x_anchor",
                "patch_h","patch_w","out_path","src_path","src_w","src_h"
            ])
            writer.writeheader()

        for class_dir in class_dirs:
            label_name = class_dir.name if class_dir != split_dir else "noclass"
            out_dir = out_root / split_name / label_name
            if args.class_subdir:
                out_dir = out_dir / args.class_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            imgs = sorted([p for p in class_dir.rglob("*") if p.is_file() and is_image(p)])
            if not imgs:
                continue

            for img_path in imgs:
                total += save_tiles(img_path, out_dir, args.patch, args.stride, args.to_png,
                                    writer=writer, split=split_name, label=label_name)
            print(f"[OK] {class_dir}: cumulative tiles = {total}")

        if mf_fp:
            mf_fp.close()
            print(f"[INFO] Wrote manifest: {mf_path}")

    print(f"Done. Total tiles: {total}. Output: {out_root}")

if __name__ == "__main__":
    main()
