import os
import shutil
from pathlib import Path
import tifffile
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths — update these
input_root = Path("../NCT-CRC-HE-100K-NONORM")  # original dataset root with subfolders per class, containing TIFFs
output_root = Path("../colorectal_split")  # output root

train_ratio = 0.8  # train split ratio

def convert_tif_to_png(tif_path, png_path):
    img_array = tifffile.imread(tif_path)
    # Convert grayscale to RGB
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[-1] > 3:
        img_array = img_array[..., :3]
    # Normalize if not uint8
    if img_array.dtype != 'uint8':
        img_array = (255 * (img_array / img_array.max())).astype('uint8')
    img = Image.fromarray(img_array)
    img.save(png_path)

def main():
    import numpy as np

    classes = [d.name for d in input_root.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    # Gather all image paths and their labels
    all_images = []
    all_labels = []

    for cls in classes:
        cls_path = input_root / cls
        tif_files = list(cls_path.glob("*.tif")) + list(cls_path.glob("*.tiff"))
        print(f"Class '{cls}' has {len(tif_files)} images")
        all_images.extend(tif_files)
        all_labels.extend([cls] * len(tif_files))

    # Split into train/test
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        all_images, all_labels, stratify=all_labels, test_size=1-train_ratio, random_state=42
    )

    print(f"Train samples: {len(train_imgs)}, Test samples: {len(test_imgs)}")

    # Helper to save images
    def save_images(images, labels, subset):
        for img_path, label in zip(images, labels):
            out_dir = output_root / subset / label
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / (img_path.stem + ".png")

            try:
                convert_tif_to_png(img_path, out_path)
            except Exception as e:
                print(f"Failed converting {img_path}: {e}")

    print("Saving training images...")
    save_images(train_imgs, train_labels, "train")
    print("Saving testing images...")
    save_images(test_imgs, test_labels, "test")
    print("Done!")

if __name__ == "__main__":
    main()
