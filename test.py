import os
import csv
import argparse

import torch
import torch.optim as optim
import time
from torch import nn
from tqdm import tqdm

import numpy as np 
from util import thresh_cal, _tissue_mask
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models
from torchvision import datasets
from torch.utils.data import DataLoader
from norm import ReinhardNormalizerTransform, MacenkoNormalizerTransform

torch.backends.cudnn.benchmark = True
class DebugTap:
    """
    A no-op transform that saves the image and logs approximate tissue coverage.
    Runs on PIL images *before* ToTensor().
    """
    counter = 0
    def __init__(self, tag: str, limit: int = 8):
        self.tag = tag
        self.limit = limit

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        if DebugTap.counter < self.limit:
            os.makedirs("debug_viz", exist_ok=True)
            arr = np.asarray(img_pil).astype(np.float32) / 255.0  # HxWxC in [0,1]
            # "Background" ≈ very bright near-white
            bg = (arr > 0.98).all(axis=2)
            coverage = 1.0 - bg.mean()

            img_pil.save(f"debug_viz/{self.tag}_{DebugTap.counter}.png")
            with open(f"debug_viz/{self.tag}_coverage.txt", "a") as f:
                f.write(f"{DebugTap.counter}\tcoverage={coverage:.4f}\n")
            print(f"[viz] {self.tag} #{DebugTap.counter}: tissue coverage ≈ {coverage*100:.1f}%")

            DebugTap.counter += 1
        return img_pil


class OtsuTissueMask:
    """
    mode='mask' -> set background to white (255)
    mode='crop' -> crop image to the bounding box of the tissue mask
    """
    def __init__(self, mode='mask', fill=255, min_tissue_ratio=0.01):
        assert mode in ('mask', 'crop')
        self.mode = mode
        self.fill = fill
        self.min_tissue_ratio = min_tissue_ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        img_rgb = np.asarray(img.convert('RGB'))
        img_hsv = np.asarray(img.convert('HSV'))

        thrR, thrG, thrB, thrH = thresh_cal(img_rgb, img_hsv)
        mask = _tissue_mask(img_rgb, img_hsv, thrR, thrG, thrB, thrH)  # True = tissue

        # Bail out if almost no tissue is detected
        if mask.mean() < self.min_tissue_ratio:
            return img

        if self.mode == 'mask':
            out = img_rgb.copy()
            out[~mask] = self.fill  # make background white
            return Image.fromarray(out)

        # mode == 'crop'
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        cropped = img_rgb[y0:y1+1, x0:x1+1]
        return Image.fromarray(cropped)

def parse_args():
    parser = argparse.ArgumentParser(description="Train CRC Model")
    parser.add_argument('--pretrained', type=str, default=False, nargs = '?', const = True,
                        help='Path to a checkpoint to resume from. If not set, will use latest in ./checkpoints')
    parser.add_argument('--batch_size', type=int, default=128, nargs = '?', const = 128,
                        help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=8, nargs = '?', const = 15,
                        help='Workers')
    parser.add_argument('--checkpoint', type = str, default = None, nargs= '?')
    parser.add_argument('--otsu', type=str, default='none', choices=['none', 'mask', 'crop'],
                help="Apply Otsu tissue masking: 'none', 'mask' (white-out bg), or 'crop' (tight crop).")
    parser.add_argument('--norm', type=str, default='no_norm', choices = ['no_norm','reinhard', 'macenko'], nargs = '?', const = True,
            help='Batch Size')
    return parser.parse_args()

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory")
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print("No pth files")
        return None
    args = parse_args()
    latest = max(checkpoints, key=lambda x: int(x.split('_')[2][1:]))
    return os.path.join(checkpoint_dir, latest)

def main():
    args = parse_args()
    test_dir = "../datasets/test_png"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.otsu == 'none':
        preproc = []
    else:
        preproc = [OtsuTissueMask(mode=args.otsu, fill=255)]

    # Normalisation read
    # Set up normalisation
    # target_image = Image.open('../datasets/train_png/BACK/BACK-HKCDQKHD.png').convert("RGB")
    target_image = Image.open('../datasets/train_png/STR/STR-MPRKVSPT.png').convert("RGB")
    match args.norm:
        case "reinhard":
            norm_transform = ReinhardNormalizerTransform(target_image)

        case "macenko":
            norm_transform = MacenkoNormalizerTransform(target_image)

    if args.norm != 'no_norm':
        test_transform = transforms.Compose(preproc + 
        # [DebugTap('after_otsu', limit = 8)] +
        [
            transforms.Resize(224),
            norm_transform,
            transforms.ToTensor(),
        ] + ([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))] if args.pretrained else [])
            )
    else:
        test_transform = transforms.Compose(preproc + 
        # [DebugTap('after_otsu', limit = 8)] +
        [
        transforms.Resize(224),
        transforms.ToTensor(),
        
        ] + ([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))] if args.pretrained else [])
        )

    args = parse_args()
    testset = datasets.ImageFolder(test_dir, transform = test_transform)
    testloader = DataLoader(dataset = testset,
                    batch_size = args.batch_size,
                    num_workers = args.num_workers,
                    pin_memory = True,
                    shuffle = True)
    
    imgs, _ = next(iter(testloader))  # N,C,H,W
    print("[debug] batch dtype:", imgs.dtype)
    print("[debug] batch range:", float(imgs.min()), float(imgs.max()))
    print("[debug] batch mean/std:", float(imgs.mean()), float(imgs.std()))
    x = imgs.clone()
    bg = (x > 0.98).all(dim=1, keepdim=True)   # very bright ~white as background
    coverage = 1.0 - bg.float().mean().item()
    print(f"[debug] tissue coverage ≈ {coverage*100:.1f}%")
    if not args.checkpoint:
        checkpoint_path = get_latest_checkpoint()

    else: 
        checkpoint_path = os.path.join("checkpoints", args.checkpoint)
        print(checkpoint_path)

    if not checkpoint_path:
        print("No checkpoint path was detected! Please train a model before testing")
        return

    num_classes = len(testset.classes)
    if args.pretrained == True:
        print("Training with pretrained weights")
        model = torchvision.models.resnet18(num_classes = num_classes)

    else: 
        model = torchvision.models.resnet18(num_classes = num_classes)

    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct= 0, 0
    running_loss = 0.0
    epoch_start_time = time.time()
    progress_bar = tqdm(enumerate(testloader, 0), total=len(testloader), desc="Testing", leave=False)
    for i, data in progress_bar:
        images = data[0].to(device)
        labels = data[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total
    loss = running_loss / len(testloader.dataset)
    print(f'[Test acc: {accuracy:.2f} Test loss: {loss:.2f} Time taken: {time.time() - epoch_start_time} seconds')
    log_path = f"test_log{'_pt' if args.pretrained else ''}.csv" 
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["pretrained", "accuracy", "loss", "time"])
        writer.writerow([args.pretrained, accuracy, loss, time.time() - epoch_start_time])


if __name__ == "__main__":
    main()