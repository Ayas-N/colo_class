import os
import csv
import argparse

import torch
import torch.optim as optim
import time
from util import thresh_cal, _tissue_mask
from torch import nn
from tqdm import tqdm


from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader, random_split

import numpy as np
from PIL import Image

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
    parser.add_argument('--resume', type=str, default=False, nargs = '?', const = True,
                        help='Path to a checkpoint to resume from. If not set, will use latest in ./checkpoints')
    parser.add_argument('--batch_size', type=int, default=256, nargs = '?', const = True,
                        help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=12, nargs = '?', const = True,
                        help='Batch Size')
    parser.add_argument('--gpu_id', type=int, default=1, nargs = '?', const = True,
                help='Batch Size')
    parser.add_argument('--otsu', type=str, default='mask', choices=['none', 'mask', 'crop'],
                    help="Apply Otsu tissue masking: 'none', 'mask' (white-out bg), or 'crop' (tight crop).")

    return parser.parse_args()

torch.backends.cudnn.benchmark = True

def pil_loader_tif(path):
    # Ensure all images are converted to RGB
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest)

def main(epoch_checkpoint=30, num_epochs = 30, val_ratio = 0.2):
    '''epoch_checkpoint: Int that determines how many checkpoints of models are made between saves'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    if args.otsu == 'none':
        preproc = []
    else:
        preproc = [OtsuTissueMask(mode=args.otsu, fill=255)]

    train_transform = transforms.Compose(preproc + 
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dir = "../datasets/train_png"
    train_path = train_dir.split("/")[1]
    

    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform, loader=pil_loader_tif)
    num_classes = len(full_dataset.classes)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    # Split dataset
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    trainloader = DataLoader(dataset = train_subset,
                        batch_size = args.batch_size,
                        num_workers = args.num_workers,
                        pin_memory = True,
                        shuffle = True)
    
    testloader = DataLoader(
    dataset= val_subset, batch_size=args.batch_size,
    num_workers=args.num_workers, pin_memory=True, shuffle=False
    )
    # Pretrained resnet
    model = models.resnet18(num_classes = num_classes).to(device)
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    if args.resume == True:
        checkpoint_path = get_latest_checkpoint()

    elif isinstance(args.resume, str):
        checkpoint_path = args.resume

    else:
        checkpoint_path = None

    if checkpoint_path and os.path.exists(checkpoint_path): 
        checkpoint = torch.load(checkpoint_path, weights_only = True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from checkpoint: {checkpoint_path}")

    else: 
        print("Starting training from scratch.")
        if not os.path.exists("./checkpoints"):
            os.mkdir("./checkpoints")

    for epoch in range(start_epoch, num_epochs):
        total, correct= 0, 0
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        progress_bar = tqdm(enumerate(trainloader, 0), total=len(trainloader), desc="Training", leave=False)
        for i, data in progress_bar:
            images = data[0].to(device)
            labels = data[1].to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(trainloader.dataset)
        train_acc = 100 * correct / total

        # ---- Validation loop ----
        model.eval()
        val_total, val_correct = 0, 0
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_running_loss += loss.item() * images.size(0)

        val_loss = val_running_loss / len(testloader.dataset)
        val_acc = 100 * val_correct / val_total

        # ---- Logging ----
        log_path = "training_log.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["epoch", "train_acc", "train_loss", "val_acc", "val_loss", "time"])
            writer.writerow([epoch + 1, train_acc, train_loss, val_acc, val_loss, time.time() - epoch_start_time])

        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"train_acc: {train_acc:.2f} | train_loss: {train_loss:.4f} || "
              f"val_acc: {val_acc:.2f} | val_loss: {val_loss:.4f} | "
              f"Time: {time.time() - epoch_start_time:.2f}s")

        if (epoch+1) % epoch_checkpoint == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss
            }

            torch.save(checkpoint, f'checkpoints/{train_path}ptepoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()
