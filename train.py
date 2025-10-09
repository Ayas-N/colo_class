import os
import csv
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import ResNet18_Weights
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.optim as optim
import time
from torch import nn
from tqdm import tqdm
import torchvision.models
import random
import cv2 

from norm import ReinhardNormalizerTransform, MacenkoNormalizerTransform
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import wandb

torch.backends.cuda.matmul.allow_tf32 = True  # TF32 on Ampere
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
CUDA_VISIBLE_DEVICES=0,1

import torch.distributed as dist

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_dist()) or dist.get_rank() == 0

def ddp_reduce_mean(value: float, device) -> float:
    """Average a scalar across ranks; noop on single GPU."""
    t = torch.tensor([value], device=device, dtype=torch.float64)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())

def get_state_dict(model):
    return model.module.state_dict() if is_dist() else model.state_dict()


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
        img_rgb = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        thr, mask = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tissue_mask = mask == 0

        if tissue_mask.mean() < self.min_tissue_ratio:
            # fallback: just return original
            return img

        if self.mode == "mask":
            out = img_rgb.copy()
            out[~tissue_mask] = self.fill
            img_out = Image.fromarray(out)
            setattr(img_out, "filename", img.filename)
            return img_out

        ys, xs = np.where(tissue_mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        cropped = img_rgb[y0:y1+1, x0:x1+1]
        img_out = Image.fromarray(cropped)
        setattr(img_out, "filename", img.filename)
        return img_out

def parse_args():
    parser = argparse.ArgumentParser(description="Train CRC Model")
    parser.add_argument('--resume', type=str, default=False, nargs = '?', const = True,
                        help='Path to a checkpoint to resume from. If not set, will use latest in ./checkpoints')
    parser.add_argument('--epoch', type=int, default=100, nargs = '?', const = True,
                        help='Number of epochs to train')
    parser.add_argument('--epoch_checkpoint', type=int, default=30, nargs = '?', const = True,
                        help='Number of epochs per checkpoint')
    parser.add_argument('--dataroot', type=str, default="train_png", nargs = '?', const = True,
                    help='Path to a checkpoint to resume from. If not set, will use latest in ./checkpoints')
    parser.add_argument('--testroot', type = str, default = "test_png", nargs ='?', const = True,
                        help = 'Path of the test directory')
    parser.add_argument('--batch_size', type=int, default=256, nargs = '?', const = True,
                        help='Batch Size')
    parser.add_argument('--num_workers', type=int, default=4, nargs = '?', const = True,
                        help='Batch Size')
    parser.add_argument('--gpu_id', type=int, default=0, nargs = '?', const = True,
                    help='Batch Size')
    parser.add_argument('--pretrained', type=str, default=False, nargs='?', const=True,
                    help='Use ImageNet-pretrained ResNet-18 (adds ImageNet mean/std)')
    parser.add_argument('--norm', type=str, default='no_norm', choices = ['no_norm','reinhard', 'macenko'], nargs = '?', const = True,
                help='Batch Size')
    parser.add_argument('--otsu', type=str, default='none', choices=['none', 'mask', 'crop'],
                    help="Apply Otsu tissue masking: 'none', 'mask' (white-out bg), or 'crop' (tight crop).")
    parser.add_argument('--dist', action='store_true', help='Enable DistributedDataParallel')
    parser.add_argument('--wandb', action='store_true',
                    help='Enable Weights & Biases logging (rank0 only in DDP)')
    parser.add_argument('--wandb_project', type=str, default='crc-class',
                    help='wandb project name')
    return parser.parse_args()

def pil_loader_tif(path):
    # Ensure all images are converted to RGB
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        setattr(img, "filename", str(path))
        return img


def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest)

def select_random_image(root: str):
    '''Selects a random image at root'''
    dirs = os.listdir(root)
    cls_num = random.randint(0, len(dirs)-1)
    cls_name = str(os.listdir(root)[cls_num])
    cls_dir = root + '/' + cls_name
    img_path = cls_dir + '/' + os.listdir(cls_dir)[random.randint(0, len(cls_dir)-1)]
    print("Random Template Image found at: " + img_path)
    return img_path

def main(val_ratio = 0.2):
    '''epoch_checkpoint: Int that determines how many checkpoints of models are made between saves'''
    # Compute split sizes
    args = parse_args()
    # Set up Template for colour normalisation
    target_image = Image.open(select_random_image(f'../datasets/{args.dataroot}')).convert('RGB')
    target_image = Image.open('../datasets/train_png/STR/STR-MPRKVSPT_p000.png').convert("RGB") 

    # Normalisation read
    match args.norm:
        case "reinhard":
            norm_transform = ReinhardNormalizerTransform(target_image)

        case "macenko":
            norm_transform = MacenkoNormalizerTransform(target_image)

    if args.otsu == 'none':
        preproc = []
    else:
        preproc = [OtsuTissueMask(mode=args.otsu, fill=255)]

    if args.pretrained:
        IMNET_MEAN = (0.485, 0.456, 0.406)
        IMNET_STD  = (0.229, 0.224, 0.225)

    else:
        IMNET_MEAN = IMNET_STD = None

    resize_size = 224 if args.dataroot == "train_png" else 512
    if args.norm != 'no_norm':
        train_transform = transforms.Compose(preproc+
            [
                transforms.Resize(resize_size),
                norm_transform,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
            + ([transforms.Normalize(IMNET_MEAN, IMNET_STD)] if args.pretrained else [])
        )
        test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            norm_transform,
            transforms.ToTensor(),
        ] + ([transforms.Normalize(IMNET_MEAN, IMNET_STD)] if args.pretrained else []))

    else: 
        train_transform = transforms.Compose(preproc+
            [
                transforms.Resize(resize_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
            + ([transforms.Normalize(IMNET_MEAN, IMNET_STD)] if args.pretrained else []))

        test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ] + ([transforms.Normalize(IMNET_MEAN, IMNET_STD)] if args.pretrained else []))

    if args.dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    train_dir = f"../datasets/{args.dataroot}/"
    full_dataset = datasets.ImageFolder(train_dir, loader=pil_loader_tif, transform = train_transform)
    val_size = int(len(full_dataset) * val_ratio)
    train_subset = full_dataset
    test_dir = f"../datasets/{args.testroot}"
    test_set = datasets.ImageFolder(test_dir, loader = pil_loader_tif, transform = test_transform)

    train_sampler = DistributedSampler(train_subset, shuffle=True) if args.dist else None
    test_sampler = DistributedSampler(test_set, shuffle = False) if args.dist else None

    trainloader = DataLoader(dataset = train_subset,
                        batch_size = args.batch_size,
                        num_workers = args.num_workers,
                        shuffle = (train_sampler is None),
                        sampler = train_sampler,
                        pin_memory = True,
                        pin_memory_device=f"cuda:{local_rank}",
                        persistent_workers = True,
                        prefetch_factor = 4,
                        )

    testloader = DataLoader(
    dataset= test_set, batch_size=args.batch_size,
    num_workers=args.num_workers, pin_memory=True, shuffle=False,
    persistent_workers=True, sampler = test_sampler,
    pin_memory_device=f"cuda:{local_rank}",
    prefetch_factor=4,
    )
    num_classes = len(full_dataset.classes)
    if args.pretrained:
        print("Using ImageNet-pretrained ResNet-18")
        model = torchvision.models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )
        # Change the FC layer to 9
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        model.to(device)
    else:
        model = torchvision.models.resnet18(num_classes = num_classes).to(device)


    if args.dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
        fused = True
    )
    min_learning_rate = 5.0e-6
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma = 0.8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=10,
        eta_min=min_learning_rate,
     )
    criterion = nn.CrossEntropyLoss()
    if args.wandb and is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=f"crc_{args.dataroot}",
            config={
                "arch": "resnet18",
                "pretrained": bool(getattr(args, "pretrained", False)),
                "epochs": args.epoch,
                "batch_size": args.batch_size,
                "lr": getattr(args, "lr", None),
                "optimizer": type(optimizer).__name__,
                "scheduler": type(scheduler).__name__ if 'scheduler' in globals() else None,
                "num_classes": num_classes,
                "ddp_world_size": dist.get_world_size() if is_dist() else 1,
            }
        )
    # If model may be wrapped by DDP later, pass the underlying module if wrapped.
    # Watch ONLY when a run exists
    _model_for_watch = model.module if hasattr(model, "module") else model
    wandb.watch(_model_for_watch, log="gradients", log_freq=100)
    try:
        wandb.watch(_model_for_watch, log="gradients", log_freq=100)  # optional; can be "all" or "parameters"
    except Exception:
        pass  # safe if disabled or not desired
    num_epochs = args.epoch
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

    best_val = float('inf')
    patience = 8
    bad_epochs = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_t0 = time.time()
        if args.dist:
            train_sampler.set_epoch(epoch)
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
        train_acc_sync = ddp_reduce_mean(train_acc, device)
        train_loss_sync = ddp_reduce_mean(train_loss, device)
        val_acc_sync   = ddp_reduce_mean(val_acc, device)
        val_loss_sync  = ddp_reduce_mean(val_loss, device)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time_seconds = time.time() - epoch_t0

        # Wandb Log
        if args.wandb and is_main_process():
            wandb.log({
                "epoch": epoch,
                "train/loss": float(train_loss_sync),
                "train/acc":  float(train_acc_sync),
                "val/loss":   float(val_loss_sync),
                "val/acc":    float(val_acc_sync),
                "lr":         float(current_lr),
                "time/epoch_sec": float(epoch_time_seconds) if 'epoch_time_seconds' in locals() else None,
            }, step=epoch)

        # --- step the scheduler properly ---
        if scheduler is not None:
            # ReduceLROnPlateau expects a metric; cosine/step/etc. do not
            import torch.optim.lr_scheduler as _s
            if isinstance(scheduler, _s.ReduceLROnPlateau):
                scheduler.step(val_loss_sync)
            else:
                scheduler.step()

        # --- logging & checkpoint (rank 0 only) ---
        if is_main_process():
            epoch_time = time.time() - epoch_start_time

            os.makedirs(os.path.dirname("training_log.csv") or ".", exist_ok=True)
            log_path = "training_log.csv"
            write_header = not os.path.exists(log_path)
            with open(log_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(["epoch", "train_acc", "train_loss", "val_acc", "val_loss", "time"])
                writer.writerow([epoch + 1, train_acc_sync, train_loss_sync, val_acc_sync, val_loss_sync, epoch_time])

            print(f"[Epoch {epoch + 1}/{num_epochs}] "
                f"train_acc: {train_acc_sync:.2f} | train_loss: {train_loss_sync:.4f} || "
                f"val_acc: {val_acc_sync:.2f} | val_loss: {val_loss_sync:.4f} | "
                f"Time: {epoch_time:.2f}s")

        # --- early stopping (decide on rank 0, broadcast to all) ---
        stop_flag = torch.zeros(1, device=device)  # 0=keep going, 1=stop

        if is_main_process():
            if val_loss_sync + 1e-6 < best_val:
                best_val = val_loss_sync
                bad_epochs = 0

                os.makedirs("checkpoints", exist_ok=True)
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': get_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'loss': train_loss_sync,
                }
                torch.save(checkpoint, f'checkpoints/{args.dataroot}_best_{args.norm}_{("pt" if args.pretrained else "")}.pth')
            else:
                bad_epochs += 1
                torch.save(checkpoint, f'checkpoints/{args.dataroot}_e{epoch+1}_{args.norm}_{("pt" if args.pretrained else "")}.pth')   
                if bad_epochs >= patience:
                    print("Early stopping.")
                    stop_flag[:] = 1  # tell everyone to stop                

        # make all ranks see the decision
        if is_dist():
            dist.broadcast(stop_flag, src=0)

        if stop_flag.item() == 1:
            if is_dist(): dist.barrier()
            break
            
    if args.wandb and is_main_process():
        wandb.finish()

if __name__ == "__main__":
    main()
