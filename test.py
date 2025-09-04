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
from models import ResNet, resnet18

torch.backends.cudnn.benchmark = True
test_dir = "../colorectal_split/test"
test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory")
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print("No pth files")
        return None
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest)

def main(batch_size = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        exit(1)
        
    model = resnet18(testset.classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
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