import os
import csv
import argparse

import torch
import torch.optim as optim
import time
from torch import nn
from tqdm import tqdm
from models import ResNet, resnet18


from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRC Model")
    parser.add_argument('--resume', type=str, default=False, nargs = '?', const = True,
                        help='Path to a checkpoint to resume from. If not set, will use latest in ./checkpoints')
    return parser.parse_args()


train_dir = "../colorectal_split/train"
test_dir = "../colorectal_split/test"
torch.backends.cudnn.benchmark = True

train_transform = transforms.Compose(
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

def pil_loader_tif(path):
    # Ensure all images are converted to RGB
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

trainset = datasets.ImageFolder(
    train_dir, transform=train_transform, loader=pil_loader_tif
)

def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest)

def main(epoch_checkpoint=5, batch_size = 64, num_epochs = 100):
    '''epoch_checkpoint: Int that determines how many checkpoints of models are made between saves'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size
    trainset = datasets.ImageFolder(train_dir, transform = train_transform)
    trainloader = DataLoader(dataset = trainset,
                        batch_size = batch_size,
                        num_workers = 4,
                        pin_memory = True,
                        shuffle = True)
    img, label = trainset[0]
    num_classes = len(trainset.classes)
    model = resnet18(num_classes).to(device)
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    criterion = nn.CrossEntropyLoss()
    num_epochs = num_epochs
    args = parse_args()
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

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_accuracy = 100 * correct / total
        log_path = "training_log.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["epoch", "accuracy", "loss", "time"])
            writer.writerow([epoch + 1, epoch_accuracy, epoch_loss, time.time() - epoch_start_time])
        print(f'[Epoch {epoch + 1} / {num_epochs} train acc: {epoch_accuracy:.2f} training loss: {epoch_loss:.2f} Time taken: {time.time() - epoch_start_time} seconds')
        if (epoch+1) % epoch_checkpoint == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss
            }
            torch.save(checkpoint, f'checkpoints/epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()
