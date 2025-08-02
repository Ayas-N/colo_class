import os
import csv
import argparse

import torch
import torch.optim as optim
import time
from torch import nn
from tqdm import tqdm

from PIL import Image
import torchvision.transforms as transforms
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
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest)

def main(batch_size = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testset = datasets.ImageFolder(test_dir, transform = test_transform)
    testloader = DataLoader(dataset = testset,
                    batch_size = batch_size,
                    num_workers = 4,
                    pin_memory = True,
                    shuffle = True)
    
    checkpoint_path = get_latest_checkpoint()
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
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
    progress_bar = tqdm(enumerate(testloader, 0), total=len(testloader), desc="Training", leave=False)
    for i, data in progress_bar:
        images = data[0].to(device)
        labels = data[1].to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


if __name__ == "__main__":
    main()