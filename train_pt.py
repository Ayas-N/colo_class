import os

import torch
import torch.optim as optim
from torch import nn
import time

import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

train_dir = "../colorectal/train0"
test_dir = "../colorectal/test/"

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = datasets.ImageFolder(train_dir, transform = train_transform)
    trainloader = DataLoader(dataset = trainset,
                        batch_size = 4,
                        shuffle = True)

    trainset = datasets.CIFAR10(root='/tmp/CIFAR10', train=True,
                                            download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=16,
                                        shuffle=True, num_workers=2)
    num_epochs = 5
    num_classes = len(trainset.classes)
    model = models.resnet18().to(device)
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=5e-4,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)
    criterion = nn.CrossEntropyLoss()
    total, correct, predicted = 0, 0, 0
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        print(epoch)
        for i, data in enumerate(trainloader,0):
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
        print(f'[Epoch {epoch + 1} / {num_epochs} train acc: {epoch_accuracy} training loss: {epoch_loss} Time taken: {time.time() - epoch_start_time} seconds')

if __name__ == "__main__":
    main()
