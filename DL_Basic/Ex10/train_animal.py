import torch.cuda
import argparse
from src.datasets import AnimalDataset
from torchvision.transforms import ToTensor, Compose, Resize, ColorJitter, Normalize, RandomAffine
from torch.utils.data import DataLoader
from src.models import AdvancedCNN
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score


def get_args():
    parser = argparse.ArgumentParser("""Train model for Animal Dataset""")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size of dataset")
    args = parser.parse_args()
    return args

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    num_epochs = 100
    batch_size = 4
    train_transforms = Compose([
        Resize(size=(224,224)),
        ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
        RandomAffine(degrees=(-5,5), translate=(0.15, 0.15), scale=(0.9, 1.1)),
        ToTensor()
    ])
    train_set = AnimalDataset(root= 'data', train=True, transform=train_transforms)
    train_dataloader = DataLoader(
        dataset = train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_transforms = Compose([
        Resize(size=(224,224)),
        ToTensor()
    ])
    val_set = AnimalDataset(root='data', train=False, transform=val_transforms)
    val_dataloader = DataLoader(
        dataset = val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    model = AdvancedCNN(10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_iters = len(train_dataloader)

    for epoch in range(num_epochs):
        model.train()
        for iter, (img, label) in enumerate(train_dataloader):
            img = img.to(device)
            label = label.to(device)
            #Forward
            output = model(img)
            loss = criterion(output, label)
            # print(loss)

            #Backward + Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter%10==0:
                print("Epoch: {}/{}. Iter {}/{}. Loss {}".format(epoch+1, num_epochs, iter+1, num_iters, loss))
        model.eval()
        all_labels = []
        all_predictions = []
        for iter, (img, label) in enumerate(val_dataloader):
            img = img.to(device)
            label = label.to(device)
            with torch.no_grad():
            # with torch.inference_mode():
                output = model(img)
                _, predictions = torch.max(output, dim=1)
                all_predictions.extend(predictions.tolist())
                all_labels.extend(label.tolist())
        print(accuracy_score(all_labels, all_predictions))

if __name__ == '__main__':
    train()