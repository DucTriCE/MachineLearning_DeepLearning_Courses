import torch.cuda
import argparse
from src.datasets import AnimalDataset
from torchvision.transforms import ToTensor, Compose, Resize, ColorJitter, Normalize, RandomAffine
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.models import AdvancedCNN
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
import os
import shutil
import numpy as np
from tqdm.autonotebook import tqdm

def get_args():
    parser = argparse.ArgumentParser("""Train model for Animal Dataset""")
    parser.add_argument("--batch_size", "-b",type=int, default=8, help="batch size of dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="number of epochs")
    parser.add_argument("--log_path", "-l", type=str, default="./tensorboard/animal", help="path to tensorboard")
    parser.add_argument("--save_path", "-s", type=str, default="./trained_model/animal", help="path to save model")
    parser.add_argument("--load_checkpoint", "-m", type=str, default=None, help="path to checkpoint loaded")
    args = parser.parse_args()
    return args

def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    num_epochs = args.epochs
    batch_size = args.batch_size
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

    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    writer = SummaryWriter(args.log_path)
    best_acc = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour='green')
        for iter, (img, label) in enumerate(progress_bar):
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
            train_loss.append(loss.item())
            writer.add_scalar("Train/Loss", np.mean(train_loss), num_iters*epoch+iter)
            progress_bar.set_description("Epoch: {}/{}. Loss {:0.4f}".format(epoch+1, num_epochs, np.mean(train_loss)))

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
        acc = accuracy_score(all_labels, all_predictions)
        writer.add_scalar("Val/Accuracy", accuracy_score(all_labels, all_predictions), epoch)

        # Save model here
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.save_path, "last.pth"))
        if acc>best_acc:
            torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))
            best_acc = acc

    # print(args.batch_size)
    # print(args.epochs)
if __name__ == '__main__':
    args = get_args()
    train(args)