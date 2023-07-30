import torch.cuda
from src.dataset import MyDataset
from torch.utils.data import DataLoader
from src.models import CNN
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision.transforms import ToTensor, Compose, Resize
from src.val import SegmentationMetric, AverageMeter

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    num_epochs = 10
    batch_size = 8
    train_set = MyDataset(transform=False, valid=False)
    train_dataloader = DataLoader(
        dataset = train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_set = MyDataset(transform=False, valid=True)
    val_dataloader = DataLoader(
        dataset = val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    total_batches = len(train_dataloader)
    # def train16fp(args, train_dataloader, model, criterion, optimizer, epoch, scaler):
    model.train()
    # scaler = torch.cuda.amp.GradScaler()
    all_labels = []
    all_predictions = []
    for epoch in range(num_epochs):
        for iter, (_, input, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            # print(target.shape)
            output = model(input/255.0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if iter%10==0:
                print("Epoch: {}/{}. Iter {}/{}. Loss {}".format(epoch+1, num_epochs, iter+1, total_batches, loss))
        model.eval()
        metric = SegmentationMetric(2)
        da_mIoU_seg = AverageMeter()
        for iter, (_, input, target) in enumerate(val_dataloader):
            with torch.no_grad():
                output = model(input/255.0)
                metric.reset()
                metric.addBatch(output, target)
                da_mIoU = metric.meanIntersectionOverUnion()
                da_mIoU_seg.update(da_mIoU, input.size(0))
        print(da_mIoU_seg.avg)

if __name__ == '__main__':
    train()