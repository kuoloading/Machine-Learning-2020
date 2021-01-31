# Import需要的套件
import glob
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import sys

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        try:
          img = cv2.imread(os.path.join(path, file))
          x[i, :, :] = cv2.resize(img,(128, 128))
        except:
            break
        if label:
          #y[i] = int(file.split("_")[0])
          y[i] = labels["label"][i]
    if label:
      return x, y
    else:
      return x

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),
            #nn.AdaptiveAvgPool2d(2, 2, 0)
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, 1, 1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.25),

            nn.Conv2d(512, 512, 3, 1, 1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.LeakyReLU(), 
            nn.Linear(1024, 512),
            nn.LeakyReLU(), 
            nn.Linear(512, 7)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


def train(model, num_epoch, train_loader, data_count):

    loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        correct = 0

        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].to(device)) 
            batch_loss = loss(train_pred, data[1].to(device)) 
            batch_loss.backward() 
            optimizer.step()

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
        print('epoch:', epoch)
    torch.save(model.state_dict(), './1115.pickle')

if __name__ == '__main__':

    labels = pd.read_csv(sys.argv[2])
    print("Reading data")
    train_x, train_y = readfile(sys.argv[1], True)
    print("Size of training data = {}".format(len(train_x)))
    train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(), 
    ])
    batch_size = 128
    train_set = ImgDataset(train_x, train_y, train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = Classifier().to(device)
    print("device :", device)
    train(model, 30, train_loader, len(train_x))