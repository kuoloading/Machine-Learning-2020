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
          y[i] = labels["label"][i]
    if label:
      return x, y
    else:
      return x

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
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
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            #nn.AdaptiveAvgPool2d(2, 2, 0)
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            nn.Dropout(0.25),

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(), 
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
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

def out(ans, name):
    with open(name, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(ans):
            f.write('{},{}\n'.format(i,v))

if __name__ == '__main__':
    test_transform = transforms.Compose([
                     transforms.ToPILImage(),
                     transforms.ToTensor(),
                     ])
    os.system('wget https://www.dropbox.com/s/hvqohyx7mqbfbzu/1114_1.pickle --no-check-certificate')
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("device is:",device)
    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load("./1114_1.pickle", map_location='cpu'))
    model_best.eval()

    test_x = readfile(sys.argv[1], False)
    test_set = ImgDataset(test_x, transform=test_transform)
    batch_size = 128
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model_best.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)
    out(prediction, sys.argv[2])