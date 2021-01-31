import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import sys

def preprocess(image_list):
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images

def cal_acc(gt, pred):
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    return max(acc, 1-acc)

def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    # First Dimension Reduction
    transformer = KernelPCA(n_components=200, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)
    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2).fit_transform(kpca)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

class AE(nn.Module):
    def __init__(self, dim):
        super(AE, self).__init__()
        self.dim = dim
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32))
        
        self.encoder2 = nn.Sequential(
            nn.Linear(4*4*32, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.dim))
        
        self.decoder1 = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 4*4*32))
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.ConvTranspose2d(16, 8, 2, 2),
            nn.ConvTranspose2d(8, 3, 2, 2),
            nn.Tanh())
    def forward(self, x):
        encoded1 = self.encoder1(x)
        encoded1 = encoded1.view(-1, 4*4*32)
        encoded = self.encoder2(encoded1)
        decoded1 = self.decoder1(encoded)
        decoded1 = decoded1.view(decoded1.size(0), 32, 4, 4)
        decoded = self.decoder2(decoded1)
        return encoded, decoded

if __name__ == '__main__':
  trainX = np.load(sys.argv[1])
  trainX_preprocessed = preprocess(trainX)
  img_dataset = Image_Dataset(trainX_preprocessed)
  same_seeds(0)
  model = AE(128).cuda()
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
  model.train()
  n_epoch = 100
  # 準備 dataloader, model, loss criterion 和 optimizer
  img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)
  epoch_loss = 0
  for epoch in range(n_epoch):
      epoch_loss = 0
      for data in img_dataloader:
          img = data
          img = img.cuda()

          output1, output = model(img)
          loss = criterion(output, img)
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          if (epoch+1) % 10 == 0:
              torch.save(model.state_dict(), './model/1217.pickle')
          epoch_loss += loss.item()
              
      print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, epoch_loss))
  torch.save(model.state_dict(), './model/1217.pickle')

  model = AE(128).cuda()
  model.load_state_dict(torch.load("./model/1217.pickle"))
  model.eval()
  latents = inference(X=trainX, model=model)
  pred, X_embedded = predict(latents)
  #save_prediction(pred, sys.argv[2])
  save_prediction(pred, sys.argv[2])