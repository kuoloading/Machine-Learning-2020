import warnings
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from gensim.models import Word2Vec
from gensim.models import word2vec
import torch
from torch import nn
from torch.utils import data
import os
import argparse
from sklearn.model_selection import train_test_split
import os
import sys
warnings.filterwarnings('ignore')

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    def get_w2v_model(self):
        # 把之前訓練好的 word to vec 模型讀進來
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
    def add_embedding(self, word):
        # 把 word 加進 embedding，並賦予他一個隨機生成的 representation vector
        # word 只會是 "<PAD>" 或 "<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得訓練好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 將 "<PAD>" 跟 "<UNK>" 加進 embedding 裡面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    def pad_sequence(self, sentence):
        # 將每個句子變成一樣的長度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    def sentence_word2idx(self):
        # 把句子裡面的字轉成相對應的 index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 將每個句子變成一樣的長度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把 labels 轉成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

class TwitterDataset(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)

def train_word2vec(x,iter):
    model = word2vec.Word2Vec(x, size=256, window=5, min_count=8, workers=12, iter=iter, sg=1)
    return model

def load_training_data(path='training_label.txt'):
    # 把 training 時需要的 data 讀進來
    # 如果是 'training_label.txt'，需要讀取 label，如果是 'training_nolabel.txt'，不需要讀取 label
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
 
            lines = [line.strip('\n').split(' ') for line in lines]
        print(lines[:5])
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def evaluation(outputs, labels):
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為有惡意
    outputs[outputs<0.5] = 0 # 小於 0.5 為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.8, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True )
        self.classifier = nn.Sequential(
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_dim*2, 32),
                                        nn.LeakyReLU(),
                                        nn.BatchNorm1d(32)
                                       )
        self.fc2 = nn.Linear(32, 2)
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        x = x[:, -1, :] 
        x = self.classifier(x)
        x = self.fc2(x)
        return x

def train(model, embedding_dim, hidden_size,batch_size,lr,epochs,n_l):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    best_acc = 0
    history = {'val_loss': [], 'val_acc': [], 'train_loss': [], 'train_acc': []}
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_acc = []
        for comments, labels in train_loader:
            if use_gpu:
                comments = comments.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(comments)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            predict = torch.max(outputs, 1)[1]
            acc = np.mean((labels == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        print(f'Epoch: {"%03d" % (epoch)}, train loss: {"%.4f" % train_loss}, train acc: {"%.4f" % train_acc}')
        # validation
        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for comments, labels in val_loader:
                if use_gpu:
                    comments = comments.cuda()
                    labels = labels.cuda()
                outputs = model(comments)
                loss = loss_fn(outputs, labels)
                predict = torch.max(outputs, 1)[1]
    #             print(predict)
                acc = np.mean((labels == predict).cpu().numpy())
                valid_acc.append(acc)
                valid_loss.append(loss.item())
            valid_loss = np.mean(valid_loss)
            valid_acc = np.mean(valid_acc)
            print(f'Epoch: {"%03d" % (epoch)}, valid loss: {"%.4f" % valid_loss}, valid acc: {"%.4f" % valid_acc}')
            if valid_acc > best_acc:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                best_acc = valid_acc
                torch.save(model.state_dict(), './1211.pickle')
                print('saving model with acc {:.3f}'.format(valid_acc))

if __name__ == "__main__":
    path_prefix = "./"
    sen_len = 15
    batch_size = 32 #128
    print("loading training data ...")
    train_x, y = load_training_data(sys.argv[1])
    train_x_no_label = load_training_data(sys.argv[2])
    model = train_word2vec(train_x + train_x_no_label,iter=20)    #model = train_word2vec(train_x)
    print("saving model ...")
    model.save(os.path.join(path_prefix, 'w2v_all20.model'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_with_label = os.path.join(path_prefix, 'training_label.txt')
    train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')
    w2v_path = os.path.join(path_prefix, 'w2v_all20.model') # 處理 word to vec model 的路徑

    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)
    preprocess = Preprocess(train_x_no_label, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x_no_label = preprocess.sentence_word2idx()
    X_train, X_val, y_train, y_val = train_x[:160000], train_x[160000:], y[:160000], y[160000:]
    print(X_train[:10])
    print(y_train[:10])

    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               num_workers = 8)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = 8)
    model = LSTM_Net(embedding, embedding_dim=256, hidden_dim=5, num_layers=1, dropout=0.8, fix_embedding=True)
    model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）

    train(model = model, embedding_dim = 200, hidden_size = 96 ,batch_size = 64 ,lr = 5e-4 ,epochs = 40 ,n_l = 3)