import warnings
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from gensim.models import Word2Vec
import torch
from torch.utils import data
import os
import argparse
import sys
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def load_testing_data(path='testing_data.txt'):
    # 把 testing 時需要的 data 讀進來
    with open(path, 'r') as f:
        lines = f.readlines()
       
        #x=[line for line in lines]
        #print(len(x))
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        #print(len(X))
        X = [sen.split(' ') for sen in X]
        #print(len(X))
    return X

def evaluation(outputs, labels):
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為有惡意
    outputs[outputs<0.5] = 0 # 小於 0.5 為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

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

def test(model,embedding):
  print('\nload model ...')
  model = model.to(device) 
  model.eval()
  results = []
  with torch.no_grad():
      for comments in test_loader:
          comments = comments.to(device, dtype=torch.long)
          outputs = model(comments)
          predict = torch.max(outputs, 1)[1]
          results += predict.cpu().tolist()
  print("Finish Predicting")
  return results

def output_ans(result, path):
  print(f'Writing ans to {path}')
  dirname = os.path.dirname(path)
  if dirname and not os.path.exists(dirname):
      os.makedirs(dirname) 
  df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
  df.to_csv(path, index=False)

import torch
from torch import nn
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

if __name__ == '__main__':
  path_prefix = './'
  sen_len = 15
  fix_embedding = True # fix embedding during training
  batch_size = 32 #128
  os.system('wget https://www.dropbox.com/s/q4zh8oaatf5mmv2/w2v_all20.model --no-check-certificate')
  os.system('wget https://www.dropbox.com/s/9e9fxrb2l5p3doc/1211.pickle --no-check-certificate')
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("loading testing data . .")
  testing_data = os.path.join(sys.argv[1])
  model_dir = 'model/'
  w2v_path = os.path.join(path_prefix, 'w2v_all20.model') # 處理 word to vec model 的路徑
  print("loading testing data ...")
  test_x = load_testing_data(testing_data)
  preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
  embedding = preprocess.make_embedding(load=True)
  test_x = preprocess.sentence_word2idx()
  test_dataset = TwitterDataset(X=test_x, y=None)
  test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                              batch_size = batch_size,
                                              shuffle = False,
                                              num_workers = 8)
  model = LSTM_Net(embedding, embedding_dim=256, hidden_dim=5, num_layers=1, dropout=0.5, fix_embedding=True)
  model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）
  model.load_state_dict(torch.load("./1211.pickle"))
  results = test(model,embedding)
  print("save csv ...")
  output_ans(results, sys.argv[2])
