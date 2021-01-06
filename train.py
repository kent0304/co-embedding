import pickle
import random

import MeCab
import numpy as np
import torch
from gensim.models import Word2Vec
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from tqdm import tqdm

from dataset import MyDataset

from model import TripletModel
from matplotlib import pyplot as plt
plt.switch_backend('agg')

device = torch.device('cuda:0')

def load_data():
    # print("Reading train tuple data(image, text) (591753, 2048) (591753, 300) dim data as tensor...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_datasetvec.pkl', 'rb') as f:
        train2017_datasetvec = pickle.load(f) 
    # print("Reading train tuple data(image, text)  dim data as tensor...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_datasetvec.pkl', 'rb') as f:
        val2017_datasetvec = pickle.load(f) 
    train_dataset = MyDataset(train2017_datasetvec)
    valid_dataset = MyDataset(val2017_datasetvec)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)
    return train_dataset, valid_dataset, train_loader, valid_loader

# モデル評価
def eval_net(triplet_model, data_loader, dataset, loss, device):
    triplet_model.eval()
    triplet_model = triplet_model.to(device)
    outputs = []
    for i, (image, text) in enumerate(data_loader):
        # neg selection by random function
        while True:
            negative_image, negative_text = random.choice(dataset)
            if (negative_image in image) or (negative_text in text):
                continue
            else:
                break
 
        with torch.no_grad():
            # GPU setting
            image = image.to(device)
            text = text.to(device)
            negative_image = negative_image.to(device)
            negative_text = negative_text.to(device)
            pos_imagevec, pos_textvec, neg_imagevec, neg_textvec = triplet_model(image, text, negative_image, negative_text)
        output = loss(pos_imagevec, pos_textvec, neg_imagevec) + loss(pos_imagevec, pos_textvec, neg_textvec)
        outputs.append(output.item())

    return sum(outputs) / i 
    

# モデルの学習
def train_net(triplet_model, train_loader, valid_loader, train_dataset, valid_dataset, loss, n_iter, device):
    train_losses = []
    valid_losses = []
    optimizer = optim.SGD(triplet_model.parameters(), lr=0.01)
    for epoch in range(n_iter):
        running_loss = 0.0
        triplet_model = triplet_model.to(device)
        # ネットワーク訓練モード
        triplet_model.train()
        for i, (image, text) in enumerate(tqdm(train_loader, total=len(train_loader))):
            # neg selection by random function
            while True:
                negative_image, negative_text = random.choice(train_dataset)
                if (negative_image in image) or (negative_text in text):
                    continue
                else:
                    break
            
            # GPU setting
            image = image.to(device)
            text = text.to(device)
            negative_image = negative_image.to(device)
            negative_text = negative_text.to(device)

            # triplet_model
            pos_imagevec, pos_textvec, neg_imagevec, neg_textvec = triplet_model(image, text, negative_image, negative_text)
            output = loss(pos_imagevec, pos_textvec, neg_imagevec) + loss(pos_imagevec, pos_textvec, neg_textvec)
   
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            running_loss += output.item()
    
        # 訓練用データでのloss値
        train_losses.append(running_loss / i)
        # 検証用データでのloss値
        pred_valid =  eval_net(triplet_model, valid_loader, valid_dataset, loss, device)
        valid_losses.append(pred_valid)
        print('epoch:' +  str(epoch+1), 'train loss:'+ str(train_losses[-1]), 'valid loss:' + str(valid_losses[-1]), flush=True)
        # 学習モデル保存
        if (epoch+1)%1==0:
            # 学習させたモデルの保存パス
            model_path =  f'model/model_epoch{epoch+1}.pth'
            # モデル保存
            torch.save(triplet_model.to('cpu').state_dict(), model_path)
            # loss保存
            # with open(train_losses_path, 'wb') as f:
            #     pickle.dump(train_losses, f) 
            # with open(valid_losses_path, 'wb') as f:
            #     pickle.dump(valid_losses, f) 
            # グラフ描画
            my_plot(train_losses, valid_losses)

def my_plot(train_losses, valid_losses):
    # グラフの描画先の準備
    fig = plt.figure()
    # 画像描画
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    #グラフタイトル
    plt.title('Triplet Margin Loss')
    #グラフの軸
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #グラフの凡例
    plt.legend()
    # グラフ画像保存
    fig.savefig("loss.png")

def main():
    train_dataset, valid_dataset, train_loader, valid_loader = load_data()
    print("データのロード完了")
    triplet_loss = nn.TripletMarginLoss(margin=2)
    triplet_model = TripletModel()
    print("訓練開始")
    train_net(triplet_model=triplet_model, 
              train_loader=train_loader, 
              valid_loader=valid_loader, 
              train_dataset=train_dataset, 
              valid_dataset=valid_dataset, 
              loss=triplet_loss, 
              n_iter=700, 
              device=device)

if __name__ == '__main__':
    main()