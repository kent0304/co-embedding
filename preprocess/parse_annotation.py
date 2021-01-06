import os
import json 
import glob
import nltk
nltk.download('punkt')
import torch
from torch import nn
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from gensim.models import KeyedVectors
from gensim.models import FastText
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel

# GPU対応
device = torch.device('cuda:0')


def build_img2cap(json_obj):
    # image idをkey キャプションをvalue
    img2cap = {}
    for dic in json_obj["annotations"]:
        if dic["image_id"] in img2cap:   
            img2cap[dic["image_id"]].append(dic["caption"]) 
        else:
            caption = [dic["caption"]]
            img2cap[dic["image_id"]] = caption
    return img2cap

def get_images(path):
    files = os.listdir(path)
    imgs = [os.path.join(path,f) for f in files if os.path.isfile(os.path.join(path,f))]
    return imgs

def image2vec(image_net, image_paths):
    # 画像を Tensor に変換
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # stackはミニバッチに対応できる
    images = torch.stack([
        transformer(Image.open(image_path).convert('RGB'))
        for image_path in image_paths
    ])
    images = images.to(device)
    images = image_net(images)
    return images.cpu()

def make_imagedata(imgs):
    # resnet呼び出し
    image_net = models.resnet50(pretrained=True)
    image_net.fc = nn.Identity()
    image_net.eval()
    image_net = image_net.to(device)
    data_num = len(imgs)
    image_vec = torch.zeros((data_num, 2048))
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            image_paths = [imgs[j] for j in range(i, len(imgs))[:batch_size]]
            images = image2vec(image_net, image_paths)
            image_vec[i:i + batch_size] = images

            # if i >= 10*batch_size:
            #     exit(0)
        
    return image_vec

def load_fasttext():
    # # 1回目（modelが保存されていない時）
    # # gensimからfastTextの学習済み学習済み単語ベクトル表現を利用
    # model = KeyedVectors.load_word2vec_format('/mnt/LSTA5/data/tanaka/fasttext/cc.en.300.vec.gz', binary=False)
    # with open('/mnt/LSTA5/data/tanaka/fasttext/gensim-vecs.cc.en.300.vec.pkl', mode='wb') as fp:
    #     pickle.dump(model, fp)

    # 2回目以降（modelが保存されている時）
    with open('/mnt/LSTA5/data/tanaka/fasttext/gensim-vecs.cc.en.300.vec.pkl', mode='rb') as fp:
        model = pickle.load(fp)
    return model

def doc2vec(imgs_list, img2cap, fasttext_model):
    text_vec = torch.zeros(len(imgs_list), 10, 300)
    for i, img in enumerate(tqdm(imgs_list, total=len(imgs_list))):
        # new_captions = torch.zeros(len(img2cap[int(img[-16:-4])]), 300)
        for j, caption in enumerate(img2cap[int(img[-16:-4])]):
            morph = nltk.word_tokenize(caption)
            sen2vec = torch.zeros(300,)
            cnt = 0
            for token in morph:
                if token in fasttext_model:
                    sen2vec += fasttext_model[token]
                    cnt += 1
            sen2vec = sen2vec/cnt
            text_vec[i][j] += sen2vec

    return text_vec

def make_dataset(imagevec, textvec):
    data_num = len(textvec) * len(textvec[0])
    imagedata = torch.zeros(data_num,2048)
    textdata = torch.zeros(data_num,300)
    # print(imagedata)
    # print(type(imagedata))
    # print(imagedata.shape)
    # print(textdata)
    # print(type(textdata))
    # print(textdata.shape)
    idx = 0
    for i, image in enumerate(tqdm(imagevec, total=len(imagevec))):
        for caption in textvec[i]: 
            if not torch.equal(caption, torch.zeros(300)):
                textdata[idx] = caption
                imagedata[idx] = image 
                idx += 1

            else:
                continue
    
    dataset = (imagedata[:idx], textdata[:idx])
    return dataset


def main():
    # open file
    json_train2017 = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/annotations/captions_train2017.json', 'r')
    json_val2017 = open('/mnt/LSTA5/data/tanaka/lang-learn/coco/annotations/captions_val2017.json', 'r')
    
    # read as json
    train2017 = json.load(json_train2017)
    val2017 = json.load(json_val2017)
    
    # Get image2caption dictionary
    train_img2cap = build_img2cap(train2017)
    val_img2cap = build_img2cap(val2017)


    # Get images path as a list
    trainimg_path = '/mnt/LSTA5/data/tanaka/lang-learn/coco/train2017'
    valimg_path = '/mnt/LSTA5/data/tanaka/lang-learn/coco/val2017'
    train_imgs = get_images(trainimg_path)
    valid_imgs = get_images(valimg_path)
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_imagepaths.pkl', 'wb') as f:
        pickle.dump(train_imgs, f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/valid2017_imagepaths.pkl', 'wb') as f:
        pickle.dump(valid_imgs, f) 

   

    # Make image data 2048 dim
    print("画像をベクトル化")
    train_images = make_imagedata(train_imgs)
    valid_images = make_imagedata(valid_imgs)
    print("picleで保存")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_images.pkl', 'wb') as f:
        pickle.dump(train_images, f) 
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_images.pkl', 'wb') as f:
        pickle.dump(valid_images, f) 


    # imageid 2 text 300dim
    print("テキストをベクトル化")
    fasttext_model = load_fasttext()
    train_textvec = doc2vec(train_imgs, train_img2cap, fasttext_model)
    print("picleで保存")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_textvec.pkl', 'wb') as f:
        pickle.dump(train_textvec, f) 
    
    val_textvec = doc2vec(valid_imgs, val_img2cap, fasttext_model)
    print("picleで保存")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_textvec.pkl', 'wb') as f:
        pickle.dump(val_textvec, f) 
    print("保存完了")



    # テキストと画像を一つにまとめる
    print("Reading train image (118287, 2048) dim data as tensor...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_images.pkl', 'rb') as f:
        train_imagevec = pickle.load(f) 
    print("Reading valid image (5000, 2048) dim data as tensor...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_images.pkl', 'rb') as f:
        valid_imagevec = pickle.load(f)

    print("Reading train text (118287, 10, 300) dim data as tensor...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_textvec.pkl', 'rb') as f:
        train_textvec = pickle.load(f) 
    print("Reading valid text (5000, 10, 300) dim data as tensor...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_textvec.pkl', 'rb') as f:
        valid_textvec = pickle.load(f) 

    print("train: (画像, テキスト）のタプルで対応させて保存")
    train2017_datasetvec = make_dataset(train_imagevec, train_textvec)
    print("保存中...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_datasetvec.pkl', 'wb') as f:
        pickle.dump(train2017_datasetvec, f, protocol=4) 
    print("valid: (画像, テキスト）のタプルで対応させて保存")
    val2017_datasetvec = make_dataset(valid_imagevec, valid_textvec)
    print("保存中...")
    with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_datasetvec.pkl', 'wb') as f:
        pickle.dump(val2017_datasetvec, f, protocol=4) 

    





    


   
    


  
    

    


if __name__ == '__main__':
    main()