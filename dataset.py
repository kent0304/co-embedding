import pickle
import torch 
from torch.utils.data import DataLoader, Dataset


# print("Reading train tuple data(image, text) (591753, 2048) (591753, 300) dim data as tensor...")
# with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/train2017_datasetvec.pkl', 'rb') as f:
#     train2017_datasetvec = pickle.load(f) 
# print("Reading train tuple data(image, text)  dim data as tensor...")
# with open('/mnt/LSTA5/data/tanaka/lang-learn/coco/vector/val2017_datasetvec.pkl', 'rb') as f:
#     val2017_datasetvec = pickle.load(f) 


class MyDataset(Dataset):
    def __init__(self, datasetvec):
        self.imagevec = datasetvec[0]
        self.textvec = datasetvec[1]
        self.data_num = len(self.imagevec)
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        image = self.imagevec[idx]
        text = self.textvec[idx]
        return image, text 


# train = MyDataset(train2017_datasetvec)
# valid = MyDataset(val2017_datasetvec)

