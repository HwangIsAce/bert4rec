import random
import numpy as np
import copy
import os
import torch
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split

from einops import rearrange, repeat

import bootstrap
from constants import TRAIN_CONSTANTS

b4r_config = bootstrap.B4RConfig(
    path = "/home/jaesung/jaesung/study/bert4rec/config.json"
)

train_constants = TRAIN_CONSTANTS()

def seed_everything(seed = 21):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def make_list_of_seq(df, seq_len=5):
    df.reset_index(inplace=True)
    df.sort_values(['user_id', 'date'], inplace=True)
    df = df.groupby('user_id')['recipe_id'].apply(list).to_frame()
    df.reset_index(inplace=True)
    df.rename(columns={'recipe_id':'user_seq'}, inplace=True)

    seq_len_tmp = []
    for v in df['user_seq']:
        seq_len_tmp.append(len(v))
    df['seq_len'] = seq_len_tmp

    df = df[df['seq_len'] > seq_len]

    return list(df['user_seq'])

def list_to_list_of_tensor(_list: list)-> list :
    
    tensor_list = []
    for i, v in enumerate(_list):
        tensor_list.append(torch.tensor(v))

    return tensor_list

def make_random_mask(_list : list) -> list: # 수정하기
    
    masked_tensor_list = []
    for _tensor in _list :
        _mask = torch.rand(_tensor.shape[0])
        _mask = _mask < 0.15
        # _mask = (torch.randint(low=0, high=2, size=(len(_tensor),))) # 비율
        masked_tensor = _tensor.masked_fill(_mask == 1, train_constants.MASK) 
        masked_tensor_list.append(masked_tensor.int())

    return masked_tensor_list

def make_last_mask(_list: list) -> list: # labels 는 왜..?
    
    masked_tensor_list = []
    for v in _list:
        tmp = v
        tmp[-1] = train_constants.MASK
        masked_tensor_list.append(tmp)

    return masked_tensor_list

def preprocessing():
    # data path
    df_path = b4r_config.df

    # data load
    raw_df = pd.read_csv(df_path, index_col=0)

    seq_list = make_list_of_seq(raw_df)

    train_seq_list, test_seq_list = train_test_split(seq_list, test_size=0.10, random_state=21)

    train_seq_list, valid_seq_list = train_test_split(train_seq_list, test_size=0.10, random_state=21)

    # train
    ## labels of train_seq_list 
    train_labels = list_to_list_of_tensor(train_seq_list)
    train_labels_temp = copy.deepcopy(train_labels)
    ## input_ids of train_seq_list
    train_input_tensor = make_random_mask(train_labels_temp)

    # valid
    ## labels of valid_seq_list
    valid_labels = list_to_list_of_tensor(valid_seq_list)
    valid_labels_temp = copy.deepcopy(valid_labels)
    ## input_ids of valid_seq_list    
    valid_input_tensor = make_last_mask(valid_labels_temp)

    # test
    ## labels of tes_seq_list
    test_labels = list_to_list_of_tensor(test_seq_list)
    test_labels_temp = copy.deepcopy(test_labels)
    ## input_ids of test_seq_list
    test_input_tensor = make_last_mask(test_labels_temp)

    input_ids= {} ; labels = {}
    input_ids['train'] = train_input_tensor
    input_ids['valid'] = valid_input_tensor
    input_ids['test'] = test_input_tensor

    labels['train'] = train_labels
    labels['valid'] = valid_labels
    labels['test'] = test_labels

    return input_ids, labels        


class TensorData(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels
        self.len = len(self.labels)

    def __getitem__(self, index):
        return {"input_ids" : self.input_ids[index], "labels" : self.labels[index]}

    def __len__(self):
        return self.len

def collate_fn(batch):
    collate_input_ids = []
    collate_labels = []

    max_len = max([len(sample['input_ids']) for sample in batch])

    for sample in batch:
        diff = max_len - len(sample['input_ids'])
        if diff > 0 :
            zero_pad = torch.zeros(size= (diff,))

            collate_input_ids.append(torch.cat([sample['input_ids'].view([len(sample['input_ids'])]), zero_pad], dim=0))
            collate_labels.append(torch.cat([sample['labels'].view([len(sample['labels'])]), zero_pad], dim=0))
        else :
            collate_input_ids.append(sample['input_ids'].view(len(sample['input_ids'])))
            collate_labels.append(sample['labels'].view(len(sample['labels'])))

    return {'input_ids': torch.stack(collate_input_ids), 'labels' : torch.stack(collate_labels)}

def MyDataLoader(batch_size=32):

    seed_everything()
    input_ids, labels = preprocessing()     

    train_data = TensorData(input_ids['train'], labels['train'])
    valid_data = TensorData(input_ids['valid'], labels['valid'])
    test_data = TensorData(input_ids['test'], labels['test'])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    return train_loader, valid_loader, test_loader

if __name__ == "__main__":

    train_loader, valid_loader, test_loader = MyDataLoader(batch_size=32) # input_ids tensor type -> float?
