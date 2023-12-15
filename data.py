import torch
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import os
import json
from copy import deepcopy
from multiprocessing import Pool
import argparse
import time

device = torch.device("cpu")

def collate_fn(datas):
    # print('collate input',datas)

    max_len=-1
    for data in datas:
        if len(data[0])>max_len:
            max_len=len(data[0])
    # print('max_len',max_len)
    n_samples = len(datas)
    state_size = len(datas[0][0][0])
    # print('max_len',max_len)
    # new_data_x=torch.FloatTensor([[[0]*len(datas[0][0][0]) for _ in range(max_len)] for __ in range(len(datas)) ]).to(device)
    new_data_x = torch.zeros((n_samples, max_len, state_size), dtype=torch.float, device=device)
    # print(new_data_x.size())
    # new_data_y=torch.FloatTensor([[[0] for _ in range(max_len)] for __ in range(len(datas)) ]).to(device)
    new_data_y = torch.zeros((n_samples, max_len, 1), dtype=torch.float, device=device)
    # print(new_data_y.size())
    # predict_mask=torch.FloatTensor([[[0] for _ in range(max_len)] for __ in range(len(datas)) ]).to(device)
    predict_mask = torch.zeros((n_samples, max_len, 1), dtype=torch.float, device=device)
    # print(predict_mask.size())

    cnt=0
    for data in datas:
        data_x=data[0]
        if len(data[0])<max_len:
            data_x=torch.cat((torch.FloatTensor([[0]*len(data[0][0]) for _ in range(max_len-len(data[0]))]).to(device),data_x))
        new_data_x[cnt]=data_x
        new_data_y[cnt][max_len-len(data[0])][0]=data[1]
        predict_mask[cnt][max_len-len(data[0])][0]=1
        cnt+=1
    # new_data_x=torch.FloatTensor(new_data_x)
    # new_data_y=torch.FloatTensor(new_data_y)
    # print('newx',new_data_x)
    # print('newy',new_data_y)
    # print('start_idx',start_idx)

    return new_data_x,new_data_y,predict_mask
# torch.save(net.state_dict(), save_name)
class Trace_Dataset(Dataset):
    def __init__(self, raw_data):
        self.item = raw_data
        for i in range(len(self.item)):
            self.item[i][0]=torch.FloatTensor(self.item[i][0]).to(device)

    def __getitem__(self, idx):
        return self.item[idx][0],self.item[idx][1]

    def __len__(self):
        return len(self.item)


def list2tuple(ltl,vocab):
    if len(ltl)==0:
        return ltl
    if ltl in vocab:
        return ltl
    if len(ltl)==2:
        return (ltl[0],list2tuple(ltl[1],vocab))
    return (ltl[0],list2tuple(ltl[1],vocab),list2tuple(ltl[2],vocab))

def get_data(fname,word2idx=-1,vocab=-1):
    with open(fname,'r') as f:
        data=json.load(f)

    if word2idx==-1:
        word2idx={}
        vocab=data['vocab']+['true']
        cnt=0
        for v in vocab:
            word2idx[v]=cnt
            cnt+=1

    datas=[]
    # print('pos traces:',len(data['traces_pos']),'neg traces',len(data['traces_neg']))
    cnt=0
    for trace in data['traces_pos']:
        cnt+=1
        # print(trace)
        idx_trace=[]
        for state in trace:
            idx_state=[0]*(len(vocab))
            for v in state:
                idx_state[word2idx[v]]=1
            idx_state[word2idx['true']]=1

            idx_state=idx_state
            idx_trace.append(idx_state)
        datas.append([idx_trace,1])
    # print('pos_trace',len(datas))
    cnt=0
    for trace in data['traces_neg']:
        cnt+=1
        idx_trace=[]
        for state in trace:
            idx_state=[0]*(len(vocab))
            for v in state:
                idx_state[word2idx[v]]=1
            idx_state[word2idx['true']] = 1
            idx_state=idx_state
            idx_trace.append(idx_state)
        datas.append([idx_trace, 0])

    # print('pos_trace+neg_traces', len(datas))
    # for data in datas:
    #     print(data)
    # print(datas)
    dataset=Trace_Dataset(datas)
    return dataset,word2idx,vocab,list2tuple(data['ltlftree'],vocab)
