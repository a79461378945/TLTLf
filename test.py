import torch
import torch.nn.functional as F     # 激励函数都在这
from torch.nn.parameter import Parameter
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

from data import get_data,collate_fn
from model import Net
os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cpu")
# device = torch.device(
#     "cuda") if torch.cuda.is_available() else torch.device("cpu")

def test(test_args,word2idx=-1,vocab=-1):
    test_file=test_args.test_file
    model_file=test_args.save_model
    save_file=model_file+test_file.split('/')[-1]
    if os.path.exists(save_file):
        with open(save_file,'r') as f:
            datas=json.load(f)
        return datas

    model_param = torch.load(model_file)
    # print(model_param)
    # exit(0)

    test_dataset, word2idx, vocab, _ = get_data(test_file, word2idx, vocab)
    formula_len=len(model_param['chose_right.weight'])
    vocab_len=len(vocab)
    model = Net(formula_len,vocab_len)
    model.load_state_dict(model_param)

    torch.set_num_threads(1)

    torch.set_printoptions(threshold=10000,precision=1,sci_mode=False)
    # print((model.state_dict()['predict.weight']).cpu())
    # print((model.state_dict()['predict.bias']).cpu())

    total_cnt=0
    correct_cnt=0
    test_result=[]
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            data = test_dataset[idx]
            prediction = model(torch.reshape(data[0],(1,len(data[0]),len(data[0][0]))).to(device))[0][0][0]
            if (data[1]-0.5)* (prediction-0.5) >0:
                correct_cnt+=1
            total_cnt+=1
            test_result.append([1 if data[1] >0.5 else 0, 1 if prediction>0.5 else 0])



    # print('correct',correct,'TP,FP,FN',(TP,FP,FN),'total',len(E_T_dic['traces_pos']+E_T_dic['traces_neg']))
    # a = input()
    # print('TP',TP,'FP',FP,'FN',FN)
    with open(save_file,'w') as f:
        json.dump(test_result,f)
    return test_result
    # return (correct_cnt/total,TP/(TP+FP),TP/(TP+FN),test_result)  # acc,pre,rec

    # return {'correct':correct_cnt,'total':total_cnt,'result':test_result}

def get_test_argument(args_list=None):
    test_parser = argparse.ArgumentParser(description='Main script for test')
    test_parser.add_argument('--test_file', type=str, required=True)
    test_parser.add_argument('--save_model', type=str, required=True)
    if args_list==None:
        test_args = test_parser.parse_args()
    else:
        test_args = test_parser.parse_args(args_list)
    # train(train_args)
    return test_args



if __name__ == '__main__':
    args = get_test_argument(['--test_file','data/test.json','--save_model','model/tmodel'])
    result=test(args)
    print(result)
    print('network accuracy:',sum([1 if i[0]==i[1] else 0 for i in result])/len(result))
