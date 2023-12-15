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

def train(train_args):

    train_file=train_args.train_file
    formula_dep=train_args.netlen
    save_name=train_args.save_model
    epoch_num=train_args.epoch
    log_file=train_args.log_file
    learn_rate = train_args.lr
    batch_size = train_args.batch_size
    test_correctness = train_args.test_correctness

    stime=time.time()
    train_dataset,word2idx,vocab,ltlf_tree=get_data(train_file)
    torch.set_printoptions(threshold=10000, precision=1, sci_mode=False)
    min_loss = 1e6

    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)


    writer = SummaryWriter(log_file)

    torch.set_num_threads(1)
    # test_correctness=True
    if test_correctness:
        print('testing ',ltlf_tree)
        model = Net(formula_dep,len(vocab),ltlf_tree,word2idx)
    else:
        model = Net(formula_dep, len(vocab))
    model.to(device)
    torch.save(model.state_dict(), save_name)
    # print('chose left',torch.softmax(model.chose_left.weight,dim=1))
    # print('chose right',torch.softmax(model.chose_right.weight,dim=1))
    # print('chose op', torch.softmax(model.chose_op.weight.transpose(0,1), dim=1))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learn_rate)
    wrongmark=0
    for epoch in range(epoch_num):
        # print('model_time',model_time,'loss_time',loss_time-model_time)
        epoch_loss=0
        for data_x,data_y,predict_mask in train_dataloader:
            prediction = model(data_x)
            loss = torch.sum((torch.mul(prediction[:,:,:1],predict_mask)-data_y)**2)
            epoch_loss+= loss.detach().cpu().numpy()

            if test_correctness:
                for i in range(len(prediction)):
                    if torch.sum((torch.mul(prediction[i][:,:1],predict_mask[i])-data_y[i])**2)>0.1 and wrongmark==0:
                        print(data_x[i], prediction[i],data_y[i])
                        print('something wrong:',ltlf_tree,torch.sum((torch.mul(prediction[i][:,:1],predict_mask[i])-data_y[i])**2))
                        wrongmark=1
            loss.backward(retain_graph=True)

            loss2=[]
            # model.cal_softmax()
            temp_a = model.chose_right_.transpose(0, 1)
            temp_b = model.op_chose_weight.transpose(0, 1)
            for i in range(1,formula_dep):
                (train_args.a1*torch.pow((torch.sum(temp_a[i])+torch.sum(temp_b[i-1])-1),2)).backward(retain_graph=True)

            # for i in range(formula_dep-1):
            torch.sum(train_args.a2*torch.relu(model.op_chose_weight[0][:formula_dep-1]-model.op_chose_weight[0][1:])).backward(retain_graph=True)
            for i in range(formula_dep-2):
                for j in range(i+2,formula_dep):
                    # for t1 in range(i+1,j):
                    #     for t2 in range(j+1,formula_dep):
                    torch.sum(torch.relu(model.chose_right_[i][j]+model.chose_right_[i+1:j][j+1:formula_dep]-1)*train_args.a3).backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()



        if epoch_loss<min_loss:
            min_loss=epoch_loss
            torch.save(model.state_dict(), save_name)

        writer.add_scalar("Loss/train-%s-netlen%d"%(train_file.split('/')[-1],formula_dep), epoch_loss, epoch)

        # for param in model.parameters():
        #     print(param)
        if epoch % 10 == 0 and epoch>0:
            if epoch_loss<1:
                break
            print(
                'train_file:%s \t epoch:%d \t loss:%f \t %d time:%f' % (train_file, epoch, epoch_loss, len(train_dataset),time.time()-stime))

    writer.close()
    return time.time()-stime

def get_train_argument(args_list=None):
    train_parser = argparse.ArgumentParser(description='Main script')
    train_parser.add_argument('--train_file', type=str, required=True)
    train_parser.add_argument('--save_model', type=str, required=True)
    train_parser.add_argument('--log_file', type=str, required=True)
    train_parser.add_argument('--netlen', type=int, required=False, default=10)
    train_parser.add_argument('--epoch', type=int, required=False, default=100, help='epoch number')
    train_parser.add_argument('--lr', type=float, required=False, default=1e-2, help='learn rate')
    train_parser.add_argument('--a1', type=float, required=False, default=0.1, help='regular cof')
    train_parser.add_argument('--a2', type=float, required=False, default=0.1, help='regular cof')
    train_parser.add_argument('--a3', type=float, required=False, default=0.1, help='regular cof')
    train_parser.add_argument('--batch_size', type=int, required=False, default=100, help='batch size')
    train_parser.add_argument('--test_correctness', type=int, required=False, default=0, help='regular cof')
    if args_list==None:
        train_args = train_parser.parse_args()
    else:
        train_args = train_parser.parse_args(args_list)
    # train(train_args)
    return train_args



if __name__ == '__main__':
    args = get_train_argument(['--train_file','data/train.json','--save_model','model/tmodel','--log_file','log/tlog'])
    train(args)
