import torch
import torch.nn.functional as F     # 激励函数都在这
from torch.nn.parameter import Parameter
import math
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
device = torch.device("cpu")
# device = torch.device(
#     "cuda") if torch.cuda.is_available() else torch.device("cpu")


class CenteredLayer(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        # x=x-0.5
        # x=torch.sigmoid(x*5)
        #
        x[x<0]=x[x<0]*0.01
        x[x>1]=x[x>1]*0.01+0.99
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

CHOOSE_NUM=5 # 直接选叶子，U,&,X,!
class Net(torch.nn.Module):
    def __init__(self,net_k,vocab_len,ltl='',vocab2idx={}):
        super(Net,self).__init__()
        self.net_k=net_k # 公式树深度
        self.vocab_len=vocab_len
        self.chose_right = torch.nn.Linear(net_k+1, net_k , bias=False).to(device)# matrix of netk*(netk_vocablen)
        self.chose_op = torch.nn.Linear(5+vocab_len,net_k, bias=False).to(device)#atoms,not,and,next,until, none



        self.zero_grad_matrix = [[1 if i+1 < j else 0 for j in range(net_k+1)] for i in range(net_k)]
        self.zero_grad_matrix = torch.FloatTensor(self.zero_grad_matrix).to(device)
        if ltl=='':
            new_state_dict=self.chose_right.state_dict() # can only chose sub formula
            for i in range(net_k): #
                for j in range(net_k):
                    if j>i+1:
                        new_state_dict['weight'][i][j] = 0
                    else:
                        new_state_dict['weight'][i][j] = -1000
            self.chose_right.load_state_dict(new_state_dict, strict=False)

            self.chose_right.weight.register_hook(lambda grad: grad.mul_(self.zero_grad_matrix))
            new_state_dict=self.chose_op.state_dict() # can only chose sub formula
            for i in range(net_k): #
                for j in range(5+vocab_len):
                    new_state_dict['weight'][i][j] = 0
            self.chose_op.load_state_dict(new_state_dict,strict=False)
        else:
            new_state_dict = self.chose_right.state_dict()
            for i in range(net_k):  #
                new_state_dict['weight'][i][-1]=20
                for j in range(net_k):
                    new_state_dict['weight'][i][j] = -1000
            self.chose_right.load_state_dict(new_state_dict, strict=False)

            new_state_dict = self.chose_op.state_dict()
            for i in range(net_k): #
                new_state_dict['weight'][i][0]=20
                for j in range(1,5+vocab_len):
                    new_state_dict['weight'][i][j] = -1000
            self.chose_op.load_state_dict(new_state_dict, strict=False)


            self.chose_right.requires_grad_(False)
            self.chose_op.requires_grad_(False)
            used_node=[0]
            self.set2ltl(ltl,vocab2idx,0,used_node)
            print('after set:',used_node)
            self.chose_right.requires_grad_(True)
            self.chose_op.requires_grad_(True)


        self.myrelu = CenteredLayer()



    def cal_softmax(self):
        # print('torch.softmax(self.chose_op.weight, dim=1)',self.chose_op.weight)
        self.op_chose_weight=torch.softmax(self.chose_op.weight, dim=1)[:,:5].transpose(0, 1).to(device)
        self.chose_atom_ = torch.softmax(self.chose_op.weight, dim=1)[:,5:].transpose(0,1).to(device)
        self.chose_right_=torch.softmax(self.chose_right.weight, dim=1)[:,:-1].transpose(0,1).to(device)


    def forward(self,x): # x : (batch_size,trace_len,vocab_len)

        # all_x :(batch_size, trace_len, net_k)
        # print(x.shape)
        # exit(0)
        self.cal_softmax()
        batch_size, trace_len, vocab_len = x.size()
        all_x = torch.zeros((batch_size, trace_len, self.net_k), dtype=torch.float, device=device, requires_grad=False)
        # end_x :(batch_size, 1, net_k)
        # end_x=torch.FloatTensor([[[0]*(self.net_k)] for _ in range(len(x))]).to(device)
        end_x = torch.zeros((batch_size, 1, self.net_k), dtype=torch.float, device=device, requires_grad=False)
        # last_x: (batch_size, trace_len,1)
        # last_x=torch.FloatTensor([[[0] for _ in range(len(x[0]))] for __ in range(len(x))]).to(device)
        last_x = torch.zeros((batch_size, trace_len, 1), dtype=torch.float, device=device, requires_grad=False)
        # op_chose_weight=torch.softmax(self.chose_op.weight,dim=0)
        # print('inputx: ',all_x)
        for i in range(len(x[0])+self.net_k):
            # print('all_x ',all_x)
            # print('x',x)
            # print('self.chose_atom_',self.chose_atom_)
            # print(all_x.shape,self.chose_right_.shape)
            # print(x.shape, self.chose_atom_.shape)
            atom_x=x.matmul(self.chose_atom_)
            left_x=torch.cat((all_x[:,:,1:],last_x),dim=2) # chose_left: self.net_k+len(vocab) -> self.net_k

            right_x = all_x.matmul(self.chose_right_)
            not_x = (left_x) * (-1) + 1
            and_x=left_x+right_x-1

            next_x = all_x[:,1:]
            # print('next_x',next_x[:,1:],last_x)
            X_left_x = torch.cat((next_x[:,:,1:],last_x[:,1:]),dim=2)
            X_left_x= torch.cat((X_left_x, end_x), dim=1)

            until_x=right_x+self.myrelu(left_x+torch.cat((next_x,end_x),dim=1)-1)


            not_x=not_x*self.op_chose_weight[1]
            and_x = and_x * self.op_chose_weight[2]
            next_x = X_left_x * self.op_chose_weight[3]
            until_x = until_x* self.op_chose_weight[4]
            all_x=self.myrelu(atom_x+not_x+next_x+and_x+until_x)
            #
            # print('tx',tx.shape)

        #     print('allx: ',all_x)
        # a=input()

        all_x = all_x-0.5
        all_x = torch.sigmoid(all_x * 5)
        # print('one time',time.time()-stime)

        return all_x





    def set2ltl(self,ltl,vocab2idx,root_idx,used_node):
        high_weight=20
        self.chose_op.weight[root_idx][0]=-1000
        if ltl in vocab2idx.keys():
            self.chose_op.weight[root_idx][5+vocab2idx[ltl]]=high_weight
            used_node[0]+=1
        else: #none,not,and,next,until,

            op_dic={'!':1,'not':1,'and':2,'&&':2,'&':2,'X':3,'U':4}
            self.chose_op.weight[root_idx][op_dic[ltl[0]]] = high_weight
            used_node[0] += 1

            self.set2ltl(ltl[1], vocab2idx, used_node[0], used_node)
            if len(ltl)==3:
                self.chose_right.weight[root_idx][used_node[0]] = high_weight
                self.chose_right.weight[root_idx][-1] = -1000
                self.set2ltl(ltl[2], vocab2idx, used_node[0], used_node)
            else:
                self.chose_right.weight[root_idx][-1]=high_weight
