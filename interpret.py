import torch
import math
import json
import numpy as np
import random
import time
import os
from copy import deepcopy
from multiprocessing import Pool
import argparse
import heapq

device = torch.device("cpu")
check_res=False
def show_depend(model_file):
    parameter=torch.load(model_file,map_location=device)
    torch.set_printoptions(threshold=10000, precision=2, sci_mode=False)


    right_weight =(torch.softmax(parameter['chose_right.weight'], dim=1)+1e-6).numpy()
    op_weight = (torch.softmax(parameter['chose_op.weight'], dim=1)+1e-6).numpy()
    # print(torch.softmax(parameter['chose_right.weight'], dim=1))
    # print(torch.softmax(parameter['chose_op.weight'], dim=1))
    # exit(0)
    if check_res:
        print('right_weight',torch.softmax(parameter['chose_right.weight'], dim=1))
        print('op_weight',torch.softmax(parameter['chose_op.weight'], dim=1))

    # exit(0)
    return right_weight,op_weight

def matrix2ltl(model_name,test_file_name,top_num):
    right_weight, op_weight=show_depend(model_name)
    np.set_printoptions(suppress=True,precision=2,linewidth=400)

    with open(test_file_name,'r') as f:
        vocab=json.load(f)['vocab']+['true']
    formula_len=len(right_weight) # the last one is to choose no right subtree
    # print(vocab)
    vocab_len=len(vocab)
    # print('ori subformulae',subformulae)

    # print('nleaf_weight\n \tnone_x\tnot_x\tnext_x\tand_x\tuntil_x')
    # for i in range(len(nleaf_weight)):
    #     print('i:%d'%i,end='\t')
    #     for j in range(5):
    #         print("%.2f"%nleaf_weight[i][i+j*len(nleaf_weight[i])//5],end='\t')
    #     print()
    # print('leaf_weight,\n',leaf_weight)

    # loss=op_weight*left_weight*right_weight*subtreeweight
    row_idx=formula_len-1
    global_best=[0]*formula_len

    while row_idx>=0:
        row_op_weight=op_weight[row_idx][:5]   # the row'th formula weights on none,not,and,next,until
        row_atom_weight = op_weight[row_idx][5:]  # the row'th formula weights on atoms
        # print('op_weight',op_weight)
        # print('atom',row_atom_weight)
        row_right_weight = right_weight[row_idx][:formula_len] # the row'th formula right subformula weights


        # ,
        turple2loss = []
        # none

        none_weight = row_op_weight[0]
        not_weight=row_op_weight[1]
        and_weight = row_op_weight[2]
        next_weight=row_op_weight[3]
        until_weight=row_op_weight[4]


        right_weight_sort=[(row_right_weight[i]+1e-6,i) for i in range(row_idx+1,formula_len)]
        right_weight_sort.sort(reverse=True)
        # print('right_weight',right_weight_sort)

        tcnt = 0
        #leaf
        for i in range(vocab_len):
            score=math.log(row_atom_weight[i])
            heapq.heappush(turple2loss, (score, tcnt,  vocab[i]))
            tcnt+=1

        # not,next
        op_score=[math.log(not_weight),math.log(next_weight)]

        for op_idx in range(2):
            l_idx=row_idx+1
            score1=op_score[op_idx]
            if l_idx>=formula_len:
                break
            left_subs=global_best[l_idx]
            for left_sub in left_subs:
                score2=score1+left_sub[0]
                if len(turple2loss) < top_num or score2 > turple2loss[0][0]:
                    if op_idx==0:
                        heapq.heappush(turple2loss, (score2,tcnt, ('!',left_sub[2]) ))
                    elif op_idx==1:
                        heapq.heappush(turple2loss, (score2,tcnt, ('X', left_sub[2])))
                    tcnt+=1
                else:
                    break
                if len(turple2loss) > top_num:
                    heapq.heappop(turple2loss)

        op_score = [math.log(and_weight), math.log(until_weight)]
        #and
        for op_idx in range(2):
            l_idx=row_idx+1
            if l_idx>=formula_len-1:
                break
            for r_idx in range(len(right_weight_sort)):
                score1=op_score[op_idx]+math.log(right_weight_sort[r_idx][0])
                left_subs = global_best[l_idx]
                right_subs = global_best[right_weight_sort[r_idx][1]]
                for left_sub in left_subs:
                    break_mark=True
                    for right_sub in right_subs:
                        score2 = score1 + left_sub[0] + right_sub[0]
                        if len(turple2loss) < top_num or score2 > turple2loss[0][0]:
                            break_mark=False
                            if op_idx == 0:
                                heapq.heappush(turple2loss, (score2,tcnt, ('&&', left_sub[2],right_sub[2])))
                            else:
                                heapq.heappush(turple2loss, (score2,tcnt, ('U', left_sub[2],right_sub[2])))
                            tcnt+=1
                        else:
                            break
                        if len(turple2loss) > top_num:
                            heapq.heappop(turple2loss)
                    if break_mark:
                        break

        # print('turple2loss',turple2loss)
        turple2loss.sort(key=lambda x:x[0],reverse=True)
        turple2loss=turple2loss[:top_num]
        global_best[row_idx]=turple2loss
        row_idx-=1



    ret_formulae=[]

    # for best in global_best:
    #     print(best)
    for i in global_best[0]:
        if check_res:
            print('i:',i)
        ret_formulae.append(i[-1])
    # print('ret_formulae',ret_formulae)
    # a=input()
    return ret_formulae


class LTLf():

    def __init__(self, vocab, LTLf_tree):
        self.vocab = vocab
        self.LTLf_tree = LTLf_tree
        self.cache = {}

    def _checkLTL(self, f, t, trace, vocab, c=None, v=False, orif=None):
        """ Checks satisfaction of a LTL formula on an execution trace

            NOTES:
            * This works by using the semantics of LTL and forward progression through recursion
            * Note that this does NOT require using any off-the-shelf planner

            ARGUMENTS:
                f       - an LTL formula (must be in TREE format using nested tuples
                        if you are using LTL dict, then use ltl['str_tree'])
                t       - time stamp where formula f is evaluated
                trace   - execution trace (a dict containing:
                            trace['name']:    trace name (have to be unique if calling from a set of traces)
                            trace['trace']:   execution trace (in propositions format)
                            trace['plan']:    plan that generated the trace (unneeded)
                vocab   - vocabulary of propositions
                c       - cache for checking LTL on subtrees
                v       - verbosity

            OUTPUT:
                satisfaction  - true/false indicating ltl satisfaction on the given trace
        """
        if orif == None:
            orif = f
        if v:
            print('\nCurrent t = ' + str(t))
            print('Current f =', f)

        ###################################################

        # Check if first operator is a proposition
        if type(f) is str:
            if f in vocab:
                return f in trace['trace'][t]
            if f == 'true':
                return True
            if f=='false':
                return False

        # Check if sub-tree info is available in the cache
        key = (f, t, trace['name'])
        if c is not None:
            if key in c:
                if v: print('Found subtree history')
                return c[key]

        # Check for standard logic operators
        # if len(f)==0:
        # 	print('f', f, 't', t, 'trace', trace, 'vocab', vocab, 'c', c, 'v', v)
        if f[0] in ['not', '!']:
            value = not self._checkLTL(f[1], t, trace, vocab, c, v, orif)
        elif f[0] in ['and', '&', '&&']:
            value = all((self._checkLTL(f[i], t, trace, vocab, c, v, orif) for i in range(1, len(f))))
        elif f[0] in ['or', '|', '||']:
            value = any((self._checkLTL(f[i], t, trace, vocab, c, v, orif) for i in range(1, len(f))))
        elif f[0] in ['imp', '->']:
            value = not (self._checkLTL(f[1], t, trace, vocab, c, v, orif)) or self._checkLTL(f[2], t, trace, vocab, c,
                                                                                              v, orif)

        # Check if t is at final time step
        elif t == len(trace['trace']) - 1:
            # Confirm what your interpretation for this should be.
            if f[0] in ['G', 'F']:
                value = self._checkLTL(f[1], t, trace, vocab, c, v,
                                       orif)  # Confirm what your interpretation here should be
            elif f[0] == 'U':
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif)
            elif f[0] == 'W':  # weak-until
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) or self._checkLTL(f[1], t, trace, vocab, c, v,
                                                                                            orif)
            elif f[0] == 'R':  # release (weak by default)
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif)
            elif f[0] == 'X':
                value = False
            elif f[0] == 'N':
                value = True
            else:
                # Does not exist in vocab, nor any of operators
                print('f', f, 't', t, 'trace', trace, 'vocab', vocab, 'c', c, 'v', v)
                sys.exit('LTL check - something wrong 1')

        else:
            # Forward progression rules
            if f[0] == 'X' or f[0] == 'N':
                value = self._checkLTL(f[1], t + 1, trace, vocab, c, v, orif)
            elif f[0] == 'G':
                value = self._checkLTL(f[1], t, trace, vocab, c, v, orif) and self._checkLTL(('G', f[1]), t + 1, trace,
                                                                                             vocab, c, v, orif)
            elif f[0] == 'F':
                value = self._checkLTL(f[1], t, trace, vocab, c, v, orif) or self._checkLTL(('F', f[1]), t + 1, trace,
                                                                                            vocab, c, v, orif)
            elif f[0] == 'U':
                # Basically enforces f[1] has to occur for f[1] U f[2] to be valid.
                if t == 0:
                    if not self._checkLTL(f[1], t, trace, vocab, c, v, orif) and not self._checkLTL(f[2], t, trace,
                                                                                                    vocab, c, v,
                                                                                                    orif):  # if f[2] is ture at time 0,then it is true
                        value = False
                    else:
                        value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) or (
                                    self._checkLTL(f[1], t, trace, vocab, c, v) and self._checkLTL(('U', f[1], f[2]),
                                                                                                   t + 1, trace, vocab,
                                                                                                   c, v, orif))
                else:
                    value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) or (
                                self._checkLTL(f[1], t, trace, vocab, c, v) and self._checkLTL(('U', f[1], f[2]), t + 1,
                                                                                               trace, vocab, c, v,
                                                                                               orif))

            elif f[0] == 'W':  # weak-until
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) or (
                            self._checkLTL(f[1], t, trace, vocab, c, v, orif) and self._checkLTL(('W', f[1], f[2]),
                                                                                                 t + 1, trace, vocab, c,
                                                                                                 v, orif))
            elif f[0] == 'R':  # release (weak by default)
                value = self._checkLTL(f[2], t, trace, vocab, c, v, orif) and (
                            self._checkLTL(f[1], t, trace, vocab, c, v, orif) or self._checkLTL(('R', f[1], f[2]),
                                                                                                t + 1, trace, vocab, c,
                                                                                                v, orif))
            else:
                # Does not exist in vocab, nor any of operators
                print('f', f, 't', t, 'trace', trace, 'vocab', vocab, 'c', c, 'v', v, ' orif', orif)
                sys.exit('LTL check - something wrong 2 ' + f[0])

        if v: print('Returned: ' + str(value))

        # Save result
        if c is not None and type(c) is dict:
            key = (f, t, trace['name'])
            c[key] = value  # append

        return value

    def pathCheck(self, trace, trace_name):

        trace_dir = {'name': trace_name, 'trace': tuple(trace)}
        return self._checkLTL(self.LTLf_tree, 0, trace_dir, self.vocab, self.cache)

    def evaluate(self, cluster1, cluster2):
        # print(self.LTLf_tree)
        check_pos_mark = []
        for i in range(len(cluster1)):
            st = self.pathCheck(cluster1[i], 'pos' + str(i))
            check_pos_mark.append(st)

        check_neg_mark = []
        for i in range(len(cluster2)):
            st = self.pathCheck(cluster2[i], 'neg' + str(i))
            check_neg_mark.append(st)
        return check_pos_mark, check_neg_mark



def get_best_ltlfs(train_dic,ltlfs):
    best_ltlfs=[]
    best_score_train=0
    for ltlf_tree in ltlfs:
        ltlf=LTLf(train_dic['vocab'],ltlf_tree)
        check_pos_mark, check_neg_mark = ltlf.evaluate(train_dic['traces_pos'], train_dic['traces_neg'])
        cur_score= (sum(check_pos_mark)+len(train_dic['traces_neg'])-sum(check_neg_mark))/(len(train_dic['traces_pos'])+len(train_dic['traces_neg']))
        if cur_score>best_score_train:
            best_score_train=cur_score
            best_ltlfs=[ltlf_tree]
        elif cur_score==best_score_train:
            best_ltlfs.append(ltlf_tree)
    return best_ltlfs




def test_matrix(model_name,train_file_name,test_file_name,max_num=100):

    with open(test_file_name,'r') as f:
        E_T_dic=json.load(f)
    with open(train_file_name,'r') as f:
        E_dic=json.load(f)

    int_time=time.time()
    ltlfs=matrix2ltl(model_name, test_file_name,max_num)
    int_time=time.time()-int_time



    rw_time=time.time()
    best_ltltree=get_best_ltlfs(E_dic,ltlfs)
    rw_time=time.time()-rw_time
    ltlf_tree= best_ltltree[0]
    ltlf = LTLf(E_T_dic["vocab"], ltlf_tree)
    # print('target ltl:',E_T_dic["ltlftree"])
    print('best_ltltree',ltlf_tree)
    # print('matrix:')

    try:
        check_pos_mark, check_neg_mark = ltlf.evaluate(E_T_dic['traces_pos'], E_T_dic['traces_neg'])
    except:
        print('error')
        return (-1,-1,-1,int_time,rw_time)

    test_result=[(1,int(i)) for i in check_pos_mark]+[(0,int(i)) for i in check_neg_mark]
    # TP=sum(check_pos_mark)
    # FN=len(E_T_dic['traces_pos'])-TP
    # FP=sum(check_neg_mark)
    # total=len(E_T_dic['traces_pos'])+len(E_T_dic['traces_neg'])
    # print('correct',correct,'TP,FP,FN',(TP,FP,FN),'total',len(E_T_dic['traces_pos']+E_T_dic['traces_neg']))
    # a = input()
    # print('TP',TP,'FP',FP,'FN',FN)
    # if TP==0:
    #     return (correct/total,0,0,int_time,rw_time,check_pos_mark, check_neg_mark)
    # return (correct/total,TP/(TP+FP),TP/(TP+FN),int_time,rw_time,check_pos_mark, check_neg_mark)  # acc,pre,rec
    return (test_result,int_time,rw_time)

def get_interpret_args(arg_list=None):
    parser = argparse.ArgumentParser(description='Main script for parallel')
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--max_num', type=int, required=False, default=100)
    if arg_list==None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_list)
    return args

if __name__ == '__main__':

    # python matrix2ltl_templatev2L2.py -tag 1001 -max_num 10 -noise 0 -fix_formula_len 3 -domains='mindata_f3 mindata_f6 mindata_f9 mindata_f12 mindata_f15' -inss="2"
    args=get_interpret_args(['--save_model','model/tmodel','--train_file','data/train.json','--test_file','data/test.json'])
    result = test_matrix(args.save_model, args.train_file, args.test_file, args.max_num)[0]
    print('formula accuracy',sum([1 if i[0]==i[1] else 0 for i in result])/len(result))
    # main(args)