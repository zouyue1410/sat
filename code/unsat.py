import os
import random

import torch
import pickle
import argparse
import torch.nn as nn
import torch.optim as optim
# 加载输入数据
from unsur import NeuroSAT
from config import parser
from tqdm import tqdm

args = parser.parse_args()


def compute_acc(n_vars, outputs, target):
    for i in range(0, n_vars):
        p = random.random()
        if p < outputs[i]:
            outputs[i] = 1.0
        else:
            outputs[i] = 0.0
    count = 0
    for i in range(0, n_vars):
        if outputs[i] == target[i]:
            count = count + 1
    count = float(count)
    return count / n_vars


def compute_output(n_vars, outputs):
    for i in range(0, n_vars):
        outputs[i] = 1 if (random.random() < outputs[i]) else 0
        '''
        p=0.9
        if random.random()<p:
            outputs[i]=outputs[i]
        else :
            if outputs[i]==1:
                outputs[i]=0
            else:
                outputs[i]=1
        '''

    return outputs



# 评估数据集


net = NeuroSAT(args)
net = net.cuda()
loss_fn = nn.BCELoss()
optim = optim.Adam(net.parameters(), lr=0.00002, weight_decay=1e-10)
sigmoid = nn.Sigmoid()
best_acc = 0.0

TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
net.train()
for i, filename in enumerate(os.listdir(args.data_dir)):
    with open(os.path.join(args.data_dir, filename), 'rb') as f:
        prob = pickle.load(f)

    # for _, prob in enumerate(train_sets):

    optim.zero_grad()

    L,C,adj = net(prob)
    losses=0
    for t in range(0,16):
        l=L[t].transpose(0,1)
        LM=torch.matmul(l,adj)
        loss=torch.matmul(LM,C[t])/torch.norm(adj,p=1)
        loss=-torch.log(loss)
        loss=loss*pow(0.95,16-t)
        losses+=loss



    optim.step()




    losses.requires_grad_(True)
    losses.backward(retain_graph=True)

    # outputs_assign = compute_output(prob.n_vars, outputs)
    outputs_assign = compute_output(prob.n_vars, L[15])
    desc = 'loss: %.4f; ' % (losses.item())

    target = torch.Tensor(prob.label).cuda().float().view(prob.n_vars, 1)
    acc = compute_acc(prob.n_vars, outputs_assign, target)
    desc += 'acc: %.3f' % acc
    if (i + 1) % 100 == 0:
        print(i)
        print(desc)
        net.eval()
        for eval_, eval_filename in enumerate(os.listdir(args.eval_data_dir)):
            with open(os.path.join(args.eval_data_dir, eval_filename), 'rb') as f:
                eval_prob = pickle.load(f)
            if eval_ > 9:
                break
            optim.zero_grad()
            eval_L, eval_C, eval_adj = net(prob)
            eval_outputs_assign = compute_output(eval_prob.n_vars, eval_L[15])
            eval_target = torch.Tensor(eval_prob.label).cuda().float()
            eval_target = eval_target.view(eval_prob.n_vars, 1)
            eval_acc = compute_acc(eval_prob.n_vars, eval_outputs_assign, eval_target)
            torch.save({'acc': eval_acc, 'state_dict': net.state_dict()},
                       os.path.join(args.model_dir, args.taskname + '_last.pth.tar'))
            if eval_acc >= best_acc:
                best_acc = eval_acc
                torch.save({'acc': best_acc, 'state_dict': net.state_dict()},
                           os.path.join(args.model_dir, args.taskname + '_best.pth.tar'))
        net.train()
    # train_bar.set_description(desc)
