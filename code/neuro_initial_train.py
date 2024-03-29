import os
import random

import torch
import pickle
import argparse
import torch.nn as nn
import torch.optim as optim
# 加载输入数据
from neurosat import NeuroSAT
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
            count=count+1
    count=float(count)
    return count/n_vars
def compute_output(n_vars,outputs):

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
'''
with open(args.data_dir, 'rb') as f:
    train_sets = pickle.load(f)
with open(args.eval_data_dir, 'rb') as f:
    eval_sets = pickle.load(f)    
'''

# 评估数据集


net = NeuroSAT(args)
net = net.cuda()
loss_fn = nn.BCELoss()
optim = optim.Adam(net.parameters(), lr=0.00002, weight_decay=1e-10)
sigmoid  = nn.Sigmoid()
best_acc=0.0

TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
net.train()
for i, filename in enumerate(os.listdir(args.data_dir)):
    with open(os.path.join(args.data_dir,filename),'rb') as f:
        prob=pickle.load(f)

#for _, prob in enumerate(train_sets):
    optim.zero_grad()
    #utputs, p0 = net(prob)
    outputs,p0= net(prob)
    #print(p0)
    outputs=outputs.unsqueeze(1)
    target = torch.Tensor(prob.label).cuda().float().view(prob.n_vars,1)
    #outputs = sigmoid(outputs)




    optim.step()
    '''
    p_outputs=[]
    for j in range(0,prob.n_vars):
        p_outputs.append(outputs[j]) if random.random() <p0 else p_outputs.append(1-outputs[j])
    '''

    p_outputs=[]
    for j in range(0, prob.n_vars):
        t=p0*outputs[j]+(1-p0)*0.5
        p_outputs.append(t*1.0)
        '''
                if random.random() < p0:
            p_outputs.append(1.0) if random.random() < outputs[j] else p_outputs.append(0.0)
        else:
            p_outputs.append(0.0) if random.random() < outputs[j] else p_outputs.append(1.0)
        '''

    
    p_outputs=torch.tensor(p_outputs).view(-1,1).cuda()

    '''
    
    '''

    loss = loss_fn(p_outputs, target)
    #loss = loss_fn(outputs_assign, target)
    #loss = loss_fn(outputs, target)
    loss.requires_grad_(True)
    loss.backward(retain_graph=True)

    #outputs_assign = compute_output(prob.n_vars, outputs)
    outputs_assign = p_outputs
    desc = 'loss: %.4f; ' % (loss.item())

    #outputs_assign=compute_output(prob.n_vars,outputs,p0)
    acc = compute_acc(prob.n_vars,outputs_assign,target)
    desc += 'acc: %.3f' % acc
    if (i + 1) % 100 == 0:
        print(i)
        print(desc)
        net.eval()
        for eval_, eval_filename in enumerate(os.listdir(args.eval_data_dir)):
            with open(os.path.join(args.eval_data_dir, eval_filename), 'rb') as f:
                eval_prob = pickle.load(f)
            if eval_>9:
                break
            optim.zero_grad()
            eval_outputs,eval_p0 = net(eval_prob)
            #eval_outputs= net(eval_prob)
            eval_outputs=eval_outputs.unsqueeze(1)
            eval_target = torch.Tensor(eval_prob.label).cuda().float()

            eval_target = eval_target.view(eval_prob.n_vars, 1)
            #eval_outputs = sigmoid(eval_outputs)
            eval_outputs_assign=compute_output(eval_prob.n_vars,eval_outputs)
            eval_acc=compute_acc(eval_prob.n_vars,eval_outputs_assign,eval_target)
            torch.save({'acc': eval_acc, 'state_dict': net.state_dict()},
                    os.path.join(args.model_dir,args.taskname + '_last.pth.tar'))
            if eval_acc >= best_acc:
                best_acc = eval_acc
                torch.save({'acc': best_acc, 'state_dict': net.state_dict()},
                           os.path.join(args.model_dir ,args.taskname+ '_best.pth.tar'))
        net.train()
    # train_bar.set_description(desc)
