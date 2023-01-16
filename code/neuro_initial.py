import argparse
import pickle
import os
import random
import time

import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from neurosat import NeuroSAT


from config import parser
sigmoid=nn.Sigmoid()
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
def compute_output(n_vars,outputs,p0):
    for i in range(0, n_vars):
        eval_p = random.random()
        outputs[i] = 1 if (random.random() < outputs[i]) else 0
        outputs[i] = outputs[i] if (eval_p < p0) else -outputs[i]

    return outputs
def load_model(args):
    net = NeuroSAT(args)
    #start=time.time()
    net = net.cuda()
    #end=time.time()
    #print(1000*(end-start))
    model = torch.load(args.model_dir)
    net.load_state_dict(model['state_dict'])

    return net


def predict(net, data):
    net.eval()
    outputs,p0 = net(data)
    outputs=outputs.unsqueeze(1)
    #outputs = sigmoid(outputs)
    outputs = compute_output(data.n_vars, outputs,p0)

    return outputs


def pre_assign(args,filename):
    #args = parser.parse_args()
    net = load_model(args)
    times = []
    with open(os.path.join(args.dir_path,'pkl',filename.replace('cnf','pkl')), 'rb') as f:
        x = pickle.load(f)
    start_time = time.time()
    outputs = predict(net, x)
    end_time = time.time()
    duration = (end_time - start_time) * 1000
    target = torch.Tensor(x.label).cuda().float()

    target = target.view(x.n_vars, 1)

    acc=compute_acc(x.n_vars,outputs,target)


    desc = " time: %.2f ms; the pred acc is %.2f" \
           % (duration * 1.0 ,acc)
    return outputs

