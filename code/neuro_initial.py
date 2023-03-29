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






def compute_output(n_vars,outputs):
    #print(p0)
    for i in range(0, n_vars):
        eval_p = random.random()
        outputs[i] = 1 if (random.random() < outputs[i]) else 0
        #outputs[i] = outputs[i] if (eval_p < p0) else -outputs[i]

    return outputs



'''
def compute_output(n_vars,outputs):

    for i in range(0, n_vars):

        outputs[i] = 1 if (random.random() < outputs[i]) else 0
        p=0.9
        if random.random()<p:
            outputs[i]=outputs[i]
        else :
            if outputs[i]==1:
                outputs[i]=0
            else:
                outputs[i]=1

    return outputs
'''












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
    #outputs = net(data)
    outputs=outputs.unsqueeze(1)
    #outputs = sigmoid(outputs)


    p_outputs = []
    for j in range(0, data.n_vars):
        t = p0*1.0 * outputs[j] + (1.0 - p0) * 0.5
        p_outputs.append(t * 1.0)
    p_outputs = torch.tensor(p_outputs).view(-1, 1).cuda()



    print(p_outputs)





    #outputs = compute_output(data.n_vars, outputs)
    outputs = compute_output(data.n_vars, p_outputs)

    return outputs


def pre_assign(args,filename):
    #args = parser.parse_args()
    net = load_model(args)
    times = []
    with open(os.path.join(args.dir_path,'pkl',filename.replace('cnf','pkl')), 'rb') as f:
        x = pickle.load(f)

    outputs = predict(net, x)




    return outputs

