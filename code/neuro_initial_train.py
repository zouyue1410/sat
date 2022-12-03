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

'''
def mk_filenames(dir_path,n):
    prefix = "{}/{}".format(dir_path,n)
    return ("", "{}.pkl".format(prefix))
'''

'''
problems=[]
n_vars=5
problems_idx=1
iclauses=40
is_sat=1
problems.append(("sr_n=%.4d_pk2=%.2f_pg=%.2f_t=%d_sat=0" % (n_vars, args.p_k_2, args.p_geo, problems_idx), n_vars, iclauses, is_sat))
'''




#train_filename=mk_filenames("/home/zhang/zouy/sat/data/sr_pkl",n)
with open(args.data_dir, 'rb') as f:
    train_sets = pickle.load(f)


#修改网络模型：迭代次数，输出每个变量节点的预测值
net = NeuroSAT(args)
net = net.cuda()
#自己编写损失函数
loss_fn = nn.BCELoss()
optim = optim.Adam(net.parameters(), lr=0.00002, weight_decay=1e-10)
sigmoid  = nn.Sigmoid()


TP, TN, FN, FP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
net.train()
for _, prob in enumerate(train_sets):

    optim.zero_grad()
    outputs = net(prob)
    # 把标签变成float型

    target = torch.Tensor(prob.label).cuda().float()
    target = target.view(prob.n_vars,1)
    # print(outputs.shape, target.shape)
    # print(outputs, target)
    outputs = sigmoid(outputs)
    #把outputs变成行列对换
    #outputs = outputs.view(1,prob.n_vars)
    loss=0
    for i in range(0,prob.n_vars):
        loss =loss + loss_fn(outputs[i], target[i])
    loss=loss/prob.n_vars
    #loss = loss_fn(outputs, target)
    desc = 'loss: %.4f; ' % (loss.item())

    loss.backward()
    optim.step()

    preds = torch.where(outputs > 0.5, torch.ones(outputs.shape).cuda(), torch.zeros(outputs.shape).cuda())

    TP += (preds.eq(1) & target.eq(1)).cpu().sum()
    TN += (preds.eq(0) & target.eq(0)).cpu().sum()
    FN += (preds.eq(0) & target.eq(1)).cpu().sum()
    FP += (preds.eq(1) & target.eq(0)).cpu().sum()
    TOT = TP + TN + FN + FP

    desc += 'acc: %.3f, TP: %.3f, TN: %.3f, FN: %.3f, FP: %.3f' % ((TP.item() + TN.item()) * 1.0 / TOT.item(), TP.item() * 1.0 / TOT.item(), TN.item() * 1.0 / TOT.item(),FN.item() * 1.0 / TOT.item(), FP.item() * 1.0 / TOT.item())
    if (_ + 1) % 100 == 0:
        print(_)
        print(desc)
    # train_bar.set_description(desc)
    '''
        if (_ + 1) % 100 == 0:
        print(desc, file=detail_log_file, flush=True)
print(desc, file=log_file, flush=True)
    '''



