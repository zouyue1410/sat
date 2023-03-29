import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP


class NeuroSAT(nn.Module):
    def __init__(self, args):
        super(NeuroSAT, self).__init__()
        self.args = args

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, args.dim)
        self.C_init = nn.Linear(1, args.dim)

        self.L_msg = nn.Linear(self.args.dim,  self.args.dim)
        self.C_msg = nn.Linear(self.args.dim,  self.args.dim)


        self.C_update = nn.LSTM(self.args.dim, self.args.dim)
        self.L_update = nn.LSTM(self.args.dim * 2, self.args.dim)

        # self.L_norm   = nn.LayerNorm(self.args.dim)

        # self.C_norm   = nn.LayerNorm(self.args.dim)

        self.L_vote =nn.Sequential(
            nn.Linear(self.args.dim,1),

        )
        self.C_vote=nn.Sequential(
            nn.Linear(self.args.dim,1),
        )
        self.L_prob=MLP(self.args.dim, self.args.dim, 1)
        self.C_prob = MLP(self.args.dim, self.args.dim, 1)

        self.denom = torch.sqrt(torch.Tensor([self.args.dim]))

    def forward(self, problem):
        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        #n_probs = len(problem.is_sat)
        # print(n_vars, n_lits, n_clauses, n_probs)

        ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()

        init_ts = self.init_ts.cuda()
        # 1 x n_lits x dim & 1 x n_clauses x dim
        L_init = self.L_init(init_ts).view(1, 1, -1)
        # print(L_init.shape)
        L_init = L_init.repeat(1, n_lits, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        # print(C_init.shape)
        C_init = C_init.repeat(1, n_clauses, 1)

        # print(L_init.shape, C_init.shape)

        L_state = (L_init, torch.zeros(1, n_lits, self.args.dim).cuda())
        C_state = (C_init, torch.zeros(1, n_clauses, self.args.dim).cuda())
        L_unpack = torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(problem.n_cells),
                                            torch.Size([n_lits, n_clauses])).to_dense().cuda()

        # print(ts_L_unpack_indices.shape)

        Ls=[]
        Cs=[]
        pos_adj = L_unpack[:n_vars, :]
        neg_adj = L_unpack[n_vars:, :]
        adj=pos_adj-neg_adj
        for _ in range(self.args.n_rounds):
            L_hidden = L_state[0].squeeze(0)
            L_pre_msg = self.L_msg(L_hidden)
            # (n_clauses x n_lits) x (n_lits x dim) = n_clauses x dim
            prac=L_unpack.t()

            LC_msg = torch.matmul(L_unpack.t(), L_pre_msg)
            # print(L_hidden.shape, L_pre_msg.shape, LC_msg.shape)

            _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)
            # print('C_state',C_state[0].shape, C_state[1].shape)

            # n_clauses x dim
            C_hidden = C_state[0].squeeze(0)
            C_pre_msg = self.C_msg(C_hidden)
            # (n_lits x n_clauses) x (n_clauses x dim) = n_lits x dim
            CL_msg = torch.matmul(L_unpack, C_pre_msg)
            # print(C_hidden.shape, C_pre_msg.shape, CL_msg.shape)

            _, L_state = self.L_update(
                torch.cat([CL_msg, self.flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0), L_state)

            logits = L_state[0].squeeze(0)
            logits_c = C_state[0].squeeze(0)
            L=self.L_prob(logits)
            L = torch.cat([L[:n_vars, :], L[n_vars:, :]], dim=1)
            L=F.softmax(L,dim=1)
            L=torch.cat([L[:n_vars,0:1],L[:n_vars,1:]],dim=0)
            #L=L[:,0]
            C = self.C_prob(logits_c)
            C=F.sigmoid(C)
            Ls.append(L)

            Cs.append(C)



        return Ls,Cs,L_unpack

    def flip(self, msg, n_vars):
        return torch.cat([msg[n_vars:2 * n_vars, :], msg[:n_vars, :]], dim=0)

