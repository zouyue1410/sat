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



        '''
        self.L_update = nn.Sequential(
            nn.Linear(self.args.dim * 2, self.args.dim),
            nn.LayerNorm(self.args.dim),
        )
        '''
        '''
        self.C_update = nn.Sequential(
            nn.Linear(self.args.dim, self.args.dim),
            nn.LayerNorm(self.args.dim),
        )
        '''
        self.C_update = nn.LSTM(self.args.dim, self.args.dim)
        self.L_update = nn.LSTM(self.args.dim * 2, self.args.dim)

        # self.L_norm   = nn.LayerNorm(self.args.dim)

        # self.C_norm   = nn.LayerNorm(self.args.dim)

        self.L_vote =nn.Sequential(
            nn.Linear(self.args.dim,1),

        )
        self.L_prob=MLP(self.args.dim, self.args.dim, 1)

        self.denom = torch.sqrt(torch.Tensor([self.args.dim]))

    def forward(self, problem):
        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()
        init_ts = self.init_ts.cuda()
        L_init = self.L_init(init_ts).view(1, 1, -1)
        L_init = L_init.repeat(1, n_lits, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        C_init = C_init.repeat(1, n_clauses, 1)

        L_state = (L_init, torch.zeros(1, n_lits, self.args.dim).cuda())
        #L_state = L_init
        #C_state = C_init
        C_state = (C_init, torch.zeros(1, n_clauses, self.args.dim).cuda())
        L_unpack = torch.sparse.FloatTensor(ts_L_unpack_indices, torch.ones(problem.n_cells),
                                            torch.Size([n_lits, n_clauses])).to_dense().cuda()



        for _ in range(self.args.n_rounds):

            L_hidden = L_state[0].squeeze(0)
            L_pre_msg = self.L_msg(L_hidden)
            # (n_clauses x n_lits) x (n_lits x dim) = n_clauses x dim
            LC_msg = torch.matmul(L_unpack.t(), L_hidden)
            # print(L_hidden.shape, L_pre_msg.shape, LC_msg.shape)

            _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)

            #LC_msg = torch.matmul(L_unpack.t(), L_state)
            # print('C_state',C_state[0].shape, C_state[1].shape)

            #C_state=self.C_update(torch.mul(LC_msg,C_state))

            C_hidden = C_state[0].squeeze(0)
            #C_pre_msg = self.C_msg(C_hidden)

            CL_msg = torch.matmul(L_unpack, C_hidden)
            #CL_msg = torch.matmul(L_unpack, C_state)
            # print(C_hidden.shape, C_pre_msg.shape, CL_msg.shape)
            #b=self.flip(L_state.squeeze(0), n_vars).unsqueeze(0)
            #a=torch.cat([CL_msg, b], dim=2)
            #L_state = self.L_update(torch.cat([torch.mul(CL_msg,L_state),torch.mul(b,L_state)],dim=2))
            _, L_state = self.L_update(torch.cat([CL_msg, self.flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0), L_state)
            # print('L_state',C_state[0].shape, C_state[1].shape)

        logits = L_state[0].squeeze(0)

        #vote = self.L_vote(logits)
        #加两层MLP 输出概率
        vote = self.L_prob(logits)
        #vote_join = vote[0:n_vars]
        vote_join = torch.cat([vote[:n_vars, :], vote[n_vars:, :]], dim=1)
        vote_join=F.softmax(vote_join,dim=1)
        #vote_join=self.L_prob(vote_join)

        return vote_join[:,0]

    def flip(self, msg, n_vars):
        return torch.cat([msg[n_vars:2 * n_vars, :], msg[:n_vars, :]], dim=0)

