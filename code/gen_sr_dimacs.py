import math
import os
import numpy as np
import random
import argparse
import PyMiniSolvers.minisolvers as minisolvers
import pickle
from data_to_pkl import to_pkl
from data_to_pkl import  mk_batch_problem

def write_dimacs_to(n_vars, iclauses, out_filename):
    with open(out_filename, 'w') as f:
        f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
        for c in iclauses:
            for x in c:
                f.write("%d " % x)
            f.write("0\n")

def write_label_to(label,out_filename):
    with open(out_filename, 'w') as f:
        for l in label:
            f.write("%d " % l)
        f.write("0\n")

def mk_out_filenames(out_dir, n_vars, t):

    prefix = "{}/id={}_n={}".format(out_dir, t, n_vars)
    return ("", "{}.cnf".format(prefix))

def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]

def gen_iclause_pair(opts):
    n = random.randint(opts.min_n, opts.max_n)

    solver = minisolvers.MinisatSolver()
    for i in range(n): solver.new_var(dvar=True)

    iclauses = []

    while True:
        k_base = 1 if random.random() < opts.p_k_2 else 2
        k = k_base + np.random.geometric(opts.p_geo)#k是第i个子句中的变量数
        iclause = generate_k_iclause(n, k)#第i个子句

        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            iclauses.append(iclause)

        else:
            break


    iclause_unsat = iclause
    iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]#翻转icluse_unsat中的一个文字就变成了可满足的
    if(opts.label_out_dir) is not None:
        label = solver.get_model()
        return n, iclauses, iclause_unsat, iclause_sat,label
    return n, iclauses, iclause_unsat, iclause_sat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir', action='store', type=str)
    parser.add_argument('n_pairs', action='store', type=int)

    parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=40)
    parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=40)

    parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)

    parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
    parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)

    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)

    parser.add_argument('--neuro_initial', action='store', type=bool,default=False)
    parser.add_argument('--label_out_dir', action='store', type=str, default=None)
    opts = parser.parse_args()

    os.makedirs(opts.out_dir)
    os.makedirs(opts.label_out_dir)

    if opts.py_seed is not None: random.seed(opts.py_seed)
    if opts.np_seed is not None: np.random.seed(opts.np_seed)

    problems = []
    batches = []
    n_nodes_in_batch = 0
    prev_n_vars = None



    for pair in range(opts.n_pairs):
        if pair % opts.print_interval == 0: print("[%d]" % pair)

        if opts.neuro_initial is not None:
            n_vars, iclauses, iclause_unsat, iclause_sat,label = gen_iclause_pair(opts)

            label_out_filename = mk_out_filenames(opts.label_out_dir, n_vars, pair)
            #write_label_to(label,label_out_filename[1])
        else :
            n_vars, iclauses, iclause_unsat, iclause_sat= gen_iclause_pair(opts)

        out_filenames = mk_out_filenames(opts.out_dir, n_vars, pair)

        iclauses.append(iclause_unsat)
        # write_dimacs_to(n_vars, iclauses, out_filenames[0])

        iclauses[-1] = iclause_sat
        write_dimacs_to(n_vars, iclauses, out_filenames[1])

        if opts.neuro_initial is not None:
            n_clauses = len(iclauses)
            n_cells = sum([len(iclause) for iclause in iclauses])
            n_nodes = 2 * n_vars + n_clauses
            problems.append((n_vars, iclauses, 1))
            # problems.append(n_vars,iclauses, label)
            if len(problems) > 0:
                batches.append(mk_batch_problem(problems))

                del problems[:]
    pkl_file = open('/home/zhang/zouy/sat/data/sr_pkl/1'+ '.pkl', mode='w')
    with open('/home/zhang/zouy/sat/data/sr_pkl/1' + '.pkl', "wb") as f:
        pickle.dump(batches, f)
        '''
        pkl_file = open('/home/zhang/zouy/sat/data/sr_pkl/{}'.format(pair)+'.pkl', mode ='w')
        with open('/home/zhang/zouy/sat/data/sr_pkl/{}'.format(pair)+'.pkl',"wb") as f:
            pickle.dump(iclauses, f)

        '''

