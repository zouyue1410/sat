import  pickle
import numpy as np
def ilit_to_var_sign(x):
    assert(abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign

# TODO(dhs): duplication
def ilit_to_vlit(x, n_vars):
    assert(x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign: return var + n_vars
    else: return var

class Problem(object):
    def __init__(self, n_vars, iclauses, label, n_cells_per_batch):
        self.n_vars = n_vars
        self.n_lits = 2 * n_vars
        self.n_clauses = len(iclauses)

        self.n_cells = sum(n_cells_per_batch)
        self.n_cells_per_batch = n_cells_per_batch

        self.label = label
        self.compute_L_unpack(iclauses)

        # will be a list of None for training problems


    def compute_L_unpack(self, iclauses):
        self.L_unpack_indices = np.zeros([self.n_cells, 2], dtype=np.int)
        cell = 0
        for clause_idx, iclause in enumerate(iclauses):
            vlits = [ilit_to_vlit(x, self.n_vars) for x in iclause]
            for vlit in vlits:
                self.L_unpack_indices[cell, :] = [vlit, clause_idx]
                cell += 1

        assert(cell == self.n_cells)

def shift_ilit(x, offset):
    assert(x != 0)
    if x > 0: return x + offset
    else:     return x - offset

def shift_iclauses(iclauses, offset):
    return [[shift_ilit(x, offset) for x in iclause] for iclause in iclauses]



def mk_batch_problem(problems):
    all_iclauses = []
    all_label = []
    all_n_cells = []

    offset = 0

    prev_n_vars = None
    #for dimacs, n_vars, iclauses, is_sat in problems:
    for n_vars, iclauses,label in problems:
        assert(prev_n_vars is None or n_vars == prev_n_vars)
        prev_n_vars = n_vars

        all_iclauses.extend(shift_iclauses(iclauses, offset))
        all_label.append(label)
        all_n_cells.append(sum([len(iclause) for iclause in iclauses]))
        #all_dimacs.append(dimacs)

        offset += n_vars

    #return Problem(offset, all_iclauses, all_is_sat, all_n_cells, all_dimacs)
    return Problem(offset, all_iclauses, all_label, all_n_cells)


def to_pkl(label_out_filename, n_vars, iclauses,label,pair):
    problems = []
    batches = []
    n_nodes_in_batch = 0
    prev_n_vars = None

    n_clauses = len(iclauses)
    n_cells = sum([len(iclause) for iclause in iclauses])
    n_nodes = 2 * n_vars + n_clauses


    n_nodes_in_batch = 0
    problems.append((n_vars, iclauses, 1))
    #problems.append(n_vars,iclauses, label)
    if len(problems) > 0:
        batches.append(mk_batch_problem(problems))

        del problems[:]
    pkl_file = open('/home/zhang/zouy/sat/data/sr_pkl/{}'.format(pair) + '.pkl', mode='w')
    with open('/home/zhang/zouy/sat/data/sr_pkl/{}'.format(pair) + '.pkl', "wb") as f:
        pickle.dump(batches, f)