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
    def __init__(self, n_vars, iclauses, label, n_cells):
        self.n_vars = n_vars
        self.n_lits = 2 * n_vars
        self.n_clauses = len(iclauses)
        self.label = label
        self.n_cells = n_cells
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
    n_vars=problems[0]
    iclauses=problems[1]
    label=problems[2]
    n_cells=problems[3]
    #return Problem(offset, all_iclauses, all_is_sat, all_n_cells, all_dimacs)
    return Problem(n_vars,iclauses,label,n_cells)


