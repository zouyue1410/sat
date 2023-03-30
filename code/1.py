import logging
import os
import statistics
import sys
import random as r
import time
import numpy as np
from neuro_initial import pre_assign
MAX_TRIES_FLAG = "--max_tries"
MAX_TRIES_DEFAULT = 300
MAX_FLIPS_FLAG = "--max_flips"
MAX_FLIPS_DEFAULT = 300
INPUT_FILE_FLAG = "--data"
FILES_FLAG = '--files'
CM_FLAG = "--cm"
CM = 0
CB_FLAG = "--cb"
CB = 2.3
REPEAT_FLAG = "-r"
REPEAT = 1
SORT_DATA_FILES_FLAG = '-s'

logger = logging.getLogger(__name__)


def find_indices(list_to_check, item_to_find):
    return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]


class SAT:
    def __init__(self, data: list, vars_count: int, clause_count: int, random: bool = False) -> None:
        self.vars_count = int(vars_count)
        self.clause_count = int(clause_count)
        self.clauses = data
        self.evaluation = [not not r.getrandbits(1) if random else True for _ in range(self.vars_count)]

    def __str__(self) -> str:
        return f"vars: {self.vars_count}\nclauses: {self.clause_count}\ndata: {self.clauses}\nactual evaluation: {self.evaluation}"

    def is_eval_satisfying(self):
        return not False in self.clauses_evaluation()

    def clauses_evaluation(self):
        return [
            True in [(self.evaluation[abs(int(y)) - 1] if int(y) > 0 else not self.evaluation[abs(int(y)) - 1]) for y in
                     x] for x in self.clauses]

    def satisfied_clauses_count(self):
        return sum(1 for i in self.clauses_evaluation() if i)

    def flip(self, index_of_var_to_flip):
        self.evaluation[index_of_var_to_flip] = not self.evaluation[index_of_var_to_flip]

    def _get_weights(self, clause, actual_eval, cm, cb):
        weights = []
        for var in clause:
            self.flip(abs(int(var)) - 1)
            m, b = 0, 0
            new_eval = self.clauses_evaluation()
            for n_eval, a_eval in zip(new_eval, actual_eval):
                if n_eval == True and a_eval == False:
                    m += 1
                elif n_eval == False and a_eval == True:
                    b += 1
            weights.append(m ** cm / (b + sys.float_info.epsilon) ** cb)
            self.flip(abs(int(var)) - 1)
        return [x / sum(weights) for x in weights]

    def probSAT_flip(self, cm, cb, flipped, backflipped):
        # print(flipped)
        actual_evaluation = self.clauses_evaluation()
        indecis = find_indices(actual_evaluation, False)
        clause = self.clauses[r.choice(indecis)]
        # print(f"{clause=}")
        weights = self._get_weights(clause, actual_evaluation, cm, cb)
        # print(f"{weights=}")
        variable = r.choices(clause, weights=weights)
        # print(variable)
        if abs(int(variable[0])) - 1 not in flipped:
            flipped.add(abs(int(variable[0])) - 1)
        else:
            # print("yyy")
            backflipped = backflipped + 1
            # print(backflipped)
        # print(f"{variable=}")
        self.flip(abs(int(variable[0])) - 1)
        return flipped, backflipped

    def random_evaluation_assign(self,file_name,args):


        outputs=pre_assign(args,file_name)
        self.evaluation=[]
        #self.evaluation = [not not r.getrandbits(1) for _ in range(self.vars_count)]
        for i in range (self.vars_count):
            self.evaluation.append(True) if outputs[i]==1 else self.evaluation.append(False)



        #self.evaluation = [not not r.getrandbits(1) for _ in range(self.vars_count)]


def handle_input_sat(file_name):
    vars_count = 0
    clauses_count = 0
    data = []
    with open(file_name, "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                vars_count = line.split()[-2]
                clauses_count = line.split()[-1]
            else:
                data.append(line.split()[:-1])

    return SAT(data, vars_count, clauses_count, True)


def probSAT(file_name,sat: SAT, max_tries, max_flips, cm, cb,args):
    tries = 0

    flip = 0
    flips_to_solution = []
    backflips = []
    unsat_clauses = []

    for i in range(max_tries):
        sat.random_evaluation_assign(file_name, args)
        tries += 1

        flipped = set()
        backflipped = 0
        # print(flipped)

        for j in range(max_flips):
            # print(j)
            # rint(backflipped)
            if sat.is_eval_satisfying():
                return ( backflips, flips_to_solution, sat.satisfied_clauses_count(), sat.clause_count)
            else:
                flip += 1
                flipped, backflipped = sat.probSAT_flip(cm, cb, flipped, backflipped)
            # print("back")
            # print(backflipped)




    return (backflips, flips_to_solution, sat.satisfied_clauses_count(), sat.clause_count)


def main(args):
    max_tries = MAX_TRIES_DEFAULT

    max_tries = args.max_tries


    max_flips = MAX_FLIPS_DEFAULT

    max_flips = args.max_flips


    cm = CM


    cb = CB


    repeat = REPEAT





    if True:
        try:
            path = args.dir_path
        except Exception:
            raise "Missing path"
        for (dirpath, dirnames, filenames) in os.walk(path):
            files = filenames
            break

        # sort files in folder by number specific for school data

        files.sort(key=lambda file_name: int(file_name[3:-9]))

        # create new path if don't exist
        newpath = path + '-out'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        start_time = time.time()

        avg_flips = []

        solved = []

        avg_backflips = []


        for i, file_name in enumerate(files):

            if i >= 100:
                break
            print(f"{file_name}")
            sat = handle_input_sat(os.path.join(path, file_name))

            if repeat == 1:
                med_flips = []

                # print(max_tries)
                backflips, flips_to_solution, n_satisfied_clauses, n_all_clauses = probSAT(file_name,sat, max_tries, max_flips,
                                                                                           cm, cb,args)

                print(n_satisfied_clauses, n_all_clauses)


                solved.append(1) if n_all_clauses==n_satisfied_clauses else solved.append(0)
                print(solved)
                # print(probSAT(sat, max_tries, max_flips, cm, cb))


        end = time.time()
        duration = (end - start_time) * 1000
        print("time%.4f" % duration)
        acc = 100 * np.mean(solved)

        print("Acc%.4f" % acc)



if __name__ == "__main__":
    main(sys.argv[1:])