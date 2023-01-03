import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--taskname', type=str, default='neuro_initial', help='task name')

parser.add_argument('--dim', type=int, default=128, help='Dimension of variable and clause embeddings')
parser.add_argument('--n_rounds', type=int, default=16, help='Number of rounds of message passing')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--max_nodes_per_batch', action='store', type=int)
parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)
parser.add_argument('--py_seed', action='store', dest='py_seed', type=int, default=None)
parser.add_argument('--np_seed', action='store', dest='np_seed', type=int, default=None)
parser.add_argument('--one', action='store', dest='one', type=int, default=0)
parser.add_argument('--model_dir', type=str, help='model folder dir')
parser.add_argument('--data_dir', type=str,help='data folder dir')
parser.add_argument('--restore', type=str, default=None, help='continue train from model')
parser.add_argument('--eval_data_dir', type=str,help='eval data folder dir')

