import argparse
import sys
import probsat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('-d', '--dir_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_tries', type=int, default=100)
    parser.add_argument('--max_flips', type=int, default=50000)
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--neuro_ini', type=bool, default=False)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--dim', type=int, default=128, help='Dimension of variable and clause embeddings')
    parser.add_argument('--n_rounds', type=int, default=16, help='Number of rounds of message passing')
    parser.add_argument('-r', type=int)
    args = parser.parse_args()
    probsat.main(args)