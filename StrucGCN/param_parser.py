"""Parsing the parameters."""

import argparse
import os
import copy
from tabulate import tabulate

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """
    parser = argparse.ArgumentParser(description="Run role_awareSAGE.")

    parser.add_argument("--N",
                        type=int,
                        default=10,
                        help="times")

    parser.add_argument("--EPOCHS",
                        nargs="?",
                        type=int,
                        default=1000,
                        help="epoch")

    parser.add_argument("--train_ra",
                        type=float,
                        default=60,
                        help="Ratio of test set. Default is 0.6.")

    parser.add_argument("--val_ra",
                        type=float,
                        default=20,
                        help="Ratio of test set. Default is 0.2.")

    parser.add_argument("--dataset",
                        nargs="?",
                        default="actor",
                        help="cora, citeseer, pubmed, actor, brazil, europe, usa, chameleon, cornell, squirrel, "
                             "texas, wisconsin")

    parser.add_argument("--feat_flag",
                        nargs="?",
                        type=bool,
                        default="True",
                        help="node with/without initial features")

    parser.add_argument("--gcn",
                        type=bool,
                        default="True")

    parser.add_argument("--normalize",
                        type=str,
                        default="degree",
                        )

    parser.add_argument("--aggr",
                        type=str,
                        default="mean_concat",
                        )

    parser.add_argument("--alpha",
                        nargs="?",
                        type=float,
                        default=0.3,
                        help="I")

    parser.add_argument("--beta",
                        nargs="?",
                        type=float,
                        default=0.1,
                        help="A")

    parser.add_argument("--gama",
                        nargs="?",
                        type=float,
                        default=0.3,
                        help="A‘")

    parser.add_argument("--lr",
                        nargs="?",
                        type=float,
                        default=0.001,
                        help="learning rate")

    parser.add_argument("--dr",
                        nargs="?",
                        type=float,
                        default=0.4,
                        help="dropout")

    parser.add_argument("--h_dim",
                        type=int,
                        default=64,
                        help="Default is 64.")

    parser.add_argument("--factor",
                        nargs="?",
                        type=float,
                        default=0.5)

    parser.add_argument("--patience",
                        type=int,
                        default=20)

    parser.add_argument("--early_stopping_patience",
                        type=int,
                        default=400,
                        help="early stop")

    parser.add_argument("--weight_decay",
                        nargs="?",
                        type=float,
                        default=5e-4,
                        help="Default is 5e-4.")

    parser.add_argument("--role_generate",
                        nargs="?",
                        default="struc2vec",
                        help="wl, motif, motif_tri, struc2vec, degree")

    parser.add_argument("--feat_generate",
                        nargs="?",
                        default="onehot",
                        help="deepwalk, node2vec")

    parser.add_argument("--embedding_dim",
                        type=int,
                        default=1024,
                        help="Default is 64.")

    parser.add_argument("--workers",
                        type=int,
                        default=4,
                        help="Number of cores. Default is 4.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs. Default is 10.")

    parser.add_argument("--labeling-iterations",
                        type=int,
                        default=2,
                        help="Number of WL labeling iterations. Default is 2.")

    parser.add_argument("--log-base",
                        type=int,
                        default=1.5,
                        help="Log base for label creation. Default is 1.5.")

    parser.add_argument("--motif_graphlet_size",
                        type=int,
                        default=4,
                        help="Maximal graphlet size. Default is 4.")

    parser.add_argument("--motif_quantiles",
                        type=int,
                        default=5,
                        help="Number of quantiles for binning. Default is 5/8.")

    parser.add_argument("--motif_compression",
                        nargs="?",
                        default="string",
                        help="Motif compression procedure -- string or factorization.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Sklearn random seed. Default is 42.")

    parser.add_argument("--motif_factors",
                        type=int,
                        default=8,
                        help="Number of factors for motif compression. Default is 8.")

    parser.add_argument("--motif_clusters",
                        type=int,
                        default=20,
                        help="Number of motif based labels. Default is 10.")

    parser.add_argument("--motif_beta",
                        type=float,
                        default=0.01,
                        help="Motif compression factorization regularizer. Default is 0.01.")

    return parser.parse_args(args=[])


def print_args_as_table(args):
    args_dict = vars(args)
    table = [[key, value] for key, value in args_dict.items()]
    print(tabulate(table, headers=["Parameter", "Value"]))

def para_load():
    path = 'experiment_runs.txt'
    args_runs = []
    args_runs_element = {}
    try:
        with open(path) as para_file:
            first_line = para_file.readline().strip()
            dict_key = first_line.split('\t')
            for line in para_file:
                line = line.rstrip().split('\t')
                if line[0] == '#':
                    continue
                assert (len(dict_key) == len(line))
                for i in range(len(line)):
                    args_runs_element[dict_key[i]] = line[i]
                args_runs.append(copy.deepcopy(args_runs_element))
    except Exception as e:
        print(f"An error occurred: {e}")
    return args_runs


def para_update(args, args_runs):
    keys = list(args_runs.keys())
    sum = float(args_runs['alpha']) + float(args_runs['beta']) + float(args_runs['gama'])
    args_runs['alpha'] = float(args_runs['alpha']) / sum
    args_runs['beta'] = float(args_runs['beta']) / sum
    args_runs['gama'] = float(args_runs['gama']) / sum
    for key in keys:
        if key in vars(args):
            expected_type = type(vars(args)[key])
            try:
                if expected_type == bool:
                    if args_runs[key].lower() in ['false', 'False', 'FALSE']:
                        vars(args)[key] = False
                    else:
                        vars(args)[key] = True
                else:
                    vars(args)[key] = expected_type(args_runs[key])
            except (ValueError, TypeError):
                print(f"Warning: Could not convert {key} to {expected_type}. Using default value.")
        else:
            print(f"Warning: {key} is not a valid attribute of args.")

    print_args_as_table(args)
    return args