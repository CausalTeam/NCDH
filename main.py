import os
import argparse
import pickle
import numpy as np
from pathlib import Path
from generate_data import gen_data
from train import train




def parse():
    """Argument parser for all configs.
    """
    parser = argparse.ArgumentParser(description='')

    # data generation args
    parser.add_argument('-n', type=int, default=5,
                        help="number of latent components")
    parser.add_argument('-k', type=int, default=11,
                        help="number of latent states")
    # parser.add_argument('-t', type=int, default=100000,
                        # help="number of time steps")
    parser.add_argument('-t', type=int, default=10000,
                        help="number of time steps")
    parser.add_argument('--mix-depth', type=int, default=4,
                        help="number of mixing layers")
    parser.add_argument('--prob-stay', type=float, default=0.99,
                        help="probability of staying in a state")
    parser.add_argument('--whiten', action='store_true', default=True,
                        help="PCA whiten data as preprocessing")

    # set seeds
    parser.add_argument('--data-seed', type=int, default=0,
                        help="seed for initializing data generation")
    parser.add_argument('--mix-seed', type=int, default=0,
                        help="seed for initializing mixing mlp")
    parser.add_argument('--est-seed', type=int, default=7,
                        help="seed for initializing function estimator mlp")
    parser.add_argument('--distrib-seed', type=int, default=7,
                        help="seed for estimating distribution paramaters")
    # training & optimization parameters
    parser.add_argument('--hidden-units', type=int, default=10,
                        help="num. of hidden units in function estimator MLP")
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help="learning rate for training")
    parser.add_argument('--num-epochs', type=int, default=100,
                        help="number of training epochs")
    parser.add_argument('--subseq-len', type=int, default=100,
                        help="length of subsequences")
    parser.add_argument('--minibatch-size', type=int, default=64,
                        help="number of subsequences in a minibatch")
    parser.add_argument('--decay-rate', type=float, default=1.,
                        help="decay rate for training (default to no decay)")
    parser.add_argument('--decay-interval', type=int, default=15000,
                        help="interval (in iterations) for full decay of LR")
    # CUDA settings
    parser.add_argument('--cuda', action='store_true', default=True,
                        help="use GPU training")
    # saving
    parser.add_argument('--out-dir', type=str, default="output/",
                        help="location where data is saved")
    args = parser.parse_args()
    return args


def main():
    args = parse()

    # check theoretical assumption satisfied
    assert args.k > 2*args.n, "K not set high enough for given N"

    # generate source data
    s_data, state_seq, mu, D, A = gen_data(args.n, args.k, args.t,
                                                  args.prob_stay,
                                                  random_seed=args.data_seed)



    x_data = np.zeros([args.t, args.n])
    for i in range(s_data.shape[0]):
        x_data[i] = np.array([(s_data[i][0])*(s_data[i][0]),
                              s_data[i][1] * np.log((s_data[i][0])*(s_data[i][0]) + 1),
                              np.abs(s_data[i][0]) * s_data[i][2],
                              np.sin(np.abs(s_data[i][0]) * s_data[i][2])+s_data[i][3]*s_data[i][3],
                              np.sin(np.abs(s_data[i][0]) * s_data[i][1])+s_data[i][3]])
    # x_data = np.loadtxt("arth150_500.txt", dtype=np.int)
    print(x_data)

    # preprocessing
    # if args.whiten:
    #     pca = PCA(whiten=True)
    #     x_data = pca.fit_transform(x_data)
    # print(x_data)
    # create variable dicts for training
    data_dict = {'x_data': x_data,
                 's_data': s_data,
                 'state_seq': state_seq}

    train_dict = {'mix_depth': args.mix_depth,
                  'hidden_size': args.hidden_units,
                  'learning_rate': args.learning_rate,
                  'num_epochs': args.num_epochs,
                  'subseq_len': args.subseq_len,
                  'minib_size': args.minibatch_size,
                  'decay_rate': args.decay_rate,
                  'decay_steps': args.decay_interval}

    seed_dict = {'est_mlp_seed': args.est_seed,
                 'est_distrib_seed': args.distrib_seed}

    # set up dict to save results
    results_dict = {}
    results_dict['data_config'] = {'N': args.n, 'K': args.k, 'T': args.t,
                                   'mix_depth': args.mix_depth,
                                   'p_stay': args.prob_stay,
                                   'data_seed': args.data_seed,
                                   'mix_seed': args.mix_seed}
    results_dict['train_config'] = {'train_vars': train_dict,
                                    'train_seeds': seed_dict}
    results_dict['results'] = []

    # train HM-nICA model
    s_est_all, sort_idx, results_dict, est_params = train(
        data_dict, train_dict, seed_dict, results_dict)
    # print(s_est.shape)

    if not os.path.exists(args.out_dir):
        Path(args.out_dir).mkdir(parents=True)
    with open(args.out_dir+"all_results.pickle", 'ab') as out:
        pickle.dump(results_dict, out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # sys.exit(main())
    main()
