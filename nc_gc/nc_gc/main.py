import argparse
from log import *
from train import *
# from simulate import *
from utils import *
import sys
sys.path.append('C:/Users/85264/Desktop/studyinust/MPhil/PycharmCoding/NAS4GNN/extra/v1')


def main():
    parser = argparse.ArgumentParser('Interface for Auto-GNN framework')

    parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of the train against whole')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the val against whole')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test against whole')
    parser.add_argument('--model', type=str, default='Auto-GNN', help='model to use')
    parser.add_argument('--metric', type=str, default='acc', help='metric for evaluating performance', choices=['acc', 'auc'])
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--directed', type=bool, default=False, help='(Currently unavailable) whether to treat the graph as directed')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='(Currently unavailable) whether to use multi cpu cores to prepare data')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')

    parser.add_argument('--task', type=str, default='graph', help='type of task', choices=['node', 'graph'])
    parser.add_argument('--dataset', type=str, default='MUTAG', help='dataset name')  # choices=['PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COX2', 'DD', 'ENZYMES', 'NCI1']
    parser.add_argument('--seed', type=int, default=18, help='seed to initialize all the random modules')
    parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
    # general model and training setting
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs to search')
    parser.add_argument('--retrain_epoch', type=int, default=100, help='number of epochs to retrain')

    parser.add_argument('--layers', type=int, default=4, help='largest number of layers')
    parser.add_argument('--hidden_features', type=int, default=64, help='hidden dimension')
    parser.add_argument('--bs', type=int, default=128, help='minibatch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    # parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')


    # logging & debug
    parser.add_argument('--log_dir', type=str, default='./log/', help='log directory')

    parser.add_argument('--summary_file', type=str, default='result_summary.log', help='brief summary of training result')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='whether to use debug mode')


    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    check(args)
    logger = set_up_log(args, sys_argv)
    set_random_seed(args)

    if args.dataset != 'PPI':
        dataset, in_features, out_features = get_data(args=args, logger=logger)
        train_mask, val_mask, test_mask = reset_mask(dataset, args=args, logger=logger, stratify=None)
        train_loader, val_loader, test_loader = get_loader(dataset, args=args)
    else:
        in_features, out_features, train_mask, val_mask, test_mask, train_loader, val_loader, test_loader = helper_ppi(args)

    model = get_model(layers=args.layers, in_features=in_features, out_features=out_features, args=args, logger=logger)

    model, results = search(model, train_loader, val_loader, train_mask, val_mask, args, logger)
    model, results = retrain(model, train_loader, test_loader, train_mask, test_mask, args, logger)

    save_performance_result(args, logger, model)


if __name__ == '__main__':
    main()
