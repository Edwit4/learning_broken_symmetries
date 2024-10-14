import os
import sys
import random
import socket
import argparse
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import numpy as np
from src.utilities.models import PFN
from src.utilities.datasets import toy_pfn_dataset, toy_pfn_aug_dataset
from src.utilities.torch_utils import trainer, test
from src.utilities.preprocessing import make_toy_pfn_standardscaler
from src.utilities.data_utils import split_data, kfold_split
from src.utilities.logging import initialize_experiment

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('fold_idx', metavar='f', type=int, nargs=1,
        help='Index of kfold')
    parser.add_argument('train_size', metavar='t', type=float, nargs=1,
        help='Proportion of total available training set to use')
    parser.add_argument('layer_size', metavar='s', type=int, nargs=1,
        help='Size of hidden layers')
    parser.add_argument('learning_rate', metavar='l', type=float, nargs=1,
        help='Learning rate')
    parser.add_argument('batch_size', metavar='b', type=int, nargs=1,
        help='Batch size')
    parser.add_argument('version', metavar='v', type=str, nargs=1,
        help='Experiment version name')
    parser.add_argument('port', metavar='p', type=int, nargs=1,
        help='Port for Torch multiprocessing')
    parser.add_argument('--num_epochs', metavar='e', type=int, default=None,
        help='Number of training epochs, if None use early stopping with some max epochs')
    args = parser.parse_args()
    fold_idx = args.fold_idx[0]
    train_size = args.train_size[0]
    layer_size = args.layer_size[0]
    learning_rate = args.learning_rate[0]
    batch_size = args.batch_size[0]
    version = args.version[0]
    port = args.port[0]
    num_epochs = args.num_epochs

    world_size = 4
    seed = 123
    test_size = 0.2
    n_splits = 5
    pixelization = 'fullrect'

    if num_epochs is None:
        max_epochs = 300
        early_stop_params = {'patience': 10, 'min_change': 1e-3}
    else:
        max_epochs = num_epochs
        early_stop_params = None

    script_path = os.path.abspath(sys.argv[0])
    script_dir = os.path.dirname(script_path)
    save_dir = os.path.join(script_dir, "versions")
    data_dir = 'src/data'
    version_name = f'{version}_trainsize_{train_size}_layersize_{layer_size}_learningrate_{learning_rate}_batchsize_{batch_size}_fold_{fold_idx}'

    torch.manual_seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)

    initialize_experiment(save_dir, version_name)

    full_indices = np.arange(len(np.load(f'{data_dir}/rect_labels_600000_events.npy')))

    train_valid_idx, test_idx = split_data(full_indices, split_size=test_size, 
                                           shuffle=True, rng=rng)
    train_valid_idx = train_valid_idx[:int(train_size*len(train_valid_idx))]
    train_idx, valid_idx = kfold_split(train_valid_idx, fold_idx=fold_idx, n_splits=n_splits, 
                                       shuffle=True, rng=rng)
    print('Train count: ', len(train_idx), 'Valid count: ', len(valid_idx), 'Test count: ', len(test_idx))

    scaler = make_toy_pfn_standardscaler(data_dir, rotation=0, indices=train_idx,
                                         pixelization=pixelization)

    model = PFN(Phi_size=layer_size, l_size=layer_size, F_size=layer_size)
    train_dataset = toy_pfn_aug_dataset(data_dir, indices=train_idx, scaler=scaler,
                                    pixelization=pixelization)
    val_dataset = toy_pfn_aug_dataset(data_dir, indices=valid_idx, scaler=scaler,
                                  pixelization=pixelization)
    datasets = {'train': train_dataset, 'valid': val_dataset}

    train_kwargs = {'learning_rate': learning_rate*world_size, 
                    'batch_size': batch_size//world_size, 
                    'max_epochs': max_epochs, 'version_name': version_name,
                    'early_stop_params': early_stop_params}
    train_args = (world_size, model, datasets, save_dir, port,
                  train_kwargs)
    mp.spawn(trainer, args=train_args, nprocs=world_size, join=True)

    test_checkpoint = os.listdir(f'{save_dir}/{version_name}/checkpoints')[-1]

    test_model = PFN(Phi_size=layer_size, l_size=layer_size, F_size=layer_size)
    test_dataset = toy_pfn_dataset(data_dir, rotation=0, indices=test_idx, scaler=scaler,
                                   pixelization=pixelization)
    saved_model_path = f'{save_dir}/{version_name}/checkpoints/{test_checkpoint}'

    test_kwargs = {'version_name': version_name, 'batch_size': batch_size//world_size}
    test_args = (world_size, test_model, test_dataset, save_dir, 
                 saved_model_path, port, test_kwargs)
    mp.spawn(test, args=test_args, nprocs=world_size, join=True)
