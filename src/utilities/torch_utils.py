import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.autograd.profiler as profiler
import numpy as np
import pandas as pd
from contextlib import contextmanager
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from src.utilities.logging import plot_loss_history, plot_outputs, \
    plot_roc_curve, plot_auc_history, write_csv_log

@contextmanager
def conditional_profiler(profile_flag, profiler_context):
    if profile_flag:
        with profiler_context as c:
            yield c
    else:
        yield None

def setup(rank, world_size, port):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup(world_size):

    if world_size > 1:
        dist.destroy_process_group()

def get_indices_from_sampler(sampler):
    indices = []
    for idx in sampler:
        indices.append(idx)
    return indices

# Suggested to keep num_workers = 0 / pin_memory = False for DDP
def prepare_dataloader(rank, world_size, dataset, batch_size=32, pin_memory=False,
                       num_workers=0, persistent_workers=False):

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                      shuffle=False, drop_last=False)

        rank_indices = get_indices_from_sampler(sampler)

    else:
        sampler = None
        rank_indices = np.arange(len(dataset)).tolist()
        
    dataset.set_sample_device(rank, rank_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                             num_workers=num_workers, drop_last=False, shuffle=False,
                             sampler=sampler, persistent_workers=persistent_workers)

    return dataloader

def train_epoch(rank, world_size, model, dataloader, criterion, optimizer,
                epoch, save_dir, version_name, criterion_2=None, crit_scale=1,
                crit_2_scale=200, profile=False, epoch_metrics=False):

    with conditional_profiler(profile, profiler.profile(
                              profile_memory=False, with_stack=True,
                              record_shapes=False, use_cuda=True)) as prof:
        model.train()

        if world_size > 1:
            dataloader.sampler.set_epoch(epoch)       

        total_loss = 0.0
        temp_loss = torch.tensor(0.0, device=rank)
        all_labels, all_outputs = [], []
        with tqdm(dataloader, unit="batch", leave=False) as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                optimizer.zero_grad(set_to_none=True)
                if criterion_2 is None:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                else:
                    labels = labels.reshape(labels.size()[0]*labels.size()[1], 1)
                    outputs, std_dev = model(inputs)
                    outputs = outputs
                    std_dev = std_dev
                    loss_1 = criterion(outputs, labels)
                    loss_2 = criterion_2(std_dev, 
                                         torch.zeros(std_dev.size(),device=rank))
                    loss = crit_scale*loss_1 + crit_2_scale*loss_2
                loss.backward()
                optimizer.step()
                temp_loss += loss

                if epoch_metrics:
                    all_labels.append(labels)
                    all_outputs.append(outputs)

            total_loss = temp_loss.item()
            
        if epoch_metrics:
            all_labels = torch.cat(all_labels)
            all_outputs = torch.cat(all_outputs)

            # Initialize the tensors that will hold the gathered data on rank 0
            if rank == 0:
                gathered_labels = [torch.zeros_like(all_labels) 
                                   for _ in range(world_size)]
                gathered_outputs = [torch.zeros_like(all_outputs) 
                                    for _ in range(world_size)]
            else:
                gathered_labels = gathered_outputs = None

            # Gather the data from all the GPUs
            if world_size > 1:
                dist.gather(all_labels, gather_list=gathered_labels, dst=0)
                dist.gather(all_outputs, gather_list=gathered_outputs, dst=0)
            else:
                gathered_labels = all_labels
                gathered_outputs = all_outputs

            train_auc = torch.tensor(0, dtype=torch.float32, device=rank)
            if rank == 0:

                # Concatenate the gathered data
                gathered_labels = torch.cat(gathered_labels)
                gathered_outputs = torch.cat(gathered_outputs)

                # Convert tensors to numpy arrays
                gathered_labels = gathered_labels.detach().cpu().numpy()
                gathered_outputs = gathered_outputs.detach().cpu().numpy()

                fpr, tpr, thresholds = roc_curve(gathered_labels, gathered_outputs)
                train_auc = auc(fpr, tpr)
                train_auc = torch.tensor(train_auc, dtype=torch.float32, device=rank)

                plot_roc_curve(fpr, tpr, train_auc, save_dir, version_name,
                               'train_roc_curve.png')
                plot_outputs(gathered_labels, gathered_outputs, save_dir, version_name,
                             'train_outs.png')

            if world_size > 1:
                dist.broadcast(train_auc, src=0)

            train_auc = train_auc.item()
        else:
            train_auc = np.nan

        gpu_loss = torch.tensor(total_loss/len(dataloader), device=rank)

        if world_size > 1:
            dist.all_reduce(gpu_loss, op=dist.ReduceOp.SUM) 

    if profile:
        with open(f'{save_dir}/{version_name}/train_epoch_profiler_output.txt', 
                  'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total"))

    return gpu_loss.item() / world_size, train_auc

def eval_epoch(rank, world_size, model, dataloader, criterion, epoch, save_dir,
               version_name, criterion_2=None, crit_scale=1, crit_2_scale=200,
               profile=False, epoch_metrics=False):

    with conditional_profiler(profile, profiler.profile(
                              profile_memory=True, with_stack=True,
                              record_shapes=True, use_cuda=True)) as prof:
        model.eval()

        if world_size > 1:
            dataloader.sampler.set_epoch(epoch)       

        total_loss = 0.0
        temp_loss = torch.tensor(0.0, device=rank)
        all_labels, all_outputs = [], []
        with torch.no_grad():
            with tqdm(dataloader, unit="batch", leave=False) as tepoch:
                for inputs, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")

                    if criterion_2 is None:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    else:
                        labels = labels.reshape(labels.size()[0]*labels.size()[1], 1)
                        outputs, std_dev = model(inputs)
                        outputs = outputs
                        std_dev = std_dev
                        loss_1 = criterion(outputs, labels)
                        loss_2 = criterion_2(std_dev,
                                             torch.zeros(std_dev.size(),device=rank))
                        loss = crit_scale*loss_1 + crit_2_scale*loss_2

                    temp_loss += loss

                    if epoch_metrics:
                        all_labels.append(labels)
                        all_outputs.append(outputs)
            total_loss = temp_loss.item()

        if epoch_metrics:
            all_labels = torch.cat(all_labels)
            all_outputs = torch.cat(all_outputs)

            # Initialize the tensors that will hold the gathered data on rank 0
            if rank == 0:
                gathered_labels = [torch.zeros_like(all_labels) 
                                   for _ in range(world_size)]
                gathered_outputs = [torch.zeros_like(all_outputs) 
                                    for _ in range(world_size)]
            else:
                gathered_labels = gathered_outputs = None

            # Gather the data from all the GPUs
            if world_size > 1:
                dist.gather(all_labels, gather_list=gathered_labels, dst=0)
                dist.gather(all_outputs, gather_list=gathered_outputs, dst=0)
            else:
                gathered_labels = all_labels
                gathered_outputs = all_outputs

            valid_auc = torch.tensor(0, dtype=torch.float32, device=rank)
            if rank == 0:

                # Concatenate the gathered data
                gathered_labels = torch.cat(gathered_labels)
                gathered_outputs = torch.cat(gathered_outputs)

                # Convert tensors to numpy arrays
                gathered_labels = gathered_labels.detach().cpu().numpy()
                gathered_outputs = gathered_outputs.detach().cpu().numpy()

                fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
                valid_auc = auc(fpr, tpr)
                valid_auc = torch.tensor(valid_auc, dtype=torch.float32, device=rank)

                plot_roc_curve(fpr, tpr, valid_auc, save_dir, version_name, 
                               'valid_roc_curve.png')
                plot_outputs(gathered_labels, gathered_outputs, save_dir, version_name, 
                             'valid_outs.png')

            if world_size > 1:
                dist.broadcast(valid_auc, src=0)

            valid_auc = valid_auc.item()
        else:
            valid_auc = np.nan

        gpu_loss = torch.tensor(total_loss/len(dataloader), device=rank)

        if world_size > 1:
            dist.all_reduce(gpu_loss, op=dist.ReduceOp.SUM)

    if profile:
        with open(f'{save_dir}/{version_name}/eval_epoch_profiler_output.txt', 
                  'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total"))

    return gpu_loss.item() / world_size, valid_auc

def trainer(rank, world_size, model, datasets, save_dir, port, kwargs):

    try:
        criterion = kwargs['criterion'] 
    except Exception:
        criterion = nn.BCELoss()
    try:
        criterion_2 = kwargs['criterion_2']
    except Exception:
        criterion_2 = None
    try:
        crit_scale = kwargs['crit_scale']
    except Exception:
        crit_scale = 1
    try:
        crit_2_scale = kwargs['crit_2_scale']
    except Exception:
        crit_2_scale = 200
    try:
        max_epochs = kwargs['max_epochs']
    except Exception:
        max_epochs = 300
    try:
        version_name = kwargs['version_name']
    except Exception:
        version_name = 'default_version'
    try:
        early_stop_params = kwargs['early_stop_params']
    except Exception:
        early_stop_params = None
    try:
        learning_rate = kwargs['learning_rate']
    except Exception:
        learning_rate = 1e-3
    try:
        batch_size = kwargs['batch_size']
    except Exception:
        batch_size = 32 
    try:
        optimizer = kwargs['optimizer'] 
    except Exception:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    try:
        profile = kwargs['profile']
        if profile == 'trainer':
            profile = True
            profile_train_epoch = False
            profile_eval_epoch = False
        elif profile == 'train_epoch':
            profile = False
            profile_train_epoch = True
            profile_eval_epoch = False
        elif profile == 'eval_epoch':
            profile = False
            profile_train_epoch = False
            profile_eval_epoch = True
        else:
            profile = False
            profile_train_epoch = False
            profile_eval_epoch = False
    except Exception:
        profile = False
        profile_train_epoch = False
        profile_eval_epoch = False
    try:
        epoch_metrics = kwargs['epoch_metrics'] 
    except Exception:
        epoch_metrics = False
    
    
    with conditional_profiler(profile, profiler.profile(
                              profile_memory=True, with_stack=True,
                              record_shapes=True, use_cuda=True)) as prof:
        setup(rank, world_size, port)
        model = model.to(rank)
        if world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank)

        dataloaders = {}
        for key in datasets.keys():
            dataloaders[key] = prepare_dataloader(rank, world_size, 
                                                  datasets[key], batch_size=batch_size)

        if rank == 0:
            with open(f'{save_dir}/{version_name}/params.txt', 'w') as f:
                if early_stop_params is not None:
                    f.write('early stopping parameters:\n')
                    f.write(f'patience: {early_stop_params["patience"]}\n')
                    f.write(f'min_change: {early_stop_params["min_change"]}\n')
                else:
                    f.write(f'max_epochs: {max_epochs}\n')
                f.write(f'learning_rate: {learning_rate}\n')
                f.write(f'batch_size: {batch_size}\n')
                f.write('\n')
                f.write(f'total number of model parameters: {sum(p.numel() for p in model.parameters())}\n')
                f.write('model summary:\n')
                for name, layer in model.named_children():
                    f.write(f'{name}, {layer}\n')

        best_val_loss = float('inf')
        epochs_since_improvement = 0

        early_stop_flag = torch.tensor(0, dtype=torch.int32, device=rank)
        for epoch in range(max_epochs):
            train_loss, train_auc = train_epoch(rank, world_size, model,
                                                dataloaders['train'],
                                                criterion, optimizer, epoch,
                                                save_dir, version_name,
                                                criterion_2=criterion_2,
                                                crit_scale=crit_scale,
                                                crit_2_scale=crit_2_scale,
                                                profile=profile_train_epoch,
                                                epoch_metrics=epoch_metrics)
            if 'valid' in dataloaders:
                val_loss, val_auc = eval_epoch(rank, world_size, model,
                                               dataloaders['valid'], criterion,
                                               epoch, save_dir, version_name,
                                               criterion_2=criterion_2,
                                               crit_scale=crit_scale,
                                               crit_2_scale=crit_2_scale,
                                               profile=profile_eval_epoch,
                                               epoch_metrics=epoch_metrics)
            else:
                val_loss, val_auc = np.nan, np.nan

            if rank == 0:
                row_dict = {'epoch': epoch, 'train_loss': train_loss,
                            'val_loss': val_loss, 'train_auc': train_auc,
                            'val_auc': val_auc}

                write_csv_log(save_dir, version_name, row_dict)

                plot_loss_history(save_dir, version_name, 'loss_history.png')
                plot_auc_history(save_dir, version_name, 'auc_history.png')

            if early_stop_params is not None:
                if val_loss < (best_val_loss - early_stop_params['min_change']):
                    best_val_loss = val_loss
                    if rank == 0:
                        checkpoints = os.listdir(
                                f'{save_dir}/{version_name}/checkpoints')
                        checkpoints = [os.path.abspath(os.path.join(
                            f'{save_dir}/{version_name}/checkpoints', f)) 
                                    for f in checkpoints]
                        if len(checkpoints) > 0:
                            for c in checkpoints:
                                os.remove(c)
                        torch.save(model.state_dict(),
                                    f'{save_dir}/{version_name}/checkpoints/epoch_{epoch}.pt')
                        stop_epoch = epoch
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= early_stop_params['patience']:
                    early_stop_flag = torch.tensor(1, dtype=torch.int32, device=rank)

                if world_size > 1:
                    dist.all_reduce(early_stop_flag, op=dist.ReduceOp.SUM)

                if early_stop_flag.item() > 0:
                    if rank == 0:
                        plot_loss_history(save_dir, version_name, 'loss_history.png',
                                          early_stop=stop_epoch)
                        plot_auc_history(save_dir, version_name, 'auc_history.png', 
                                         early_stop=stop_epoch)
                        print("Early stopping")
                    break
            else:
                if rank == 0:
                    checkpoints = os.listdir(f'{save_dir}/{version_name}/checkpoints')
                    checkpoints = [os.path.abspath(os.path.join(
                        f'{save_dir}/{version_name}/checkpoints', f)) 
                                for f in checkpoints]
                    if len(checkpoints) > 0:
                        for c in checkpoints:
                            os.remove(c)
                    torch.save(model.state_dict(),
                                f'{save_dir}/{version_name}/checkpoints/epoch_{epoch}.pt')

    if profile:
        with open(f'{save_dir}/{version_name}/trainer_profiler_output.txt', 
                  'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total"))

    cleanup(world_size)

def test(rank, world_size, model, test_dataset, save_dir, saved_model_path,
         port, kwargs):

    try:
        version_name = kwargs['version_name'] 
    except Exception:
        version_name = 'default_version'
    try:
        batch_size = kwargs['batch_size'] 
    except Exception:
        batch_size = 32
    try:
        profile = kwargs['profile']
    except Exception:
        profile = False


    with conditional_profiler(profile, profiler.profile(
                              profile_memory=False, with_stack=True,
                              record_shapes=False, use_cuda=True)) as prof:

        setup(rank, world_size, port)

        # Load the saved model
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        state_dict = torch.load(saved_model_path, map_location=map_location)
        modified_state_dict = {}
        for key, value in state_dict.items():
            new_key = '.'.join(key.split('.')[-3:])
            modified_state_dict[new_key] = value
        model.load_state_dict(modified_state_dict)
        
        # Move the model to the desired device
        model.to(rank)
        if world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank)

        # Prepare the test DataLoader
        test_dataloader = prepare_dataloader(rank, world_size, test_dataset,
                                             batch_size=batch_size)

        all_inputs, all_labels, all_outputs = [], [], []
        model.eval()
        with torch.no_grad():
            with tqdm(test_dataloader, unit="batch", leave=False) as tepoch:
                for inputs, labels in tepoch:
                    all_labels.append(labels)
                    all_inputs.append(inputs)
                    outs = model(inputs.to(rank))
                    if isinstance(outs, tuple):
                        outs = outs[0]
                    all_outputs.append(outs)
        all_labels = torch.cat(all_labels).to(torch.int32)
        all_inputs = torch.cat(all_inputs)
        all_outputs = torch.cat(all_outputs)

        # Initialize the tensors that will hold the gathered data on rank 0
        if rank == 0:
            gathered_labels = [torch.zeros_like(all_labels,dtype=torch.int32)
                               for _ in range(world_size)]
            gathered_inputs = [torch.zeros_like(all_inputs) 
                               for _ in range(world_size)]
            gathered_outputs = [torch.zeros_like(all_outputs) 
                                for _ in range(world_size)]
        else:
            gathered_labels = gathered_inputs = gathered_outputs = None

        # Gather the data from all the GPUs
        if world_size > 1:
            dist.gather(all_labels, gather_list=gathered_labels, dst=0)
            dist.gather(all_inputs, gather_list=gathered_inputs, dst=0)
            dist.gather(all_outputs, gather_list=gathered_outputs, dst=0)
        else:
            gathered_labels = all_labels
            gathered_inputs = all_inputs
            gathered_outputs = all_outputs

        if rank == 0:

            # Concatenate the gathered data
            if world_size > 1:
                gathered_labels = torch.cat(gathered_labels)
                gathered_inputs = torch.cat(gathered_inputs)
                gathered_outputs = torch.cat(gathered_outputs)

            # Convert tensors to numpy arrays
            gathered_labels = gathered_labels.detach().cpu().numpy()
            gathered_inputs = gathered_inputs.detach().cpu().numpy()
            gathered_outputs = gathered_outputs.detach().cpu().numpy()

            fpr, tpr, thresholds = roc_curve(gathered_labels, gathered_outputs)
            test_auc = auc(fpr, tpr)

            np.save(f'{save_dir}/{version_name}/test_npys/test_labels.npy', 
                    gathered_labels)
            np.save(f'{save_dir}/{version_name}/test_npys/test_inputs.npy', 
                    gathered_inputs)
            np.save(f'{save_dir}/{version_name}/test_npys/test_outputs.npy', 
                    gathered_outputs)
            roc_dict = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
                        'auc': test_auc}
            pd.DataFrame(roc_dict).to_csv(f'{save_dir}/{version_name}/csv_logs/test_roc.csv',
                                          index=False)

            print(f'Test AUC: {test_auc:.4f}')

            plot_outputs(gathered_labels, gathered_outputs, save_dir,
                         version_name, 'test_outs.png')
            plot_roc_curve(fpr, tpr, test_auc, save_dir, version_name,
                           'test_roc_curve.png')

    if profile:
        with open(f'{save_dir}/{version_name}/test_profiler_output.txt', 
                  'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total"))

    cleanup(world_size)
