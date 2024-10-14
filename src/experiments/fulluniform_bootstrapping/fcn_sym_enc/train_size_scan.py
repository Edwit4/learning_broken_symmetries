import os
import sys
import time
import torch
import numpy as np
import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor

num_parallel = 5
num_bootstraps = 100
start_port = 12355
version_name = 'cvparams'

script_path = os.path.abspath(sys.argv[0])
experiment_dir = os.path.dirname(script_path)
cv_dir = experiment_dir.split('/')
cv_dirname = f'{cv_dir[-2].split("_")[0]}_cross_validation'
cv_dir[-2] = cv_dirname
cv_dir = '/'.join(cv_dir)

train_sizes = [0.001, 0.002, 0.003, 0.004, 0.005]

with open(f'{cv_dir}/optimal_model.txt', 'r') as f:
    for line in f:
        if 'Optimal' in line:
            print(line)
            layer_size = int(line.split('_')[4])
            learning_rate = float(line.split('_')[6])
            batch_size = int(line.split('_')[8])
            crit_scale = float(line.split('_')[10])
            crit_2_scale = float(line.split('_')[12])
        if 'Epochs' in line:
            num_epochs = int(np.ceil(float(line.split(' ')[-1])))

parameterizations = list(itertools.product(np.arange(num_bootstraps), train_sizes))
parameterizations = [[*p, start_port+i] for p,i in 
                     zip(parameterizations, np.arange(len(parameterizations)))]
cmds = [['python', f'{experiment_dir}/train_test.py', str(bs), str(t_size), 
    str(layer_size), str(learning_rate), str(batch_size), str(crit_scale),
    str(crit_2_scale), version_name, str(port), str(num_epochs)] 
    for bs,t_size,port in parameterizations]

dir_names = [f'{experiment_dir}/versions/{version_name}_trainsize_{t_size}_layersize_{layer_size}_learningrate_{learning_rate}_batchsize_{batch_size}_bootstrap_{bs}'
             for bs, t_size, port in parameterizations]

def worker(cmd, name):
    if not os.path.exists(name):
        print(f'Starting {cmd}')
        subprocess.call(cmd)
    else:
        print(f'Skipping {cmd}')

with ProcessPoolExecutor(max_workers=num_parallel) as executor:
    for name, cmd in zip(dir_names, cmds):
        time.sleep(2)
        executor.submit(worker, cmd, name)
