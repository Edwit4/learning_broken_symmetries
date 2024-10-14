import os
import sys
import time
import torch
import numpy as np
import os
import itertools
import subprocess
from concurrent.futures import ProcessPoolExecutor

seed = 123
train_size = 0.01
num_rand_search = 60
num_parallel = 5
num_folds = 5
start_port = 12355
version_name = 'modelselect'
script_path = os.path.abspath(sys.argv[0])
experiment_dir = os.path.dirname(script_path)

learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]
batch_sizes = [32, 64, 128, 256, 512, 1024]
layer_sizes = [32, 64, 128, 256, 512]
crit_scales = [0.01, 1., 10, 100, 300]
crit_2_scales = [0.01, 1., 10, 100, 300]

rng = np.random.default_rng(seed=seed)
parameterizations = np.array(list(itertools.product([train_size], layer_sizes,
                                                    learning_rates, batch_sizes, crit_scales,
                                                    crit_2_scales, [version_name])))
param_idx = np.arange(len(parameterizations))
param_idx = rng.choice(param_idx, replace=False, size=num_rand_search)
parameterizations = [[i, *p] for p in parameterizations[param_idx] for i in range(num_folds)]
parameterizations = [[*p, start_port+i] for p,i in 
                     zip(parameterizations, np.arange(len(parameterizations)))]

cmds = [['python', f'{experiment_dir}/train_test.py', str(kf), str(t_size), 
    str(l_size), str(lr), str(b_size), str(c1), str(c2), v, str(p)] 
    for kf,t_size,l_size,lr,b_size,c1,c2,v,p in parameterizations]

dir_names = [f'{experiment_dir}/versions/{v}_trainsize_{t_size}_layersize_{l_size}_learningrate_{lr}_batchsize_{b_size}_crit1_{c1}_crit2_{c2}_fold_{kf}'
             for kf,t_size,l_size,lr,b_size,c1,c2,v,p in parameterizations]

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