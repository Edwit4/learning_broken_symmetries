import os
import sys
import numpy as np
import pandas as pd

num_folds = 5
version = 'modelselect'
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
version_dir = os.path.join(script_dir, "versions")

versions = os.listdir(version_dir)
versions = [v for v in versions if version in v]
versions = np.array([f'{version_dir}/{v}'.rsplit('_',1)[0] for v in versions])
unique_versions = np.unique(versions)

mean_aucs = []
std_aucs = []
mean_epochs = []
for i,v in enumerate(unique_versions):
    print(f'Version: {v.rsplit("_",1)[0]}')
    fold_aucs = []
    fold_epochs = []
    for j in range(num_folds):
        test_log = pd.read_csv(f'{v}_{j}/csv_logs/test_roc.csv')
        fold_aucs.append(test_log['auc'].values[0])
        fold_epochs.append(int(os.listdir(
            f'{v}_{j}/checkpoints')[0].split('_')[1].split('.')[0]))
    mean_aucs.append(np.mean(fold_aucs))
    std_aucs.append(np.std(fold_aucs))
    mean_epochs.append(np.mean(fold_epochs))
    print(mean_aucs[-1], std_aucs[-1], mean_epochs[-1])

mean_aucs = np.array(mean_aucs)
std_aucs = np.array(std_aucs)
mean_epochs = np.array(mean_epochs)

std_bound = np.mean(std_aucs)
bounded_idx = (std_aucs <= std_bound) 

range_aucs = mean_aucs[bounded_idx]
range_stds = std_aucs[bounded_idx]
range_epochs = mean_epochs[bounded_idx]
range_versions = unique_versions[bounded_idx]

optimal_idx = np.argmax(range_aucs)

print('')
print(f'Optimal model: {range_versions[optimal_idx]}, AUC: {range_aucs[optimal_idx]}, std. dev. {range_stds[optimal_idx]}, Epochs: {range_epochs[optimal_idx]}')

summary_dir = f'{version_dir.rsplit("/",1)[0]}/optimal_model.txt'
with open(summary_dir, 'w') as f:
    f.write(f'Optimal model: {range_versions[optimal_idx].split("/")[-1]}\n')
    f.write(f'AUC: {range_aucs[optimal_idx]}\n')
    f.write(f'Std. Dev.: {range_stds[optimal_idx]}\n')
    f.write(f'Epochs: {range_epochs[optimal_idx]}\n')
