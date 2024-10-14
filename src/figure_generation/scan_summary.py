import os
import sys
import numpy as np
import pandas as pd
from src.utilities.figure_generation import LinePlotter

train_sizes = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
ylim = (0.506, 0.794)
label_sizes = train_sizes / train_sizes.max()
version = 'extended'
script_path = os.path.abspath(sys.argv[0])
experiments_dir = \
        f'{"/".join(os.path.dirname(script_path).split("/")[:-1])}/experiments'
fig_dir = f'{"/".join(os.path.dirname(script_path).split("/")[:-2])}/figures'
dataset_name = 'full'
network_types = ['fcn', 'pfn']
method_types = ['sym_enc', 'sym_encpix', 'with_aug', 'with_augpix', 'no_aug']
network_labels = ['FCN', 'PFN']
method_labels = ['Pre-Det. Enc. Inv.', 'Post-Det Enc. Inv.', 
                 'Pre-Det. Aug.', 'Post-Det. Aug.', 'No Aug.']
data_versions = ['uniform', 'rect']
rectboot_dir = f'{experiments_dir}/fullrect_bootstrapping/'
unifboot_dir = f'{experiments_dir}/fulluniform_bootstrapping/'
colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'black']
styles = ['solid', 'solid', (0, (2.5,5)), (0, (2.5,5)), 'solid']
method_labels = {t:ml for t,ml in zip(method_types, method_labels)}
color_dict = {method_labels[t]:c for t,c in zip(method_types, colors)}
style_dict = {method_labels[t]:sty for t,sty in zip(method_types, styles)}

for data_version in data_versions:
    bootstrap_dir = f'{experiments_dir}/{dataset_name}{data_version}_bootstrapping'
    for i,network in enumerate(network_types):
        performance_dict = {method_labels[t]: 
                            {str(label): [] for label in label_sizes} 
                            for t in method_types}
        for method in method_types:

            model = f'{network}_{method}'

            version_dir = f'{bootstrap_dir}/{model}/versions'
            versions = os.listdir(version_dir)
            if method == 'sym_enc':
                versions = [v for v in versions 
                            if (v.split('_')[0]==version 
                                and v.split('_')[2]==str(0.001))
                            or (v.split('_')[0]=='cvparams' 
                                and v.split('_')[2]!=str(0.001))]
            elif method == 'sym_encpix':
                versions = [v for v in versions if v.split('_')[0]=='cvparams']
            else:
                versions = [v for v in versions if v.split('_')[0]==version]

            for v in versions:
                try:
                    size = str(float(v.split('_')[2]) / train_sizes.max())
                    test_log = pd.read_csv(
                            f'{bootstrap_dir}/{model}/versions/{v}/csv_logs/test_roc.csv')
                    performance_dict[method_labels[method]][size].append(
                            test_log['auc'].values[0])
                except Exception:
                    pass

        data_dict = {}
        error_dict = {}
        for key in performance_dict:
            auc_means = []
            auc_stds = []
            sizes = []
            for s in label_sizes:
                s = str(s)
                sizes.append(s)
                auc_means.append(np.mean(performance_dict[key][s]))
                num_bootstraps = len(performance_dict[key][s])
                auc_stds.append(np.std(performance_dict[key][s],
                                 ddof=1) / np.sqrt(num_bootstraps))
            data_dict[key] = [sizes, auc_means]
            error_dict[key] = auc_stds

        for key in data_dict.keys():
            print(f'{network} - {key}: {data_dict[key][1]} +- {error_dict[key]}')

        if data_version == 'uniform':
            title = f'{network_labels[i]}, Uniform Pixels'

        if data_version == 'rect':
            title = f'{network_labels[i]}, Non-Uniform Pixels'

        if network_labels[i] == 'FCN':
            legend_col = 2
        else:
            legend_col = 1

        if network_labels[i] == 'PFN' and data_version == 'uniform':
            legend = True
        else:
            legend = False

        plotter = LinePlotter(title=title,
                              legend_col=legend_col, legend=legend,
                              legend_fontsize=15,
                              xlabel='Train Set Size (Proportion)',
                              ylabel='AUC')
        plotter.ax.set_ylim(ylim[0],ylim[1])
        plotter.make_plot(data_dict, error_dict=error_dict,
                          color_dict=color_dict, style_dict=style_dict,
                          legend_error=True)
        plotter.save_plot(f'{fig_dir}/{dataset_name}{data_version}_{network}_train_set_scan.pdf')

