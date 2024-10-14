import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
from src.utilities.figure_generation import LinePlotter

fixed_fpr = 0.5
ylim = (0.48,0.86)
num_interp_points = 1000
train_sizes = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
label_sizes = train_sizes / train_sizes.max()
version = 'extended'
script_path = os.path.abspath(sys.argv[0])
experiments_dir = \
        f'{"/".join(os.path.dirname(script_path).split("/")[:-1])}/experiments'
fig_dir = f'{"/".join(os.path.dirname(script_path).split("/")[:-2])}/figures'
network_types = ['fcn', 'pfn']
method_types = ['sym_enc', 'sym_encpix', 'with_aug', 'with_augpix', 'no_aug']
network_labels = ['FCN', 'PFN']
method_labels = ['Pre-Det. Enc. Inv.', 'Post-Det. Enc. Inv.', 
                 'Pre-Det. Aug.', 'Post-Det. Aug.', 'No Aug.']
dataset_name = 'full'
data_versions = ['uniform', 'rect']
rectboot_dir = f'{experiments_dir}/fullrect_bootstrapping/'
unifboot_dir = f'{experiments_dir}/fulluniform_bootstrapping/'
colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue', 'black']
styles = ['solid', 'solid', (0, (2.5,5)), (0, (2.5,5)), 'solid']
labels_to_types = {ml:t for ml,t in zip(method_labels,method_types)}
types_to_labels = {t:ml for t,ml in zip(method_types,method_labels)}
color_labels = [f'{m_label}' for m_label in method_labels]
method_keys = {t:ml for t,ml in zip(method_types, method_labels)}
color_dict = {cl:c for cl,c in zip(color_labels, colors)}
style_dict = {method_keys[t]:sty for t,sty in zip(method_types, styles)}

def init_empty_structure(d):
    if isinstance(d, dict):
        return {k: init_empty_structure(v) for k, v in d.items()}
    else:
        return None

for data_version in data_versions:
    model_dir = f'{experiments_dir}/{dataset_name}{data_version}_bootstrapping'
    for i,network in enumerate(network_types):
        performance_dict = {f'{network}_{m}': {str(label): {'fpr': [], 'tpr': [], 
                                                            'thresh': [], 'auc': []}
                                               for label in label_sizes} 
                            for m in method_types}
        interp_dict = init_empty_structure(performance_dict)
        fixed_dict = {f'{network}_{m}': {str(label): {'avg': None, 'se': None}
                                         for label in label_sizes}
                      for m in method_types}
        for method in method_types:
            model = f'{network}_{method}'
            for t_size,l_size in zip(train_sizes,label_sizes):
                l_size = str(l_size)
                version_dir = f'{model_dir}/{model}/versions'
                versions = os.listdir(version_dir)
                if (method == 'sym_enc') and (t_size != 0.001):
                    versions = [v for v in versions if v.split('_')[0]=='cvparams']
                elif method == 'sym_encpix':
                    versions = [v for v in versions if v.split('_')[0]=='cvparams']
                else:
                    versions = [v for v in versions if v.split('_')[0]==version]
                versions = [v for v in versions if v.split('_')[2]==str(t_size)]

                failed_counter = 0 
                for v in versions:
                    bootstrap_dir = os.path.join(version_dir, v)

                    try:
                        test_outs = np.load(
                                f'{bootstrap_dir}/test_npys/test_outputs.npy')
                        test_labels = np.int32(np.load(
                                f'{bootstrap_dir}/test_npys/test_labels.npy'))
                        assert(len(np.unique(test_labels)) == 2)
                    except Exception:
                        failed_counter += 1
                        continue

                    fpr, tpr, thresh = roc_curve(test_labels, test_outs)
                    roc_auc = auc(fpr,tpr)

                    performance_dict[model][l_size]['fpr'].append(fpr)
                    performance_dict[model][l_size]['tpr'].append(tpr)
                    performance_dict[model][l_size]['thresh'].append(thresh)
                    performance_dict[model][l_size]['auc'].append(roc_auc)

                interp_dict[model][l_size]['fpr'] = np.linspace(0, 1, 
                                                                num_interp_points)
                interp_dict[model][l_size]['tpr'] = np.empty((len(versions) -
                                                              failed_counter,
                                                             num_interp_points))

                for j in range(len(versions)-failed_counter):

                    interp = np.interp(interp_dict[model][l_size]['fpr'],
                                       performance_dict[model][l_size]['fpr'][j],
                                       performance_dict[model][l_size]['tpr'][j])
                    interp_dict[model][l_size]['tpr'][j] = interp

                tprs_at_fixed_fprs = []
                for j in range(len(versions)-failed_counter):

                    idx = np.abs(interp_dict[model][l_size]['fpr']-fixed_fpr).argmin()
                    tprs_at_fixed_fprs.append(interp_dict[model][l_size]['tpr'][j][idx])

                tpr_at_fixed_fpr_avg = np.mean(tprs_at_fixed_fprs)
                tpr_at_fixed_fpr_se = np.std(tprs_at_fixed_fprs,
                                             ddof=1) / np.sqrt(len(versions)-failed_counter)
                fixed_dict[model][l_size]['avg'] = tpr_at_fixed_fpr_avg
                fixed_dict[model][l_size]['se'] = tpr_at_fixed_fpr_se

        data_dict = {}
        error_dict = {}
        for key in method_labels:
            fixed_key = f'{network}_{labels_to_types[key]}'
            data_dict[key] = [np.float32(list(fixed_dict[fixed_key].keys())),
                              [fixed_dict[fixed_key][str(s)]['avg']
                               for s in label_sizes]]
            error_dict[key] = [fixed_dict[fixed_key][str(s)]['se'] for s in
                               label_sizes]

        for key in data_dict.keys():
            print(f'{network} - {key}: {data_dict[key][1]} +- {error_dict[key]}')

        if data_version == 'uniform':
            title = f'{network_labels[i]}, Uniform Pixels'

        if data_version == 'rect':
            title = f'{network_labels[i]}, Non-Uniform Pixels'

        if network_labels[i] == 'PFN' and data_version == 'uniform':
            legend = True
        else:
            legend = False

        plotter = LinePlotter(title=title,
                              legend_col=1, legend=legend,
                              legend_fontsize=15,
                              xlabel='Train Set Size (Proportion)',
                              ylabel='Signal Efficiency')
        plotter.ax.set_ylim(ylim[0],ylim[1])
        plotter.make_plot(data_dict, error_dict=error_dict,
                          color_dict=color_dict, style_dict=style_dict,
                          legend_error=True)
        plotter.save_plot(f'{fig_dir}/{dataset_name}{data_version}_{network}_sig_eff_at_fixed.pdf')

