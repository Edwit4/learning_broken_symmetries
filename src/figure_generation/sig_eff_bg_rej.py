import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
from src.utilities.figure_generation import LinePlotter

num_bootstraps = 100
num_interp_points = 1000
train_sizes = np.array([0.001, 0.005])
label_sizes = train_sizes / train_sizes.max()
version = 'extended'
script_path = os.path.abspath(sys.argv[0])
experiments_dir = \
        f'{"/".join(os.path.dirname(script_path).split("/")[:-1])}/experiments'
fig_dir = f'{"/".join(os.path.dirname(script_path).split("/")[:-2])}/figures'
network_types = ['fcn', 'pfn']
method_types = ['sym_enc', 'sym_encpix', 'with_aug', 'with_augpix', 'no_aug']
network_labels = ['FCN', 'PFN']
method_labels = ['Pre Det. Encouraged Inv.', 'Post Det Encouraged Inv.', 
                 'Pre Det. Aug.', 'Post Det. Aug.', 'No Aug.']
dataset_name = 'full'
data_versions = ['uniform', 'rect']
rectboot_dir = f'{experiments_dir}/fullrect_bootstrapping/'
unifboot_dir = f'{experiments_dir}/fulluniform_bootstrapping/'
styles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
color_labels = [f'{m_label}' for m_label in method_labels]
color_dict = {cl:c for cl,c in zip(color_labels, colors)}
style_dict = {mlabel:sty for mlabel,sty in zip(method_labels, styles)}

def init_empty_structure(d):
    if isinstance(d, dict):
        return {k: init_empty_structure(v) for k, v in d.items()}
    else:
        return None

for data_version in data_versions:
    model_dir = f'{experiments_dir}/{dataset_name}{data_version}_bootstrapping'
    for i,network in enumerate(network_types):
        performance_dict = {f'{network}_{m}': {'fpr': [], 'tpr': [], 'thresh':
                                               [], 'auc': []}
                            for m in method_types}
        interp_dict = init_empty_structure(performance_dict)
        for t_size,l_size in zip(train_sizes,label_sizes):
            l_size = str(l_size)
            for method in method_types:
                model = f'{network}_{method}'
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

                    performance_dict[model]['fpr'].append(fpr)
                    performance_dict[model]['tpr'].append(tpr)
                    performance_dict[model]['thresh'].append(thresh)
                    performance_dict[model]['auc'].append(roc_auc)

                interp_dict[model]['fpr'] = np.linspace(0, 1,
                                                        num_interp_points)
                interp_dict[model]['tpr'] = np.empty((len(versions) -
                                                      failed_counter,
                                                      num_interp_points))

                for j in range(len(versions)-failed_counter):

                    interp = np.interp(interp_dict[model]['fpr'],
                                       performance_dict[model]['fpr'][j],
                                       performance_dict[model]['tpr'][j])
                    interp_dict[model]['tpr'][j] = interp

            data_dict = {}
            for m_label,p_key in zip(method_labels, interp_dict.keys()):
                x = np.mean(interp_dict[p_key]['tpr'], axis=0)
                y = 1/interp_dict[p_key]['fpr']
                data_dict[f'{m_label}'] = [x,y] 

            plotter = LinePlotter(xlabel='Signal Efficiency',
                                  ylabel='1 / Background Rejection',
                                  title=f'{network_labels[i]}, Training Size = {l_size}')
            plotter.ax.set_yscale('log')
            plotter.make_plot(data_dict, color_dict=color_dict, style_dict=style_dict)
            plotter.save_plot(f'{fig_dir}/{dataset_name}{data_version}_{network}_{l_size}_sig_eff_bg_rej.pdf')

