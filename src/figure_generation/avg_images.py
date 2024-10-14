import os
import sys
import numpy as np
from src.utilities.figure_generation import Hist2DPlotter

script_path = os.path.abspath(sys.argv[0])
data_dir = f'{"/".join(os.path.dirname(script_path).split("/")[:-1])}/data'
fig_dir = f'{"/".join(os.path.dirname(script_path).split("/")[:-2])}/figures'
xlabel = 'x'
ylabel = 'y'
clabel = 'Density'
cmap = 'Reds'
uniform_bins = [np.linspace(-1,1,32+1),np.linspace(-1,1,32+1)]
rect_bins = [np.linspace(-1,1,32+1), np.linspace(-1,1,4+1)]
xticks = [-1, -0.5, 0, 0.5, 1]
yticks = [-1, -0.5, 0, 0.5, 1]
dataset_name = 'full'
versions = ['uniform', 'rect']

for version in versions:

    if version == 'uniform':
        bins = uniform_bins
    else:
        bins = rect_bins

    file_prefix = f'{dataset_name}_'
    shape = np.load(f'{data_dir}/{file_prefix}x_raw_600000_events.npy').shape
    data = np.empty((shape[0], shape[1], 3))
    data[:,:,0] = np.load(f'{data_dir}/{file_prefix}x_raw_600000_events.npy')
    data[:,:,1] = np.load(f'{data_dir}/{file_prefix}y_raw_600000_events.npy')
    data[:,:,2] = np.load(f'{data_dir}/{file_prefix}z_raw_600000_events.npy')

    labels = np.load(f'{data_dir}/{file_prefix}labels_600000_events.npy')

    sig_data = data[labels==1]
    bg_data = data[labels==0]

    plotter = Hist2DPlotter(title='Signal', xlabel=xlabel, ylabel=ylabel,
                            clabel=clabel, xticks=xticks, yticks=yticks)
    plotter.make_plot([sig_data[:,:,0].ravel(),sig_data[:,:,1].ravel()],
                      weights=sig_data[:,:,2].ravel(), bins=bins, cmap=cmap,
                      density=True)
    plotter.save_plot(f'{fig_dir}/{dataset_name}{version}_sig_avg.pdf')

    plotter = Hist2DPlotter(title='Background', xlabel=xlabel, ylabel=ylabel,
                            clabel=clabel, xticks=xticks, yticks=yticks)
    plotter.make_plot([bg_data[:,:,0].ravel(),bg_data[:,:,1].ravel()],
                      weights=bg_data[:,:,2].ravel(), bins=bins, cmap=cmap,
                      density=True)
    plotter.save_plot(f'{fig_dir}/{dataset_name}{version}_bg_avg.pdf')

