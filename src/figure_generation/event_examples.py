import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.utilities.figure_generation import Hist2DPlotter, ScatterPlotter
from src.utilities.data_utils import rotate_event

script_path = os.path.abspath(sys.argv[0])
data_dir = f'{"/".join(os.path.dirname(script_path).split("/")[:-1])}/data'
fig_dir = f'{"/".join(os.path.dirname(script_path).split("/")[:-2])}/figures'
xlabel = 'x'
ylabel = 'y'
clabel = 'Density'
cmap = 'Reds'
event_index = 1
uniform_bins = [np.linspace(-1,1,32+1),np.linspace(-1,1,32+1)]
rect_bins = [np.linspace(-1,1,32+1), np.linspace(-1,1,4+1)]
bin_size_uniform = (2/32,2/32)
grid_shape_uniform = (32,32)
bin_size_rect = (2/32,2/4)
grid_shape_rect = (32,4)
xticks = [-1, -0.5, 0, 0.5, 1]
yticks = [-1, -0.5, 0, 0.5, 1]
rotations = [0, 45]
dataset_name = 'full'
versions = ['uniform', 'rect']

for version in versions:
    for rotation in rotations:

        if version == 'uniform':
            bins = uniform_bins
        else:
            bins = rect_bins

        file_prefix = f'{dataset_name}{version}_'
        shape = np.load(
                f'{data_dir}/{file_prefix}x_{rotation}_deg_600000_events.npy').shape
        pix_event= np.empty((shape[1], 3))
        pix_event[:,0] = np.load(
                f'{data_dir}/{file_prefix}x_{rotation}_deg_600000_events.npy')[event_index]
        pix_event[:,1] = np.load(
                f'{data_dir}/{file_prefix}y_{rotation}_deg_600000_events.npy')[event_index]
        pix_event[:,2] = np.load(
                f'{data_dir}/{file_prefix}z_{rotation}_deg_600000_events.npy')[event_index]

        if rotation == 0:
            bbox = 'tight'
            plot_clabel = clabel
            plot_xlabel = xlabel
            plot_ylabel = ylabel
            plot_xticks = xticks
            plot_yticks = yticks
            if version == 'uniform':
                title = 'Uniform Binning'
            elif version == 'rect':
                title = 'Non-Uniform Binning'
        elif rotation == 45:
            bbox = None
            if version == 'uniform':
                title = 'Pre-Detector'
                plot_xticks = []
                plot_yticks = yticks
                plot_xlabel = None
                plot_ylabel = ylabel
                plot_clabel = None
            elif version == 'rect':
                title = None
                plot_xticks = xticks
                plot_yticks = yticks
                plot_xlabel = xlabel
                plot_ylabel = ylabel
                plot_clabel = None

        plotter = Hist2DPlotter(title=title, xlabel=plot_xlabel, ylabel=plot_ylabel,
                                clabel=plot_clabel, xticks=plot_xticks,
                                yticks=plot_yticks)
        if rotation == 45:
            plotter.fig.set_figheight(12)
            plotter.fig.set_figwidth(12)
            plotter.ax.set_title(title, fontsize=2*plotter.title_fontsize)
            plotter.ax.set_xlabel(plot_xlabel, fontsize=2*plotter.label_fontsize)
            plotter.ax.set_ylabel(plot_ylabel, fontsize=2*plotter.label_fontsize)
            plotter.ax.tick_params(labelsize=2*plotter.tick_fontsize)
        plotter.ax.set_aspect('equal', adjustable='box')
        plotter.make_plot([pix_event[:,0].ravel(),pix_event[:,1].ravel()],
                          weights=pix_event[:,2].ravel(), bins=bins, cmap=cmap,
                          density=True)
        plotter.save_plot(f'{fig_dir}/{dataset_name}{version}_{rotation}_example.pdf',
                          bbox_inches=bbox)

for version in versions:
    for rotation in rotations[1:]:

        if version == 'uniform':
            grid_shape = grid_shape_uniform
            bin_size = bin_size_uniform
            bins = uniform_bins
        else:
            grid_shape = grid_shape_uniform
            bin_size = bin_size_uniform
            bins = rect_bins

        file_prefix = f'{dataset_name}{version}_'
        shape = np.load(
                f'{data_dir}/{file_prefix}x_{rotations[0]}_deg_600000_events.npy').shape
        pix_event= np.empty((shape[1], 3))
        pix_event[:,0] = np.load(
                f'{data_dir}/{file_prefix}x_{rotations[0]}_deg_600000_events.npy')[event_index]
        pix_event[:,1] = np.load(
                f'{data_dir}/{file_prefix}y_{rotations[0]}_deg_600000_events.npy')[event_index]
        pix_event[:,2] = np.load(
                f'{data_dir}/{file_prefix}z_{rotations[0]}_deg_600000_events.npy')[event_index]
        pix_event = rotate_event(pix_event, grid_shape, rotation, bin_size=bin_size)

        if rotation == 45:
            bbox = None
            if version == 'uniform':
                title = 'Post-Detector'
                plot_xlabel = None
                plot_ylabel = None
                plot_xticks = []
                plot_yticks = []
                plot_clabel = None
            elif version == 'rect':
                title = None
                plot_xlabel = xlabel
                plot_ylabel = None
                plot_xticks = xticks
                plot_yticks = []
                plot_clabel = None
        else:
            bbox = 'tight'
            plot_xticks = xticks
            plot_yticks = yticks
            plot_clabel = None

        plotter = Hist2DPlotter(title=title, xlabel=plot_xlabel, ylabel=plot_ylabel,
                                clabel=plot_clabel, xticks=plot_xticks,
                                yticks=plot_yticks)
        if rotation == 45:
            plotter.fig.set_figheight(12)
            plotter.fig.set_figwidth(12)
            plotter.ax.set_title(title, fontsize=2*plotter.title_fontsize)
            plotter.ax.set_xlabel(plot_xlabel, fontsize=2*plotter.label_fontsize)
            plotter.ax.set_ylabel(plot_ylabel, fontsize=2*plotter.label_fontsize)
            plotter.ax.tick_params(labelsize=2*plotter.tick_fontsize)

        plotter.ax.set_aspect('equal', adjustable='box')
        plotter.make_plot([pix_event[:,0].ravel(),pix_event[:,1].ravel()],
                          weights=pix_event[:,2].ravel(), bins=bins, cmap=cmap,
                          density=True)
        plotter.save_plot(f'{fig_dir}/{dataset_name}{version}_{rotation}_postdet_example.pdf',
                          bbox_inches=bbox)

shape = np.load(
        f'{data_dir}/{dataset_name}_x_raw_600000_events.npy').shape
raw_event= np.empty((shape[1], 3))
raw_event[:,0] = np.load(
        f'{data_dir}/{dataset_name}_x_raw_600000_events.npy')[event_index]
raw_event[:,1] = np.load(
        f'{data_dir}/{dataset_name}_y_raw_600000_events.npy')[event_index]
raw_event[:,2] = np.load(
        f'{data_dir}/{dataset_name}_z_raw_600000_events.npy')[event_index]

plotter = ScatterPlotter(title='Raw Event', xlabel=xlabel, ylabel=ylabel,
                        xticks=xticks, yticks=yticks)
plotter.ax.set_aspect('equal', adjustable='box')
plotter.make_plot({'raw': [raw_event[:,0].ravel(), raw_event[:,1].ravel()]},
                  weight_dict={'raw': raw_event[:,2].ravel()})
plotter.save_plot(f'{fig_dir}/{dataset_name}_raw_example.pdf')

shape = np.load(
        f'{data_dir}/{dataset_name}_x_spread_600000_events.npy').shape
spread_event= np.empty((shape[1], 3))
spread_event[:,0] = np.load(
        f'{data_dir}/{dataset_name}_x_spread_600000_events.npy')[event_index]
spread_event[:,1] = np.load(
        f'{data_dir}/{dataset_name}_y_spread_600000_events.npy')[event_index]
spread_event[:,2] = np.load(
        f'{data_dir}/{dataset_name}_z_spread_600000_events.npy')[event_index]

plotter = ScatterPlotter(title='Spread Event', xlabel=xlabel, ylabel=ylabel,
                        xticks=xticks, yticks=yticks)
plotter.ax.set_aspect('equal', adjustable='box')
plotter.make_plot({'spread': [spread_event[:,0].ravel(),
                              spread_event[:,1].ravel()]},
                  weight_dict={'spread': spread_event[:,2].ravel()})
plotter.save_plot(f'{fig_dir}/{dataset_name}_spread_example.pdf')

plotter = ScatterPlotter(title=None, xlabel=xlabel, ylabel=ylabel,
                        xticks=xticks, yticks=yticks)
plotter.ax.set_aspect('equal', adjustable='box')
plotter.make_plot({'raw': [raw_event[:,0].ravel(), raw_event[:,1].ravel()],
                   'spread': [spread_event[:,0].ravel(),
                              spread_event[:,1].ravel()]}, weight_dict=
                  {'raw': raw_event[:,2].ravel(), 'spread':
                   spread_event[:,2].ravel()}, alpha=0.5)
                  
plotter.save_plot(f'{fig_dir}/{dataset_name}_raw_spread_example.pdf')
