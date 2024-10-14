import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from sklearn.datasets import fetch_openml
from src.utilities.figure_generation import Hist2DPlotter, ImPlotter
from src.utilities.data_utils import rotate_event

script_path = os.path.abspath(sys.argv[0])
data_dir = f'{"/".join(os.path.dirname(script_path).split("/")[:-1])}/data'
fig_dir = f'{"/".join(os.path.dirname(script_path).split("/")[:-2])}/figures'

uniform_bins = [np.linspace(-1,1,32+1),np.linspace(-1,1,32+1)]
bin_size_uniform = (2/32,2/32)
grid_shape_uniform = (32,32)
xticks = [-1, -0.5, 0, 0.5, 1]
yticks = [-1, -0.5, 0, 0.5, 1]
cmap = 'Reds'

# Load CIFAR-10 dataset from OpenML
cifar_10 = fetch_openml('CIFAR_10', version=1)

# The data is flattened, so we need to reshape it to its original shape (32x32x3)
images = cifar_10['data'].values.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
labels = cifar_10['target'].values.astype("int")

# Select a random image from the dataset
cifar_index = 4
class_label = 3
cifar_sample_image = images[labels==class_label][cifar_index]
cifar_sample_label = class_label

event_index = 10
file_prefix = 'fulluniform_'
shape = np.load(
        f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy').shape
pix_event= np.empty((shape[1], 3))
pix_event[:,0] = np.load(
        f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy')[event_index]
pix_event[:,1] = np.load(
        f'{data_dir}/{file_prefix}y_0_deg_600000_events.npy')[event_index]
pix_event[:,2] = np.load(
        f'{data_dir}/{file_prefix}z_0_deg_600000_events.npy')[event_index]
rot_event = rotate_event(pix_event, grid_shape_uniform, 45,
                         bin_size=bin_size_uniform)

def rotate_with_full_border(image, angle):
    # Calculate the length of the diagonal of the original image
    diagonal_length = int(np.ceil(np.sqrt(image.shape[0]**2 + image.shape[1]**2)))
    
    # Calculate padding sizes
    pad_height = diagonal_length - image.shape[0]
    pad_width = diagonal_length - image.shape[1]
    
    # Create a new empty (black) image with padding
    padded_image = np.zeros((diagonal_length, diagonal_length, image.shape[2]), dtype=image.dtype)
    
    # Place the input image at the center of the padded image
    padded_image[pad_height//2 : -pad_height//2, pad_width//2 : -pad_width//2] = image
    
    # Rotate the padded image
    rotated_image = rotate(padded_image, angle, reshape=False, mode='constant', cval=0)
    
    return padded_image, rotated_image

# Rotate the image by 45 degrees
cifar_sample_image, cifar_rotated_image = rotate_with_full_border(cifar_sample_image, 45)

plotter = ImPlotter(title=None, xticks=[], yticks=[])
plotter.ax.set_aspect('equal', adjustable='box')
plotter.make_plot(cifar_sample_image)
plotter.save_plot(f'{fig_dir}/cifar_example.pdf')

plotter = ImPlotter(title=None, xticks=[], yticks=[])
plotter.ax.set_aspect('equal', adjustable='box')
plotter.make_plot(cifar_rotated_image)
plotter.save_plot(f'{fig_dir}/cifar_rotated_example.pdf')

plotter = Hist2DPlotter(title=None, clabel=None, xticks=[], yticks=[])
plotter.ax.set_aspect('equal', adjustable='box')
plotter.make_plot([pix_event[:,0].ravel(),pix_event[:,1].ravel()],
                  weights=pix_event[:,2].ravel(), bins=uniform_bins, cmap=cmap,
                  density=True)
plotter.save_plot(f'{fig_dir}/cifar_toy_example.pdf')

plotter = Hist2DPlotter(title=None, clabel=None, xticks=[], yticks=[])
plotter.ax.set_aspect('equal', adjustable='box')
plotter.make_plot([rot_event[:,0].ravel(),rot_event[:,1].ravel()],
                  weights=rot_event[:,2].ravel(), bins=uniform_bins, cmap=cmap,
                  density=True)
plotter.save_plot(f'{fig_dir}/cifar_toy_rotated_example.pdf')
