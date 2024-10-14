import cv2
import numpy as np
from sklearn.model_selection import KFold

def split_data(indices, split_size=0.2, shuffle=False, rng=None):

    if rng is None:
        rng = np.random.default_rng(seed=123)

    if shuffle is True:
        rng.shuffle(indices)

    split_idx = rng.choice(indices, size=int(split_size*len(indices)), replace=False)
    remainder_idx = np.delete(indices, split_idx)

    return remainder_idx, split_idx

def kfold_split(indices, fold_idx=0, n_splits=5, shuffle=False, rng=None):

    if rng is None:
        rng = np.random.default_rng(seed=123)

    if shuffle is True:
        rng.shuffle(indices)

    kf = KFold(n_splits=n_splits, shuffle=False)
    split = list(kf.split(indices))
    train_split, valid_split = split[fold_idx]

    train_idx = indices[train_split] 
    valid_idx = indices[valid_split]

    return train_idx, valid_idx

def bootstrap(indices, bootstrap_idx=0, shuffle=False, bootstrap_seed=123):

    bootstrap_rng = np.random.default_rng(seed=bootstrap_seed+bootstrap_idx)
    if shuffle is True:
        bootstrap_rng.shuffle(indices)
    bootstrap_idx = bootstrap_rng.choice(indices, size=len(indices), replace=True, shuffle=False)

    return bootstrap_idx

def multiple_histogram2d(x, y, z, num_bins):
    # Ensure x, y, and z have the same shape
    assert x.shape == y.shape == z.shape, "x, y, and z must have the same shape"

    # Compute bin edges
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_edges = np.linspace(x_min, x_max, num_bins + 1)
    y_edges = np.linspace(y_min, y_max, num_bins + 1)

    # Determine bin indices for x and y
    x_indices = np.searchsorted(x_edges, x, side='right') - 1
    y_indices = np.searchsorted(y_edges, y, side='right') - 1

    # Clip bin indices to be within the range [0, num_bins-1]
    x_indices = np.clip(x_indices, 0, num_bins - 1)
    y_indices = np.clip(y_indices, 0, num_bins - 1)

    # Create an array for the output histograms
    histograms = np.zeros((x.shape[0], num_bins, num_bins))

    # Use advanced indexing and broadcasting to compute the 2D histograms
    np.add.at(histograms, (np.arange(x.shape[0])[:, None], x_indices, y_indices), z)

    return histograms, x_edges, y_edges

def image_to_event(img, bin_size):
    coords = np.column_stack(np.where(img > 0))
    
    # Convert back to the range [-1, 1]
    x_coords = (coords[:, 0] + 0.5) * bin_size[0] - 1
    y_coords = (coords[:, 1] + 0.5) * bin_size[1] - 1
    
    z_values = img[img > 0]
    
    return np.column_stack((x_coords, y_coords, z_values))

def event_to_image(event, grid_shape):
    img = np.zeros(grid_shape)
    x_bins = np.linspace(-1, 1, grid_shape[0] + 1)
    y_bins = np.linspace(-1, 1, grid_shape[1] + 1)
    
    x_indices = np.digitize(event[:, 0], x_bins) - 1
    y_indices = np.digitize(event[:, 1], y_bins) - 1

    img[x_indices, y_indices] = event[:, 2]
    
    return img

def rotate_event(event, grid_shape, angle, bin_size=(1, 1)):
    # Convert event data to image
    img = event_to_image(event, grid_shape)
    
    # Scale to square
    target_size = max(grid_shape)
    resized_img = cv2.resize(img, (target_size, target_size))
    
    # Get rotation matrix
    center = (target_size / 2, target_size / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # Apply rotation
    rotated_img_resized = cv2.warpAffine(resized_img, rotation_matrix, (target_size, target_size))

    # Scale back to original shape
    rotated_img = cv2.resize(rotated_img_resized, grid_shape[::-1])

    # Convert rotated image back to event data
    rotated_event = image_to_event(rotated_img, bin_size)

    return rotated_event
