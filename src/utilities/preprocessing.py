import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utilities.data_utils import multiple_histogram2d

def make_toy_fcn_standardscaler(data_dir, rotation=0, indices=None,
                                pixelization='square'):

    if pixelization == 'square':
        file_prefix = '' 
    else:
        file_prefix = f'{pixelization}_'

    if indices is None:
        indices = slice(None)

    shape = np.load(f'{data_dir}/{file_prefix}x_{rotation}_deg_600000_events.npy',
            mmap_mode='r')[indices].shape

    data = np.empty((shape[0], shape[1], 3))
    data[:,:,0] = np.load(
            f'{data_dir}/{file_prefix}x_{rotation}_deg_600000_events.npy',
            mmap_mode='r')[indices]
    data[:,:,1] = np.load(
            f'{data_dir}/{file_prefix}y_{rotation}_deg_600000_events.npy',
            mmap_mode='r')[indices]
    data[:,:,2] = np.load(
            f'{data_dir}/{file_prefix}z_{rotation}_deg_600000_events.npy',
            mmap_mode='r')[indices]
    data = data.reshape([data.shape[0], data.shape[1]*data.shape[2]])

    scaler = StandardScaler()
    scaler.fit(data)

    return scaler

def make_toy_fcn_aug_standardscaler(data_dir, indices=None, pixelization='square'):

    if pixelization == 'square':
        file_prefix = '' 
    else:
        file_prefix = f'{pixelization}_'

    if indices is None:
        indices = slice(None)

    shape = np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                    mmap_mode='r')[indices].shape

    rotations = [0, 45, 90, 135, 180, 225, 270, 315]
    num_events = 600000
    data = np.empty((shape[0],len(rotations),shape[1],3))
    for i,r in enumerate(rotations):
        data[:,i,:,0] = np.load(f'{data_dir}/{file_prefix}x_{r}_deg_{num_events}_events.npy',
                                mmap_mode='r')[indices]
        data[:,i,:,1] = np.load(f'{data_dir}/{file_prefix}y_{r}_deg_{num_events}_events.npy',
                                mmap_mode='r')[indices]
        data[:,i,:,2] = np.load(f'{data_dir}/{file_prefix}z_{r}_deg_{num_events}_events.npy',
                                mmap_mode='r')[indices]

    scaler = StandardScaler()

    scaler.fit(data[:,0,:,:].reshape(data.shape[0],data.shape[2]*data.shape[3]))

    return scaler

def make_toy_pfn_standardscaler(data_dir, rotation=0, indices=None,
                                pixelization='square'):

    if pixelization == 'square':
        file_prefix = '' 
    else:
        file_prefix = f'{pixelization}_'

    if indices is None:
        indices = slice(None)

    shape = np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                    mmap_mode='r')[indices].shape
    data = np.zeros((shape[0],shape[1],3))

    data[:,:,0] = np.load(
            f'{data_dir}/{file_prefix}x_{rotation}_deg_600000_events.npy',
            mmap_mode='r')[indices]
    data[:,:,1] = np.load(
            f'{data_dir}/{file_prefix}y_{rotation}_deg_600000_events.npy',
            mmap_mode='r')[indices]
    data[:,:,2] = np.load(
            f'{data_dir}/{file_prefix}z_{rotation}_deg_600000_events.npy',
            mmap_mode='r')[indices]

    scaler = StandardScaler()
    scaler.fit(data[:,:,2].flatten().reshape(-1,1))

    return scaler

def make_toy_pfn_aug_standardscaler(data_dir, indices=None, pixelization='square'):

    if pixelization == 'square':
        file_prefix = '' 
    else:
        file_prefix = f'{pixelization}_'

    if indices is None:
        indices = slice(None)

    rotations = [0, 45, 90, 135, 180, 225, 270, 315]
    shape = np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                    mmap_mode='r')[indices].shape
    num_events = 600000
    data = np.empty((shape[0],len(rotations),shape[1],3))
    for i,r in enumerate(rotations):
        data[:,i,:,0] = np.load(f'{data_dir}/{file_prefix}x_{r}_deg_{num_events}_events.npy',
                                mmap_mode='r')[indices]
        data[:,i,:,1] = np.load(f'{data_dir}/{file_prefix}y_{r}_deg_{num_events}_events.npy',
                                mmap_mode='r')[indices]
        data[:,i,:,2] = np.load(f'{data_dir}/{file_prefix}z_{r}_deg_{num_events}_events.npy',
                                mmap_mode='r')[indices]

    if indices is None:
        indices = np.arange(len(data))

    data = data[indices]
    shape = data.shape

    scaler = StandardScaler()

    scaler.fit(data[:,0,:,2].flatten().reshape(-1,1))

    return scaler

def make_toy_cnn_standardscaler(data_dir, rotation=0, indices=None, pixelization='square'):

    if pixelization == 'square':
        file_prefix = '' 
    else:
        file_prefix = f'{pixelization}_'

    if indices is None:
        indices = slice(None)

    x_coord = np.load(f'{data_dir}/{file_prefix}x_{rotation}_deg_600000_events.npy',
                      mmap_mode='r')[indices]
    y_coord = np.load(f'{data_dir}/{file_prefix}y_{rotation}_deg_600000_events.npy',
                      mmap_mode='r')[indices]
    z_coord = np.load(f'{data_dir}/{file_prefix}z_{rotation}_deg_600000_events.npy',
                      mmap_mode='r')[indices]

    data, x_edges, y_edges = multiple_histogram2d(x_coord, y_coord, z_coord, 32)
    shape = data.shape

    scaler = StandardScaler()

    scaler.fit(data.reshape(shape[0],shape[1]*shape[2]))

    return scaler

def make_toy_cnn_aug_standardscaler(data_dir, indices=None, pixelization='square'):

    if pixelization == 'square':
        file_prefix = '' 
    else:
        file_prefix = f'{pixelization}_'

    if indices is None:
        indices = slice(None)

    rotations = [0, 45, 90, 135, 180, 225, 270, 315]
    shape = np.array(np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                             mmap_mode='r')[indices].shape)
    shape[0] = len(indices)
    num_events = 600000
    data = np.empty((shape[0],len(rotations),shape[1],3))
    images = np.empty((shape[0],len(rotations),32,32))
    for i,r in enumerate(rotations):
        data[:,i,:,0] = np.load(f'{data_dir}/{file_prefix}x_{r}_deg_{num_events}_events.npy',
                                mmap_mode='r')[indices]
        data[:,i,:,1] = np.load(f'{data_dir}/{file_prefix}y_{r}_deg_{num_events}_events.npy',
                                mmap_mode='r')[indices]
        data[:,i,:,2] = np.load(f'{data_dir}/{file_prefix}z_{r}_deg_{num_events}_events.npy',
                                mmap_mode='r')[indices]

    data_shape = data.shape
    image_shape = images.shape
    data = data.reshape((data_shape[0]*data_shape[1],data_shape[2],data_shape[3]))
    images = images.reshape((images.shape[0]*images.shape[1],
                                images.shape[2],images.shape[3]))
    images, x_edges, y_edges = multiple_histogram2d(data[:,:,0], data[:,:,1],
                                                        data[:,:,2], 32)
    images = images.reshape(image_shape)

    scaler = StandardScaler()

    images = scaler.fit(images[:,0,:,:].reshape(
        image_shape[0], image_shape[2]*image_shape[3]))

    return scaler
