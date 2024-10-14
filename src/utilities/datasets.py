import torch 
import numpy as np
from torch.utils.data import Dataset
from src.utilities.data_utils import multiple_histogram2d, rotate_event

class toy_fcn_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None,
                 rotation=0, scaler=None):

        if pixelization == 'square':
            file_prefix = '' 
        else:
            file_prefix = f'{pixelization}_'

        if indices is None:
            indices = slice(None)

        shape = np.load(
                f'{data_dir}/{file_prefix}x_{rotation}_deg_600000_events.npy',
                mmap_mode='r')[indices].shape
        self.data = np.empty((shape[0], shape[1], 3))
        self.data[:,:,0] = np.load(
                f'{data_dir}/{file_prefix}x_{rotation}_deg_600000_events.npy',
                mmap_mode='r')[indices]
        self.data[:,:,1] = np.load(
                f'{data_dir}/{file_prefix}y_{rotation}_deg_600000_events.npy',
                mmap_mode='r')[indices]
        self.data[:,:,2] = np.load(
                f'{data_dir}/{file_prefix}z_{rotation}_deg_600000_events.npy',
                mmap_mode='r')[indices]
        self.data = self.data.reshape([self.data.shape[0], 
                                       self.data.shape[1]*self.data.shape[2]])

        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)

        if scaler is not None:
            self.data = scaler.transform(self.data)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})

        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_fcn_aug_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

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
        self.data = np.empty((shape[0],len(rotations),shape[1],3))
        for i,r in enumerate(rotations):
            self.data[:,i,:,0] = np.load(
                    f'{data_dir}/{file_prefix}x_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]
            self.data[:,i,:,1] = np.load(
                    f'{data_dir}/{file_prefix}y_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]
            self.data[:,i,:,2] = np.load(
                    f'{data_dir}/{file_prefix}z_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]

        self.data = self.data.reshape((self.data.shape[0]*self.data.shape[1],
                                      self.data.shape[2]*self.data.shape[3]))

        if scaler is not None:
            self.data = scaler.transform(self.data)

        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        self.labels = np.repeat(self.labels, len(rotations), axis=0)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_fcn_aug_pix_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

        if pixelization == 'square':
            file_prefix = '' 
            bin_size = (2/32,2/32)
            grid_shape = (32,32)
        else:
            file_prefix = f'{pixelization}_'

        if 'uniform' in pixelization:
            bin_size = (2/32,2/32)
            grid_shape = (32,32)
        elif 'rect' in pixelization:
            bin_size = (2/32,2/4)
            grid_shape = (32,4)

        if indices is None:
            indices = slice(None)

        rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        shape = np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                        mmap_mode='r')[indices].shape
        num_events = 600000
        self.data = np.zeros((shape[0],len(rotations),shape[1],3))
        self.data[:,0,:,0] = np.load(
                f'{data_dir}/{file_prefix}x_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]
        self.data[:,0,:,1] = np.load(
                f'{data_dir}/{file_prefix}y_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]
        self.data[:,0,:,2] = np.load(
                f'{data_dir}/{file_prefix}z_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]

        for i,r in enumerate(rotations):
            if i == 0: 
                continue
            for j in range(len(self.data)):
                rotated_event = rotate_event(self.data[j,0,:,:], grid_shape, r,
                                                  bin_size=bin_size)
                rotated_event = rotated_event[:self.data.shape[2]]
                self.data[j,i,:rotated_event.shape[0],:] = rotated_event

        self.data = self.data.reshape((self.data.shape[0]*self.data.shape[1],
                                      self.data.shape[2]*self.data.shape[3]))

        if scaler is not None:
            self.data = scaler.transform(self.data)

        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        self.labels = np.repeat(self.labels, len(rotations), axis=0)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_fcn_aug_set_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

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
        self.data = np.empty((shape[0],len(rotations),shape[1],3))
        for i,r in enumerate(rotations):
            self.data[:,i,:,0] = np.load(
                    f'{data_dir}/{file_prefix}x_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]
            self.data[:,i,:,1] = np.load(
                    f'{data_dir}/{file_prefix}y_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]
            self.data[:,i,:,2] = np.load(
                    f'{data_dir}/{file_prefix}z_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]

        shape = self.data.shape
        self.data = self.data.reshape((shape[0]*shape[1],shape[2]*shape[3]))

        if scaler is not None:
            self.data = scaler.transform(self.data)

        self.data = self.data.reshape((shape[0],shape[1],shape[2]*shape[3]))

        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        self.labels = np.repeat(self.labels, len(rotations), axis=0)
        self.labels = self.labels.reshape(self.labels.shape[0]//len(rotations),
                                          len(rotations))

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_fcn_aug_set_pix_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

        if pixelization == 'square':
            file_prefix = '' 
            bin_size = (2/32,2/32)
            grid_shape = (32,32)
        else:
            file_prefix = f'{pixelization}_'

        if 'uniform' in pixelization:
            bin_size = (2/32,2/32)
            grid_shape = (32,32)
        elif 'rect' in pixelization:
            bin_size = (2/32,2/4)
            grid_shape = (32,4)

        if indices is None:
            indices = slice(None)

        rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        num_events = 600000
        shape = np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                        mmap_mode='r')[indices].shape
        self.data = np.zeros((shape[0],len(rotations),shape[1],3))
        self.data[:,0,:,0] = np.load(
                f'{data_dir}/{file_prefix}x_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]
        self.data[:,0,:,1] = np.load(
                f'{data_dir}/{file_prefix}y_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]
        self.data[:,0,:,2] = np.load(
                f'{data_dir}/{file_prefix}z_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]

        for i,r in enumerate(rotations):
            if i == 0: 
                continue
            for j in range(len(self.data)):
                rotated_event = rotate_event(self.data[j,0,:,:], grid_shape, r,
                                                  bin_size=bin_size)
                rotated_event = rotated_event[:self.data.shape[2]]
                self.data[j,i,:rotated_event.shape[0],:] = rotated_event

        shape = self.data.shape
        self.data = self.data.reshape((shape[0]*shape[1],shape[2]*shape[3]))

        if scaler is not None:
            self.data = scaler.transform(self.data)

        self.data = self.data.reshape((shape[0],shape[1],shape[2]*shape[3]))

        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        self.labels = np.repeat(self.labels, len(rotations), axis=0)
        self.labels = self.labels.reshape(self.labels.shape[0]//len(rotations),
                                          len(rotations))

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_pfn_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None,
                 rotation=0, scaler=None):

        if pixelization == 'square':
            file_prefix = '' 
        else:
            file_prefix = f'{pixelization}_'

        if indices is None:
            indices = slice(None)

        shape = np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                        mmap_mode='r')[indices].shape
        self.data = np.zeros((shape[0],shape[1],3))

        self.data[:,:,0] = np.load(
                f'{data_dir}/{file_prefix}x_{rotation}_deg_600000_events.npy',
                mmap_mode='r')[indices]
        self.data[:,:,1] = np.load(
                f'{data_dir}/{file_prefix}y_{rotation}_deg_600000_events.npy',
                mmap_mode='r')[indices]
        self.data[:,:,2] = np.load(
                f'{data_dir}/{file_prefix}z_{rotation}_deg_600000_events.npy',
                mmap_mode='r')[indices]

        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)

        if scaler is not None:
            energy_shape = self.data[:,:,2].shape
            self.data[:,:,2] = scaler.transform(
                self.data[:,:,2].flatten().reshape(-1,1)).reshape(energy_shape)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_pfn_aug_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

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
        self.data = np.empty((shape[0],len(rotations),shape[1],3))
        for i,r in enumerate(rotations):
            self.data[:,i,:,0] = np.load(
                    f'{data_dir}/{file_prefix}x_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]
            self.data[:,i,:,1] = np.load(
                    f'{data_dir}/{file_prefix}y_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]
            self.data[:,i,:,2] = np.load(
                    f'{data_dir}/{file_prefix}z_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]

        energy_shape = self.data[:,:,:,2].shape
        if scaler is not None:
            self.data[:,:,:,2] = scaler.transform(
                    self.data[:,:,:,2].flatten().reshape(-1,1)).reshape(energy_shape)
        self.data = self.data.reshape((self.data.shape[0]*self.data.shape[1],
                                      self.data.shape[2],self.data.shape[3]))
        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        self.labels = np.repeat(self.labels, len(rotations), axis=0)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_pfn_aug_pix_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

        if pixelization == 'square':
            file_prefix = '' 
            bin_size = (2/32,2/32)
            grid_shape = (32,32)
        else:
            file_prefix = f'{pixelization}_'

        if 'uniform' in pixelization:
            bin_size = (2/32,2/32)
            grid_shape = (32,32)
        elif 'rect' in pixelization:
            bin_size = (2/32,2/4)
            grid_shape = (32,4)

        if indices is None:
            indices = slice(None)

        rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        shape = np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                        mmap_mode='r')[indices].shape
        num_events = 600000
        self.data = np.zeros((shape[0],len(rotations),shape[1],3))
        self.data[:,0,:,0] = np.load(
                f'{data_dir}/{file_prefix}x_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]
        self.data[:,0,:,1] = np.load(
                f'{data_dir}/{file_prefix}y_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]
        self.data[:,0,:,2] = np.load(
                f'{data_dir}/{file_prefix}z_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]

        for i,r in enumerate(rotations):
            if i == 0: 
                continue
            for j in range(len(self.data)):
                rotated_event = rotate_event(self.data[j,0,:,:], grid_shape, r,
                                             bin_size=bin_size)
                rotated_event = rotated_event[:self.data.shape[2]]
                self.data[j,i,:rotated_event.shape[0],:] = rotated_event

        energy_shape = self.data[:,:,:,2].shape
        if scaler is not None:
            self.data[:,:,:,2] = scaler.transform(
                    self.data[:,:,:,2].flatten().reshape(-1,1)).reshape(energy_shape)
        self.data = self.data.reshape((self.data.shape[0]*self.data.shape[1],
                                      self.data.shape[2],self.data.shape[3]))
        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        self.labels = np.repeat(self.labels, len(rotations), axis=0)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_pfn_aug_set_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

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
        self.data = np.empty((shape[0],len(rotations),shape[1],3))
        for i,r in enumerate(rotations):
            self.data[:,i,:,0] = np.load(
                    f'{data_dir}/{file_prefix}x_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]
            self.data[:,i,:,1] = np.load(
                    f'{data_dir}/{file_prefix}y_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]
            self.data[:,i,:,2] = np.load(
                    f'{data_dir}/{file_prefix}z_{r}_deg_{num_events}_events.npy',
                    mmap_mode='r')[indices]

        if scaler is not None:
            energy_shape = self.data[:,:,:,2].shape
            self.data[:,:,:,2] = scaler.transform(
                    self.data[:,:,:,2].flatten().reshape(-1,1)).reshape(energy_shape)

        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        self.labels = np.repeat(self.labels, len(rotations), axis=0)
        self.labels = self.labels.reshape(self.labels.shape[0]//len(rotations),
                                          len(rotations))

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_pfn_aug_set_pix_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

        if pixelization == 'square':
            file_prefix = '' 
            bin_size = (2/32,2/32)
            grid_shape = (32,32)
        else:
            file_prefix = f'{pixelization}_'

        if 'uniform' in pixelization:
            bin_size = (2/32,2/32)
            grid_shape = (32,32)
        elif 'rect' in pixelization:
            bin_size = (2/32,2/4)
            grid_shape = (32,4)

        if indices is None:
            indices = slice(None)

        rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        shape = np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                        mmap_mode='r')[indices].shape
        num_events = 600000
        self.data = np.zeros((shape[0],len(rotations),shape[1],3))
        self.data[:,0,:,0] = np.load(
                f'{data_dir}/{file_prefix}x_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]
        self.data[:,0,:,1] = np.load(
                f'{data_dir}/{file_prefix}y_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]
        self.data[:,0,:,2] = np.load(
                f'{data_dir}/{file_prefix}z_0_deg_{num_events}_events.npy',
                mmap_mode='r')[indices]

        for i,r in enumerate(rotations):
            if i == 0: 
                continue
            for j in range(len(self.data)):
                rotated_event = rotate_event(self.data[j,0,:,:], grid_shape, r,
                                                  bin_size=bin_size)
                rotated_event = rotated_event[:self.data.shape[2]]
                self.data[j,i,:rotated_event.shape[0],:] = rotated_event

        if scaler is not None:
            energy_shape = self.data[:,:,:,2].shape
            self.data[:,:,:,2] = scaler.transform(
                    self.data[:,:,:,2].flatten().reshape(-1,1)).reshape(energy_shape)

        if file_prefix[:4] == 'full':
            self.labels = np.load(
                    f'{data_dir}/full_labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(
                    f'{data_dir}/{file_prefix}labels_600000_events.npy',
                    mmap_mode='r')[indices].reshape(-1,1)
        self.labels = np.repeat(self.labels, len(rotations), axis=0)
        self.labels = self.labels.reshape(self.labels.shape[0]//len(rotations),
                                          len(rotations))

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_cnn_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, rotation=0, scaler=None):

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

        self.data, x_edges, y_edges = multiple_histogram2d(x_coord, y_coord, z_coord, 32)
        shape = self.data.shape

        if scaler is not None:
            self.data = scaler.transform(self.data.reshape(
                shape[0],shape[1]*shape[2])).reshape(shape)

        if file_prefix[:4] == 'full':
            self.labels = np.load(f'{data_dir}/full_labels_600000_events.npy',
                                  mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(f'{data_dir}/{file_prefix}labels_600000_events.npy',
                                  mmap_mode='r')[indices].reshape(-1,1)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_cnn_aug_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

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
        self.images = np.empty((shape[0],len(rotations),32,32))
        for i,r in enumerate(rotations):
            data[:,i,:,0] = np.load(f'{data_dir}/{file_prefix}x_{r}_deg_{num_events}_events.npy',
                                    mmap_mode='r')[indices]
            data[:,i,:,1] = np.load(f'{data_dir}/{file_prefix}y_{r}_deg_{num_events}_events.npy',
                                    mmap_mode='r')[indices]
            data[:,i,:,2] = np.load(f'{data_dir}/{file_prefix}z_{r}_deg_{num_events}_events.npy',
                                    mmap_mode='r')[indices]

        data_shape = data.shape
        image_shape = self.images.shape
        data = data.reshape((data_shape[0]*data_shape[1],data_shape[2],data_shape[3]))
        self.images, x_edges, y_edges = multiple_histogram2d(data[:,:,1], data[:,:,2],
                                                            data[:,:,0], 32)
        self.images = self.images.reshape(image_shape)

        if scaler is not None:
            self.images = scaler.transform(self.images.reshape(
                image_shape[0]*image_shape[1],
                image_shape[2]*image_shape[3])).reshape(image_shape)
        self.images.reshape((image_shape[0]*image_shape[1],
                             image_shape[2],image_shape[3]))

        if file_prefix[:4] == 'full':
            self.labels = np.load(f'{data_dir}/full_labels_600000_events.npy',
                                  mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(f'{data_dir}/{file_prefix}labels_600000_events.npy',
                                  mmap_mode='r')[indices].reshape(-1,1)
        self.labels = np.repeat(self.labels, len(rotations), axis=0)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

class toy_cnn_aug_set_dataset(Dataset):
    def __init__(self, data_dir, pixelization='square', indices=None, scaler=None):

        if pixelization == 'square':
            file_prefix = '' 
        else:
            file_prefix = f'{pixelization}_'

        if indices is None:
            indices = slice(None)

        rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        shape = np.array(np.load(f'{data_dir}/{file_prefix}x_0_deg_600000_events.npy',
                                 mmap_mode='r').shape)
        shape[0] = len(indices)
        num_events = 600000
        data = np.empty((shape[0],len(rotations),shape[1],3))
        self.images = np.empty((shape[0],len(rotations),32,32))
        for i,r in enumerate(rotations):
            data[:,i,:,0] = np.load(f'{data_dir}/{file_prefix}x_{r}_deg_{num_events}_events.npy',
                                    mmap_mode='r')[indices]
            data[:,i,:,1] = np.load(f'{data_dir}/{file_prefix}y_{r}_deg_{num_events}_events.npy',
                                    mmap_mode='r')[indices]
            data[:,i,:,2] = np.load(f'{data_dir}/{file_prefix}z_{r}_deg_{num_events}_events.npy',
                                    mmap_mode='r')[indices]

        data_shape = data.shape
        image_shape = self.images.shape
        data = data.reshape((data_shape[0]*data_shape[1],data_shape[2],data_shape[3]))
        self.images, x_edges, y_edges = multiple_histogram2d(data[:,:,1], data[:,:,2],
                                                            data[:,:,0], 32)
        self.images = self.images.reshape(image_shape)

        if scaler is not None:
            self.images = scaler.transform(self.images.reshape(
                image_shape[0]*image_shape[1],
                image_shape[2]*image_shape[3])).reshape(image_shape)

        if file_prefix[:4] == 'full':
            self.labels = np.load(f'{data_dir}/full_labels_600000_events.npy',
                                  mmap_mode='r')[indices].reshape(-1,1)
        else:
            self.labels = np.load(f'{data_dir}/{file_prefix}labels_600000_events.npy',
                                  mmap_mode='r')[indices].reshape(-1,1)

        self.data = torch.from_numpy(self.data).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.gpu_data = {}
        self.gpu_labels = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rank = torch.cuda.current_device()
        x = self.gpu_data[rank][index]
        y = self.gpu_labels[rank][index]
        return x, y

    def set_sample_device(self, rank, indices):
        device = torch.device(f'cuda:{rank}')
        self.gpu_data.setdefault(rank, {})
        self.gpu_labels.setdefault(rank, {})
        for i, x, y in zip(indices, self.data[indices].to(device),
                           self.labels[indices].to(device)):
            self.gpu_data[rank][i] = x
            self.gpu_labels[rank][i] = y

