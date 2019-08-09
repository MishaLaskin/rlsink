import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from vqvae.models.vqvae import VQVAE, TemporalVQVAE


class ImageDataset(Dataset):
    """
    Creates image dataset of 32X32 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, train=True, transform=None, make_temporal=False, path_length=100, include_state=False):
        print('Loading data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading data')
        state_data = np.array(data.item().get('observation'))
        data = np.array(data.item().get('image_observation'))

        self.include_state = include_state
        self.state_data = None

        self.n = data.shape[0]
        self.cutoff = self.n//10
        self.data = data[:-self.cutoff] if train else data[-self.cutoff:]
        if self.include_state:

            self.state_data = state_data[:-
                                         self.cutoff] if train else state_data[-self.cutoff:]
        self.transform = transform
        self.make_temporal = make_temporal
        self.path_length = path_length
        self.train = train

    def __getitem__(self, index):
        img = self.data[index]

        if self.transform is not None:
            img = self.transform(img)

        label = index

        # if we want to keep track of adjacent states
        # e.g.s obs_t and obs_{t-1} we use make_temporal
        if self.train and self.make_temporal:
            if index % self.path_length == 0:
                img2 = self.data[index+1]
            else:
                img2 = self.data[index-1]
            img2 = self.transform(img2)

            return img, img2, label

        if self.include_state:
            state = self.state_data[index]
            return img, state, label
        else:

            return img, label

    def __len__(self):
        return len(self.data)


def cpc_data_loader(sequential_loader):
    """
    needs to
    1. 
    """
    pass


def load_data(data_file_path=None, include_state=False):

    if data_file_path is None:
        raise ValueError('Please provide a data_file_path input string')

    train = ImageDataset(data_file_path,
                         train=True,
                         include_state=include_state,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = ImageDataset(data_file_path,
                       train=False,
                       include_state=include_state,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val


def data_loaders(train_data, val_data, batch_size, shuffle=True):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(data_file_path, batch_size, shuffle=True, include_state=False):

    training_data, validation_data = load_data(
        data_file_path, include_state=include_state)
    training_loader, validation_loader = data_loaders(
        training_data, validation_data, batch_size, shuffle)

    return training_data, validation_data, training_loader, validation_loader


def load_model(model_filename, temporal=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        data = torch.load(model_filename)
    else:
        data = torch.load(
            model_filename, map_location=lambda storage, loc: storage)

    params = data["hyperparameters"]

    if temporal:
        model = TemporalVQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                              params['n_residual_layers'], params['n_embeddings'],
                              params['embedding_dim'], params['beta']).to(device)
    else:
        model = VQVAE(params['n_hiddens'], params['n_residual_hiddens'],
                      params['n_residual_layers'], params['n_embeddings'],
                      params['embedding_dim'], params['beta']).to(device)

    model.load_state_dict(data['model'])

    return model, data
