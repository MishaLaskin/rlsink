
import torch
import torchvision
import torch.utils.data
import random
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
What we want this file(s) to accomplish:

1. Given latent states supplied by a VQ VAE {z_t} - i.e. converted from some data source (on or offline)
2. Use CPC to learn a distance metric between two states z_i, z_j
2a. Create CPC nn.module that computes log densities for the CPC loss
2b. Define the CPC loss within the CPC module
3. Define a train function that
3a. Inputs a cpc_model, a vqvae model, (optional) external data source for observations
3b. Samples a batch of positive / negative samples from data
3c. Computes CPC loss, backprops, and steps forward
3d. Logs loss output

Then run this on a dataset to sense check
"""


class CPCSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self,
                 data_source,
                 path_length=100,
                 proximity_threshold=10,
                 n_negative=10,
                 batch_size=32,
                 ):

        self.data_source = data_source
        self.path_length = path_length
        self.n_paths = len(self.data_source)//self.path_length
        self.proximity_threshold = proximity_threshold
        self.n_negative = n_negative
        self.batch_size = batch_size

    def _get_positive_samples(self):
        # path id
        n = np.random.randint(self.n_paths)
        # step ids
        m1 = self.path_length*n + \
            np.random.randint(self.path_length-self.proximity_threshold)
        m2 = m1 + np.random.randint(1, self.proximity_threshold+1)
        return [m1, m2]

    def _get_negative_samples(self, pos_index):
        neg_indices = []

        while len(neg_indices) < self.n_negative:
            index = np.random.randint(len(self.data_source))
            # check that sample isn't close using oracle
            # [:2] is specific to Reacher
            pos_state = self.data_source.state_data[pos_index][:2]
            neg_state = self.data_source.state_data[index][:2]
            L2 = np.linalg.norm(pos_state-neg_state)
            if L2 > 0.05:
                neg_indices.append(index)

        return neg_indices

    def _one_batch(self):
        pos = self._get_positive_samples()
        neg = self._get_negative_samples(pos[0])
        return pos+neg

    def __iter__(self):

        minibatch = np.array([self._one_batch()
                              for _ in range(self.batch_size)]).reshape(-1)

        return iter(minibatch)

    def __len__(self):
        return len(self.data_source)


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def vqvae_encoder_for_cpc(x, model):
    x = x.to(device)
    z = model.pre_quantization_conv(model.encoder(x))
    return z


class CPC(nn.Module):

    def __init__(self,
                 z_dim=None,
                 cpc_batch=None,
                 batch_size=None,
                 loader=None,
                 ):
        super(CPC, self).__init__()

        self.z_dim = z_dim
        self.cpc_batch = cpc_batch
        self.batch_size = batch_size
        self.loader = loader

        self._check_inputs()

        self.z_dim = z_dim

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.W = nn.Parameter(torch.rand(z_dim, z_dim), requires_grad=True)

    def _check_inputs(self):
        assert self.z_dim is not None
        assert self.cpc_batch is not None
        assert self.batch_size is not None
        assert self.loader is not None

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        cpc_batch_size = z_j.shape[0]
        z_dim = z_i.shape[-1]

        z_i = z_i.view(z_dim, batch_size)

        right = torch.matmul(self.W, z_i)
        f = torch.bmm(z_j.view(batch_size, cpc_batch_size, z_dim),
                      right.view(batch_size, z_dim, 1))
        f_scores = f.squeeze(2)/1000

        return f_scores

    def naive_loss(self, z_t, z_others, verbose=False):
        # maybe try making z
        z_t = z_t.view(128, 1)
        z_others = z_others.view(51, 128)
        left_side = torch.matmul(z_others, self.W)
        scores = torch.matmul(left_side, z_t)
        scores /= 1000
        scores = scores.squeeze(1)
        positive = scores[0]

        # be careful - goes to NaN
        # use log sum exp trick

        # visualizing CPC - sample 100 data points, rank distance from 1 point
        # visualize
        def log_sum_exp_trick(arr):
            max_arr = torch.max(arr)
            return max_arr + torch.log(torch.sum(torch.exp(arr - max_arr)))

        loss = log_sum_exp_trick(scores) - \
            torch.log(torch.exp(positive))

        if verbose:
            print('LOSS')
            print(positive)
            print(torch.sum(scores))
            print('z_t', z_t.shape)
            print('z_others', z_others.shape)
            print('scores shape is:', scores.shape)

        return loss

    def naive_forward(self, z1, z2):
        z1 = z1.view(128, 1)
        z2 = z2.view(-1, 128)
        return torch.exp(torch.matmul(torch.matmul(z2, self.W), z1).squeeze(0)/1000)

    def loss(self, z_i, z_j, verbose=False):
        batch_size = 32  # z_i.shape[0]
        cpc_batch_size = 51  # z_j.shape[0]
        z_dim = 128  # z_i.shape[-1]

        z_i = z_i.view(z_dim, batch_size)

        right = torch.matmul(self.W, z_i)
        f = torch.bmm(z_j.view(batch_size, cpc_batch_size, z_dim),
                      right.view(batch_size, z_dim, 1))
        f = f.squeeze(2)/1000

        f_pos = torch.exp(f[:, 0])
        f_sums = torch.sum(torch.exp(f), dim=1)
        loss = -torch.log(f_pos / f_sums)
        loss = torch.mean(loss)
        if verbose:
            print('z_i', z_i.shape)
            print('z_j', z_j.shape)
            print('f shape is:', f.shape, 'should be:',
                  str([batch_size, cpc_batch_size]))
            print('f_pos', f_pos.shape)
            print('f_sums', print(f_sums.shape))
            print('final loss', loss)

        return loss


def train_cpc():
    n_updates = 100000
    lr = 3e-1
    n = 100
    path_length = 100
    T = 10
    n_neg = 50
    batch_size = 1
    cpc_batch_size = n_neg+2
    cpc_model_path = '/home/misha/research/rlsink/saved/cpc_weights3.pth'
    total_samples = batch_size*cpc_batch_size

    from rlsink.utils.data import load_data_and_data_loaders, load_model

    data_file_path = '/home/misha/research/vqvae/data/reacher_no_target_length100_paths_2000.npy'
    model_filename = '/home/misha/research/vqvae/results/vqvae_data_reacher_aug7_ne128nd2.pth'

    model, _ = load_model(model_filename)

    tr_data, _, _, _ = load_data_and_data_loaders(
        data_file_path, path_length, shuffle=False, include_state=True)

    sampler = CPCSampler(tr_data, path_length=n, proximity_threshold=T,
                         n_negative=n_neg, batch_size=batch_size)

    loader = torch.utils.data.DataLoader(
        tr_data, sampler=sampler, batch_size=total_samples)

    cpc = CPC(z_dim=128,
              cpc_batch=52,
              batch_size=1,
              loader=loader).cuda()

    optimizer = optim.Adam(cpc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40, 60, 80], gamma=0.1)

    # scheduler =  ReduceLROnPlateau(optimizer, 'min')

    for i in range(n_updates):
        x = next(iter(loader))[0]
        # print(x.shape)5
        z = vqvae_encoder_for_cpc(x, model)
        z = z.reshape(cpc_batch_size, batch_size, -1)
        # print(z.shape)
        z_t = z[:1]
        z_others = z[1:]
        # print(z_t.shape,z_others.shape)
        loss = cpc.naive_loss(z_t, z_others)

        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print('epoch', i//1000, 'loss', loss.detach().cpu().item())

            torch.save(cpc.state_dict(), cpc_model_path)
            scheduler.step()


if __name__ == "__main__":
    train_cpc()
