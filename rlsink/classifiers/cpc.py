
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

BATCH_SIZE = 32


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
            if L2 > 0.1:
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


def vqvae_encoder_for_cpc(x, model, device):

    x = x.to(device)
    z = model.pre_quantization_conv(model.encoder(x))
    return z


def log_sum_exp(arr):

    max_arr = torch.max(arr, 1, keepdim=True).values
    return max_arr + torch.log(torch.sum(torch.exp(arr - max_arr), 1))


class CPC(nn.Module):

    def __init__(self,
                 z_dim=None,
                 cpc_batch=None,
                 batch_size=None,
                 loader=None,
                 gpu_id=0
                 ):
        super(CPC, self).__init__()

        self.z_dim = z_dim
        self.cpc_batch = cpc_batch
        self.batch_size = batch_size
        self.loader = loader

        self._check_inputs()

        self.z_dim = z_dim

        self.device = torch.device(
            "cuda:"+str(gpu_id)) if torch.cuda.is_available() else torch.device("cpu")

        self.W = nn.Parameter(torch.rand(z_dim, z_dim), requires_grad=True)

    def _check_inputs(self):
        assert self.z_dim is not None
        assert self.cpc_batch is not None
        assert self.batch_size is not None
        assert self.loader is not None

    def encode(self, x):
        return x

    def loss(self, x, x_next):
        # Same as density
        assert x_next.size(0) == x.size(0)
        z = self.encode(x)
        z_next = self.encode(x_next)
        # z = z.unsqueeze(2)  # bs x z_dim x 1
        #z_next = z_next.unsqueeze(2)
        w = self.W

        w = w.expand(x.shape[0], w.shape[0], w.shape[1])

        f_out = torch.bmm(torch.bmm(z_next, w), z.permute(0, 2, 1))
        f_out = f_out.squeeze()
        f_out /= 1000

        f_pos = f_out[:, :1]

        loss = log_sum_exp(f_out) - torch.log(torch.exp(f_pos))

        return torch.mean(loss), None

    def forward(self, z1, z2):
        z1 = z1.view(128, 1)
        z2 = z2.view(-1, 128)
        return torch.exp(torch.matmul(torch.matmul(z2, self.W), z1).squeeze(0)/1000)

    """
    def naive_batch_loss(self, z, z_all):
        # ref sample (32,128,1)
        z = z.permute(1, 2, 0)
        # cpc batch samples (32,51,128)
        z_all = z_all.permute(1, 0, 2)
        # positive sample (32,1,128)
        z_pos = z_all[:, :1, :]

        losses = []

        for i in range(BATCH_SIZE):
            z_i = z[i:i+1, :, :].squeeze(0)
            z_all_i = z_all[i:i+1, :, :].squeeze(0)

            loss = self.naive_loss(z_i, z_all_i)[0]
            losses.append(loss)

        # print(torch.stack(losses))
        return torch.mean(torch.stack(losses)), None

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

        return loss, None

    def naive_forward(self, z1, z2):
        z1 = z1.view(128, 1)
        z2 = z2.view(-1, 128)
        return torch.exp(torch.matmul(torch.matmul(z2, self.W), z1).squeeze(0)/1000)

    def loss(self, z, z_others, verbose=False):
        batch_size = BATCH_SIZE  # z_i.shape[0]
        cpc_batch_size = 51  # z_j.shape[0]
        z_dim = 128  # z_i.shape[-1]

        # ref sample (32,128,1)
        z = z.permute(1, 2, 0)  # .repeat(1,1,batch_size)
        # cpc batch samples (32,51,128)
        z_others = z_others.permute(1, 0, 2)
        # positive sample (32,1,128)
        z_pos = z_others[:, :1, :]

        repeated_w = self.W.unsqueeze(2).expand(
            z_dim, z_dim, batch_size).permute(2, 0, 1)

        # z_all.T W z (32,51,128) x (32,128,128) x (32,128,1) = (32,51,1) -> squeeze(2) = (32,51)
        z_all_W_z = torch.bmm(z_others, torch.bmm(
            repeated_w, z)).squeeze(2)/1000

        # z_pos.T W z (32,1,128) x (32,128,128) x (32,128,1) = (32,1,1) -> squeeze(2) = (32,1)

        z_pos_W_z = z_all_W_z[:, :1]

        def log_sum_exp_batch(arr):
            # arr has shape (batch,cpc_batch) so (32,51) for example
            # max_arr has shape (32,1)
            max_arr = torch.max(arr, 1, keepdim=True).values
            #max_arr = torch.max(arr).unsqueeze(0)
            # print(max_arr.shape)
            # print(torch.log(torch.sum(torch.exp(arr),1)).shape)
            # print(arr.shape)
            return max_arr + torch.log(torch.sum(torch.exp(arr - max_arr), 1))

        # calculate the loss
        # Loss = - E [log (exp(z_pos.T W z) / sum_i exp( z_i.T W z))]
        #      = E [log sum_i exp (z_i.T W z) - log (exp (z_pos.T W z ))]
        log_exp_f_all = log_sum_exp_batch(z_all_W_z).squeeze(1)
        log_exp_f_pos = torch.log(torch.exp(z_pos_W_z).squeeze(1))
        loss = torch.mean(log_exp_f_all - log_exp_f_pos)
        # print('log_exp_f_neg',torch.mean(log_exp_f_neg))
        # print('log_exp_f_pos',torch.mean(log_exp_f_pos))
        if verbose:
            print('repeated w', repeated_w.shape)
            print('z', z.shape)
            print('z_pos w', z_pos.shape)
            print('z_neg', z_others.shape)
            print('z_pos_W_z', z_pos_W_z.shape)
            print('z_neg_W_z', z_all_W_z.shape)
            print('log_exp_f_neg', log_exp_f_all.shape)
            print('loss', loss)
        info = dict(
            # z_pos=z,
            # z_all=torch.mean(z_others),
            W=torch.mean(self.W),
            z_pos_W_z=z_pos_W_z,
            z_all_W_z=torch.mean(z_all_W_z),
            log_exp_f_pos=log_exp_f_pos,
            log_exp_f_all=log_exp_f_all,

        )
        return loss, info

    def forward(self, z_i, z_j):
        batch_size = 32  # z_i.shape[0]
        cpc_batch_size = 51  # z_j.shape[0]
        z_dim = 128  # z_i.shape[-1]

        z_i = z_i.view(batch_size, z_dim, 1)  # .repeat(1,1,batch_size)
        z_j = z_j.view(batch_size, z_dim, 1).permute(0, 2, 1)

        repeated_w = self.W.unsqueeze(2).repeat(
            1, 1, batch_size).permute(2, 0, 1)

        # stopped here 
        score = torch.bmm(z_j, torch.bmm(repeated_w, z_i)).squeeze(2)
        print('score', score.shape)
        assert False

    def encode(self, x):
        return x

    def log_density(self, x, x_next):
        # Same as density
        assert x_next.size(0) == x.size(0)
        z, _ = self.encode(x)
        z_next, _ = self.encode(x_next)
        z = z.unsqueeze(2)  # bs x z_dim x 1
        z_next = z_next.unsqueeze(2)

        w = self.W
        w = w.repeat(z.size(0), 1, 1)
        f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z)
        f_out = f_out.squeeze()

        return f_out / 1000
    """


def train_cpc():
    n_updates = 100000
    lr = 3e0
    n = 100
    path_length = 100
    T = 10
    n_neg = 8
    batch_size = BATCH_SIZE
    cpc_batch_size = n_neg+2
    cpc_model_path = '/home/misha/research/rlsink/saved/cpc_weights4.pth'
    total_samples = cpc_batch_size
    gpu_id = 0

    device = torch.device(
        "cuda:"+str(gpu_id)) if torch.cuda.is_available() else torch.device("cpu")

    from rlsink.utils.data import load_data_and_data_loaders, load_model

    data_file_path = '/home/misha/research/vqvae/data/reacher_no_target_length100_paths_2000.npy'
    model_filename = '/home/misha/research/vqvae/results/vqvae_data_reacher_aug7_ne128nd2.pth'

    model, _ = load_model(model_filename)

    tr_data, _, _, _ = load_data_and_data_loaders(
        data_file_path, path_length, shuffle=False, include_state=True)

    sampler = CPCSampler(tr_data, path_length=n, proximity_threshold=T,
                         n_negative=n_neg, batch_size=1)

    loader = torch.utils.data.DataLoader(
        tr_data, sampler=sampler, batch_size=total_samples)

    cpc = CPC(z_dim=128,
              cpc_batch=52,
              batch_size=1,
              gpu_id=gpu_id,
              loader=loader).to(device)

    model = model.to(device)

    optimizer = optim.Adam(cpc.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40, 60], gamma=0.1)

    # scheduler =  ReduceLROnPlateau(optimizer, 'min')

    for i in range(n_updates):
        x = torch.cat([next(iter(loader))[0]
                       for _ in range(batch_size)])  # next(iter(loader))[0]
        z = vqvae_encoder_for_cpc(x, model, device)

        z = z.view(batch_size, cpc_batch_size, -1)

        z_t = z[:, :1, :]
        z_all = z[:, 1:, :]

        # print(z_t.shape,z_others.shape)
        loss, info = cpc.loss(z_t, z_all)

        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('='*20)
            print('epoch', i, 'loss', loss.detach().cpu().item())
            # for k, v in info.items():
            #    print(k, v.detach().cpu().numpy())
            scheduler.step()

        if i % 1000:
            torch.save(cpc.state_dict(), cpc_model_path)


if __name__ == "__main__":
    train_cpc()
