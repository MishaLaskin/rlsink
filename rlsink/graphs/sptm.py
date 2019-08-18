"""what we want this file to accomplish:

Create nodes
1. Given VQ VAE latent state
2. Check if there exists a node in the graph close to latent state:
2a. If yes: update as moving average
2b. If no: create new node

Create edges:
3. Given CPC retrieval network R
4. Pass though all node pairs (v_i,v_j)
5. If R(v_i,v_j) > threshold -> add an edge between vertices

Compute shortest path:
6. Given an observed latent z
7. If there exists a node close to z in graph
7a. Localize z - compute shortest path to goal node
7b. If not, 

Suppose z is not in the graph and far away"

Ideas:
- initialize infinite 

"""

"""CREATE NODES

1. sample N (e.g. 1000) starting nodes randomly 
2. traverse dataset sequentially  and encode using vq vae
2a. If proximal node is in dataset (determined by threshold e.g. 0.05) 
2aa. merge z into node and update node value with moving average
2ab. if not, create new node with value z

return list of node values 
"""

from datetime import datetime
import torch
import torchvision
import torch.utils.data
import random
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
from rlsink.classifiers.cpc import CPC, CPCSampler
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def vqvae_encoder_as_numpy(x, model):
    x = x.to(device)
    z = model.pre_quantization_conv(model.encoder(x))
    return z.detach().cpu().numpy()


def L2(x1, x2):
    return np.linalg.norm(x1-x2)


def polyak_avg(x1, x2, beta=0.95):
    return beta * x1 + (1-beta)*x2


"""
Create nodes only (no edges)
"""


def create_node_list(data,
                     vqvae_model,
                     threshold=0.1,
                     n_starting_nodes=1,
                     batch_size=100,
                     z_dim=128
                     ):
    loader = torch.utils.data.DataLoader(
        data, shuffle=False, batch_size=batch_size)
    random_loader = torch.utils.data.DataLoader(
        data, shuffle=True, batch_size=n_starting_nodes)

    init_images = next(iter(random_loader))[0]
    nodes = vqvae_encoder_as_numpy(init_images, vqvae_model)

    for i, batch in enumerate(iter(loader)):
        print(i, end='\r')
        images = batch[0]
        latents = vqvae_encoder_as_numpy(images, vqvae_model)
        for z in latents:
            # find closest node in graph
            distances = np.linalg.norm(
                nodes.reshape(-1, 128)-z.reshape(-1), axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist < threshold:
                nodes[min_idx] = polyak_avg(nodes[min_idx], z)
            else:
                new_node = np.expand_dims(z, axis=0)
                nodes = np.concatenate((nodes, new_node), axis=0)

    return nodes


"""
Build edges with retrieval network
"""


def build_edges_with_cpc(
        nodes,
        data,
        z_dim=128,
        path_length=100,
        proximity_threshold=10,
        n_negative=10,
        path='/home/misha/research/rlsink/saved/cpc_weights4.pth'):

    batch_size = 1
    cpc_batch_size = n_negative+2
    total_samples = batch_size*cpc_batch_size

    sampler = CPCSampler(data, path_length=path_length,
                         proximity_threshold=proximity_threshold, n_negative=n_negative, batch_size=1)

    loader = torch.utils.data.DataLoader(
        data, sampler=sampler, batch_size=total_samples)

    cpc_model = CPC(z_dim=z_dim,
                    cpc_batch=cpc_batch_size,
                    batch_size=1,
                    loader=loader).cuda()

    cpc_model.load_state_dict(torch.load(path))
    cpc_model.eval()

    nodes = torch.tensor(nodes).float().to(device)
    scores = []
    for i, n in enumerate(nodes):
        node_scores = cpc_model.forward(n, nodes)
        scores.append(node_scores)
    scores = torch.cat(scores, 1)
    return scores


def preprocess_scores(scores):
    clean_scores = torch.log(scores).detach().cpu()  # torch.log
    clean_scores = torch.exp(-clean_scores)
    np_scores = clean_scores.detach().cpu().numpy()
    return np_scores


def remove_false_negatives(np_scores, np_nodes):
    print('removing false negatives')
    np_scores.shape
    n = np_scores.shape[0]
    # print(nodes.shape)
    #assert False
    #np_nodes = nodes.detach().cpu().numpy()
    for i in range(n):
        print(i, end='\r')
        for j in range(n):

            if np.linalg.norm(np_nodes[i]-np_nodes[j]) > .2:
                np_scores[i, j] = float("inf")
    return np_scores


def build_sptm_graph(data, model):
    nodes = create_node_list(data,
                             model,
                             threshold=0.15,
                             n_starting_nodes=1,
                             batch_size=100,
                             z_dim=128)

    scores = build_edges_with_cpc(nodes, data)
    np_scores = preprocess_scores(scores)
    np_scores = remove_false_negatives(np_scores, nodes)

    G = nx.Graph()
    n = len(nodes)
    for i in range(n):
        for j in range(n):
            G.add_edge(i, j, weight=np_scores[i, j])

    return G, nodes


if __name__ == "__main__":
    from rlsink.utils.data import load_data, load_data_and_data_loaders, load_model
    import pickle

    def save_object(obj, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    data_file_path = '/home/misha/research/vqvae/data/reacher_no_target_length100_paths_2000.npy'
    model_filename = '/home/misha/research/vqvae/results/vqvae_data_reacher_aug7_ne128nd2.pth'

    model, vqvae_data = load_model(model_filename)
    data, _, _, _ = load_data_and_data_loaders(
        data_file_path, 100, shuffle=False, include_state=True)
    # data = load_data(
    #    data_file_path, include_state=False)
    G, nodes = build_sptm_graph(data, model)
    now = datetime.now()
    dt_string = now.strftime("%m/%d/%Y %H:%M")
    filename = '/home/misha/research/rlsink/saved/reacher_graph.pkl'

    obj = {'graph': G, 'nodes': nodes}
    save_object(obj, filename)
