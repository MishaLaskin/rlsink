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
2. traverse dataset sequentially 
2a. If proximal node is in dataset (determined by threshold e.g. 0.05) 
2aa. merge z into node and update node value with moving average
2ab. if not, create new node with value z

return list of node values 
"""
