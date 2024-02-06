from pathlib import Path
import numpy as np
import sys
import torch
from torch_geometric.nn import knn_graph
sys.path.append("..")
from src.simulations import SurfaceCodeSim
from src.graph_representation import get_batch_of_graphs

def main():
    
    # settings
    reps = 3
    code_sz = 3
    n_shots = 1000
    p = 1e-3
    n_node_features = 5
    power = 2.0
    m_nearest_nodes = 5
    n = 64
    save_path = Path("data")
    file_name = "input_"
    
    # generate data
    sim = SurfaceCodeSim(
        reps,
        code_sz,
        p,
        n_shots=n_shots,
    )
    
    # we'll only treat one syndrome at a time
    syndromes, flips, n_identities = sim.generate_syndromes(n_shots, n_shots)
    flips = flips[:, None]
    
    syndromes = syndromes.astype(np.float32)

    defect_inds = np.nonzero(syndromes)
    defects = syndromes[defect_inds]

    defect_inds = np.transpose(np.array(defect_inds))

    x_defects = defects == 1
    z_defects = defects == 3

    node_features = np.zeros((defects.shape[0], n_node_features + 1), dtype=np.float32)

    node_features[x_defects, 0] = 1
    node_features[x_defects, 2:] = defect_inds[x_defects, ...]
    node_features[z_defects, 1] = 1
    node_features[z_defects, 2:] = defect_inds[z_defects, ...]

    node_features.max(axis=0)
    x_cols = [0, 1, 3, 4, 5]
    batch_col = 2

    x = torch.tensor(node_features[:, x_cols])
    batch_labels = torch.tensor(node_features[:, batch_col]).long()
    pos = x[:, 2:]

    # get edge indices
    edge_index = knn_graph(pos, m_nearest_nodes, batch=batch_labels)
    
    # compute distances to get edge attributes
    dist = torch.sqrt(((pos[edge_index[0], :] - pos[edge_index[1], :])**2).sum(dim=1, keepdim=True))
    edge_attr = 1 / dist ** power
    
    node_range = np.arange(x.shape[0])
    sorted_edge_row, sort_indx = torch.sort(edge_index[0, :])
    
    
    for i in range(n_shots):
        nodes = np.zeros((n, n_node_features))
        adj = np.zeros((n, n))
        node_labels = np.zeros((1, n))
        graph_indx = batch_labels == i
        node_indx = node_range[graph_indx]
        nodes[:np.count_nonzero(graph_indx)] = x[graph_indx].numpy()
        
        edges = (sorted_edge_row >= node_indx[0]) & (sorted_edge_row <= node_indx[-1])
        graph_edges = edge_index[:, sort_indx][:, edges]
        if graph_edges.numel() > 0:
            graph_edges -= torch.min(graph_edges)
        graph_attr = edge_attr[sort_indx, :][edges, :]
        
        # get adjacency matrix
        graph_edges = graph_edges.numpy()
        graph_attr = graph_attr.numpy()
        
        adj[graph_edges[0, :], graph_edges[1, :]] = graph_attr[:, 0]
        node_labels[0, :nodes.shape[0]] = 1 / nodes.shape[0]
        
        path = save_path / (file_name + str(i) + ".npz")
        np.savez(path, x=nodes, adj = adj, node_labels=node_labels, flip=flips[i])
        
if __name__ == "__main__":
    main()
