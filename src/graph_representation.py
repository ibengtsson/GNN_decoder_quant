"""Package with functions for creating graph representations of syndromes."""
import numpy as np
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_dense_adj, to_dense_batch
from icecream import ic


def get_node_list_3D(syndrome_3D):
    """
    Create two arrays, one containing the syndrome defects,
    and the other containing their corresponding contiguous
    indices in the matrix representation of the syndrome.
    """
    defect_indices_triple = np.nonzero(syndrome_3D)
    defects = syndrome_3D[defect_indices_triple]
    return defects, defect_indices_triple


def get_node_feature_matrix(defects, defect_indices_triple, num_node_features=None):
    """
    Creates a node feature matrix of dimensions
    (number_of_defects, number_of_node_features), where each row
    is the feature vector of a single node.
    The feature vector is defined as
    x = (X, Z, d_north, d_west, d_time)
        X: 1(0) if defect corresponds to a X(Z) stabilizer
        Z: 1(0) if defect corresponds to a Z(X) stabilizer
        d_north: distance to north boundary, i.e. row index in syndrome matrix
        d_west: distance to west boundary, i.e. column index in syndrome matrix
        d_time: distance in time from the first measurement
    """

    if num_node_features is None:
        num_node_features = 5  # By default, use 4 node features

    # Get defects (non_zero entries), defect indices (indices of defects in
    # flattened syndrome)
    # and defect_indices_tuple (indices in 3D syndrome) of the syndrome matrix

    num_defects = defects.shape[0]

    defect_indices_triple = np.transpose(np.array(defect_indices_triple))

    # get indices of x and z type defects, resp.
    x_defects = defects == 1
    z_defects = defects == 3

    # initialize node feature matrix
    node_features = np.zeros([num_defects, num_node_features])
    # defect is x type:
    node_features[x_defects, 0] = 1
    # distance of x tpe defect from northern and western boundary:
    node_features[x_defects, 2:] = defect_indices_triple[x_defects, :]

    # defect is z type:
    node_features[z_defects, 1] = 1
    # distance of z tpe defect from northern and western boundary:
    node_features[z_defects, 2:] = defect_indices_triple[z_defects, :]

    return node_features


# Function for creating a single graph as a PyG Data object
def get_3D_graph(
    syndrome_3D,
    target=None,
    m_nearest_nodes=None,
    power=None,
    use_knn=False,
    test=False,
):
    """
    Form a graph from a repeated syndrome measurement where a node is added,
    each time the syndrome changes. The node features are 5D.
    """
    # get defect indices:
    defects, defect_indices_triple = get_node_list_3D(syndrome_3D)

    # Use helper function to create node feature matrix as torch.tensor
    # (X, Z, N-dist, W-dist, time-dist)
    X = get_node_feature_matrix(defects, defect_indices_triple, num_node_features=5)
    # set default power of inverted distances to 1
    if power is None:
        power = 1.0

    # construct the adjacency matrix!
    n_defects = len(defects)
    y_coord = defect_indices_triple[0].reshape(n_defects, 1)
    x_coord = defect_indices_triple[1].reshape(n_defects, 1)
    t_coord = defect_indices_triple[2].reshape(n_defects, 1)

    y_dist = np.abs(y_coord.T - y_coord)
    x_dist = np.abs(x_coord.T - x_coord)
    t_dist = np.abs(t_coord.T - t_coord)

    # inverse square of the supremum norm between two nodes
    adj = np.maximum.reduce([y_dist, x_dist, t_dist])
    # set diagonal elements to nonzero to circumvent division by zero
    np.fill_diagonal(adj, 1)
    # scale the edge weights
    adj = 1.0 / adj**power
    # set diagonal elements to zero to exclude self loops
    np.fill_diagonal(adj, 0)

    if target is not None:
        # does this really need shape (1, 1)?
        y = target.reshape(1, 1)
    else:
        y = None

    if test:
        adj = np.maximum(adj, adj.T)  # Make sure for each edge i->j there is edge j->i
        n_edges = np.count_nonzero(adj)  # Get number of edges

        # get the edge indices:
        edge_index = np.nonzero(adj)
        edge_attr = adj[edge_index].reshape(n_edges, 1)
        edge_index = np.array(edge_index)

        return (
            torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(edge_index.astype(np.int64)),
            torch.from_numpy(edge_attr.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
        )
        
    if use_knn:
        x = torch.tensor(X, dtype=torch.float32)
        edge_index = knn_graph(x[:, 2:], m_nearest_nodes, flow="target_to_source", batch_size=1)
        adj = torch.tensor(adj, dtype=torch.float32)
        mask = torch.zeros_like(adj, dtype=bool)
        mask[*edge_index] = True
        adj[~mask] = 0
        edge_attr = adj[mask][:, None]

        y = torch.tensor(y)

        return x, edge_index, edge_attr, y

    # remove all but the m_nearest neighbours
    if m_nearest_nodes is not None:
        for ix, row in enumerate(adj.T):
            # Do not remove edges if a node has (degree <= m)
            if np.count_nonzero(row) <= m_nearest_nodes:
                continue
            # Get indices of all nodes that are not the m nearest
            # Remove these edges by setting elements to 0 in adjacency matrix
            adj.T[
                ix, np.argpartition(row, -m_nearest_nodes)[:-m_nearest_nodes]
            ] = 0.0

    adj = np.maximum(adj, adj.T)  # Make sure for each edge i->j there is edge j->i
    n_edges = np.count_nonzero(adj)  # Get number of edges

    # get the edge indices:
    edge_index = np.nonzero(adj)
    edge_attr = adj[edge_index].reshape(n_edges, 1)
    edge_index = np.array(edge_index)

    return (
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(edge_index.astype(np.int64)),
        torch.from_numpy(edge_attr.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )

def prune_graph(x, edge_index, edge_attr, batch, m_nearest_nodes):
    
    adj = to_dense_adj(edge_index, batch, edge_attr).squeeze()
    ic(adj.shape)
    edge_index = knn_graph(x[:, 2:], m_nearest_nodes, batch=batch, flow="target_to_source")
    ic(edge_index.shape)
    mask = torch.zeros_like(adj, dtype=bool)
    mask[*edge_index] = True
    adj[~mask] = 0
    edge_attr = adj[mask][:, None]
    
    return edge_index, edge_attr