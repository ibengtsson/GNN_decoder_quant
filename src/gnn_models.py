import torch
import numpy as np
from torch.nn import Linear, ModuleList
from torch import nn
from torch_geometric.nn import global_mean_pool, GraphConv, DenseGraphConv
from torch_geometric.utils import to_dense_adj


class GNN_7(torch.nn.Module):
    """
    GNN with 7 consecutive GraphConv layers, whose final output is converted
    to a single graph embedding (feature vector) with global_mean_pool.
    This graph embedding is duplicated, and passed to a
    dense network which performs binary classification.
    The binary classifications represents the logical equivalence classes
    X or Z, respectively.
    The output of this network is a tensor with 1 element, giving a binary
    representation of the predicted equivalence class.
    0 <-> class I
    1 <-> class X or Z
    """

    def __init__(
        self,
        hidden_channels_GCN=(32, 128, 256, 512, 512, 256, 256),
        hidden_channels_MLP=(256, 128, 64),
        num_node_features=4,
        num_classes=1,
        manual_seed=12345,
    ):
        # num_classes is 1 for each head
        super(GNN_7, self).__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
        # GCN layers
        self.graph1 = GraphConv(num_node_features, hidden_channels_GCN[0])
        self.graph2 = GraphConv(hidden_channels_GCN[0], hidden_channels_GCN[1])
        self.graph3 = GraphConv(hidden_channels_GCN[1], hidden_channels_GCN[2])
        self.graph4 = GraphConv(hidden_channels_GCN[2], hidden_channels_GCN[3])
        self.graph5 = GraphConv(hidden_channels_GCN[3], hidden_channels_GCN[4])
        self.graph6 = GraphConv(hidden_channels_GCN[4], hidden_channels_GCN[5])
        self.graph7 = GraphConv(hidden_channels_GCN[5], hidden_channels_GCN[6])
        # Layers for graph embedding classifier
        self.lin1 = Linear(hidden_channels_GCN[6], hidden_channels_MLP[0])
        self.lin2 = Linear(hidden_channels_MLP[0], hidden_channels_MLP[1])
        self.lin3 = Linear(hidden_channels_MLP[1], hidden_channels_MLP[2])
        self.lin4 = Linear(hidden_channels_MLP[2], num_classes)

        # Reset parameters on initialization of model instance
        self.graph1.reset_parameters()
        self.graph2.reset_parameters()
        self.graph3.reset_parameters()
        self.graph4.reset_parameters()
        self.graph5.reset_parameters()
        self.graph6.reset_parameters()
        self.graph7.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        # Obtain node embeddings
        x = self.graph1(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.graph2(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.graph3(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.graph4(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.graph5(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.graph6(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        x = self.graph7(x, edge_index, edge_weight=edge_attr)
        x = x.relu()
        # obtain graph embedding
        x = global_mean_pool(x, batch)
        # Apply X(Z) classifier
        X = self.lin1(x)
        X = X.relu()
        X = self.lin2(X)
        X = X.relu()
        X = self.lin3(X)
        X = X.relu()
        X = self.lin4(X)

        return X
    
    class GNN_7_DenseConv(torch.nn.Module):
        """
        GNN with 7 consecutive GraphConv layers, whose final output is converted
        to a single graph embedding (feature vector) with global_mean_pool.
        This graph embedding is duplicated, and passed to a
        dense network which performs binary classification.
        The binary classifications represents the logical equivalence classes
        X or Z, respectively.
        The output of this network is a tensor with 1 element, giving a binary
        representation of the predicted equivalence class.
        0 <-> class I
        1 <-> class X or Z
        
        TODO: UPDATE!
        """

        def __init__(
            self,
            hidden_channels_GCN=(32, 128, 256, 512, 512, 256, 256),
            hidden_channels_MLP=(256, 128, 64),
            num_node_features=4,
            num_classes=1,
            manual_seed=12345,
        ):
            # num_classes is 1 for each head
            super(GNN_7, self).__init__()
            if manual_seed is not None:
                torch.manual_seed(manual_seed)
            # GCN layers
            self.graph1 = DenseGraphConv(num_node_features, hidden_channels_GCN[0])
            self.graph2 = DenseGraphConv(hidden_channels_GCN[0], hidden_channels_GCN[1])
            self.graph3 = DenseGraphConv(hidden_channels_GCN[1], hidden_channels_GCN[2])
            self.graph4 = DenseGraphConv(hidden_channels_GCN[2], hidden_channels_GCN[3])
            self.graph5 = DenseGraphConv(hidden_channels_GCN[3], hidden_channels_GCN[4])
            self.graph6 = DenseGraphConv(hidden_channels_GCN[4], hidden_channels_GCN[5])
            self.graph7 = DenseGraphConv(hidden_channels_GCN[5], hidden_channels_GCN[6])
            # Layers for graph embedding classifier
            self.lin1 = Linear(hidden_channels_GCN[6], hidden_channels_MLP[0])
            self.lin2 = Linear(hidden_channels_MLP[0], hidden_channels_MLP[1])
            self.lin3 = Linear(hidden_channels_MLP[1], hidden_channels_MLP[2])
            self.lin4 = Linear(hidden_channels_MLP[2], num_classes)

            # Reset parameters on initialization of model instance
            self.graph1.reset_parameters()
            self.graph2.reset_parameters()
            self.graph3.reset_parameters()
            self.graph4.reset_parameters()
            self.graph5.reset_parameters()
            self.graph6.reset_parameters()
            self.graph7.reset_parameters()
            self.lin1.reset_parameters()
            self.lin2.reset_parameters()
            self.lin3.reset_parameters()
            self.lin4.reset_parameters()

        def forward(self, x, edge_index, edge_attr, batch):
            # Obtain node embeddings
            
            dense_adj = to_dense_adj(edge_index, batch, edge_attr)
            
            x = self.graph1(x, dense_adj, batch)
            x = x.relu()
            x = self.graph2(x, dense_adj, batch)
            x = x.relu()
            x = self.graph3(x, dense_adj, batch)
            x = x.relu()
            x = self.graph4(x, dense_adj, batch)
            x = x.relu()
            x = self.graph5(x, dense_adj, batch)
            x = x.relu()
            x = self.graph6(x, dense_adj, batch)
            x = x.relu()
            x = self.graph7(x, dense_adj, batch)
            x = x.relu()
            # obtain graph embedding
            x = global_mean_pool(x, batch)
            # Apply X(Z) classifier
            X = self.lin1(x)
            X = X.relu()
            X = self.lin2(X)
            X = X.relu()
            X = self.lin3(X)
            X = X.relu()
            X = self.lin4(X)

            return X


class RecurrentGNN(torch.nn.Module):
    def __init__(
        self,
        repetitions,
        num_classes=1,
        num_node_features=3,
        hidden_chn_gcn=[64, 128, 256, 512, 256, 128],
        hidden_size=128,
        hidden_chn_mlp=[128, 64],
        device=None,
    ):
        super.__init__()

        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.repetitions = repetitions

        in_shape_gcn = [num_node_features] + hidden_chn_gcn[:-1]
        self.graph_layers = ModuleList(
            [
                GraphConv(in_dim, out_dim)
                for (in_dim, out_dim) in zip(in_shape_gcn, hidden_chn_gcn)
            ]
        )

        self.lstm = nn.LSTM(
            input_size=hidden_chn_gcn[-1], hidden_size=hidden_size, batch_first=True
        )

        in_shape_mlp = hidden_chn_gcn[-1:] + hidden_chn_mlp[:-1]
        self.mlp_layers = ModuleList(
            [
                Linear(in_dim, out_dim)
                for (in_dim, out_dim) in zip(in_shape_mlp, hidden_chn_mlp)
            ]
        )

        self.out_layer = Linear(hidden_chn_mlp[-1], num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_weight=edge_attr)
            x = x.relu()

        x = global_mean_pool(x, batch)

        # reshape so we get time dimension right
        x = torch.reshape(x, (-1, self.repetitions, x.shape[1]))
        x, _ = self.lstm(x)
        # mlp
        for layer in self.mlp_layers:
            x = layer(x)
            x.relu()

        return self.out_layer(x)
