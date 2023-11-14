import numpy as np
import torch.nn as nn
from src.GNN_Decoder import GNN_Decoder
from src.gnn_models import GNN_7, GNN_7_DenseConv


if __name__ == "__main__":
    
    # code and noise settings
    code_sz = 7
    p = 1e-3
    reps = 10
    
    # training settings
    n_epochs = 1
    batch_sz = 4
    lr = 1e-3
    loss = nn.BCEWithLogitsLoss()
    
    # graph settings
    n_node_feats = 5
    power = 1
    cuda = False
    
    # initialise decoder parameters
    gnn_params = {
        "model": {
            "class": GNN_7,
            "num_classes": 1,
            "loss": loss,
            "num_node_features": n_node_feats,
            "initial_learning_rate": lr,
        },
        "graph": {
            "num_node_features": n_node_feats,
            "power": power,
        },
        "cuda": cuda,
    }
    
    decoder = GNN_Decoder(gnn_params)
    print("Decoder structure was successfully created")
    
    
    