import numpy as np
import sys, getopt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.GNN_Decoder import GNN_Decoder
from src.gnn_models import GNN_7, GNN_7_DenseConv
from src.simulations import SurfaceCodeSim
from src.graph_representation import get_3D_graph

def main(argv):
    
    # read which model to use
    opts, args = getopt.getopt(argv,"m:",["model="]) 
    
    # default
    model_class = GNN_7
    for opt, arg in opts:
        if opt in ("-m", "--model"):
            if arg == "GNN_7":
                model_class = GNN_7
            elif arg == "GNN_7_DenseConv":
                model_class = GNN_7_DenseConv
            else:
                TypeError("Must pick viable model class alternative!")
        
    # code and noise settings
    code_sz = 5
    p = 3e-3
    reps = 10

    # training settings
    n_epochs = 1
    n_graphs = 20000
    batch_sz = 4
    lr = 1e-3
    loss = nn.BCEWithLogitsLoss()
    seed = 11

    # graph settings
    n_node_feats = 5
    power = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = True if torch.cuda.is_available() else False

    # initialise decoder parameters
    gnn_params = {
        "model": {
            "class": model_class,
            "num_classes": 1,
            "loss": loss,
            "num_node_features": n_node_feats,
            "initial_learning_rate": lr,
            "manual_seed": seed,
        },
        "graph": {
            "num_node_features": n_node_feats,
            "power": power,
        },
        "cuda": cuda,
    }
    
    decoder = GNN_Decoder(gnn_params)
    print("Decoder was successfully created")
    
    sim = SurfaceCodeSim(reps, code_sz, p, n_shots=20000, seed=seed)
    syndromes, flips = sim.generate_syndromes(n_graphs)

    graphs = []
    for syndrome, flip in zip(syndromes, flips):
        graph = get_3D_graph(
            syndrome_3D=syndrome,
            target=flip,
            power=power,
        )

        graphs.append(
            Data(
                x=torch.from_numpy(graph[0]).to(device),
                edge_index=torch.from_numpy(graph[1]).to(device),
                edge_attr=torch.from_numpy(graph[2]).to(device),
                y=torch.from_numpy(graph[3]).to(device),
            )
        )

    loader = DataLoader(graphs, batch_size=1024)

    print(f"We have #{len(loader)} batches.")
    # run forward pass
    for batch in tqdm(loader):
        batch.batch = batch.batch.to(device)
        out = decoder(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )
    
    print(f"Mean value of output: {out.mean()}")
    
    
if __name__ == "__main__":
    main(sys.argv[1:])