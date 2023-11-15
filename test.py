import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.GNN_Decoder import GNN_Decoder
from src.gnn_models import GNN_7, GNN_7_DenseConv
from src.simulations import SurfaceCodeSim
from src.graph_representation import get_3D_graph

def main():
    
    # default settings:
    # code and noise settings
    code_sz = 5
    p = 3e-3
    reps = 10

    # training settings
    n_epochs = 1
    n_graphs = 100000
    lr = 1e-3
    loss = nn.BCEWithLogitsLoss()
    seed = 11
    batch_size = 512

    # graph settings
    n_node_feats = 5
    power = 1
    
    # read input arguments and potentially overwrite default settings
    parser = argparse.ArgumentParser(description="Choose model and optionally GPU instance.")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-b", "--batch_size", required=False)
    parser.add_argument("-g", "--gpu", required=False)
    
    args = parser.parse_args()
    if args.model == "GNN_7":
        model_class = GNN_7
    elif args.model == "GNN_7_DenseConv":
        model_class = GNN_7_DenseConv
    else:
        print("Using default model class, GNN_7.")
        model_class = GNN_7
    
    if args.gpu:   
        device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
        print(f"Using {device}.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using default device {device}.")
    
    if args.batch_size:
        batch_size = args.batch_size
        print(f"Using batch size {batch_size}.")
    else: 
        print(f"Using default batch size {batch_size}.")

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
        "device": device,
    }
    
    decoder = GNN_Decoder(gnn_params)
    print("Decoder was successfully created")
    
    sim = SurfaceCodeSim(reps, code_sz, p, n_shots=n_graphs, seed=seed)
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
                x=torch.from_numpy(graph[0]),
                edge_index=torch.from_numpy(graph[1]),
                edge_attr=torch.from_numpy(graph[2]),
                y=torch.from_numpy(graph[3]),
            )
        )

    loader = DataLoader(graphs, batch_size=5012)

    print(f"We have #{len(loader)} batches.")
    # run forward pass
    for batch in tqdm(loader):
        
        # move what we need to gpu
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        batch_label = batch.batch.to(device)
        
        out = decoder(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_label,
        )

    print(f"Mean value of output: {out.mean()}")
    
    
if __name__ == "__main__":
    main()