import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, DynamicBatchSampler
from src.GNN_Decoder import GNN_Decoder
from src.gnn_models import GNN_7, GNN_7_DenseConv
from src.simulations import SurfaceCodeSim
from src.graph_representation import get_3D_graph, prune_graph
import torch_geometric.nn as nn_g
from torch_geometric.nn import knn_graph
from icecream import ic
# profiling
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    # default settings:
    # code and noise settings
    code_sz = 50
    p = 3e-3
    reps = 50

    # training settings
    n_epochs = 1
    n_graphs = 10
    lr = 1e-3
    loss = nn.BCEWithLogitsLoss()
    seed = 11
    batch_size = 512

    # graph settings
    n_node_feats = 5
    power = 1
    m_nearest_nodes = 5

    # read input arguments and potentially overwrite default settings
    parser = argparse.ArgumentParser(
        description="Choose model and optionally GPU instance."
    )
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
        batch_size = int(args.batch_size)
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
        # x, edge_index, edge_attr, y = get_3D_graph(
        #     syndrome_3D=syndrome, target=flip, power=power, m_nearest_nodes=3, use_knn=True
        # )
        
        x, edge_index, edge_attr, y = get_3D_graph(
            syndrome_3D=syndrome, target=flip, power=power, m_nearest_nodes=3, test=False, use_knn=True
        )
        graphs.append(Data(x, edge_index, edge_attr, y))
    loader = DataLoader(graphs, batch_size=batch_size)
    return
    # print(f"We have #{len(loader)} batches.")
    # run forward pass
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    # ) as prof:
    #     with record_function("forward pass"):
    for batch in loader:
        # move what we need to device
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        edge_attr = batch.edge_attr.to(device)
        batch_label = batch.batch.to(device)

        
        ic(x.shape)
        ic(edge_index.shape)
        ic(edge_attr.shape)
        ic(batch_label.shape)
        # prune graphs
        edge_index, edge_attr = prune_graph(x, edge_index, edge_attr, batch_label, m_nearest_nodes=3)
        
        out = decoder(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_label,
        )

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()
