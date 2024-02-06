import argparse
import torch
import torch.nn as nn
import torch_geometric.nn as nn_g
import torch.ao.quantization as tq
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import KNNGraph, Distance, Compose
import sys

sys.path.append("..")

from src.gnn_models import GNN_7
from src.simulations import SurfaceCodeSim
from src.graph_representation import get_3D_graph
from src.utils import match_and_load_state_dict, run_inference
from pathlib import Path
from tqdm import tqdm

from sys import getsizeof


def main():
    
    msg = ("WARNING: If using pre-trained weights from /models"
            + "this will likely not match results from paper"
            + "due to slight changes in graph creation."
            + "Now node pruning is done based on euclidian distances,"
            + " before Manhattan distance was used"
    )
    print(msg)
    # command line parsing
    parser = argparse.ArgumentParser(description="Choose model to load.")
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-d", "--device", required=False)

    args = parser.parse_args()

    model_path = Path(args.file)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    if model_path.is_file():
        model_data = torch.load(model_path, map_location=device)
    else:
        raise FileNotFoundError("The file was not found!")

    model = GNN_7().to(device)
    model.load_state_dict(model_data["model"])
    model.eval()

    print(f"Moved model to {device} and loaded pre-trained weights.")

    # settings
    n_graphs = int(1e7)
    n_graphs_per_sim = int(10000)
    seed = 747
    p = 1e-3
    batch_size = n_graphs_per_sim if "cuda" in device.type else 2000
    
    m_nearest_nodes = model_data["graph_settings"]["m_nearest_nodes"]

    # if we want to run inference on many graphs, do so in batches
    if n_graphs > n_graphs_per_sim:
        n_partitions = n_graphs // n_graphs_per_sim
        remaining = n_graphs % n_graphs_per_sim
    else:
        n_partitions = 0
        remaining = n_graphs

    # read code distance and number of repetitions from file name
    file_name = model_path.name
    splits = file_name.split("_")
    code_sz = int(splits[0][1])
    reps = int(splits[3].split(".")[0])

    # go through partitions
    correct_preds = 0
    n_trivial = 0
    for i in range(n_partitions):
        sim = SurfaceCodeSim(
            reps,
            code_sz,
            p,
            n_shots=n_graphs_per_sim,
            seed=seed + i,
        )

        syndromes, flips, n_identities = sim.generate_syndromes()
        flips = torch.tensor(flips[:, None], dtype=torch.float32).to(device)
        
        # add identities to # trivial predictions
        n_trivial += n_identities

        # run inference
        _correct_preds, _ = run_inference(model, syndromes, flips, m_nearest_nodes=m_nearest_nodes, device=device)
        correct_preds += _correct_preds
    # run the remaining graphs
    if remaining > 0:
        sim = SurfaceCodeSim(
            reps,
            code_sz,
            p,
            n_shots=remaining,
            seed=seed - 1,
        )

        syndromes, flips, n_identities = sim.generate_syndromes()
        flips = torch.tensor(flips[:, None], dtype=torch.float32).to(device)
        
        # add identities to # trivial predictions
        n_trivial += n_identities

        _correct_preds, _ = run_inference(model, syndromes, flips, m_nearest_nodes=m_nearest_nodes, device=device)
        correct_preds += _correct_preds

    # compute logical failure rate
    failure_rate = (n_graphs - correct_preds - n_trivial) / n_graphs
    print(f"We have a logical failure rate of {failure_rate}.")
    return 0

if __name__ == "__main__":
    main()
