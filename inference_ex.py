import argparse
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.gnn_models import GNN_7
from src.simulations import SurfaceCodeSim
from src.graph_representation import get_3D_graph
from pathlib import Path


def main():
    # command line parsing
    parser = argparse.ArgumentParser(description="Choose model to load.")
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-d", "--device", required=False)
    args = parser.parse_args()

    model_path = Path(args.file)
    if args.device:
        device = args.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # settings
    n_graphs = 1000
    seed = 11
    p = 3e-3
    batch_size = n_graphs
    power = 1

    # read code distance and number of repetitions from file name
    file_name = model_path.name
    splits = file_name.split("_")
    code_sz = int(splits[0][1])
    reps = int(splits[3].split(".")[0])

    sim = SurfaceCodeSim(reps, code_sz, p, n_shots=n_graphs, seed=seed)

    syndromes, flips = sim.generate_syndromes(n_graphs)

    graphs = []
    for syndrome, flip in zip(syndromes, flips):
        graph = get_3D_graph(
            syndrome_3D=syndrome, target=flip, power=power, m_nearest_nodes=5
        )

        graphs.append(
            Data(
                x=torch.from_numpy(graph[0]),
                edge_index=torch.from_numpy(graph[1]),
                edge_attr=torch.from_numpy(graph[2]),
                y=torch.from_numpy(graph[3]),
            )
        )
    loader = DataLoader(graphs, batch_size=batch_size)

    # load model
    if model_path.is_file():
        model_data = torch.load(model_path, map_location=device)
    else:
        FileNotFoundError("The file was not found!")
        return 1

    model = GNN_7().to(device)
    model.load_state_dict(model_data["model"])
    model.eval()

    sigmoid = nn.Sigmoid()
    correct_preds = 0
    # run inference on simulated data
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_label = batch.batch.to(device)
            target = batch.y.to(device).int()
            
            out = model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch_label,
            )
            
            prediction = (sigmoid(out.detach()) > 0.5).long()
            correct_preds += int((prediction == target).sum())
    
    accuracy = correct_preds / n_graphs
    print(f"We have an accuracy of {accuracy:.2f}.")

    return 0


if __name__ == "__main__":
    main()
