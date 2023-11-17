import argparse
import torch
import torch.nn as nn
import torch_geometric.nn as nn_g
import torch.ao.quantization as tq
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
    parser.add_argument("-q", "--quantize", required=False, action="store_true")

    args = parser.parse_args()

    model_path = Path(args.file)
    if args.device:
        device = args.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    if model_path.is_file():
        model_data = torch.load(model_path, map_location=device)
    else:
        raise FileNotFoundError("The file was not found!")

    # settings
    n_graphs = 5000
    seed = 11
    p = 1e-3
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
        x, edge_index, edge_attr, y = get_3D_graph(
            syndrome_3D=syndrome, target=flip, power=power, m_nearest_nodes=3
        )
        graphs.append(Data(x, edge_index, edge_attr, y))
    loader = DataLoader(graphs, batch_size=batch_size)

    model = GNN_7().to(device)
    model.load_state_dict(model_data["model"])
    model.eval()

    # quantize model
    if args.quantize:
        model.qconfig = tq.get_default_qconfig("x86")
        model_prepared = tq.prepare(model)
        
        # calibrate
        batch = next(iter(loader))
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_label = batch.batch
        target = batch.y
        model_prepared(x, edge_index, edge_attr, batch_label)
        
        # convert
        model_int8 = tq.convert(model_prepared)
        

    sigmoid = nn.Sigmoid()
    correct_preds = 0
    # run inference on simulated data
    with torch.no_grad():
        for batch in loader:

            if args.quantize:
                x = torch.quantize_per_tensor(batch.x.to(device), 0.1, 8, torch.quint8)
                edge_index = batch.edge_index.to(device)
                edge_attr = torch.quantize_per_tensor(batch.edge_attr.to(device), 0.1, 8, torch.quint8)
                batch_label = batch.batch.to(device)
                
                out = model_int8(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch_label,
                )
                out = torch.dequantize(out)
            else:  
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                edge_attr = batch.edge_attr.to(device)
                batch_label = batch.batch.to(device)
                target = batch.y.to(device).int()
                
            prediction = (sigmoid(out.detach()) > 0.5).long()
            correct_preds += int((prediction == target).sum())

    accuracy = correct_preds / n_graphs
    print(f"We have an accuracy of {accuracy:.2f}.")

    return 0


if __name__ == "__main__":
    main()
