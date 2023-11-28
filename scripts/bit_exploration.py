import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from pathlib import Path
import sys
sys.path.append("..")

from src.simulations import SurfaceCodeSim
from src.graph_representation import get_3D_graph
from src.gnn_models import GNN_7
from src.utils import (
    match_and_load_state_dict,
    get_scale,
    get_zero_pt,
    quantize_model_layers,
    dequantize_model_layers,
    run_inference,
    get_all_weights,
    quantize_tensor,
    dequantize_tensor,
)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from icecream import ic
ic.configureOutput(includeContext=True)


def explore_weights(
    float_model: nn.Module,
    code_sz: int,
    reps: int,
    p: float,
    min_bits: int,
    max_bits: int,
    n_graphs: int,
    n_graphs_per_sim: int,
    m_nearest_nodes: int = 5,
    batch_size: int = 5000,
    seed: int = None,
    device: torch.device = torch.device("cuda"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # get weights in FP32
    all_weights = get_all_weights(float_model)
    qt = 0.00
    lower_limit = torch.tensor(np.quantile(all_weights, qt))
    upper_limit = torch.tensor(np.quantile(all_weights, 1 - qt))

    # collect data for all iterations
    bit_widths = np.arange(min_bits, max_bits + 1, step=2, dtype=np.int64)
    bit_predictions_data = np.zeros((len(bit_widths), 2))
    float_predictions_data = np.zeros((2, 1))
    q_errors = np.zeros((len(bit_widths),))
    
    # if we want to generate many graphs, do so in chunks
    if n_graphs > n_graphs_per_sim:
        n_partitions = n_graphs // n_graphs_per_sim
        remaining = n_graphs % n_graphs_per_sim
    else:
        n_partitions = 0
        remaining = n_graphs
    
    # go through partitions
    for i in range(n_partitions):
        # print(f"Running partition {i + 1} of {n_partitions}.")
        sim = SurfaceCodeSim(
            reps,
            code_sz,
            p,
            n_shots=n_graphs_per_sim,
            seed=seed + i,
        )

        # generate syndromes and save number of trivial syndromes
        syndromes, flips, n_identities = sim.generate_syndromes()
        bit_predictions_data[:, 1] += n_identities
        float_predictions_data[1] += n_identities
        
        graphs = []
        for syndrome, flip in zip(syndromes, flips):
            x, edge_index, edge_attr, y = get_3D_graph(
                syndrome_3D=syndrome,
                target=flip,
                m_nearest_nodes=m_nearest_nodes,
                power=2.0,
            )
            graphs.append(Data(x, edge_index, edge_attr, y))
        loader = DataLoader(graphs, batch_size=batch_size)

        count = 0
        for bit_width in bit_widths:
            
            # load model and quantize/dequantize it to the given bit_width
            model = GNN_7().to(device)
            model.load_state_dict(float_model.state_dict())
            scale = get_scale(lower_limit, upper_limit, bit_width)
            zero_pt = get_zero_pt(lower_limit, upper_limit, bit_width)

            quantized_layers, scale, zero_pt = quantize_model_layers(
                model,
                bit_width,
                scale=scale,
                zero_pt=zero_pt,
                same_quantization=True,
            )
            dequantized_layers = dequantize_model_layers(model, scale, zero_pt)
            
            # run inference and add #correct predictions to data array
            bit_predictions_data[count, 0] += run_inference(model, loader, device)
            
            # save quantization error
            dq_weights = get_all_weights(model)
            q_errors[count] = np.linalg.norm(all_weights - dq_weights)
            count += 1
            
        # before running next partition we check how the floating point model performs
        float_predictions_data[0] += run_inference(float_model, loader, device)
    
    # run the remaining parts
    sim = SurfaceCodeSim(
            reps,
            code_sz,
            p,
            n_shots=remaining,
            seed=seed - 1,
        )

    # generate syndromes and save number of trivial syndromes
    syndromes, flips, n_identities = sim.generate_syndromes()
    bit_predictions_data[:, 1] += n_identities
    float_predictions_data[1] += n_identities
    
    graphs = []
    for syndrome, flip in zip(syndromes, flips):
        x, edge_index, edge_attr, y = get_3D_graph(
            syndrome_3D=syndrome,
            target=flip,
            m_nearest_nodes=m_nearest_nodes,
            power=2.0,
        )
        graphs.append(Data(x, edge_index, edge_attr, y))
    loader = DataLoader(graphs, batch_size=batch_size)

    count = 0
    for bit_width in bit_widths:
        
        # load model and quantize/dequantize it to the given bit_width
        model = GNN_7().to(device)
        model.load_state_dict(float_model.state_dict())
        scale = get_scale(lower_limit, upper_limit, bit_width)
        zero_pt = get_zero_pt(lower_limit, upper_limit, bit_width)

        quantized_layers, scale, zero_pt = quantize_model_layers(
            model,
            bit_width,
            scale=scale,
            zero_pt=zero_pt,
            same_quantization=True,
        )
        dequantized_layers = dequantize_model_layers(model, scale, zero_pt)
        
        # run inference and add #correct predictions to data array
        bit_predictions_data[count, 0] += run_inference(model, loader, device)
        
        # save quantization error
        dq_weights = get_all_weights(model)
        q_errors[count] = np.linalg.norm(all_weights - dq_weights)
        count += 1
        
    # add final floating point predictions
    float_predictions_data[0] += run_inference(float_model, loader, device)
        
    # when all partitions are finished we can compute logical failure rates
    failure_rate = (np.ones((len(bit_widths), 1)) * n_graphs - bit_predictions_data.sum(axis=1, keepdims=True)) / n_graphs
    failure_rate_fp_model = (n_graphs - float_predictions_data.sum()) / n_graphs

    return failure_rate, failure_rate_fp_model, q_errors
  
def explore_data(
    model: nn.Module,
    code_sz: int,
    reps: int,
    p: float,
    min_bits: int,
    max_bits: int,
    n_graphs: int,
    n_graphs_per_sim: int,
    m_nearest_nodes: int = 5,
    batch_size: int = 5000,
    seed: int = None,
    device: torch.device = torch.device("cuda")
) -> list[tuple[np.ndarray, np.ndarray]]:
    
    # collect data for all iterations
    bit_widths = np.arange(min_bits, max_bits + 1, step=2, dtype=np.int64)
    bit_predictions_data = np.zeros((len(bit_widths), 2))
    float_predictions_data = np.zeros((2, 1))
    q_errors = np.zeros((len(bit_widths), 1))
    q_batch_error = []
    
    # if we want to generate many graphs, do so in chunks
    if n_graphs > n_graphs_per_sim:
        n_partitions = n_graphs // n_graphs_per_sim
        remaining = n_graphs % n_graphs_per_sim
    else:
        n_partitions = 0
        remaining = n_graphs
    
    # go through partitions
    sigmoid = nn.Sigmoid()
    for i in range(n_partitions):
        sim = SurfaceCodeSim(
            reps,
            code_sz,
            p,
            n_shots=n_graphs_per_sim,
            seed=seed + i,
        )

        # generate syndromes and save number of trivial syndromes
        syndromes, flips, n_identities = sim.generate_syndromes()
        bit_predictions_data[:, 1] += n_identities
        float_predictions_data[1] += n_identities
        
        graphs = []
        for syndrome, flip in zip(syndromes, flips):
            x, edge_index, edge_attr, y = get_3D_graph(
                syndrome_3D=syndrome,
                target=flip,
                m_nearest_nodes=m_nearest_nodes,
                power=2.0,
            )
            graphs.append(Data(x, edge_index, edge_attr, y))
        loader = DataLoader(graphs, batch_size=batch_size)

        with torch.no_grad():
            for batch in loader:
                
                # unzip data
                x = batch.x.to(device)
                edge_index = batch.edge_index.to(device)
                edge_attr = batch.edge_attr.to(device)
                batch_label = batch.batch.to(device)

                # quantize and dequantize data
                data = np.concatenate(
                    [x.cpu().numpy().flatten(), edge_attr.cpu().numpy().flatten()]
                )
                lower_limit = torch.tensor(data.min())
                upper_limit = torch.tensor(data.max())
            
                count = 0
                for bit_width in bit_widths:
                    scale = get_scale(lower_limit, upper_limit, bit_width)
                    zero_pt = get_zero_pt(lower_limit, upper_limit, bit_width)
                    x = quantize_tensor(x, scale, zero_pt, bit_width)
                    x = dequantize_tensor(x, scale, zero_pt)
                    edge_attr = quantize_tensor(edge_attr, scale, zero_pt, bit_width)
                    edge_attr = dequantize_tensor(edge_attr, scale, zero_pt)

                    dq_data = np.concatenate(
                        [x.cpu().numpy().flatten(), edge_attr.cpu().numpy().flatten()]
                    )
                    q_errors[count] = np.linalg.norm(data - dq_data)
                    
                    out = model(
                        x,
                        edge_index,
                        edge_attr,
                        batch_label,
                    )
                    prediction = (sigmoid(out.detach()) > 0.5).long()
                    target = batch.y.to(device).int()
                    bit_predictions_data[count, 0] += int((prediction == target).sum())
                    count += 1
                    
                q_batch_error.append(q_errors)
                    
        float_predictions_data[0] += run_inference(model, loader, device)
        
    # run the remaining parts
    sim = SurfaceCodeSim(
            reps,
            code_sz,
            p,
            n_shots=remaining,
            seed=seed - 1,
        )

    # generate syndromes and save number of trivial syndromes
    syndromes, flips, n_identities = sim.generate_syndromes()
    bit_predictions_data[:, 1] += n_identities
    float_predictions_data[1] += n_identities
    
    graphs = []
    for syndrome, flip in zip(syndromes, flips):
        x, edge_index, edge_attr, y = get_3D_graph(
            syndrome_3D=syndrome,
            target=flip,
            m_nearest_nodes=m_nearest_nodes,
            power=2.0,
        )
        graphs.append(Data(x, edge_index, edge_attr, y))
    loader = DataLoader(graphs, batch_size=batch_size)
    
    count = 0
    with torch.no_grad():
        for batch in loader:
            
            # unzip data
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_label = batch.batch.to(device)

            # quantize and dequantize data
            data = np.concatenate(
                [x.cpu().numpy().flatten(), edge_attr.cpu().numpy().flatten()]
            )
            lower_limit = torch.tensor(data.min())
            upper_limit = torch.tensor(data.max())
        
            count = 0
            for bit_width in bit_widths:
                scale = get_scale(lower_limit, upper_limit, bit_width)
                zero_pt = get_zero_pt(lower_limit, upper_limit, bit_width)
                x = quantize_tensor(x, scale, zero_pt, bit_width)
                x = dequantize_tensor(x, scale, zero_pt)
                edge_attr = quantize_tensor(edge_attr, scale, zero_pt, bit_width)
                edge_attr = dequantize_tensor(edge_attr, scale, zero_pt)

                dq_data = np.concatenate(
                    [x.cpu().numpy().flatten(), edge_attr.cpu().numpy().flatten()]
                )
                q_errors[count] = np.linalg.norm(data - dq_data)
                
                out = model(
                    x,
                    edge_index,
                    edge_attr,
                    batch_label,
                )
                prediction = (sigmoid(out.detach()) > 0.5).long()
                target = batch.y.to(device).int()
                bit_predictions_data[count, 0] += int((prediction == target).sum())
                count += 1
                
            q_batch_error.append(q_errors)
    # add final floating point predictions
    float_predictions_data[0] += run_inference(model, loader, device)
                
    # when all partitions are finished we can compute logical failure rates
    failure_rate = (np.ones((len(bit_widths), 1)) * n_graphs - bit_predictions_data.sum(axis=1, keepdims=True)) / n_graphs
    failure_rate_fp_model = (n_graphs - float_predictions_data.sum()) / n_graphs
    
    q_batch_error = np.array(q_batch_error).mean(axis=0)
    return failure_rate, failure_rate_fp_model, q_batch_error

def main():
    experiment = "data"

    paths = [
        Path("../models/circuit_level_noise/d3/d3_d_t_5.pt"),
        Path("../models/circuit_level_noise/d5/d5_d_t_5.pt"),
        Path("../models/circuit_level_noise/d7/d7_d_t_5.pt"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_graphs = int(1e7)
    n_graphs_per_sim = int(5e4)
    batch_size = n_graphs_per_sim if "cuda" in device.type else 4000
    p = 1e-3
    min_bits = 2
    max_bits = 16
    
    # must use seed to make sure code distances are comparable
    seed = 747
    
    # collect data for each code size
    data_per_code_sz = []
    for path in paths:
        model_path = Path.cwd() / path

        # read code distance and number of repetitions from file name
        file_name = model_path.name
        splits = file_name.split("_")
        code_sz = int(splits[0][1])
        reps = int(splits[3].split(".")[0])
    
        # initialise floating point model and extract weight statistics
        float_model = GNN_7().to(device)
        trained_weights = torch.load(model_path, map_location=device)["model"]
        float_model = match_and_load_state_dict(float_model, trained_weights)
        float_model.eval()

        print(f"Running bit exploration for {n_graphs} graphs with maximal batch size {batch_size}.")
        if experiment == "weights":
            failure_rate, failure_rate_fp_model, q_error = explore_weights(
                float_model,
                code_sz,
                reps,
                p,
                min_bits,
                max_bits,
                n_graphs,
                n_graphs_per_sim,
                batch_size=batch_size,
                seed=seed,
                device=device,
            )
            print(f"Code size: {code_sz}, fr: {failure_rate_fp_model}")
        
        elif experiment == "data":
            failure_rate, failure_rate_fp_model, q_error = explore_data(
                float_model,
                code_sz,
                reps,
                p,
                min_bits,
                max_bits,
                n_graphs,
                n_graphs_per_sim,
                batch_size=batch_size,
                seed = seed,
                device=device,
            )
        else:
            print("You need to provide a valid experiment.")
            return

        data_per_code_sz.append((failure_rate, failure_rate_fp_model, q_error))

    fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
    fig_qerr, ax_qerr = plt.subplots(figsize=(12, 8))

    x = np.arange(min_bits, max_bits + 1, step=2)
    colors = ["r", "b", "g"]
    code_sz = [3, 5, 7]
    for i, data in enumerate(data_per_code_sz):
        failure_rate, failure_rate_fp_model, q_error = data

        ax_acc.axhline(
            failure_rate_fp_model,
            0,
            max_bits,
            linestyle="--",
            color=colors[i],
            label=f"FP32 logical failure rate, d={code_sz[i]}"
        )
        ax_acc.semilogy(
            x,
            failure_rate,
            color=colors[i],
            label=f"d = {code_sz[i]}",
        )

        ax_qerr.semilogy(
            x,
            q_error,
            color=colors[i],
            label=f"d = {code_sz[i]}",
        )

    ax_acc.set_xlabel("# bits")
    ax_acc.set_ylabel("Logical failure rate")
    ax_acc.legend(loc="upper right")
    ax_acc.set_title(f"Quantization of {experiment}")

    ax_qerr.set_xlabel("# bits")
    ax_qerr.set_ylabel("Quantization error")
    ax_qerr.set_title(f"Quantization of {experiment}")

    fig_acc.tight_layout()
    fig_qerr.tight_layout()

    fig_acc.savefig(f"../figures/bit_accuracies_{experiment}.pdf")
    fig_qerr.savefig(f"../figures/bit_qerror_{experiment}.pdf")


if __name__ == "__main__":
    main()
