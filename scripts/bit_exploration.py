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
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # get weights in FP32
    all_weights = get_all_weights(float_model)
    qt = 0.00
    lower_limit = torch.tensor(np.quantile(all_weights, qt))
    upper_limit = torch.tensor(np.quantile(all_weights, 1 - qt))

    # collect data for all iterations
    bit_widths = np.arange(min_bits, max_bits + 1, dtype=np.int64)
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
    correct_preds = 0
    n_trivial = 0
    for _ in range(n_partitions):
        sim = SurfaceCodeSim(
            reps,
            code_sz,
            p,
            n_shots=n_graphs_per_sim,
            seed=seed,
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
            seed=seed,
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
        
    # when all partitions are finished we can compute logical failure rates
    failure_rate = (np.ones((len(bit_widths), 1)) * n_graphs - bit_predictions_data.sum(axis=1, keepdims=True)) / n_graphs
    failure_rate_fp_model = (n_graphs - float_predictions_data.sum()) / n_graphs
    
    return failure_rate, failure_rate_fp_model, q_errors
        
        
    # run the remaining graphs
    sim = SurfaceCodeSim(
        reps,
        code_sz,
        p,
        n_shots=remaining,
        seed=seed,
    )

    syndromes, flips, n_identities = sim.generate_syndromes()
    # add identities to # trivial predictions
    n_trivial += n_identities
    n_untrivial += syndromes.shape[0]

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
    correct_preds += run_inference(model, loader, device)
    
    # compute logical failure rate
    failure_rate = (n_graphs - correct_preds - n_trivial) / n_graphs
    
    # return failure_rate

    
    for i in range(its):
        # create simulation data
        sim = SurfaceCodeSim(reps, code_sz, p, n_shots=n_graphs, seed=seed)
        syndromes, flips, n_trivial_preds = sim.generate_syndromes()

        graphs = []
        for syndrome, flip in zip(syndromes, flips):
            x, edge_index, edge_attr, y = get_3D_graph(
                syndrome_3D=syndrome, target=flip, m_nearest_nodes=5, power=2.0,
            )
            graphs.append(Data(x, edge_index, edge_attr, y))
        loader = DataLoader(graphs, batch_size=batch_size)

        # loop over bits
        bit_widths = np.arange(1, max_bits + 1, dtype=np.int64)
        accuracies = []
        q_errors = []
        for bit_width in bit_widths:
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
            dq_weights = get_all_weights(model)

            q_error = np.linalg.norm(all_weights - dq_weights)
            accuracy, n_correct_preds = run_inference(
                model,
                loader,
                n_graphs,
                n_trivial_preds,
            )

            q_errors.append(q_error)
            accuracies.append(accuracy)

        float_accuracy, n_float_correct_preds = run_inference(
            float_model,
            loader,
            n_graphs,
            n_trivial_preds,
            device=device,
        )
        accuracies = np.array(accuracies)
        q_errors = np.array(q_errors)

        accuracy_data[:, i] = accuracies
        q_error_data[:, i] = q_errors

    
    return accuracy_data, q_error_data, float_accuracy


def explore_data(
    model: nn.Module,
    code_sz: int,
    reps: int,
    p: float,
    min_bits: int,
    max_bits: int,
    n_graphs: int,
    n_graphs_per_sim: int,
    batch_size: int = 5000,
    seed: int = None,
    device: torch.device = torch.device("cpu")
) -> list[tuple[np.ndarray, np.ndarray]]:
    return 0
    # collect data for all iterations
    accuracy_data = np.zeros((max_bits, its))
    q_error_data = np.zeros((max_bits, its))

    sigmoid = nn.Sigmoid()
    # inform user of iterations
    print(f"Running inference for code distance {code_sz}.")
    for i in tqdm(range(its)):
        # create simulation data
        sim = SurfaceCodeSim(reps, code_sz, p, n_shots=n_graphs, seed=seed)
        syndromes, flips, n_trivial_preds = sim.generate_syndromes()

        graphs = []
        for syndrome, flip in zip(syndromes, flips):
            x, edge_index, edge_attr, y = get_3D_graph(
                syndrome_3D=syndrome, target=flip, m_nearest_nodes=5, power=2.0,
            )

            graphs.append(Data(x, edge_index, edge_attr, y))
        loader = DataLoader(graphs, batch_size=batch_size)

        # loop over bits
        bit_widths = np.arange(1, max_bits + 1, dtype=np.int64)
        accuracies = []
        q_errors = []
        for bit_width in bit_widths:
            correct_preds = 0
            q_batch_errors = []
            # loop over batches
            with torch.no_grad():
                for batch in tqdm(loader):
                    # unzip data
                    x = batch.x
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_attr
                    batch_label = batch.batch

                    # quantize and dequantize data
                    data = np.concatenate(
                        [x.numpy().flatten(), edge_attr.numpy().flatten()]
                    )

                    lower_limit = torch.tensor(data.min())
                    upper_limit = torch.tensor(data.max())
                    scale = get_scale(lower_limit, upper_limit, bit_width)
                    zero_pt = get_zero_pt(lower_limit, upper_limit, bit_width)
                    x = quantize_tensor(x, scale, zero_pt, bit_width)
                    x = dequantize_tensor(x, scale, zero_pt)
                    edge_attr = quantize_tensor(edge_attr, scale, zero_pt, bit_width)
                    edge_attr = dequantize_tensor(edge_attr, scale, zero_pt)

                    dq_data = np.concatenate(
                        [x.numpy().flatten(), edge_attr.numpy().flatten()]
                    )
                    q_batch_errors.append(np.linalg.norm(data - dq_data))

                    out = model(
                        x,
                        edge_index,
                        edge_attr,
                        batch_label,
                    )

                    prediction = (sigmoid(out.detach()) > 0.5).long()
                    target = batch.y.int()
                    correct_preds += int((prediction == target).sum())

            accuracy = (n_graphs - correct_preds - n_trivial_preds) / n_graphs
            q_error = np.array(q_batch_errors).mean()

            accuracies.append(accuracy)
            q_errors.append(q_error)

        float_accuracy, _ = run_inference(
            model,
            loader,
            n_graphs,
            n_trivial_preds,
        )
        accuracies = np.array(accuracies)
        q_errors = np.array(q_errors)

        accuracy_data[:, i] = accuracies
        q_error_data[:, i] = q_errors

    return accuracy_data, q_error_data, float_accuracy


def main():
    experiment = "weights"

    paths = [
        Path("../models/circuit_level_noise/d3/d3_d_t_5.pt"),
        Path("../models/circuit_level_noise/d5/d5_d_t_5.pt"),
        Path("../models/circuit_level_noise/d7/d7_d_t_5.pt"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_graphs = int(1e7)
    n_graphs_per_sim = int(5e4)
    batch_size = 80000 if "cuda" in device.type else 4000
    p = 1e-3
    min_bits = 2
    max_bits = 16

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
                batch_size,
                device=device,
            )
        
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
                batch_size,
                device=device,
            )
        else:
            print("You need to provide a valid experiment.")
            return

        data_per_code_sz.append((failure_rate, failure_rate_fp_model, q_error))

    fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
    fig_qerr, ax_qerr = plt.subplots(figsize=(12, 8))

    x = range(min_bits, max_bits + 1)
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
        )

        ax_qerr.semilogy(
            x,
            q_error,
            color=colors[i],
        )

    ax_acc.set_xlabel("# bits")
    ax_acc.set_ylabel("Logical failure rate")
    ax_acc.legend(loc="lower right")
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
