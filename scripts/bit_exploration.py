import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse

from pathlib import Path
import sys

sys.path.append("..")

from src.simulations import SurfaceCodeSim
from src.graph_representation import get_3D_graph
from src.gnn_models import GNN_7, QGNN_7
from src.utils import (
    match_and_load_state_dict,
    get_scale,
    get_zero_pt,
    quantize_model_layers,
    dequantize_model_layers,
    run_inference_old,
    get_all_weights,
    quantize_tensor,
    dequantize_tensor,
    fixed_precision_model_layers,
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

            scale, zero_pt = quantize_model_layers(
                model,
                bit_width,
                scale=scale,
                zero_pt=zero_pt,
                same_quantization=True,
            )
            dequantize_model_layers(model, scale, zero_pt)

            # run inference and add #correct predictions to data array
            bit_predictions_data[count, 0] += run_inference_old(model, loader, device=device)

            # save quantization error
            dq_weights = get_all_weights(model)
            q_errors[count] = np.linalg.norm(all_weights - dq_weights)
            count += 1

        # before running next partition we check how the floating point model performs
        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)

    # run the remaining parts
    if remaining > 0:
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

            scale, zero_pt = quantize_model_layers(
                model,
                bit_width,
                scale=scale,
                zero_pt=zero_pt,
                same_quantization=True,
            )
            dequantize_model_layers(model, scale, zero_pt)

            # run inference and add #correct predictions to data array
            bit_predictions_data[count, 0] += run_inference_old(model, loader, device=device)

            # save quantization error
            dq_weights = get_all_weights(model)
            q_errors[count] = np.linalg.norm(all_weights - dq_weights)
            count += 1

        # add final floating point predictions
        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)

    # when all partitions are finished we can compute logical failure rates
    failure_rate = (
        np.ones((len(bit_widths), 1)) * n_graphs
        - bit_predictions_data.sum(axis=1, keepdims=True)
    ) / n_graphs
    failure_rate_fp_model = (n_graphs - float_predictions_data.sum()) / n_graphs

    return failure_rate, failure_rate_fp_model, q_errors


def explore_fixed_pt_weights(
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
            # load model and quantize/dequantize it to the given fixed bit_width
            model = GNN_7().to(device)
            model.load_state_dict(float_model.state_dict())

            fixed_precision_model_layers(model, bit_width)

            # run inference and add #correct predictions to data array
            bit_predictions_data[count, 0] += run_inference_old(model, loader, device=device)

            # save quantization error
            dq_weights = get_all_weights(model)
            q_errors[count] = np.linalg.norm(all_weights - dq_weights)
            count += 1

        # before running next partition we check how the floating point model performs
        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)

    # run the remaining parts
    if remaining > 0:
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

            fixed_precision_model_layers(model, bit_width)

            # run inference and add #correct predictions to data array
            bit_predictions_data[count, 0] += run_inference_old(model, loader, device=device)

            # save quantization error
            dq_weights = get_all_weights(model)
            q_errors[count] = np.linalg.norm(all_weights - dq_weights)
            count += 1

        # add final floating point predictions
        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)

    # when all partitions are finished we can compute logical failure rates
    failure_rate = (
        np.ones((len(bit_widths), 1)) * n_graphs
        - bit_predictions_data.sum(axis=1, keepdims=True)
    ) / n_graphs
    failure_rate_fp_model = (n_graphs - float_predictions_data.sum()) / n_graphs

    return failure_rate, failure_rate_fp_model, q_errors


def explore_weights_per_layer(
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

    # create list of layers to quantize
    all_names = list(float_model.state_dict().keys())
    graph_weights = (
        np.array([name for name in all_names if "graph" in name and "weight" in name])
        .reshape(-1, 2)
        .tolist()
    )
    graph_bias = [[name] for name in all_names if "graph" in name and "bias" in name]
    lin_weights = [
        [name] for name in all_names if not "graph" in name and "weight" in name
    ]
    lin_bias = [[name] for name in all_names if not "graph" in name and "bias" in name]
    layers = graph_weights + graph_bias + lin_weights + lin_bias

    # collect data for all iterations
    bit_widths = np.arange(min_bits, max_bits + 1, step=2, dtype=np.int64)
    layer_predictions_data = np.zeros((len(bit_widths), len(layers), 2))
    float_predictions_data = np.zeros((2, 1))

    # if we want to generate many graphs, do so in chunks
    if n_graphs > n_graphs_per_sim:
        n_partitions = n_graphs // n_graphs_per_sim
        remaining = n_graphs % n_graphs_per_sim
    else:
        n_partitions = 0
        remaining = n_graphs

    # go through partitions
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
        layer_predictions_data[..., 1] += n_identities
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

        layer_count = 0
        for layer in layers:
            bit_count = 0
            for bit_width in bit_widths:
                # load model and quantize/dequantize it to the given bit_width
                model = GNN_7().to(device)
                model.load_state_dict(float_model.state_dict())
                scale = get_scale(lower_limit, upper_limit, bit_width)
                zero_pt = get_zero_pt(lower_limit, upper_limit, bit_width)

                scale, zero_pt = quantize_model_layers(
                    model,
                    bit_width,
                    scale=scale,
                    zero_pt=zero_pt,
                    layer_names=layer,
                    same_quantization=True,
                )
                dequantize_model_layers(model, scale, zero_pt, layer_names=layer)

                # run inference and add #correct predictions to data array
                layer_predictions_data[bit_count, layer_count, 0] += run_inference_old(model, loader, device=device)
                bit_count += 1
            layer_count += 1

        # before running next partition we check how the floating point model performs
        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)
        
    # run the remaining parts
    if remaining > 0:
        sim = SurfaceCodeSim(
            reps,
            code_sz,
            p,
            n_shots=remaining,
            seed=seed - 1,
        )

        # generate syndromes and save number of trivial syndromes
        syndromes, flips, n_identities = sim.generate_syndromes()
        layer_predictions_data[..., 1] += n_identities
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

        layer_count = 0
        for layer in layers:
            bit_count = 0
            for bit_width in bit_widths:
                # load model and quantize/dequantize it to the given bit_width
                model = GNN_7().to(device)
                model.load_state_dict(float_model.state_dict())
                scale = get_scale(lower_limit, upper_limit, bit_width)
                zero_pt = get_zero_pt(lower_limit, upper_limit, bit_width)

                scale, zero_pt = quantize_model_layers(
                    model,
                    bit_width,
                    scale=scale,
                    zero_pt=zero_pt,
                    layer_names=layer,
                    same_quantization=True,
                )
                dequantize_model_layers(model, scale, zero_pt, layer_names=layer)

                # run inference and add #correct predictions to data array
                layer_predictions_data[bit_count, layer_count, 0] += run_inference_old(model, loader, device=device)
                bit_count += 1
            layer_count += 1

        # add final floating point predictions
        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)

    # when all partitions are finished we can compute logical failure rates
    failure_rate = ((
        np.ones((len(bit_widths), len(layers), 1)) * n_graphs
        - layer_predictions_data.sum(axis=-1, keepdims=True)
    ) / n_graphs).squeeze()
    
    # want to check how many bits required to match accuracy of fp-model
    failure_rate_fp_model = (n_graphs - float_predictions_data.sum()) / n_graphs
    below_or_equal = failure_rate <= failure_rate_fp_model

    min_req_bits = np.tile(bit_widths, (1, len(layers)))[:, np.argmax(below_or_equal, axis=0)]
    
    return min_req_bits.squeeze(), failure_rate_fp_model


def explore_data(
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
) -> list[tuple[np.ndarray, np.ndarray]]:
    
    # collect data for all iterations
    bit_widths = np.arange(min_bits, max_bits + 1, step=2, dtype=np.int64)
    bit_predictions_data = np.zeros((len(bit_widths), 2))
    float_predictions_data = np.zeros((2, 1))

    # if we want to generate many graphs, do so in chunks
    if n_graphs > n_graphs_per_sim:
        n_partitions = n_graphs // n_graphs_per_sim
        remaining = n_graphs % n_graphs_per_sim
    else:
        n_partitions = 0
        remaining = n_graphs
        
    # create model to do data quantization
    model = QGNN_7().to(device)
    model.load_state_dict(float_model.state_dict())
    
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

                count = 0
                for bit_width in bit_widths:

                    out = model(
                        x,
                        edge_index,
                        edge_attr,
                        batch_label,
                        bit_width
                    )
                    prediction = (sigmoid(out.detach()) > 0.5).long()
                    target = batch.y.to(device).int()
                    bit_predictions_data[count, 0] += int((prediction == target).sum())
                    count += 1

        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)

    # run the remaining parts
    if remaining > 0:
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

                count = 0
                for bit_width in bit_widths:

                    out = model(
                        x,
                        edge_index,
                        edge_attr,
                        batch_label,
                        bit_width,
                    )
                    prediction = (sigmoid(out.detach()) > 0.5).long()
                    target = batch.y.to(device).int()
                    bit_predictions_data[count, 0] += int((prediction == target).sum())
                    count += 1

        # add final floating point predictions
        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)

    # when all partitions are finished we can compute logical failure rates
    failure_rate = (
        np.ones((len(bit_widths), 1)) * n_graphs
        - bit_predictions_data.sum(axis=1, keepdims=True)
    ) / n_graphs
    failure_rate_fp_model = (n_graphs - float_predictions_data.sum()) / n_graphs

    return failure_rate, failure_rate_fp_model

def explore_data_and_weights(
    float_model: nn.Module,
    code_sz: int,
    reps: int,
    p: float,
    min_bits: int,
    max_bits: int,
    const_bit_w: int,
    n_graphs: int,
    n_graphs_per_sim: int,
    m_nearest_nodes: int = 5,
    batch_size: int = 5000,
    seed: int = None,
    device: torch.device = torch.device("cuda"),
) -> list[tuple[np.ndarray, np.ndarray]]:
    
    # get weights in FP32
    all_weights = get_all_weights(float_model)
    qt = 0.00
    lower_limit = torch.tensor(np.quantile(all_weights, qt))
    upper_limit = torch.tensor(np.quantile(all_weights, 1 - qt))
    
    # collect data for all iterations
    bit_widths = np.arange(min_bits, max_bits + 1, step=2, dtype=np.int64)
    bit_predictions_data = np.zeros((len(bit_widths), 2))
    float_predictions_data = np.zeros((2, 1))

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

                count = 0
                for bit_width in bit_widths:
                    # create model to do data quantization
                    model = QGNN_7().to(device)
                    model.load_state_dict(float_model.state_dict())
                    scale = get_scale(lower_limit, upper_limit, bit_width)
                    zero_pt = get_zero_pt(lower_limit, upper_limit, bit_width)

                    scale, zero_pt = quantize_model_layers(
                        model,
                        bit_width,
                        scale=scale,
                        zero_pt=zero_pt,
                        same_quantization=True,
                    )
                    dequantize_model_layers(model, scale, zero_pt)
                    
                    out = model(
                        x,
                        edge_index,
                        edge_attr,
                        batch_label,
                        const_bit_w,
                    )
                    prediction = (sigmoid(out.detach()) > 0.5).long()
                    target = batch.y.to(device).int()
                    bit_predictions_data[count, 0] += int((prediction == target).sum())
                    count += 1

        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)

    # run the remaining parts
    if remaining > 0:
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

                count = 0
                for bit_width in bit_widths:
                    
                    # create model to do data quantization
                    model = QGNN_7().to(device)
                    model.load_state_dict(float_model.state_dict())
                    scale = get_scale(lower_limit, upper_limit, bit_width)
                    zero_pt = get_zero_pt(lower_limit, upper_limit, bit_width)

                    scale, zero_pt = quantize_model_layers(
                        model,
                        bit_width,
                        scale=scale,
                        zero_pt=zero_pt,
                        same_quantization=True,
                    )
                    dequantize_model_layers(model, scale, zero_pt)
                    
                    out = model(
                        x,
                        edge_index,
                        edge_attr,
                        batch_label,
                        const_bit_w,
                    )
                    prediction = (sigmoid(out.detach()) > 0.5).long()
                    target = batch.y.to(device).int()
                    bit_predictions_data[count, 0] += int((prediction == target).sum())
                    count += 1

        # add final floating point predictions
        float_predictions_data[0] += run_inference_old(float_model, loader, device=device)

    # when all partitions are finished we can compute logical failure rates
    failure_rate = (
        np.ones((len(bit_widths), 1)) * n_graphs
        - bit_predictions_data.sum(axis=1, keepdims=True)
    ) / n_graphs
    failure_rate_fp_model = (n_graphs - float_predictions_data.sum()) / n_graphs

    return failure_rate, failure_rate_fp_model


def main():
    
    # command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("-n", "--n_graphs", required=False)
    parser.add_argument("-ns", "--n_graphs_per_sim", required=False)
    
    args = parser.parse_args()   
    experiment = args.experiment
    
    paths = [
        Path("../models/circuit_level_noise/d3/d3_d_t_5.pt"),
        Path("../models/circuit_level_noise/d5/d5_d_t_5.pt"),
        Path("../models/circuit_level_noise/d7/d7_d_t_5.pt"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.n_graphs:
        n_graphs = int(args.n_graphs)
    else: 
        n_graphs = int(1e4)
    
    if args.n_graphs_per_sim:
        n_graphs_per_sim = int(args.n_graphs_per_sim)
    else: 
        n_graphs_per_sim = int(1e4)
    batch_size = n_graphs_per_sim if "cuda" in device.type else 4000
    p = 1e-3
    min_bits = 2
    max_bits = 18

    # must use seed to make sure code distances are comparable
    seed = 747

    # collect data for each code size
    data_per_code_sz = []
    bit_data = []
    float_data = []
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

        print(
            f"Running bit exploration for {n_graphs} graphs with maximal batch size {batch_size}."
        )
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
            data_per_code_sz.append((failure_rate, failure_rate_fp_model))
            print(f"Code size: {code_sz}, fr: {failure_rate_fp_model}")

        elif experiment == "data":
            failure_rate, failure_rate_fp_model = explore_data(
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
            data_per_code_sz.append((failure_rate, failure_rate_fp_model))
            
        elif experiment == "data_and_weights":
            const_bit_w = 16
            failure_rate, failure_rate_fp_model = explore_data_and_weights(
                float_model,
                code_sz,
                reps,
                p,
                min_bits,
                max_bits,
                const_bit_w,
                n_graphs,
                n_graphs_per_sim,
                batch_size=batch_size,
                seed=seed,
                device=device,
            )
            
            data_per_code_sz.append((failure_rate, failure_rate_fp_model))
            bit_data.append(failure_rate.squeeze())
            float_data.append(failure_rate_fp_model)

        elif experiment == "fixed_pt":
            failure_rate, failure_rate_fp_model, q_error = explore_fixed_pt_weights(
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
            data_per_code_sz.append((failure_rate, failure_rate_fp_model))

        elif experiment == "weights_per_layer":
            min_req_bit, failure_rate_fp_model = explore_weights_per_layer(
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
            data_per_code_sz.append((min_req_bit, failure_rate_fp_model))
        else:
            print("You need to provide a valid experiment.")
            return
    
    bit_data = np.array(bit_data)
    float_data = np.array(float_data)
    np.save("const_data_quantization.npy", bit_data)
    np.save("const_data_quantization_float.npy", float_data)
    return

    if "layer" not in experiment:
        fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
        # fig_qerr, ax_qerr = plt.subplots(figsize=(12, 8))

        x = np.arange(min_bits, max_bits + 1, step=2)
        colors = ["r", "b", "g"]
        code_sz = [3, 5, 7]
        for i, data in enumerate(data_per_code_sz):
            failure_rate, failure_rate_fp_model = data

            ax_acc.axhline(
                failure_rate_fp_model,
                0,
                max_bits,
                linestyle="--",
                color=colors[i],
                label=f"FP32 logical failure rate, d={code_sz[i]}",
            )
            if not x < 0 in failure_rate:
                ax_acc.semilogy(
                    x,
                    failure_rate,
                    color=colors[i],
                    label=f"d = {code_sz[i]}",
                )
            else:
                print(f"The dataset is likely to small, failure rate = {failure_rate:.2f}")

            # ax_qerr.semilogy(
            #     x,
            #     q_error,
            #     color=colors[i],
            #     label=f"d = {code_sz[i]}",
            # )

        ax_acc.set_xlabel("# bits")
        ax_acc.set_ylabel("Logical failure rate")
        ax_acc.legend(loc="upper right")
        ax_acc.set_title(f"Quantization of {experiment}")

        # ax_qerr.set_xlabel("# bits")
        # ax_qerr.set_ylabel("Quantization error")
        # ax_qerr.set_title(f"Quantization to {experiment}")

        fig_acc.tight_layout()
        fig_acc.savefig(f"../figures/bit_accuracies_{experiment}.pdf")
        # fig_qerr.savefig(f"../figures/bit_qerror_{experiment}.pdf")

    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ["r", "b", "g"]
        code_sz = [3, 5, 7]
        
        graph_weight_labels = [f"Graph layer {n} - weights" for n in range(7)]
        graph_bias_labels  = [f"Graph layer {n} - bias" for n in range(7)]
        linear_weight_labels = [f"Linear layer {n} - weights" for n in range(3)]
        linear_bias_labels = [f"Linear layer {n} - bias" for n in range(3)]
        labels = (graph_weight_labels + graph_bias_labels + linear_weight_labels + linear_bias_labels)
        labels.extend(["Output layer - weight", "Output layer - bias"])
        
        # ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        plt.xticks(rotation=90)
        # ax.set_xticklabels(labels)
        for i, data in enumerate(data_per_code_sz):
            min_req_bit, failure_rate_fp_model = data
            x = range(len(labels))
            ax.plot(
                x,
                min_req_bit,
                color=colors[i],
                label=f"d = {code_sz[i]}",
            )
        ax.legend(loc="upper left")
        ax.set_ylabel("Minimum required bits to reach accuracy of FP32")
        ax.set_title("Quantization of weights per layer")
        fig.tight_layout()
        fig.savefig(f"../figures/bit_accuracies_per_layer.pdf")

if __name__ == "__main__":
    main()
