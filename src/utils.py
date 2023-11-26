import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

def match_and_load_state_dict(
    model: nn.Module,
    trained_state_dict: OrderedDict,
) -> nn.Module:
    new_dict = OrderedDict(trained_state_dict)
    for new_key, key in zip(model.state_dict().keys(), trained_state_dict.keys()):
        new_dict[new_key] = new_dict[key]
        del new_dict[key]

    model.load_state_dict(new_dict)
    return model

def get_all_weights(model: nn.Module) -> np.ndarray:
    
    weights: OrderedDict = model.state_dict()
    all_weights = []
    for weight in weights.values():
        all_weights.append(weight.numpy().flatten())
    
    return np.concatenate(all_weights)

def plot_weights(
    n_graph_layers: int,
    n_dense_layers: int,
    weights: OrderedDict[str, torch.Tensor],
) -> (plt.figure, plt.figure):
    graph_dict = {"lin_rel.weight": 0, "lin_rel.bias": 1, "lin_root.weight": 2}
    dense_dict = {"weight": 0, "bias": 1}

    graph_fig, graph_ax = plt.subplots(
        n_graph_layers, 3, figsize=(4 * 3, 4 * n_graph_layers)
    )
    dense_fig, dense_ax = plt.subplots(
        n_dense_layers, 2, figsize=(4 * 2, 4 * n_dense_layers)
    )

    for key, weight in weights.items():
        if "output" in key:
            index = 3
        else:
            index = int(re.findall(r"\d+", key)[0])
        data = weight.numpy().flatten()
        data_sz = data.shape[0]

        if "graph" in key:
            split_key = key.split(sep=".")
            plot_key = split_key[-2] + "." + split_key[-1]
            ax = graph_ax[index][graph_dict[plot_key]]
            ax.hist(data, 50)
            ax.set_title(f"GraphConv {index}: {plot_key} ({data_sz} parameters)")

        else:
            split_key = key.split(sep=".")
            plot_key = split_key[-1]
            ax = dense_ax[index][dense_dict[plot_key]]
            ax.hist(weight.numpy().flatten(), 50)
            ax.set_title(f"Linear {index}: {plot_key} ({data_sz} parameters)")

        ax.set_xlabel(
            f"$\mu$ = {data.mean():.2f}, $\sigma^2$ = {data.std():.2f}, $x \in$ [{data.min():.2f}, {data.max():.2f}]"
        )

    graph_fig.tight_layout()
    dense_fig.tight_layout()

    return graph_fig, dense_fig


def quantize_model_layers(
    model: nn.Module,
    bit_width: np.int64,
    same_quantization: bool = False,
    scale: torch.Tensor = None,
    zero_pt: torch.Tensor = None,
    signed: bool = True,
    layer_index: int = None,
    quantile: float = 0.001,
) -> tuple[list[str], float, int]:
    weights: OrderedDict = model.state_dict()

    keys = []
    if layer_index is not None:
        key = list(model.state_dict())[layer_index]
        layer_weight = weights[key]

        if not same_quantization:
            lower_bound = torch.quantile(layer_weight, quantile)
            upper_bound = torch.quantile(layer_weight, 1 - quantile)
            scale = get_scale(lower_bound, upper_bound, bit_width=bit_width)
            zero_pt = get_zero_pt(lower_bound, upper_bound, bit_width=bit_width)

        weights[key] = quantize_tensor(
            weights[key],
            scale=scale,
            zero_point=zero_pt,
            bit_width=bit_width,
            signed=signed,
        )

        model.load_state_dict(weights)
        keys.append(key)

    else:
        for key, tensor in weights.items():
            if not same_quantization:
                lower_bound = torch.quantile(tensor, quantile)
                upper_bound = torch.quantile(tensor, 1 - quantile)
                if lower_bound == upper_bound:
                    break

                scale = get_scale(lower_bound, upper_bound, bit_width=bit_width)
                zero_pt = get_zero_pt(lower_bound, upper_bound, bit_width=bit_width)
                
            weights[key] = quantize_tensor(
                tensor,
                scale=scale,
                zero_point=zero_pt,
                bit_width=bit_width,
                signed=signed,
            )
            model.load_state_dict(weights)
            keys.append(key)

    return keys, scale, zero_pt


def dequantize_model_layers(
    model: nn.Module,
    scale: torch.Tensor,
    zero_pt: torch.Tensor,
    layer_index: int = None,
) -> list[str]:
    weights: OrderedDict = model.state_dict()

    keys = []
    if layer_index is not None:
        key = list(model.state_dict())[layer_index]
        weights[key] = dequantize_tensor(weights[key], scale, zero_pt)

        model.load_state_dict(weights)
        keys.append(key)
    else:
        for key, tensor in weights.items():
            weights[key] = dequantize_tensor(tensor, scale, zero_pt)
            model.load_state_dict(weights)
            keys.append(key)

    return keys


def get_number_of_model_layers(module: nn.Module):
    sum = 0
    for child in module.children():
        if isinstance(child, nn.modules.container.ModuleList):
            n = get_number_of_model_layers(child)
            sum += n
        else:
            sum += 1
    return sum


def quantize_tensor(
    r: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    bit_width: np.int64 = 8,
    signed: bool = True,
) -> torch.Tensor:
    if signed:
        return torch.clamp(
            torch.round(r / scale + zero_point),
            -(2 ** (bit_width - 1)),
            2 ** (bit_width - 1) - 1,
        )
    else:
        return torch.clamp(torch.round(r / scale + zero_point), 0, 2**bit_width - 1)


def dequantize_tensor(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    return (scale * (q - zero_point)).float()


def get_scale(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    bit_width: np.int64 = 8,
) -> torch.Tensor:

    return (beta - alpha) / (2**bit_width - 1)


def get_zero_pt(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    bit_width: np.int64 = 8,
    signed: bool = True,
) -> torch.Tensor:
    
    if signed:
        return torch.round(
            (2**bit_width * (alpha + beta) - 2 * beta) / (2 * (alpha - beta))
        )
    else:
        return torch.round(
            (-alpha * 2**bit_width) / (beta - alpha)
        )
        


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    n_graphs: int,
    n_trivial_preds: int,
    device: torch.device = torch.device("cpu")
) -> float:
    sigmoid = nn.Sigmoid()
    correct_preds = 0

    # loop over batches
    with torch.no_grad():
        for batch in tqdm(loader):
            # unzip data
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            batch_label = batch.batch.to(device)

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
    return accuracy, correct_preds