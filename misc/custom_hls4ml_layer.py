from pathlib import Path

import hls4ml
import torch
import torch.nn as nn


@torch.fx.wrap
def pmat_mul(x1, x2):
    return torch.mm(x1, x2)


# class PMatMul(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x1, x2):
#         return custum_matmul(x1, x2)


class HMatMul(hls4ml.model.layers.Layer):
    "hls4ml implementation of a layer doing matrix multiplication"

    def initialize(self):
        assert len(self.inputs) == 2
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])

        out_shape = [inp1.shape[0], inp2.shape[1]]
        dims = [f"OUT_MATMUL_{i}_{self.index}" for i in range(2)]
        self.add_output_variable(out_shape, dims)


def parse_matmul_layer(
    operation,
    layer_name,
    input_names,
    input_shapes,
    node,
    x,
    reader,
    config,
):
    layer = {}
    layer["class_name"] = "HMatMul"
    layer["name"] = layer_name
    layer["x_height"] = input_shapes[0][1]
    layer["x_width"] = input_shapes[0][2]
    layer["y_height"] = input_shapes[1][1]
    layer["y_width"] = input_shapes[1][2]

    if input_names is not None:
        layer["inputs"] = input_names

    return layer, [input_shapes[0][1], input_shapes[1][2]]


matmul_config_template = """struct config{index} : nnet::matmul_config {{
    static const unsigned x_height = {x_height};
    static const unsigned x_width = {x_width};
    static const unsigned y_height = {y_height};
    static const unsigned y_width = {y_width};
}};\n"""

matmul_function_template = "nnet::matmul<{input_t}, {config}>({x}, {y}, {res});"
matmul_include_list = ["nnet_utils/nnet_matmul.h"]


class HMatMulConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(HMatMul)
        self.template = matmul_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params["x_height"] = node.get_input_variable(node.inputs[0]).shape[0]
        params["x_width"] = node.get_input_variable(node.inputs[0]).shape[1]
        params["y_height"] = node.get_input_variable(node.inputs[1]).shape[0]
        params["y_width"] = node.get_input_variable(node.inputs[1]).shape[1]

        return self.template.format(**params)


class HMatMulFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(HMatMul, include_header=matmul_include_list)
        self.template = matmul_function_template

    def format(self, node):
        params = {}
        params["config"] = f"config{node.index}"
        params["input_t"] = node.get_input_variable(node.inputs[0]).type.name
        params["x"] = node.get_input_variable(node.inputs[0]).name
        params["y"] = node.get_input_variable(node.inputs[1]).name
        params["res"] = node.get_output_variable().name

        return self.template.format(**params)


class CustomGraphConv(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.weight_node = nn.Linear(input_features, output_features, bias=False)
        self.weight_adj = nn.Linear(input_features, output_features, bias=True)

    def forward(self, x, adj):
        node_term = self.weight_node(x)

        adjaceny_sum = pmat_mul(adj, x)
        adjaceny_term = self.weight_adj(adjaceny_sum)

        return node_term + adjaceny_term


class GraphWTorchNet(torch.nn.Module):
    def __init__(
        self,
        hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
        hidden_channels_MLP=[256, 128, 64],
        num_node_features=5,
        num_classes=1,
        manual_seed=12345,
    ):
        # num_classes is 1 for each head
        super().__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)

        # Activation
        self.activation = nn.ReLU()

        # Average pooling
        self.mean_pooling = nn.AvgPool1d(kernel_size=num_node_features)

        # GCN layers
        channels = [num_node_features] + hidden_channels_GCN
        self.graph_layers = nn.ModuleList(
            [
                CustomGraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Dense layers
        channels = hidden_channels_GCN[-1:] + hidden_channels_MLP
        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ]
        )

        # Output later
        self.output_layer = nn.Linear(hidden_channels_MLP[-1], num_classes)

    def forward(self, x, adj, batch, one_div_n_nodes):
        #  node embeddings
        for layer in self.graph_layers:
            x = layer(x, adj)
            x = self.activation(x)

        # global mean pool
        x = pmat_mul(batch, x) * one_div_n_nodes

        for layer in self.dense_layers:
            x = layer(x)
            x = self.activation(x)

        # output
        x = self.output_layer(x)

        return x


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        out = pmat_mul(x1, x2)
        return out


def main():
    hls4ml.converters.register_pytorch_layer_handler("pmat_mul", parse_matmul_layer)
    hls4ml.model.layers.register_layer("HMatMul", HMatMul)
    backend = hls4ml.backends.get_backend("Vivado")

    backend.register_template(HMatMulConfigTemplate)
    backend.register_template(HMatMulFunctionTemplate)

    p = Path(__file__).parent / "nnet_matmul.h"
    print(f"Registering custom template at {p}.")
    backend.register_source(p)

    # test if it works
    model = GraphWTorchNet()
    config = {}
    config["Model"] = {
        "Precision": "ap_fixed<16,6>",
        "ReuseFactor": 1,
        "ParallelizationFactor": 1,
        "Strategy": "Resource",
    }

    input_shape = [[None, 100, 5], [None, 100, 100], [None, 1, 100], [None, 1, 1]]
    hmodel = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape,
        output_dir="graph_nn_as_hls",
        project_name="quant_on_fpga",
        backend="Vivado",
        hls_config=config,
    )
    hmodel.build()
if __name__ == "__main__":
    main()
