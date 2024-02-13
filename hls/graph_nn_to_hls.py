import hls4ml
import torch
import torch.nn as nn
import numpy as np

from pathlib import Path

#workaround until container is fixed
import os
os.environ["LIBRARY_PATH"]="/usr/lib/x86_64-linux-gnu"
os.environ["PATH"]="/tools/Xilinx/Vivado/2020.1/bin/:" + os.environ["PATH"]

@torch.fx.wrap
def pmat_mul(x1, x2):
    return torch.mm(x1, x2)

class HMatMul(hls4ml.model.layers.Layer):
    "hls4ml implementation of a layer doing matrix multiplication"

    def initialize(self):
        assert len(self.inputs) == 2
        inp1 = self.get_input_variable(self.inputs[0])
        inp2 = self.get_input_variable(self.inputs[1])

        out_shape = [inp1.shape[0], inp2.shape[1]]
        dims = [f"OUT_MATMUL_{i}_{self.index}" for i in range(2)]
        dims = [inp1.dim_names[0], inp2.dim_names[1]]
        self.add_output_variable(out_shape, dims)
        
def parse_matmul_layer(
    operation,
    layer_name,
    input_names,
    input_shapes,
    node,
    class_object,
    data_reader,
    config,
):
    layer = {}
    layer["class_name"] = "HMatMul"
    layer["name"] = layer_name
    layer["x_height"] = input_shapes[0][1]
    layer["x_width"] = input_shapes[0][2]
    layer["y_height"] = input_shapes[1][1]
    layer["y_width"] = input_shapes[1][2]
    
    # layer["x_height"] = input_shapes[0][0]
    # layer["x_width"] = input_shapes[0][1]
    # layer["y_height"] = input_shapes[1][0]
    # layer["y_width"] = input_shapes[1][1]

    if input_names is not None:
        layer["inputs"] = input_names

    return layer, [None, input_shapes[0][1], input_shapes[1][2]]

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

        self.lin_root = nn.Linear(input_features, output_features, bias=False)
        self.lin_rel = nn.Linear(input_features, output_features, bias=True)

    def forward(self, x, adj):
        node_term = self.lin_root(x)
        adjaceny_sum = pmat_mul(adj, x)
        adjaceny_term = self.lin_rel(adjaceny_sum)

        return node_term + adjaceny_term   
    
class GraphWTorchNet(torch.nn.Module):
    def __init__(
        self,
        hidden_channels_GCN=[32, 128, 256, 512, 512, 256, 256],
        hidden_channels_MLP=[256, 128, 64],
        num_node_features=5,
        num_classes=1,
    ):
        # num_classes is 1 for each head
        super().__init__()

        # Activation
        self.activation = nn.ReLU()

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

    def forward(self, x, adj, batch):
        #  node embeddings
        for layer in self.graph_layers:
            x = layer(x, adj)
            x = self.activation(x)

        # global mean pool
        x = pmat_mul(batch, x)

        for layer in self.dense_layers:
            x = layer(x)
            x = self.activation(x)

        # output
        x = self.output_layer(x)

        return x
    
def main():
    
    # register custom templates
    hls4ml.converters.register_pytorch_layer_handler("pmat_mul", parse_matmul_layer)
    hls4ml.model.layers.register_layer("HMatMul", HMatMul)
    
    backend = hls4ml.backends.get_backend("Vivado")
    backend.register_template(HMatMulConfigTemplate)
    backend.register_template(HMatMulFunctionTemplate)
    
    p = Path(__file__).parent / "nnet_matmul.h"
    print(f"Registering custom template at {p}.")
    backend.register_source(p)
    
    # set seeds
    torch.manual_seed(111)
    np.random.seed(111)
    
    # create Torch model
    n_node_features = 5
    max_nodes = 64
    hidden_channels_GCN = [32, 64]
    hidden_channels_MLP = [64, 32]
    input_shape = [
        [None, max_nodes, n_node_features], 
        [None, max_nodes, max_nodes], 
        [None, 1, max_nodes]
        ]
    # input_shape = [
    #     [1, max_nodes, n_node_features], 
    #     [1, max_nodes, max_nodes], 
    #     [1, 1, max_nodes]
    #     ]
    
    model = GraphWTorchNet(
        hidden_channels_GCN,
        hidden_channels_MLP,
        n_node_features,
        )
    
    # load weights and map to network
    path = Path("../saved_models/d3_d_t_3_240125-163025_load_f_d3_d_t_3_240125-111113_load_f_d3_d_t_3_240124-141433_load_f_d3_d_t_3_240123-230657.pt")
    weights = torch.load(path, map_location="cpu")["model"]
    
    for key in weights.keys():
        weights[key] = torch.zeros_like(weights[key])
        if len(weights[key].shape) > 1 and weights[key].shape[0] > 4:
            weights[key][[0, 2, 9, 10, 5], [1, 0, 4, 2, 3]] = 1
    model.load_state_dict(weights)
    model.eval()
    
    # prepare conversion of model
    hls_config = {}
    hls_config["Model"] = {
        "Precision": "ap_fixed<512, 256>",
        "ReuseFactor": 1,
        "Strategy": "Resource",
    }
    hmodel = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape,
        output_dir="gnn_hls",
        project_name="qec_fpga",
        backend="Vivado",
        hls_config=hls_config,
        io_type="io_parallel",
        )
    hmodel.compile()
    
    # load test data
    data_folder = Path("./data")
    for i in range(10):
        file_name = f"input_{i}.npz"
        input_data = np.load(data_folder / file_name)
        
        x = input_data["x"]
        adj = input_data["adj"]
        node_labels = input_data["node_labels"]
        flip = input_data["flip"]
        
        
        x = np.ones_like(x)
        adj = np.zeros_like(adj)
        adj[[1, 2, 3, 4], [2, 7, 9, 10]] = 1
        node_labels = np.ones_like(node_labels)
        
        inputs_np = [np.ascontiguousarray(x), np.ascontiguousarray(adj), np.ascontiguousarray(node_labels)]
        inputs_torch = [torch.tensor(input, dtype=torch.float32) for input in inputs_np]
        
        # compare outputs from Torch model and C/C++-template
        sigmoid_torch = nn.Sigmoid()
        sigmoid_np = lambda x: 1 / (1 + np.exp(-x))
        out_torch = model(*inputs_torch)
        out_csim = hmodel.predict(inputs_np)[0]
        
        pred_torch = (sigmoid_torch(out_torch.detach()) > 0.5).long().item()
        pred_np = int(sigmoid_np(out_csim) > 0.5)
        
        print(f"F32 output: \n{out_torch.item()}")
        print(f"Quantized output: \n{out_csim}")
        print(f"Difference: \n{out_torch.item() - out_csim}")
        print(f"Label: \n{flip[0]}")
        print(f"F32 prediction \n{pred_torch}")
        print(f"Quantized prediction: \n{pred_np}")
        print(f"Should be the same: \n {flip[0]}, {pred_torch}, {pred_np}")
        
        
if __name__ == "__main__":
    main()