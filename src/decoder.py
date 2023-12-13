import torch
from utils import parse_yaml
from gnn_models import GNN_7
from simulations import SurfaceCodeSim
from graph_representation import get_batch_of_graphs
from icecream import ic
from timeit import default_timer as timer

class Decoder:
    def __init__(self, yaml_config=None):
        
        # load settings and initialise state
        paths, graph_settings, training_settings = parse_yaml(yaml_config)
        self.save_dir = paths["save_dir"]
        self.model_name = paths["model_name"]
        self.graph_settings = graph_settings
        self.training_settings = training_settings

        # current training status
        self.epoch = training_settings["current_epoch"]
        self.lr = training_settings["lr"]
        if training_settings["device"] == "cuda":
            self.device = torch.device(
                training_settings["device"] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")
            
        # instantiate model
        self.model = GNN_7()

    def train(self):
        
        # training settings
        current_epoch = self.epoch
        n_epochs = self.training_settings["epochs"]
        dataset_size = self.training_settings["dataset_size"]
        batch_size = self.training_settings["batch_size"]
        n_batches = dataset_size // batch_size
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fun = torch.nn.BCEWithLogitsLoss()
        
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        error_rate = self.graph_settings["error_rate"]
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_node_features = self.graph_settings["n_node_features"]
        power = self.graph_settings["power"]
        n_classes = self.graph_settings["n_classes"]

        times = 0
        sim = SurfaceCodeSim(reps, code_size, error_rate, batch_size)
        for i in range(current_epoch, n_epochs):
            
            epoch_loss = 0
            for j in range(n_batches):
                
                # simulate data as we go
                start = timer()
                syndromes, flips, n_identities = sim.generate_syndromes()
                x, edge_index, edge_attr, batch_labels = get_batch_of_graphs(
                    syndromes,
                    m_nearest_nodes=m_nearest_nodes,
                    n_node_features=n_node_features,
                    power=power,
                    device=self.device
                )
                flips = torch.tensor(flips[:, None], dtype=torch.float32)
                stop = timer()
                times += (stop - start)
                
                # forward/backward pass
                opt.zero_grad()
                out = self.model(x, edge_index, edge_attr, batch_labels)
                loss = loss_fun(out, flips)
                loss.backward()
                opt.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= n_batches
            print(f"Epoch {i}: loss = {epoch_loss:.2f}")
            
        print(f"Average time taken to create a batch of data: {(times / (n_epochs * n_batches)):.2f}s")
        print(f"Total time used for data preparation: {times:.2f}s")
                
                


decoder = Decoder()
decoder.train()