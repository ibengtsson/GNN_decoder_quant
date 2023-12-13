import torch
from src.utils import parse_yaml, run_inference
from src.gnn_models import GNN_7
from src.simulations import SurfaceCodeSim
from src.graph_representation import get_batch_of_graphs
from pathlib import Path
from datetime import datetime


class Decoder:
    def __init__(self, yaml_config=None):
        
        # load settings and initialise state
        paths, graph_settings, training_settings = parse_yaml(yaml_config)
        self.save_dir = Path(paths["save_dir"])
        self.saved_model_path = paths["saved_model_path"]
        self.graph_settings = graph_settings
        self.training_settings = training_settings

        # current training status
        self.epoch = training_settings["current_epoch"]
        if training_settings["device"] == "cuda":
            self.device = torch.device(
                training_settings["device"] if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")

        # create a dictionary saving training metrics
        training_history = {}
        training_history["epoch"] = self.epoch
        training_history["loss"] = []
        training_history["failure_rate"] = []

        self.training_history = training_history

        # instantiate model and optimizer
        self.model = GNN_7().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_settings["lr"]
        )
        
        # generate a unique name to not overwrite other models
        current_datetime = datetime.now().strftime("%y%m%d-%H%M%S") 
        self.save_name = "model_" + current_datetime

    def save_model_w_training_settings(self, model_name=None):
        
        # make sure path exists, else create it
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if model_name is not None:
            path = self.save_dir / (model_name + ".pt")
        else:
            path = self.save_dir / (self.save_name + ".pt")
        
        attributes = {
            "training_history": self.training_history,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(attributes, path)

    def load_trained_model(self):
        model_path = Path(self.saved_model_path)
        saved_attributes = torch.load(model_path, map_location=self.device)

        # update attributes and load model with trained weights
        self.training_history = saved_attributes["training_history"]
        self.epoch = saved_attributes["training_history"]["epoch"] + 1
        self.model.load_state_dict(saved_attributes["model"])
        self.optimizer.load_state_dict(saved_attributes["optimizer"])
        self.save_name = model_path.name.split(sep=".")[0]

    def train(self):
        # training settings
        current_epoch = self.epoch
        n_epochs = self.training_settings["epochs"]
        dataset_size = self.training_settings["dataset_size"]
        batch_size = self.training_settings["batch_size"]
        n_batches = dataset_size // batch_size
        loss_fun = torch.nn.BCEWithLogitsLoss()

        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        error_rate = self.graph_settings["error_rate"]
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_node_features = self.graph_settings["n_node_features"]
        power = self.graph_settings["power"]
        sim = SurfaceCodeSim(reps, code_size, error_rate, batch_size)

        # generate validation syndromes
        val_syndromes, val_flips, n_val_identities = sim.generate_syndromes()

        for epoch in range(current_epoch, n_epochs):
            
            epoch_loss = 0
            epoch_n_graphs = 0
            for _ in range(n_batches):
                
                # simulate data as we go
                syndromes, flips, _ = sim.generate_syndromes()
                x, edge_index, edge_attr, batch_labels = get_batch_of_graphs(
                    syndromes,
                    m_nearest_nodes=m_nearest_nodes,
                    n_node_features=n_node_features,
                    power=power,
                    device=self.device,
                )

                n_graphs = syndromes.shape[0]
                flips = torch.tensor(flips[:, None], dtype=torch.float32).to(
                    self.device
                )

                # forward/backward pass
                self.optimizer.zero_grad()
                out = self.model(x, edge_index, edge_attr, batch_labels)
                loss = loss_fun(out, flips)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * n_graphs
                epoch_n_graphs += n_graphs

            # compute epoch loss and check logical failure rate
            epoch_loss /= epoch_n_graphs
            n_correct_preds = run_inference(
                self.model,
                val_syndromes,
                val_flips,
                m_nearest_nodes=m_nearest_nodes,
                device=self.device,
            )
            failure_rate = (
                batch_size - n_correct_preds - n_val_identities
            ) / batch_size
            
            # save training attributes after each epoch
            self.training_history["epoch"] = epoch
            self.training_history["loss"].append(epoch_loss)
            self.training_history["failure_rate"].append(failure_rate)
            self.save_model_w_training_settings()
            

