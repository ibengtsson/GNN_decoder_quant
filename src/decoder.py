import torch
import numpy as np
from src.utils import parse_yaml, run_inference
from src.gnn_models import GNN_7
from src.simulations import SurfaceCodeSim
from src.graph_representation import get_batch_of_graphs
from pathlib import Path
from datetime import datetime
import random


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
        training_history["train_loss"] = []
        training_history["train_accuracy"] = []
        training_history["val_loss"] = []
        training_history["val_accuracy"] = []
        training_history["best_val_accuracy"] = -1

        self.training_history = training_history
        
        # only keep best found weights
        self.optimal_weights = None

        # instantiate model and optimizer
        self.model = GNN_7().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_settings["lr"]
        )

        # generate a unique name to not overwrite other models
        name = (
            "d"
            + str(graph_settings["code_size"])
            + "_d_t_"
            + str(graph_settings["repetitions"])
            + "_"
        )
        current_datetime = datetime.now().strftime("%y%m%d-%H%M%S")
        self.save_name = name + current_datetime
        
        # check if model should be loaded
        if training_settings["resume_training"]:
            self.load_trained_model()

    def save_model_w_training_settings(self, model_name=None):
        # make sure path exists, else create it
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if model_name is not None:
            path = self.save_dir / (model_name + ".pt")
        else:
            path = self.save_dir / (self.save_name + ".pt")

        # we only want to save the weights that corresponds to the best found accuracy
        if self.training_history["val_accuracy"][-1] > self.training_history["best_val_accuracy"]:
            self.training_history["best_val_accuracy"] = self.training_history["val_accuracy"][-1]
            self.optimal_weights = self.model.state_dict()
        
        attributes = {
            "training_history": self.training_history,
            "model": self.optimal_weights,
            "optimizer": self.optimizer.state_dict(),
            "graph_settings": self.graph_settings,
            "training_settings": self.training_settings, 
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
        self.save_name = self.save_name + "_load_f_" + model_path.name.split(sep=".")[0]
        
        # only keep best found weights
        self.optimal_weights = saved_attributes["model"]

    def initialise_simulations(self, n=5):
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        batch_size = self.training_settings["batch_size"]

        min_error_rate = self.graph_settings["min_error_rate"]
        max_error_rate = self.graph_settings["max_error_rate"]

        error_rates = np.linspace(min_error_rate, max_error_rate, n)

        sims = []
        for p in error_rates:
            sim = SurfaceCodeSim(reps, code_size, p, batch_size)
            sims.append(sim)

        return sims

    def create_test_set(self, n_graphs=5e5, n=5):
        # simulation settings
        code_size = self.graph_settings["code_size"]
        reps = self.graph_settings["repetitions"]
        min_error_rate = self.graph_settings["min_error_rate"]
        max_error_rate = self.graph_settings["max_error_rate"]

        error_rates = np.linspace(min_error_rate, max_error_rate, n)

        syndromes = []
        flips = []
        n_identities = 0
        for p in error_rates:
            sim = SurfaceCodeSim(reps, code_size, p, int(n_graphs / n))
            syndrome, flip, n_id = sim.generate_syndromes()
            syndromes.append(syndrome)
            flips.append(flip)
            n_identities += n_id

        syndromes = np.concatenate(syndromes)
        flips = np.concatenate(flips)

        # split into chunks to reduce memory footprint later
        batch_size = self.training_settings["batch_size"]
        n_splits = syndromes.shape[0] // batch_size + 1

        syndromes = np.array_split(syndromes, n_splits)
        flips = np.array_split(flips, n_splits)

        return syndromes, flips, n_identities

    def evaluate_test_set(self, syndromes, flips, n_identities, loss_fun, n_graphs=5e5):

        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_correct_preds = 0
        n_syndrome_graphs = 0
        val_loss = 0
        for syndrome, flip in zip(syndromes, flips):
            flip = torch.tensor(flip[:, None], dtype=torch.float32).to(self.device)

            _n_correct_preds, out = run_inference(
                self.model,
                syndrome,
                flip,
                m_nearest_nodes=m_nearest_nodes,
                device=self.device,
            )
            n_correct_preds += _n_correct_preds
            loss = loss_fun(out, flip)
            val_loss += loss.item() * syndrome.shape[0]
            n_syndrome_graphs += syndrome.shape[0]

        # compute metrics
        val_loss /= n_syndrome_graphs
        val_accuracy = (n_correct_preds + n_identities) / n_graphs

        return val_loss, val_accuracy

    def train(self):
        # training settings
        current_epoch = self.epoch
        n_epochs = self.training_settings["epochs"]
        dataset_size = self.training_settings["dataset_size"]
        batch_size = self.training_settings["batch_size"]
        n_batches = dataset_size // batch_size
        loss_fun = torch.nn.BCEWithLogitsLoss()
        sigmoid = torch.nn.Sigmoid()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1)

        # initialise simulations and graph settings
        m_nearest_nodes = self.graph_settings["m_nearest_nodes"]
        n_node_features = self.graph_settings["n_node_features"]
        power = self.graph_settings["power"]

        sims = self.initialise_simulations()

        # generate validation syndromes
        n_val_graphs = self.training_settings["validation_set_size"]
        val_syndromes, val_flips, n_val_identities = self.create_test_set(
            n_graphs=n_val_graphs,
        )

        for epoch in range(current_epoch, n_epochs):
            train_loss = 0
            epoch_n_graphs = 0
            epoch_n_correct = 0
            epoch_n_trivial = 0
            for _ in range(n_batches):
                # simulate data as we go
                sim = random.choice(sims)
                syndromes, flips, n_trivial = sim.generate_syndromes()
                epoch_n_trivial += n_trivial
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

                # update loss and accuracies
                prediction = (sigmoid(out.detach()) > 0.5).long()
                flips = flips.long()
                epoch_n_correct += int((prediction == flips).sum())

                train_loss += loss.item() * n_graphs
                epoch_n_graphs += n_graphs

            # update learning rate
            scheduler.step()

            # compute losses and logical accuracy
            # ------------------------------------

            # train
            train_loss /= epoch_n_graphs
            train_accuracy = (epoch_n_correct + epoch_n_trivial) / (
                batch_size * n_batches
            )

            # validation
            val_loss, val_accuracy = self.evaluate_test_set(
                val_syndromes,
                val_flips,
                n_val_identities,
                loss_fun,
                n_val_graphs,
            )

            # save training attributes after every epoch
            self.training_history["epoch"] = epoch
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_accuracy"].append(train_accuracy)
            self.training_history["val_accuracy"].append(val_accuracy)
            self.save_model_w_training_settings()
