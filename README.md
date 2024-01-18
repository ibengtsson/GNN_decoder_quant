# NOTE
This is a fork and contains additions to the repository developed by Moritz Lange. The code is still a work in process and although many things work, it must be considered experimental. When time comes I'll add more comments to where to find stuff.

* `scripts/` contains simple scripts used as entry-points to run source code. Often accept command-line arguments.
* `bash_scripts/` contains bash scripts used to run code on a Linux-based GPU-cluster with Slurm. 
* `notebooks/` contains Jupyter notebooks where I've visualised stuff and tested functionality during development. Experimental, can't guarantee stuff that works yet. 

# Moritz comments follows below (will remove when I've added more comments):
-----------------------------------------------------------------------------
# Graph Decoder
Graph neural network decoder for the rotated surface code.

Includes the source code for the GNN decoder, as well as scripts to run the code on a cluster.

## Getting started
Follow these steps to run the code on a cluster.

* Clone the repository with
`git clone https://github.com/LangeMoritz/GNN_decoder`
* Install the required packages in `requirements.txt`, for example using a virtual environment:
```
python3 -m venv .venv 
source .venv/bin/activate
python -m pip install -r requirements.txt
```
* Install PyTorch by following the instructions found [here](https://pytorch.org/get-started/locally/).
* Install PyTorch Geometric by following the instructions found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Repository structure
* `src/` contains the "source code" for the project, split into a number of modules:
  * `GNN_Decoder.py` A class for creating a decoder object with methods for training a GNN decoder and running decoding simulations.
  * `__init__.py`turns `src/` into an importable package.
  * `gnn_models.py` Contains PyTorch geometric graph neural network decoder models.
  * `graph_representation.py` Functions for converting syndromes to graphs.
```
├── src
│   ├── GNN_Decoder.py
│   ├── __init__.py
│   ├── gnn_models.py
│   ├── graph_representation.py
│   └── rotated_surface_code.py
```
* `models/` contains trained models corresponding to fig. 3 (`circuit_level_noise/`), fig. 5 and 6 (`repetition_code/`) and fig. 7 (`perfect_stabilizers/`)
* `results/` contains model and training history from training runs as .pt files (each run generates one file upon finishing). This directory should exist in your working directory.
* `job_outputs/` contains the standard output files from runs. This directory should exist in your working directory.
* `buffer_training.py` is the python script used to run training with a data buffer, replacing part of the buffer with new data after a fixed number of training iterations.
  * `run_buffer_training.sh` is the shell run script used to start is jobs for `buffer_training.py`.
* `.gitignore` lists files and directories in the git repository to be ignored in commits.
* `requirements.txt` lists the required python packages. See the Getting Started section above.
  
