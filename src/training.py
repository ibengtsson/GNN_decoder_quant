from timeit import default_timer as timer
from simulations import SurfaceCodeSim
from graph_representation import get_batch_of_graphs

seed = 11
n_graphs = 1000
code_sz = 7
reps = 5
p = 5e-3
m_nearest_nodes = 3
n_node_features = 5


start = timer()
sim = SurfaceCodeSim(reps, code_sz, p, n_graphs, seed=seed)
syndromes, flips, n_identities = sim.generate_syndromes()
x, edge_index, edge_attr, batch_labels = get_batch_of_graphs(
    syndromes,
    m_nearest_nodes=m_nearest_nodes,
    n_node_features=n_node_features,
)
stop = timer()

print(stop - start)
