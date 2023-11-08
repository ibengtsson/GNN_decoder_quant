#### Slice graph into 2d-slices and use a recurrent structure to embed time
* Instead of feeding a graph with a time axis directly into the network we can build graph feature vectors over spatial slices and feed them into a recurrent block (maybe use batch/layer norm to ensure same magnitude of features)
* Use output directly after $t = 1$ to compute loss and combine with loss after $t = d_t$ rounds (probably possible if we have simulated data)
* Embed timestep as a "global" node in the graph
* 