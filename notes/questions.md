#### gnn_models.py
* Why do we reset parameters using `self.graphx.reset_parameters()`? 
* Is the batch-argument in `global_mean_pool` a tensor matching the shape of the graph representation indicating which node that belongs to which graph in the batch?


#### General
* How would we handle an empty graph? If I understand correctly we do not handle it atm.
* Do we have Google's data and if so do we have $p_{ij}$ data?
* Could GraphConv be slow because we have fully connected graphs?
* How to make transfer between CPU and GPU more efficient? It's weird that "batch" is not a GPU-tensor when using the dataloader.
* I don't think the edge pruning works as expected! More edges than desired is given.
  * Implement `knn_graph` instead
  * Have we been CPU-limited before? If pruning is a bottleneck - has the forward pass waited for graphs to be generated? If this runs in parallell this cannot be a problem, can it? 