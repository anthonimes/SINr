# SINr

**SINr** is a `python` framework to compute sparse and interpretable graph embeddings. The method can rely on any community detection algoritm, and is currently implemented using Louvain's algorithm. 

## How does it work

**SINr** is implemented in two versions, classic (with better execution time) and using CSR matrices (using less memory). To allow compatibility with any graph library, the parameters expected are a set of (weighted) edges together with the list of weighted degrees of all nodes of the network. 
Edges should be represented using tuples according to the following format: 

*(id1, id2, weight)*

For similar reasons, the communities are expected as arguments of the methods using a list containing, for each node, the label of the community it belongs to. 
** For these reasons, all graphs must be numbered from `0` to `(number of nodes - 1)`. 

## Embeddings

The embeddings are returned as `numpy` matrices in both cases. In order to allow more flexibility, we chose not to implement any method to save or load embeddings. We recommand using the [load](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load) and [save](https://numpy.org/doc/stable/reference/generated/numpy.save.html) methods from `numpy` and can provide assistance for dealing with embeddings.  

## Examples

Two examples are provided for [*networkx*](https://networkx.github.io/) (`src/network_example.py`) and [*networkit*](https://networkit.github.io/) python libraries. For the former case, the additional [*python-louvain*](https://github.com/taynaud/python-louvain) is used to produce communities. In the latter case, the build-in Louvain's algorithm is used. 

## Dependencies

**SINr** relies on the following python modules:

* `numpy >= 1.18`
* `numba`
