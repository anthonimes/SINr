# SINr

**SINr** is a `python` framework to compute sparse and interpretable graph embeddings. The method can rely on any community detection algoritm, and is currently implemented using Louvain's algorithm. 

## How does it work

**SINr** is implemented in two versions, classic (with better execution time) and using CSR matrices (using less memory). To allow compatibility with any graph library, the parameters expected are a set of (weighted) edges together with the list of weighted degrees of all nodes of the network. 
Edges should be represented using tuples according to the following format: 

*(id1, id2, weight)*

For similar reasons, the communities are expected as arguments of the methods using a list containing, for each node, the label of the community it belongs to. 
** For these reasons, all graphs must be numbered from `0` to `(number of nodes - 1)`. Since most graph libraries implement them, we do not provide a relabeling function.

## Embeddings

The embeddings are returned as `numpy` matrices in both cases. Every row contains the embedding vector of the corresponding node in the network

In order to allow more flexibility, we chose not to implement any method to save or load embeddings. We recommand using the [load](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load) and [save](https://numpy.org/doc/stable/reference/generated/numpy.save.html) methods from `numpy` and can provide assistance for dealing with embeddings.  

## Examples

Two examples are provided for [*networkx*](https://networkx.github.io/) (`src/network_example.py`) and [*networkit*](https://networkit.github.io/) (`src/networit_example.py`) python libraries. In both cases one argument (the graph) is needed:

`python src/networkx_examply.py examples/citeseer.ed`

The only output produced by such examples is the time needed to compute embeddings. As mentioned previsouly, embeddings are returned by the functions as numpy matrices and can thus be dealt with using any library. 

For the *networkx* example, the additional [*python-louvain*](https://github.com/taynaud/python-louvain) is used to produce communities. Regarding *networkit*, the built-in Louvain's algorithm is used. 

## Requirements

**SINr** relies on the following python modules:

* `python >= 3.7`
* `numpy >= 1.18`
* `numba`
