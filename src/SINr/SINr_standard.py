# -*- coding: utf-8 -*-
import numpy 
import numba
from numba.typed import List

# --- DEV/TODO --- update for directed graphs!
@numba.njit(parallel=True)
def _fill(sum_degrees_com, int_degree, vector, edge):
    """updates total degree of edges incident to a community and 
    the degree of extremities of the edge towards the corresponding communities.
    
    Keyword arguments:
        sum_degrees_com -- the list of total degree of edges for communities
        int_degree -- the matrix degreexcommunity for each node and each community
        vector -- the vector describing community structure
        edge -- the edge to be considered
    """
    src,dst,weight = edge
    try:
        if vector[src]==vector[dst]:
            sum_degrees_com[vector[src]]+=weight
        else:
            sum_degrees_com[vector[src]]+=weight
            sum_degrees_com[vector[dst]]+=weight
        int_degree[src][vector[dst]]+=weight
        int_degree[dst][vector[src]]+=weight
    except:
        pass
    
@numba.njit(parallel=True)
def _do_node_fmeasure(edges, weighted_degrees, vector, number_communities, nodes):
    """Compute Node Predominance and Node Recall for every node of the network.
    
    Keyword arguments:
        edges -- the edges of the network
        weighted_degree -- the weighted degree of each node of the network
        vector -- the vector describing community structure
        number_communities -- a list containing the size of every community
        nodes -- number of nodes of the network
    """
    
    # list and matrix initialization using numba
    sum_degrees_com=numpy.zeros(number_communities, dtype=numba.float64)
    int_degree=numpy.zeros((nodes, number_communities), dtype=numba.float64)
    #np=[ [0. for _ in range(number_communities)] for _ in range(nodes) ]
    # the line below does not work if we use np, but is fine with NP. 
    # the above line woks however
    NP=numpy.zeros((nodes,number_communities), dtype=numba.float64)
    NR=numpy.zeros((nodes,number_communities), dtype=numba.float64)
            
    # filling the structures for every edge of the network
    for edge in edges:
        _fill(sum_degrees_com, int_degree, vector, edge)

    # computing Node Predominance and Node Recall for every node and every community
    for u in range(len(vector)):
        for com in range(number_communities):
            if sum_degrees_com[com] == 0:
                NP[u][com]= 0
            else:
                NP[u][com] = int_degree[u][com] / sum_degrees_com[com]
            if weighted_degrees[u] == 0:
                NR[u][com] = 0
            else:
                NR[u][com] = int_degree[u][com] / weighted_degrees[u]
                           
    return NP, NR

def get_SINr_embeddings(edges, weights, vector, number_communities, nodes):
    """Procedure to generate embedding vectors for the network G
    
    Keyword arguments:
        edges -- list of edges of the netword as triplets: (src,dst,weight)
        weighted -- list of weighted degrees of all nodes (from 0 to nodes-1)
        vector -- the vector describing community structure
        number_communities -- a list containing the size of every community
        nodes -- number of nodes of the network
    """
        
    # adapting lists to numba
    numba_edges = List()
    [numba_edges.append((edge[0],edge[1],edge[2])) for edge in edges]
    numba_weighted = List()
    [numba_weighted.append(weights[node]) for node in range(nodes)]
    
    # computing Node Predominance and Node Recall
    NP,NR=_do_node_fmeasure(numba_edges, numba_weighted, vector, number_communities, nodes)
    
    # preparing embedding vectors
    embedding_matrix=numpy.concatenate((NP, NR), axis=1)
    
    return NP,NR,embedding_matrix
