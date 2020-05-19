# -*- coding: utf-8 -*-
import numpy 
import numba
import networkit 
from numba.typed import List

# networkit generates a lot of warnings, removed for clarity purposes
import warnings
warnings.filterwarnings("ignore")

@numba.njit
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
    
@numba.njit
def _do_node_fmeasure(edges, weighted_degrees, vector, list_partition_sizes, nodes):
    """Compute Node Predominance and Node Recall for every node of the network.
    
    Keyword arguments:
        edges -- the edges of the network
        weighted_degree -- the weighted degree of each node of the network
        vector -- the vector describing community structure
        list_partition_sizes -- a list containing the size of every community
        nodes -- number of nodes of the network
    """
    
    # list and matrix initialization using numba
    sum_degrees_com=numpy.zeros(list_partition_sizes, dtype=numba.int64)
    int_degree=numpy.zeros((nodes, list_partition_sizes), dtype=numba.float64)
    np=numpy.zeros((nodes,list_partition_sizes), dtype=numba.float64)
    nr=numpy.zeros((nodes,list_partition_sizes), dtype=numba.float64)
            
    # filling the structures for every edge of the network
    for edge in edges:
        _fill(sum_degrees_com, int_degree, vector, edge)

    # computing Node Predominance and Node Recall for every node and every community
    for u in range(len(vector)):
        for com in range(list_partition_sizes):
            if sum_degrees_com[com] == 0:
                np[u][com]= 0
            else:
                np[u][com] = int_degree[u][com] / sum_degrees_com[com]
            if weighted_degrees[u] == 0:
                nr[u][com] = 0
            else:
                nr[u][com] = int_degree[u][com] / weighted_degrees[u]
                           
    return np, nr

def get_nfm_embeddings(edges, weights, vector, list_partition_sizes, nodes):
    """Procedure to generate embedding vectors for the network G
    
    Keyword arguments:
        edges -- list of edges of the netword as triplets: (src,dst,weight)
        weighted -- list of weighted degrees of all nodes (from 0 to nodes-1)
        vector -- the vector describing community structure
        list_partition_sizes -- a list containing the size of every community
        nodes -- number of nodes of the network
    """
        
    # adapting lists to numba
    numba_edges = List()
    [numba_edges.append((edge[0],edge[1],edge[2])) for edge in edges]
    numba_weighted = List()
    [numba_weighted.append(weights[node]) for node in range(nodes)]
    
    # computing Node Predominance and Node Recall
    np,nr=_do_node_fmeasure(numba_edges, numba_weighted, vector, list_partition_sizes, nodes)
    
    # preparing embedding vectors
    np=numpy.asmatrix(np)
    nr=numpy.asmatrix(nr)
    embedding_matrix=numpy.concatenate((np, nr), axis=1)
    
    return np,nr,embedding_matrix
                

if __name__ == '__main__': 
    from parallel_nfm import get_nfm_embeddings as gne
    
    small_graphs = [
                        ("../../rolesAsAVec/data/dolphins/dolphins.ed",networkit.Format.EdgeListSpaceZero)
            ]
    
    medium_graphs = [
                    #("../../data/citeseer/citeseer.renum", networkit.Format.EdgeListSpaceZero),
                    ("../../rolesAsAVec/data/citeseer/citeseer.renum_clean", networkit.Format.EdgeListSpaceZero),
                    #("../../data/cora/cora.renum", networkit.Format.EdgeListSpaceZero),
                    ("../../rolesAsAVec/data/cora/cora.renum_clean", networkit.Format.EdgeListSpaceZero),
                    #("../../data/email_eu/email-Eu-core.txt", networkit.Format.EdgeListSpaceZero), 
                    #("../../data/email_eu/email-Eu-core.txt_clean", networkit.Format.EdgeListSpaceZero),
                    #("../../data/ca-AstroPh/out.ca-AstroPh", networkit.Format.EdgeListSpaceOne),
                    ("../../rolesAsAVec/data/ca-AstroPh/out.ca-AstroPh_clean", networkit.Format.EdgeListSpaceZero) 
                ]
    
    large_graphs = [
                    #("../../data/facebook-wosn-linetworkits/out.facebook-wosn-linetworkits.ed_clean", networkit.Format.EdgeListSpaceZero),
                    #("../../data/youtube/youtube-linetworkits.txt", networkit.Format.EdgeListTabOne),
                    #("../../data/flickr/flickr-linetworkits.txt", networkit.Format.EdgeListTabOne),
                ]

    for graph, formatnetworkit in medium_graphs:
        G=networkit.readGraph(graph, formatnetworkit)
        #G=networkit.readGraph("../../../../data/facebook-wosn-linetworkits/out.facebook-wosn-linetworkits.ed_clean", networkit.Format.EdgeListSpaceZero)
        G.removeSelfLoops()
        communities=networkit.community.PLM(G)
        communities.run()
        partition=communities.getPartition()
        print("Numba version on graph", graph.split("/")[-1], ", execution time ", end='')
        edges = [(edge[0], edge[1], G.weight(edge[0],edge[1])) for edge in G.edges()]
        weights = [G.weightedDegree(u) for u in G.nodes()]
        np,nr,embedding_matrix = get_nfm_embeddings(edges, weights, partition.getVector(), partition.numberOfSubsets(), G.numberOfNodes())
        print("Parallel version on graph", graph.split("/")[-1], "execution time ", end='')
        np,nr,matrix = gne(G,partition)    
        print("Same results in parallel ? ",(matrix==embedding_matrix).all())
    