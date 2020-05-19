import nfm2vec.nfm as nfm
import sys
import networkx as nx
from community import community_louvain

if __name__ == "__main__":
    
    if(len(sys.argv) != 2):
        print("Usage: python networkx_nfm.py relative_graph_path")
        exit(0)
    else:
        graph = sys.argv[1]
        # NOTE: the graph needs to be from 0 to n-1
        G = nx.read_weighted_edgelist(sys.argv[1],nodetype=int)
        
        # computing louvain partition
        partition = community_louvain.best_partition(G)
        # extracting partition vector
        partition_vector = list(partition.values()) 
        # initializing list of edges 
        # data specifies the attribute name and default value for edges
        edges = [(edge[0],edge[1],edge[2]) for edge in G.edges.data('weight', default=1)]
        weights = [G.degree(u) for u in G.nodes()]
        number_communities = max(partition)+1
        nodes = len(G)
        # computing embedding vectors
        np,nr,embedding_matrix = nfm.get_nfm_embeddings(edges, weights, partition_vector, number_communities, nodes)