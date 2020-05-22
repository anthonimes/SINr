if __name__ == "__main__":
    import networkx as nx
    import nfm2vec.nfm as nfm
    import warnings, sys
    warnings.filterwarnings("ignore")
    
    from community import community_louvain
    
    if(len(sys.argv) != 2):
        print("Usage: python networkit.py graph_path")
    else:
        graph = sys.argv[1]
        # NOTE: the graph needs to be from 0 to n-1
        G=nx.read_weighted_edgelist(graph, nodetype=int)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        partition = community_louvain.best_partition(G)
        
        # extracting partition vector
        partition_vector = list(partition.values())
        '''partition_vector=[[]]*(max(partition)+1)
        
        for u,c in enumerate(partition):
            if(partition_vector[c]):
                partition_vector[c].append(u)
            else:
                partition_vector[c]=[u]'''
                
        # initializing list of edges 
        # data specifies the attribute name and default value for edges
        edges = [(edge[0],edge[1],edge[2]) for edge in G.edges.data('weight', default=1)]
        weights = [G.degree(u) for u in G.nodes()]
        number_communities = max(partition_vector)+1
        nodes = len(G)
        # computing embedding vectors
        np,nr,embedding_matrix = nfm.get_nfm_embeddings(edges, weights, partition_vector, number_communities, nodes)