if __name__ == "__main__":
    import networkit as nk
    import nfm2vec.nfm as nfm

    import warnings, sys
    warnings.filterwarnings("ignore")
    
    if(len(sys.argv) != 3):
        print("Usage: python networkit.py graph_path networkit_format")
    else:
        graph = sys.argv[1]
        formatnk = nk.Format.EdgeListSpaceZero
        # NOTE: the graph needs to be from 0 to n-1
        G=nk.readGraph(graph, formatnk)
        G.removeSelfLoops()
        
        communities=nk.community.PLM(G)
        communities.run()
        partition=communities.getPartition()
        
        # extracting partition vector
        partition_vector = partition.getVector()
        # initializing list of edges 
        # data specifies the attribute name and default value for edges
        edges = [(edge[0],edge[1],1.) for edge in G.edges()]
        weights = [G.degree(u) for u in G.nodes()]
        number_communities = partition.numberOfSubsets()
        nodes = G.numberOfNodes()
        # computing embedding vectors
        np,nr,embedding_matrix = nfm.get_nfm_embeddings(edges, weights, partition_vector, number_communities, nodes)