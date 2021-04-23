if __name__ == "__main__":
    import networkit as nk
    import SINr.SINr as SINr
    import time

    import warnings, sys
    warnings.filterwarnings("ignore")
    
    if(len(sys.argv) != 2):
        print("Usage: python networkit.py graph_path")
    else:
        # NOTE: the graph needs to be from 0 to n-1
        graph = sys.argv[1]
        #NOTE : format of the graph should be:
        #id1 id2 weight
        formatnk = nk.Format.EdgeListSpaceZero
        G=nk.readGraph(graph, formatnk)
        G.removeSelfLoops()
        
        communities=nk.community.PLM(G)
        communities.run()
        partition=communities.getPartition()
        
        # extracting partition vector
        partition_vector = partition.getVector()
        print(partition_vector)
        # initializing list of edges 
        # data specifies the attribute name and default value for edges
        edges = [(edge[0],edge[1],edge[2]) for edge in G.iterEdgesWeights()]
        weights = [G.degree(u) for u in G.nodes()]
        number_communities = partition.numberOfSubsets()
        nodes = G.numberOfNodes()
        # computing embedding vectors
        debut=time.time()
        print("computing embeddings...")
        np,nr,embedding_matrix = SINr.get_SINr_embeddings(edges, weights, partition_vector, number_communities, nodes)
        end=time.time()
        print("embeddings computed in {}s".format(end-debut))
