if __name__ == "__main__":
    from scipy.sparse import csr_matrix
    import networkx as nx
    import SINr.SINr as SINr
    import warnings, sys
    import time, random
    warnings.filterwarnings("ignore")
    
    from community import community_louvain
   
    if(len(sys.argv) > 2):
        print("Usage: python networkit.py graph_path")
    elif(len(sys.argv)==1):
        G = nx.erdos_renyi_graph(1500, 0.2, seed=None, directed=False)
        for (u,v,w) in G.edges(data=True):
            w['weight'] = random.random()
    else:
        graph = sys.argv[1]
        G=nx.read_weighted_edgelist(graph, nodetype=int)
        G.remove_edges_from(nx.selfloop_edges(G))

    # NOTE: the graph needs to be from 0 to n-1
    partition = community_louvain.best_partition(G)
    
    # extracting partition vector
    partition_vector = list(partition.values())
            
    # initializing list of edges 
    # data specifies the attribute name and default value for edges
    edges = [(edge[0],edge[1],edge[2]) for edge in G.edges.data('weight', default=1)]
    weights = [G.degree(u,weight='weight') for u in G.nodes()]

    number_communities = max(partition_vector)+1
    nodes = len(G)
    # computing embedding vectors --- CSR format: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    debut=time.time()
    print("computing embeddings...")
    np,nr,embedding_matrix = SINr.get_SINr_embeddings(edges, weights, partition_vector, number_communities, nodes)
    end=time.time()
    print("embeddings computed in {}s".format(end-debut))
