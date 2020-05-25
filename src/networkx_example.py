if __name__ == "__main__":
    from scipy.sparse import csr_matrix
    import numpy as np
    import networkx as nx
    import nfm2vec.nfm_sparse as nfm_sparse
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
                
        # initializing list of edges 
        # data specifies the attribute name and default value for edges
        edges = [(edge[0],edge[1],edge[2]) for edge in G.edges.data('weight', default=1)]
        weights = [G.degree(u) for u in G.nodes()]
        number_communities = max(partition_vector)+1
        nodes = len(G)
        # computing embedding vectors --- CSR numpy format: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
        np,nr,embedding_matrix = nfm_sparse.get_nfm_embeddings(edges, weights, partition_vector, number_communities, nodes)
        """le format CSR est particulier : une matrice est représentée par trois tableaux indptr, indices et data
        + indptr représente le début de chaque ligne, ou autrement dit le nombre de valeurs non nulles pour chaque ligne
        pour représenter une matrice 4x4, si on a indptr = [0, 3, 3, 7, 10] alors :
            * la première ligne a 3 valeurs non nulles
            * la seconde n'en a aucune
            * la troisième en a 7
            * la quatrième en a 3
        + indices représente le numéro de la colonne ayant une valeur non nulle sur chaque ligne
        + data contient toutes les valeurs non nulles de la matrice
        
        PAR EXEMPLE : 
            [[1,0,2,3],
             [0,0,0,0],
             [1,2,3,4],
             [1,1,0,4]]
        est représentée par :
            indptr = [0,3,3,7,10]
            indices = [0,2,3,0,1,2,3,0,1,3]
            data = [1,2,3,1,2,3,4,1,1,4]"""
        indptr = [0,3,3,7,10]
        indices = [0,2,3,0,1,2,3,0,1,3]
        data = [1,2,3,1,2,3,4,1,1,4]
        matrix=csr_matrix((data,indices,indptr))
        print(matrix,matrix.todense(),sep='\n')
        # les trois tableaux sont accessibles
        print(matrix.indptr, matrix.indices, matrix.data, sep='\n')
            
        # accéder aux éléments peut se faire très facilement
        print(matrix[0,2])
        # mais est parfois un peu lent : on a donc une méthode get_item dans nfm_sparse
        print(nfm_sparse.get_item(0,2,matrix.indptr,matrix.indices,matrix.data))
