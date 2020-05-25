# -*- coding: utf-8 -*-
import numba, timeit
from numba.typed import List 
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")

@numba.njit(parallel=True)
def _fill(sum_degrees_com, int_degree_indptr, int_degree_indices, int_degree_data, vector, edge, index):
    """updates total degree of edges incident to a community and 
    the degree of extremities of the edge towards the corresponding communities.
    
    Keyword arguments:
        sum_degrees_com -- the list of total degree of edges for communities
        int_degree_indptr -- beginning of each row of the CSR matrix
        int_degree_indices -- column indices each row of the CSR matrix
        int_degree_data -- all values stored in the CSR matrix
        vector -- the vector describing community structure
        edge -- the edge to be considered
    """
    src,dst,weight=edge
    cpt=2*index
    try:
        #to avoid counting twice edges inside a community
        if vector[src]==vector[dst]:
            sum_degrees_com[vector[src]]+=weight
        else:
            sum_degrees_com[vector[src]]+=weight
            sum_degrees_com[vector[dst]]+=weight
            
        int_degree_indptr[cpt]=src
        int_degree_indices[cpt]=vector[dst]
        int_degree_data[cpt]=weight
        cpt+=1
        int_degree_indptr[cpt]=dst
        int_degree_indices[cpt]=vector[src]
        int_degree_data[cpt]=weight
    except: 
        pass
    
@numba.njit(parallel=True)
def _arrays_nfm(edges, vector, size_partition, nb_nodes,verbose=False):
    sum_degrees_com=[0]*size_partition
    
    int_degree_indptr = [0]*(2*len(edges))
    int_degree_indices = [0]*(2*len(edges))
    int_degree_data = [0.]*(2*len(edges))
    
    if(verbose):
        print("computing community degrees and internal node degrees...")
    for index,edge in enumerate(edges):
        _fill(sum_degrees_com, int_degree_indptr, int_degree_indices, int_degree_data, vector, edge, index)
                        
    return sum_degrees_com, int_degree_indptr, int_degree_indices, int_degree_data

@numba.njit(parallel=True)
def _compute_nfm(sum_degrees_com, vector, size_partition,weighted,indptr,indices,data,positions,verbose=False):
    
    # computing the size of each needed array
    size_np = len([1 for _,com in positions if sum_degrees_com[com] != 0])
    size_nr = len([1 for u,_ in positions if weighted[u] !=0])
    
    np_indptr = [0]*size_np
    np_indices = [0]*size_np
    np_data = [0.0]*size_np

    nr_indptr = [0]*size_nr
    nr_indices = [0]*size_nr
    nr_data = [0.0]*size_nr

    if(verbose):
        print("computing Node Predominance and Node Recall...")
    index_np, index_nr=0,0
    for u, com in positions:
        if sum_degrees_com[com] != 0:
            np_indptr[index_np]=u
            np_indices[index_np]=com
            #np_data[index_np]=(value/sum_degrees_com[com])
            np_data[index_np]=get_item(u,com,indptr,indices,data) / sum_degrees_com[com]
            index_np+=1
        if weighted[u] != 0:
            nr_indptr[index_nr]=u
            nr_indices[index_nr]=com
            #nr_data[index_nr]=(value/weighted[u])
            nr_data[index_nr]=get_item(u,com,indptr,indices,data) / weighted[u]
            index_nr+=1
    return np_indptr, np_indices, np_data, nr_indptr, nr_indices, nr_data

@numba.njit(parallel=True)
def get_item(row_index, column_index, indptr, indices, data):
    # Get row values
    row_start = indptr[row_index]
    row_end = indptr[row_index + 1]
    row_values = data[row_start:row_end]

    # contains indices of occupied cells at a specific row
    row_indices = list(indices[row_start:row_end])

    # Find a positional index for a specific column index
    value_index = row_indices.index(column_index)
    return row_values[value_index]

def get_nfm_embeddings(edges, weights, vector, size_partition, nodes,verbose=False):

    numba_edges = List()
    [numba_edges.append((edge[0],edge[1],edge[2])) for edge in edges]
    numba_weighted = List()
    [numba_weighted.append(weights[node]) for node in range(nodes)]
    
    sum_degrees_com,indptr,indices,data=_arrays_nfm(numba_edges, vector, size_partition, nodes,verbose)
    int_degree=csr_matrix((data,(indptr,indices)), shape=(nodes,size_partition))
    positions=[(i,j) for i, j in zip(*int_degree.nonzero())]
    #positions=List()
    #[positions.append((i, j, int_degree[i,j])) for i, j in zip(*int_degree.nonzero())]
    
    #print(sys.getsizeof(items)/1024.,"kylobytes")    
    if(verbose):
        print("computing embeddings...")
    np_indptr, np_indices, np_data, nr_indptr, nr_indices, nr_data = _compute_nfm(sum_degrees_com, vector, size_partition, numba_weighted, int_degree.indptr,int_degree.indices,int_degree.data,positions,verbose)
    np=csr_matrix((np_data,(np_indptr,np_indices)), shape=(nodes,size_partition))
    nr=csr_matrix((nr_data,(nr_indptr,nr_indices)), shape=(nodes,size_partition))

    # FIXME: works only on identical shape matrices!
    def concatenate_csr_matrices(matrix1, matrix2):
        new_indptr = 2*matrix1.indptr
        new_indices = []
        new_data = []
        
        for r in range(len(matrix1.indptr)-1):
            row_start = matrix1.indptr[r]
            row_end = matrix1.indptr[r + 1]
            # contains indices of occupied cells at a specific row
            row_indices1 = list(matrix1.indices[row_start:row_end])
            row_indices2 = list(matrix2.indices[row_start:row_end])
            new_indices.extend(row_indices1)
            ri = [e+size_partition for e in row_indices2]
            new_indices.extend(ri)
            
            row_values1 = matrix1.data[row_start:row_end]
            row_values2 = matrix2.data[row_start:row_end]
            new_data.extend(row_values1)
            new_data.extend(row_values2)
    
        return csr_matrix((new_data, new_indices, new_indptr))
  
    return np, nr, concatenate_csr_matrices(np,nr)