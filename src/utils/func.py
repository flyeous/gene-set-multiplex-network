import anndata
import contextlib
import copy
import itertools
import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
import scanpy as sc
import scipy
from sklearn.neighbors import NearestNeighbors
import sknetwork as skn
from tqdm import tqdm
import umap
import warnings
warnings.filterwarnings('ignore')
### warnings.filterwarnings(action='once')


### monoplex/multiplex PageRank centrality
def column_normalization(A):
    A = A / np.sum(A, axis = 0)
    return A

def page_rank(A, alpha, iteration, threshold = 1e-5):
    N = A.shape[0]
    current_rank = np.repeat(1/N, N).reshape(-1, 1)
    A_power = (alpha)*A + (1-alpha)/N
    converge = False
    for i in range(iteration):
        next_rank = A_power@current_rank
        difference = current_rank - next_rank
        current_rank = next_rank
        if np.linalg.norm(difference) < threshold:
            converge = True
            break
    if not converge:
        print(f'The PageRank does not converge with {iteration} iteration times and convergence threshold {threshold}')
        return
    return current_rank.ravel()

def skn_page_rank(A, alpha = 0.85, iteration = 1000, threshold=1e-5):
    model = skn.ranking.PageRank(damping_factor = 0.85, n_iter= 1000, tol = 1e-5)
    model.fit(A)
    return model.scores_

def skn_hits(A):
    model = skn.ranking.HITS()
    model.fit(A)
    hub, authority = model.scores_row_, model.scores_col_
    return hub, authority

def multiplex_page_rank(A_lst, alpha_lst, iteration_lst, beta, gamma, threshold = 1e-7):
    N = A_lst[0].shape[0]
    init_rank = np.repeat(1/N, N).reshape(-1, 1)
    for i in range(len(A_lst)):
        if i == 0:
            current_rank = page_rank(column_normalization(A_lst[i]), alpha = alpha_lst[i], iteration = iteration_lst[0], threshold = threshold)
        else:  
            temp_adj = np.diag(current_rank.ravel()**(beta))@(A_lst[i])
            col_norm = np.sum(temp_adj, axis = 0)
            temp_adj = temp_adj/col_norm
            current_rank_iter = copy.deepcopy(init_rank)
            for j in range(iteration_lst[i]):
                next_rank_iter = alpha_lst[i]*temp_adj@current_rank_iter + (1-alpha_lst[i])*(current_rank**(gamma)/np.sum(current_rank**(gamma))).reshape(-1,1) 
                difference = next_rank_iter - current_rank_iter
                current_rank_iter = next_rank_iter
            if np.linalg.norm(difference) >= threshold:
                print(f'The multiplex PageRank coefficients do not converge in layer {i}')
                return
            else:
                current_rank = copy.deepcopy(current_rank_iter)
                
    return current_rank.ravel()

def multiplex_participation_coefficient(adj_filtered_lst, self_loop = False, replace = True, layer = False):
    assert adj_filtered_lst is not None, "adj_filtered_lst is None!"
    adj_lst = copy.deepcopy(adj_filtered_lst)
    N = adj_lst[0].shape[0]
    M = len(adj_lst)
    ### remove self-loops
    iden_mat = np.diag(np.repeat(1, N))
    res_lst = list()
    for alpha in adj_lst:
        if not self_loop:
            C_i = np.sum(alpha-iden_mat, axis = 1)
        else:
            C_i = np.sum(alpha, axis = 1)
        res_lst.append(C_i)
    O_i = np.sum(res_lst, axis = 0)
    d_i = (np.array(res_lst)/O_i)**2
    if layer:
        return d_i
    P_i = M/(M-1)*(1- np.sum(d_i, axis = 0))
    if replace:
        P_i  = replace_min(P_i , label = "multiplex participation coefficient")
    return O_i, P_i


def C_1(adj_lst, replace = True):
    adj_lst = copy.deepcopy(adj_lst)
    N = adj_lst[0].shape[0]
    M = len(adj_lst)
    ### remove self-loops
    iden_mat = np.diag(np.repeat(1, N))
    adj_lst = [(mat - iden_mat) for mat in adj_lst]
    numerator = 0
    for i,j in itertools.permutations(range(M),2): 
        numerator += np.diag((adj_lst[i]+adj_lst[i].transpose())@(adj_lst[j]+adj_lst[j].transpose())@(adj_lst[i]+adj_lst[i].transpose()))  
    denominator = 0
    for i in range(M):
        denominator += np.sum(adj_lst[i]+adj_lst[i].transpose(), axis = 1)**2 - np.sum((adj_lst[i]+adj_lst[i].transpose())**2, axis = 1)
    denominator *= 2*(M-1)
    C_1 = numerator/denominator
    if replace:
        C_1 = replace_min(C_1, label = "C1") 
    return C_1 

def C_2(adj_lst, replace = True, num_core = 30):
    adj_lst = copy.deepcopy(adj_lst)
    N = adj_lst[0].shape[0]
    M = len(adj_lst)
    ### remove self-loops
    iden_mat = np.diag(np.repeat(1, N))
    adj_lst = [(mat - iden_mat) for mat in adj_lst]
    numerator = 0
    for i,j,k in itertools.permutations(range(M),3):
        numerator += np.diag((adj_lst[i]+adj_lst[i].transpose())@(adj_lst[j]+adj_lst[j].transpose())@(adj_lst[k]+adj_lst[k].transpose()))
    denominator = 0
    for i,j in itertools.permutations(range(M),2): 
        denominator += np.sum(adj_lst[i]+adj_lst[i].transpose(), axis = 1)*np.sum(adj_lst[j]+adj_lst[j].transpose(), axis = 1) - np.sum((adj_lst[i]+adj_lst[i].transpose())*(adj_lst[j]+adj_lst[j].transpose()), axis = 1)
    
    denominator *= 2*(M-2)
    C_2 = numerator/denominator
    if replace:
         C_2 = replace_min(C_2, label = "C2")
    return C_2


def _knn_similarity(i,j, knn):
    if i < j:
        mat_i = knn[i]
        mat_j = knn[j]
        N = mat_i.shape[0]
        ### exclude itself
        K = mat_i.shape[1]-1
        return np.sum([len(set(mat_i[h,1:]).intersection(set(mat_j[h,1:]))) for h in range(N)])/(N*K)
    else:
        return 0
    



def KNN_graph_similarity(U_lst, k = 30, num_core = 60, min_feature = 4, knn_para = {'metric':'euclidean', 'algorithm':'auto', 'n_jobs': 60}):
    knn = []
    print('Finding k-nearest neighbors of gene sets...')
    for i in tqdm(range(len(U_lst))):
        U = U_lst[i]
        if U.shape[1] < min_feature:
            print(f'gene set {i} has less than {min_feature} features!')
            return
        N = U.shape[0]
        nb = NearestNeighbors(n_neighbors=k+1, **knn_para).fit(U)
        _, indices = nb.kneighbors(U)
        knn.append(indices)
    print('Computing the similarity matrix...')
    length = len(knn)
    with Pool(num_core) as proc: 
        res_mat = proc.starmap(_knn_similarity, [(i,j, knn) for i in range(length) for j in range(length)])
        res_mat = np.reshape(np.array(res_mat), (length, length)) 
        res_mat = full_matrix(res_mat)
    print('Complete!')
    
    return res_mat
    

def full_matrix(half_matrix):
    ### the diagnals are ones
    return half_matrix + np.transpose(half_matrix) + np.diag(np.ones(half_matrix.shape[0]), k = 0)


### for tissue objects
def JC(i:int, j:int, gs_list:list) -> float:
    return len(set(gs_list[i]).intersection(set(gs_list[j])))/len(set(gs_list[i]).union(set(gs_list[j])))


def _modified_JC_worker(gs_i, gs_j, gene_weight_dict):
    intersection = set(gs_i).intersection(set(gs_j))
    union = set(gs_i).union(set(gs_j))
   
    numerator = np.sum([gene_weight_dict[gene] for gene in intersection])
    denominator = np.sum([gene_weight_dict[gene] for gene in union])
    if denominator == 0:
        print("Two gene sets have zero gene expression! Missing value is replaced by 0.")
        return 0
    else:
        return numerator/denominator


def modified_JC(gene_set_lst, gene_exp:anndata.AnnData, num_core = 30):
    gene_weight = np.array(np.sum(gene_exp.X, axis = 0)/(gene_exp.X.shape[0])).ravel()
    gene_in_exp = gene_exp.var.index.to_numpy()
    gene_weight_dict = dict(zip(gene_in_exp, gene_weight))
    filtered_gs_lst = [list(set(gs).intersection(gene_in_exp)) for gs in gene_set_lst]
    
    
    length = len(gene_set_lst)
    with Pool(num_core) as proc:           
        mat = proc.starmap(_modified_JC_worker, [(gs_i,gs_j, gene_weight_dict) \
                                                     for gs_i in filtered_gs_lst for gs_j in filtered_gs_lst]) 
        mat = np.reshape(np.array(mat), (length, length))
    return mat
    
def gene_set_overlap_check(gene_set:list, gene_exp:anndata.AnnData, min_feature = 10, max_feature = 2000, zero_tr = 0.9):
    gex_mat = []
    overlap_prop = []
    filtered_index = []
    filtered = 0
    for i in tqdm(range(len(gene_set))):
        temp_gene_list = list(set(gene_set[i]).intersection(gene_exp.var.index))
        proportion = len(temp_gene_list)/len(gene_set[i])
  
        if len(temp_gene_list) < min_feature or len(temp_gene_list) > max_feature:
            filtered += 1
            continue
        ### gene expression matrix: cells * genes (in the fitered gene set)
        temp_gene_mat = gene_exp[:,temp_gene_list]
        ### large proportionate cells with zero-counts.
        zero_prop = np.sum((np.sum(temp_gene_mat.X, axis = 1) == 0))/temp_gene_mat.X.shape[0]
        if zero_prop > zero_tr:
            filtered += 1
            continue
        gex_mat.append(temp_gene_mat.X)
        overlap_prop.append(proportion)
        filtered_index.append(i)
    print(f'{len(gene_set) - filtered} over {len(gene_set)} pass the filteration!')
    return gex_mat, overlap_prop, filtered_index


### for network objects
def self_loop_exclude(sim_mat_lst):
    mat_lst = copy.deepcopy(sim_mat_lst)
    N = mat_lst[0].shape[0]
    iden_mat = np.diag(np.repeat(1, N))
    mat_lst = [(mat - iden_mat) for mat in mat_lst]
    return mat_lst
    
def adj_filter_tr(adj_mat,q = None, tr = 0):
    adj_mat = copy.deepcopy(adj_mat)
    ### negative edges are removed
    adj_mat[adj_mat < 0] = 0 
    ### weights lower than a threshold or quantile truncated to zero
    if tr is None:
        tr = np.quantile(adj_mat, q)
        print(f'The {q*100} quantile value is {tr}.')
    adj_mat[adj_mat < tr] = 0 
    ### min-max normalization
    (adj_mat - np.min(adj_mat))/(np.max(adj_mat) - np.min(adj_mat))
    return adj_mat
        

### Replace missing values by the non-missing minimal value
def replace_min(x, label = None ):
    vec = copy.deepcopy(x)
    missing_index = np.where(np.isnan(vec) == True)[0]
    print(f'The raw {label} coefficients contain {len(missing_index)} missing values.')
    if len(missing_index) != 0:
        min_val = np.min(np.delete(vec, missing_index))
        vec[missing_index] = min_val
        print(f"The missing values have been replaced by the minimal non-missing value:{min_val:.3f}.")
    return vec
    
    
def abridge_GO(GO, sep = ' ', cut = 15):
    GO = GO.replace(sep, ' ')
    voc =  GO.split(' ')
    length = np.cumsum([len(item) for item in voc])
    if length[-1] <= cut:
        return GO
    index = np.min(np.where(length >= cut)[0])
    GO_abridge = GO[:(length[index]+index+1)] + '\n' + abridge_GO(GO[(length[index]+index+1):], cut = cut)
    return GO_abridge

def annotation_table(index:dict, df_vis, num_layer = 6, sep = ' ', nb_lst = None,  position_only = False, ID_only = False, No_ID = False, first_col = False):
#     for item in list(index.keys()):
#         print('The keys contain ' + item)
    gs_names = df_vis['Description']
    if not No_ID:
        gs_ID = df_vis['ID']
    marker = []
    index_lst = []
    ### add a Jaccard layer
    for i in range(num_layer):
        index_sub_lst = []
        marker_sub = []
        for j in range(len(index)):
            index_sub_lst.append(list(index.values())[j])
            label = gs_names[index_sub_lst[-1]] 
            if position_only:
                if nb_lst is not None:
                    if j in nb_lst:
                        marker_sub.append(gs_ID[index_sub_lst[-1]])
                    else:
                        marker_sub.append(' ')
                else:
                    marker_sub.append(' ')
            elif ID_only:
                marker_sub.append(gs_ID[index_sub_lst[-1]])
            elif not No_ID:
                marker_sub.append(gs_ID[index_sub_lst[-1]] + ":\n" + abridge_GO(label, sep = sep))
            else:
                marker_sub.append(abridge_GO(label, sep = sep))
        index_lst.append(index_sub_lst)
        marker.append(marker_sub)
    loc = []
    for i in range(num_layer):
        loc_sub = []
        for j in range(len(index)):
            if  not first_col:
                loc_sub.append(df_vis.iloc[index_lst[i][j],[i*2+1,i*2+2]])
            else:
                loc_sub.append(df_vis.iloc[index_lst[i][j],[i*2,i*2+1]])
        loc.append(loc_sub)
    df_annotate = pd.DataFrame(dict(marker = marker, loc = loc))
    return df_annotate
    
    
def xytext_loc(t, loc, r = 2):
    '''
    loc: df_annotate['loc']
    
    '''
    return_lst = []
    for i in range(len(loc)):
        layer_lst = []
        for j in range(len(loc[i])):
            theta = t[i][j]
            dist = np.array([[np.cos(theta), -1*np.sin(theta)], 
                                                  [np.sin(theta), np.cos(theta)]])@np.array([r,0])
            layer_lst.append((np.array(loc[i][j]) + dist).ravel())
        return_lst.append(layer_lst)
    return return_lst

def membership_by_property(res_gsea, top = 20, GO_ID = True, keys = ['NES', 'Multiplex.PageRank', 'PageRank.Jaccard', 'P', 'C1', 'C2']):
    if GO_ID:
        select = 'ID'
    else:
        select = 'Description'
    return_dict = {}
    for key in keys:
        if key == "NES":
            return_dict[key] = res_gsea.sort_values(by = key, ascending = False, key = np.abs)[select][:top]
        elif key == "P":
            return_dict[key] = res_gsea.sort_values(by = key, ascending = True)[select][:top]
        else:
            return_dict[key] = res_gsea.sort_values(by = key, ascending = False)[select][:top]
    return return_dict

### for the multipartite plot of GO sets
class GO_entity:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name

def term_to_entity(term_lst, layer_index):
    return_lst = []
    for item in term_lst:
            exec(f'{item.replace(":", "_")}_{layer_index} = GO_entity(item)') 
            return_lst.append(eval(f'{item.replace(":", "_")}_{layer_index}'))
    return return_lst

def entity_next_layer_index(entity, entity_lst):
    entity_lst_name = [item.name for item in entity_lst]
    if entity.name not in entity_lst_name:
        return None, False
    else:
        index = np.where(entity.name == np.array(entity_lst_name))[0][0]
        return index, True

def edge_lst_from_layers(layer_lst_ordered):
    return_lst = []
    length = len(layer_lst_ordered)
    for i in range(length):
        if i == length-1:
            break
        else:
            current_layer = layer_lst_ordered[i]
            next_layer = layer_lst_ordered[i+1]
            current_edge_lst = [(current_layer[j], next_layer[entity_next_layer_index(current_layer[j],  next_layer)[0]])\
                  for j in range(len(current_layer)) if entity_next_layer_index(current_layer[j], next_layer)[1]] 
            return_lst += current_edge_lst
    return return_lst


def _fold_change(i,j,root_ID, descendent, GO_ID, adj_Jaccard, adj_filtered_lst):
    if i <= j:
        index_node_i = [np.where(item == np.array(GO_ID))[0][0] for item in descendent[i]]
        index_node_j = [np.where(item == np.array(GO_ID))[0][0] for item in descendent[j]]
        mat_Jaccard = adj_Jaccard[index_node_i,:][:,index_node_j]
        similarity_Jaccard = np.mean(mat_Jaccard)
        if similarity_Jaccard == 0:
            return np.nan
        similarity_multiplex = []
        for j in range(len(adj_filtered_lst)):
            mat_multiplex_layer = adj_filtered_lst[j][index_node_i,:][:,index_node_j]
            similarity_layer = np.mean(mat_multiplex_layer)
            similarity_multiplex.append(similarity_layer)
        return np.mean(similarity_multiplex)/similarity_Jaccard
    else:
        return 0


def GO_hierarchy_variation_detection(nt, go, GO_ID:list, GO_level:list, \
                                     level = 3, num_core = 5):
    root_ID = GO_ID[GO_level == level]
    ### filteration gene sets with zero child.
    num_descendent = []
    descendent = []
    for i in range(len(root_ID)):
        _str = list(root_ID)[i]
        d = go[_str].get_all_children().intersection(np.array(GO_ID))
        descendent.append(list(d))
        num_descendent.append(len(d))
    root_ID = root_ID[np.array(num_descendent) > 0]
    descendent = np.array(descendent)[np.array(num_descendent) > 0]
    length = len(root_ID)

    with Pool(num_core) as proc: 
        res_mat = proc.starmap(_fold_change,\
                               [(i,j,root_ID,descendent, GO_ID, nt.adj_Jaccard, nt.adj_filtered_lst) for i in range(length)\
                                for j in range(length)])
        res_mat = np.reshape(np.array(res_mat), (length, length)) 
        res_mat = res_mat + res_mat.transpose() - np.diag(np.diag(res_mat, k = 0))
    return res_mat, root_ID, descendent


def fetch_similarity_mat_descendent(i,j, GO_ID, descendent, adj_Jaccard, adj_filtered_lst):
    index_node_i = [np.where(item == np.array(GO_ID))[0][0] for item in descendent[i]]
    index_node_j = [np.where(item == np.array(GO_ID))[0][0] for item in descendent[j]]
    mat_Jaccard = adj_Jaccard[index_node_i,:][:,index_node_j]
    mat_multiplex_layer = [adj_filtered_lst[k][index_node_i,:][:,index_node_j] for k in range(len(adj_filtered_lst))]
    return mat_Jaccard, mat_multiplex_layer
    