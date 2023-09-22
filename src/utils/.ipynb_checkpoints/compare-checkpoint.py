import anndata
import contextlib
import dcor
import hoggorm
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import scanpy as sc
import scipy.spatial
# import skbio.stats (dependency conflict)
from sklearn.metrics import pairwise_distances
import time
from . import func
from . import visualization



def worker_KNN(gs_lst:list, sampled_data:anndata.AnnData, num_core = 60):
    gex_mat, overlap_prop, _ = func.gene_set_overlap_check(gene_set = gs_lst, gene_exp = sampled_data, zero_tr = 0.8)
    N = gex_mat[0].shape[0]
    begin_time = time.time()
    sim_mat = func.KNN_graph_similarity(gex_mat, k = int(np.sqrt(N)), num_core = num_core, knn_para = {'metric':'euclidean', 'algorithm':'auto', 'n_jobs': num_core})
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'KNN takes {end_time - begin_time:.2f} s')
    
    return sim_mat, time_cost

def pairwise_compute(mat_i, mat_j, func, func_parameters):
    return func(mat_i, mat_j, **func_parameters)


def mantel_worker(sampled_gs:list, sampled_data:anndata.AnnData)-> (float, np.ndarray):
    gex_mat, overlap_prop, _ = func.gene_set_overlap_check(gene_set = sampled_gs, gene_exp = sampled_data, zero_tr = 0.8)
    length = len(gex_mat)
    begin_time = time.time()
    dist_mat_lst = [scipy.spatial.distance_matrix(gex_mat[i].toarray(), gex_mat[i].toarray()) for i in range(length)]
    res = [pairwise_compute(dist_i, dist_j, skbio.stats.distance.mantel, {"permutations":0})[0] for dist_i in dist_mat_lst for dist_j in dist_mat_lst]
    res = np.array(res).reshape(length, length)
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'mantel takes {end_time - begin_time:.2f} s')
    
    return (res, time_cost)
    
def RV2_worker(sampled_gs:list, sampled_data:anndata.AnnData)-> (float, np.ndarray):
    gex_mat, overlap_prop, _ = func.gene_set_overlap_check(gene_set = sampled_gs, gene_exp = sampled_data, zero_tr = 0.8)  
    length = len(gex_mat)
    begin_time = time.time()
    res_RV2 = hoggorm.mat_corr_coeff.RV2coeff([gex_mat[i] - np.mean(gex_mat[i], axis = 0) for i in range(length)])
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'RV2coeff takes {end_time - begin_time:.2f} s')
    return (res_RV2, time_cost)


def modified_Jaccard_worker(sampled_gs, sampled_data:anndata.AnnData, num_core = 60):
    _, _, filter_index = func.gene_set_overlap_check(gene_set = sampled_gs, gene_exp = sampled_data, zero_tr = 0.8)
    begin_time = time.time()
    res = func.modified_JC(gene_set_lst = np.array(sampled_gs)[filter_index], gene_exp = sampled_data, num_core = num_core)
    end_time = time.time()
    time_cost = end_time - begin_time
    print(f'Modified Jaccard takes {end_time - begin_time:.2f} s')
    return (res, time_cost)


### Hyperparameters
def run_duration_comparison_hyperparameter(sampled_gs:list, sampled_data:anndata.AnnData, worker, hyper_lst, num_core ):
    time_lst = []
    res_lst = []
    
    ### reference:
    res_RV2, _ = RV2_worker(sampled_gs, sampled_data)
    res_mantel, _ = mantel_worker(sampled_gs, sampled_data)
    
    for hp in hyper_lst:
        res_temp, time_temp = worker(sampled_gs, sampled_data, hp,  num_core = num_core)
        time_lst.append(time_temp)
        res_lst.append(res_temp)
        
    
    return res_lst, time_lst, res_RV2, res_mantel


def run_duration_comparison(sampled_gs:list, sampled_data:anndata.AnnData, num_core):
    time_list = []
    
    res_knn, time_cost = worker_KNN(sampled_gs, sampled_data, num_core = num_core)
    time_list.append(('KNN',time_cost))
    
    res_RV2, time_cost = RV2_worker(sampled_gs, sampled_data)
    time_list.append(('RV2coeff',time_cost))
    
    res_mantel, time_cost = mantel_worker(sampled_gs, sampled_data)
    time_list.append(('mantel',time_cost))
    
    res_mod, time_cost = modified_Jaccard_worker(sampled_gs, sampled_data)
    time_list.append(('Jaccard_mod',time_cost))
    
    res_list = [res_knn, res_RV2, res_mantel, res_mod]
    return time_list, res_list


