import anndata
import copy
import contextlib
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import pandas as pd
import scanpy as sc
import time
from . import func

# import warnings
# warnings.filterwarnings('ignore')
# ### warnings.filterwarnings(action='once')
# ### Disable warnings: reference https://stackoverflow.com/questions/8391411
    
class tissue:
    
    _Jaccard_proj = {}
    def __init__(self, name, scRNAseq, processed, gene_set_collection_name, gene_set_gene_symbols, gene_set_names):
        self.name = name
        self.scRNAseq = scRNAseq
        self.processed = processed
        self.gene_set = {gene_set_collection_name: geneSet(raw_gs = gene_set_gene_symbols, raw_gs_names = gene_set_names)}
        self.proj = {}
        self.sampled_data = None
        
    def scRNAseq_release(self):
        self.scRNAseq = None
        self.sampled_data = None
    
    def cell_sampling(self, size, seed):
        assert self.scRNAseq is not None, "Load scRNAseq data first!"
        assert size <= self.scRNAseq.X.shape[0], f'sample size should be less than {self.scRNAseq.X.shape[0]}!'

        np.random.seed(seed)
        sampling = np.random.choice(self.scRNAseq.X.shape[0], size = size, replace = False)
        self.sampled_data = (seed, size, self.scRNAseq[sampling,])
        
    
    def add_scRNAseq(self, scRNAseq):
        self.scRNAseq = scRNAseq
        
    def add_gene_set(self,gene_set_collection_name, gene_set_gene_symbols, gene_set_names):
        self.gene_set[gene_set_collection_name] = geneSet(raw_gs = gene_set_gene_symbols, raw_gs_names = gene_set_names)
        
    def call_Jaccard(self, gene_set_collection_name, num_core = 30):
        self.gene_set[gene_set_collection_name]._Jaccard(num_core)
        
    def call_knn(self, gene_set_collection_name, k = 30, num_core = 60, \
                 knn_para = {'metric':'euclidean', 'algorithm':'auto', 'n_jobs': 60}):
        assert self.sampled_data is not None, 'Call cell_sampling first!'
        self.gene_set[gene_set_collection_name]._knn(sampled_data = self.sampled_data[2],k = k, num_core = num_core, knn_para = knn_para)
        
    def call_modified_Jaccard(self, gene_set_collection_name, num_core = 30):
        assert self.sampled_data is not None, 'Call cell_sampling first!'
        self.gene_set[gene_set_collection_name]._modified_Jaccard(sampled_data = self.sampled_data[2], num_core = num_core)      
        
class geneSet():
    
    methods = ['Jaccard', 'KNN']

    def __init__(self, raw_gs:list, raw_gs_names:list):
        self.gene_set = raw_gs
        self.gs_names = raw_gs_names
        self.cluster_labels = {}
        self.filter_index = {}
        self.filter_overlap = {}
        self.miscellaneous = {}
        self.sim_mat = {}

    def _Jaccard(self, num_core):

        length = len(self.gene_set)
        print("Computing the Jaccard coefficient similarity matrix ...")
        with Pool(num_core) as proc:           
            res = proc.starmap(func.JC, [(i,j, self.gene_set) for i in range(length) for j in range(length)]) 
        res = np.reshape(np.array(res), (length, length))   
        
        geneSet._Jaccard_mat = res
        self.sim_mat['Jaccard'] = geneSet._Jaccard_mat
        
        
    
    def _modified_Jaccard(self, sampled_data, num_core):
        assert self.gene_set is not None, 'self.gene_set is None.'
        data = sampled_data
        print("Computing the modified Jaccard coefficient similarity matrix ...")
        res = func.modified_JC(self.gene_set, data, num_core = num_core)  
        
        self.sim_mat['modified_Jaccard'] = res
 
        
    def _knn(self, sampled_data,  k, num_core, \
                 knn_para):
        assert self.gene_set is not None, 'gene_set is None.'
        data = sampled_data
        gex_mat, overlap_prop, filtered_index = func.gene_set_overlap_check(gene_set = self.gene_set, gene_exp = data)
        res = func.KNN_graph_similarity(U_lst = gex_mat, k = k, num_core = num_core, knn_para = knn_para)
        self.sim_mat['knn'] = res
        self.filter_index['knn'] = filtered_index
        self.filter_overlap['knn'] = overlap_prop
        

    
    
        
    
        
        
        
             
        