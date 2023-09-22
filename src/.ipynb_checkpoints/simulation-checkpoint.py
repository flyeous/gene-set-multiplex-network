import numpy as np
import scipy
import hoggorm
import dcor
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
# import skbio.stats (dependency conflict, older version required)



def graph_KNN(mat_A, mat_B, K = None, n_jobs = 60, metric= 'euclidean' ):
    N = mat_A.shape[0]
    if K is None:
        K = int(np.sqrt(N))
    nbrs = NearestNeighbors(n_neighbors = K+1, algorithm='auto', n_jobs = n_jobs, metric = metric).fit(mat_A)
    _, indices_A = nbrs.kneighbors(mat_A)
    nbrs = NearestNeighbors(n_neighbors = K+1, algorithm='auto', n_jobs = n_jobs, metric = metric).fit(mat_B)
    _, indices_B = nbrs.kneighbors(mat_B)
    return np.sum([len(set(indices_A[i,1:]).intersection(set(indices_B[i,1:]))) for i in range(N)])/(N*K)


def dispatch_func(method, mat_A, mat_B, metric = 'l2', print_if = True):
    if print_if:
        print(f'{method}:')
    if method == 'RV_modified':
        return hoggorm.mat_corr_coeff.RV2coeff([mat - np.mean(mat, axis = 0) for mat in [mat_A, mat_B]])[0,1]
    elif method == 'dcor':
        return dcor.distance_correlation(mat_A, mat_B, exponent = 1)
    elif method == 'mantel':
        dist_mat = [scipy.spatial.distance_matrix(mat, mat) for mat in [mat_A, mat_B]]
        return skbio.stats.distance.mantel(dist_mat[0], dist_mat[1], permutations = 0)[0]
    elif method == 'KNN':
        return graph_KNN(mat_A, mat_B, metric = metric)
    else:
        print('unsupported!')
        return