from src.utils import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
import seaborn as sns
import copy
import os

def s_knn(processed_scRNAseq_obj_lst, population_labels, gene_set_dict, output_folder,\
                      num_core = 60, sample_size = None, parameter =  {'k':None,
                        'knn_para': {'metric':'euclidean', 'algorithm':'auto', 'n_jobs': 60}},
                      ):

    def _create(processed_scRNAseq_obj, tissue_label,\
               sample_size, gene_set_gene_symbols, gene_set_annotation, gene_set_collection_name,\
               output_folder):
        
        def _init(tissue_obj, sample_size = sample_size):
            k = parameter['k']
            if sample_size is None:
                sample_size = tissue_obj.scRNAseq.X.shape[0]
            if k is None:
                k = int(np.sqrt(sample_size))
            tissue_obj.cell_sampling(size = sample_size, seed = 123)
            tissue_obj.call_knn(gene_set_collection_name = gene_set_collection_name, k = k, num_core = num_core, knn_para = parameter['knn_para'])
            tissue_obj.scRNAseq_release()
            ### save tissue objects
            np.save(f'{output_folder}/{tissue_label}_{gene_set_collection_name}.npy', tissue_obj)
    
            print(f'The pipeline for {tissue_label} finishes!')
           
    
        obj = tissue.tissue(name = tissue_label, processed = True, 
                 scRNAseq = processed_scRNAseq_obj,
                 gene_set_collection_name = gene_set_collection_name,
                 gene_set_gene_symbols = gene_set_gene_symbols,
                 gene_set_names = gene_set_annotation)
        print(f'Population {tissue_label} has been instantiated.')
        _init(tissue_obj = obj)
            
    for i in range(len(processed_scRNAseq_obj_lst)):
        _create(processed_scRNAseq_obj_lst[i], population_labels[i],\
               **gene_set_dict,\
                    sample_size = sample_size, output_folder = output_folder)
        
        
        
### cd into the directory where tissue objects are stored.
def tissue_obj_multiplex_network(collection, labels = None, folder = '.'):
    file_ns_lst = os.listdir(folder)
    assert file_ns_lst != None, 'the folder is empty!'
    file_ns_lst = [item for item in file_ns_lst if (item[item.find('.'):] == ".npy")]
    
    if (labels == None):
        labels = [item[:item.find('_')] for item in file_ns_lst]
    
    tissue_obj = [np.load(folder + '/' + ns, allow_pickle = True).tolist() for ns in file_ns_lst]

    length = len(labels)
    nt = network.multiplex_tissue_network(gene_set_collection_name = collection, layer = tissue_obj,\
                                                layer_ns = labels, method = "knn", self_loop = True)
    return(nt)


def gene_set_similarity_multiplex_network(processed_scRNAseq_obj_lst, population_labels, gene_set_dict, output_folder = None,\
                      collection = "GO-BP", sample_size = None, labels = None,
                      num_core = 60, parameter =  {'k':None,
                        'knn_para': {'metric':'euclidean', 'algorithm':'auto', 'n_jobs': 60}}):
    
    if output_folder is None:
        os.mkdir('gene_set_multiplex')
        output_folder = './gene_set_multiplex'
    s_knn(processed_scRNAseq_obj_lst = processed_scRNAseq_obj_lst, sample_size = sample_size, population_labels = population_labels, 
          gene_set_dict = gene_set_dict, output_folder = output_folder,\
                      num_core = num_core, parameter =  parameter)
    nt = tissue_obj_multiplex_network(collection = collection, folder = output_folder)
    return nt
    
    
def vis_3D(x, y, z, title, colors,  shrink = 1, df_GO = None, index = None, axis_off = False, labels = ['Participation coefficient', 'Multiplex PageRank', 'Clustering coefficient'], font_size = 18, annotate_font_size = 12, tick_pad = (0,0,0), title_size = 18, line_alpha = 0.02,\
                colorbar = None, z_tick_pad = 100, alpha = 0.8,  view_para = {'elev':30., 'azim':-60},figsize= (12,12)):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= figsize )
    ### feature colors
    if colorbar is not None:
        cmap = sns.color_palette(palette = colorbar, as_cmap = True)
        norm = matplotlib.colors.Normalize(vmin=-4, vmax=4)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm = norm, cmap=cmap), \
                     orientation='vertical',\
                     spacing='proportional', shrink = 0.2, anchor = (0.8,0.8))
        cbar.ax.text(-1, 5,'NES',rotation=0, fontsize = 12)
        ax.scatter(xs = x, ys = y, zs= z, zdir='z', c = colors, alpha = alpha, cmap = cmap)
    else:
        ax.scatter(xs = x, ys = y, zs= z, zdir='z', c = colors, alpha = alpha)
    ax.view_init(**view_para)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_tick_params(labelsize = 12, pad = tick_pad[0])
    ax.yaxis.set_tick_params(labelsize = 12, pad = tick_pad[1])
    ax.zaxis.set_tick_params(labelsize = 12, pad = tick_pad[2])
    ax.set_xlabel(f'{labels[0]}', loc = 'center', fontsize=font_size,labelpad=20 )
    ax.set_ylabel(f'{labels[1]}', loc = 'center', fontsize=font_size,labelpad=20)
    ax.set_zlabel(f'{labels[2]}',  fontsize=font_size,labelpad=20)
    ax.set_title(title, fontsize = title_size)
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    if (df_GO is not None) and (index is not None):
        try:
            df_temp = df_GO.iloc[list(index.values()),].loc[:,['P', 'C2', 'Multiplex PageRank', 'ID', 'Description']]
        except:
            df_temp = df_GO.iloc[list(index.values()),].loc[:,['P', 'C2', 'Multiplex.PageRank', 'ID', 'Description']]
        for i in range(df_temp.shape[0]):
            x, z, y = df_temp.iloc[i,:3]
            text = df_temp['ID'].iloc[i,] + ":\n" + func.abridge_GO(df_temp['Description'].iloc[i,], cut = 25)
            ax.text(x*shrink, y*shrink, z*shrink, text, ha = 'center', fontsize = annotate_font_size, zorder=10 )
    if axis_off:
        ax.set_axis_off()
    
    plt.tight_layout()
    return fig

# convert continuous values to colors in the hex format
def value_to_hex(data, cmap = 'Spectral', norm = True):
    cmap = sns.color_palette(palette = cmap, as_cmap = True)
    data = copy.deepcopy(data)
    if norm:
        data = (data - np.min(data))/(np.max(data) - np.min(data))
    hex_lst = np.array([matplotlib.colors.to_hex(x, keep_alpha = False) \
    for x in cmap(X = data, alpha = None).tolist()])
    return hex_lst, cmap