import copy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mycolorpy import colorlist as mcp
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy
import scipy.stats as stats
from scipy.stats import spearmanr
import umap
from umap import umap_


def vis_hyper(cor_RV_lst, cor_mantel_lst, label_lst, K_num, title, figsize=(12,12), axvline_kw = {'x': np.sqrt(1000), 'color':'r', 'label': r'$\sqrt(1000)$'}, anno_font_size = 16, title_size = 18):
    fig, ax = plt.subplots(figsize= figsize, dpi = 300)
    for cor_RV, cor_mantel, label in zip(cor_RV_lst, cor_mantel_lst, label_lst):
        ax.plot(np.array(K_num), cor_RV, label = label[0])
        ax.plot(np.array(K_num), cor_mantel, label = label[1])
    ax.set_xlabel('number of K', fontsize=anno_font_size)
    ax.set_ylabel('Pearson correlation coefficient', fontsize=anno_font_size)
    ax.tick_params(axis='both', which='major', labelsize=anno_font_size-2)
    ax.tick_params(axis='both', which='minor', labelsize=anno_font_size-4)
    ax.axvline(**axvline_kw)
    ax.legend(frameon=False,bbox_to_anchor= (1.0, 1.0), prop={'size': anno_font_size})
    ax.set_title(title, fontsize = title_size)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim(0.5,1)
    fig.tight_layout()
    return fig
    
def vis_heatmap(mat, labels, figsize=(12,12), title = "ImmuneSig",  legend_title = "NMI", x_rotation = 0, y_rotation = 0, ticks = np.arange(0.5,10.5,1), title_size = 16, anno_font_size = 14, cbar_font = 14, para_heatmap = {'annot':True, 'vmin':0, 'vmax':1, 'cmap':'magma', 'square':True, 'cbar_kws': {"shrink": .25, "anchor" :(0.0, 0.85)}} ):
    fig, ax = plt.subplots(figsize=figsize) 
    ax_sns = sns.heatmap(np.round(mat,2), ax = ax, mask = np.triu(mat) , **para_heatmap, annot_kws = {"fontsize":anno_font_size})
    plt.xticks(ticks=ticks, labels=labels, rotation= x_rotation, fontsize= anno_font_size)
    plt.yticks(ticks=ticks, labels=labels,  rotation= y_rotation, fontsize= anno_font_size)
    ax.legend(title=legend_title, frameon=False, title_fontsize = anno_font_size)
    ax.set_title('{}'.format(title),
             fontsize = title_size)
    ### font size of the colorbar
    color_bar = ax_sns.collections[0].colorbar
    color_bar.ax.tick_params(labelsize=cbar_font)
    fig.tight_layout()
    return fig

def vis_clustermap(sim_mat, label_lst,figsize=(10, 10), row = True, dendro_mov = (-0.2, 0), cbar = True, cbar_kws ={"label":"NMI", "shrink":0.3},  cbar_pos = (1 ,0.5, 0.03,0.10),\
                   heatmap_kw = {'annot':True, 'vmin':0, 'vmax':1, 'cmap':'magma'}, fontsize = 12, tick_rotation = 15,\
                  title = "", title_font_size = 14, title_pad = 10):
    def true_column(index):
        true_index = []
        for i in range(len(index)):
            true_index.append(np.where(i == index)[0][0])
        return true_index

    g = sns.clustermap(sim_mat)
    plt.close()
    mask = np.triu(np.ones_like(sim_mat))[true_column(g.mask.index),][:,true_column(g.mask.index)]
    g = sns.clustermap(sim_mat, figsize=figsize, tree_kws = {'colors':"black", 'linewidths':2}, mask = mask, cbar_pos = cbar_pos, cbar = cbar, cbar_kws = cbar_kws, **heatmap_kw, annot_kws = {'fontsize': fontsize})
    plt.close()
    _ = g.ax_heatmap.set_title(title, fontsize = title_font_size, ha = 'center', pad = title_pad)
    _ = g.ax_heatmap.set_xticklabels(np.array(label_lst)[g.data2d.index], fontsize = fontsize, rotation =tick_rotation)
    _ = g.ax_heatmap.set_yticklabels(np.array(label_lst)[g.data2d.index], fontsize = fontsize, rotation =tick_rotation)
    _ = g.ax_heatmap.yaxis.tick_left()
    if row:
        g.ax_row_dendrogram.set_visible(True)
        g.ax_col_dendrogram.set_visible(False)
        pos = g.ax_row_dendrogram.get_position() 
        new_pos = [pos.x0 + dendro_mov[0], pos.y0 + dendro_mov[1] , pos.width, pos.height]
        g.ax_row_dendrogram.set_position(new_pos)
    else:
        g.ax_row_dendrogram.set_visible(False)
        g.ax_col_dendrogram.set_visible(True)
        pos = g.ax_col_dendrogram.get_position() 
        new_pos = [pos.x0 + dendro_mov[0], pos.y0 + dendro_mov[1] , pos.width, pos.height]
        g.ax_col_dendrogram.set_position(new_pos)
        g.ax_col_dendrogram.set_ylim(g.ax_col_dendrogram.get_ylim()[::-1])
    g.fig.tight_layout()
    return g


def vis_UMAP(adj_tr, feature, title, annotate_kw, para_UMAP,  para_plot, colorbar_kw, circle = False, circle_kw = {'target_lst':None},  title_size = 16, UMAP_coord = None, annotate_df = None ):
    if UMAP_coord is None:
        model = umap_.UMAP(**para_UMAP, metric='precomputed')
        res = model.fit_transform(1 - adj_tr)
        vis = pd.DataFrame(res, columns = ['umap1','umap2'])
    else:
        vis = UMAP_coord.iloc[:,:3]
    vis['feature'] = feature

    fig = plt.figure(figsize=para_plot['figsize'], dpi=para_plot['dpi'])
    plt.scatter(x = vis['umap1'], y = vis['umap2'], c = vis['feature'], cmap = para_plot['cmap'], alpha = para_plot['alpha'])
    plt.box(on=None)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(title, fontsize=title_size)
    if colorbar_kw['colorbar']:
        plt.colorbar( location=colorbar_kw['location'], anchor = colorbar_kw['anchor'], fraction = 0.2, shrink = 0.25 )
    if annotate_df is not None:
        x_lst, y_lst = annotate_df
        for i in range(len(x_lst)):
            text = x_lst[i]
            x, y = y_lst[i]
            if not circle:
                annotate_kw_cur = copy.deepcopy(annotate_kw)
                rotation = annotate_kw['rotation_lst'][i]
                annotate_kw_cur['rotation'] = rotation
                xytext = annotate_kw['xytext_lst'][i]
                annotate_kw_cur['xytext'] = xytext
                annotate_kw_cur.pop('rotation_lst')
                annotate_kw_cur.pop('xytext_lst')
                plt.annotate(text, xy=(x, y), **annotate_kw_cur)
            else:
                if (circle_kw['target_lst'] is not None) and (i in circle_kw['target_lst']):
                    plt.scatter(x, y, s=10, c=circle_kw['annotate_kw']['color'])
                    plt.plot(x, y, 'o', ms=30, mec=circle_kw['annotate_kw']['color'], mfc='none', mew=1)
                    current_circle_kw = copy.deepcopy(circle_kw['annotate_kw'])
                    mv = current_circle_kw['xytext_lst'][i]
                    current_circle_kw.pop('xytext_lst')
                    plt.annotate(text, xy = (x, y), xytext = mv,  **current_circle_kw)
                else:
                    plt.scatter(x, y, s=5, c=circle_kw['annotate_kw']['color'])
        
    fig.tight_layout()
    plt.show()
    return fig, vis


def vis_multiplex_network(multiplex_nw, UMAP_coord = None, annotate_df = None, title_lst = None, annotate_kw_lst = None,  display_feature = 'hub', title_size = 16, circle = False,
             circle_kw = {'target_lst':None, 'annotate_kw':{'xytext' : (-20, -50), 'textcoords':'offset points',
                   'color':'black', 'size':10, 'arrowprops':dict(arrowstyle='-', facecolor='black', shrinkB=20 * 1.2, alpha = 0.5)}},
                           para_vis_UMAP ={ 'para_UMAP': {'random_state':111, 'n_neighbors':50}, \
             'para_plot' : {'figsize':(5,5), 'dpi':100, 'cmap':None, 'alpha':1}, \
             'colorbar_kw': {'colorbar':True, 'cmap':'Spectral', 'anchor':(0.0,1.0), 'location':'right'}
              }):    
    
    fig_lst = [] 
    vis_lst = [] 
    gs_length = len(multiplex_nw.common_gs_index)
    if title_lst is None:
        title_lst = copy.deepcopy(multiplex_nw.layer_ns)
        title_lst.insert(0, 'Jaccard')
        
    for i in range(len(title_lst)):
        if i == 0:
            if UMAP_coord is not None:
                cor = UMAP_coord[i]
                adj_tr = None
            else:
                cor = None
                adj_tr = multiplex_nw.adj_Jaccard
                
            if display_feature is 'hub':
                assert len(multiplex_nw.multiplex_property) != 0, print('multiplex_nw.multiplex_property is empty!')
                ### ratio
                feature = multiplex_nw.multiplex_property['Jaccard_hub']
            elif display_feature is "pagerank":
                assert len(multiplex_nw.multiplex_property) != 0, print('multiplex_nw.multiplex_property is empty!')
                feature = multiplex_nw.multiplex_property['Jaccard_page_rank']/(1/gs_length)
            elif display_feature is 'membership':
                assert len(multiplex_nw.community_detection)!= 0, print('multiplex_nw.community_detection is empty!')
                membership = multiplex_nw.community_detection['MVP'][0] 
                feature = membership
            else:
                print('The display_feature should be a list of colors ...')
                feature = display_feature[0]
           
        else:
            if UMAP_coord is not None:
                cor = UMAP_coord[i]
                adj_tr = None
            else:
                cor = None
                adj_tr =  multiplex_nw.adj_filtered_lst[i-1]
            ### ratio
            if display_feature is 'hub':
                feature =  multiplex_nw.multiplex_property['per_layer_hub'][i-1]
            elif display_feature is 'pagerank':
                feature =  multiplex_nw.multiplex_property['per_layer_page_rank'][i-1]/(1/gs_length)
            elif display_feature is 'membership':
                feature = membership
            else:
                feature = display_feature[i-1]
                
            
        if display_feature is 'membership':
            len_mem = len(np.unique(membership))
            assert len_mem <= 10, print("Too many communities...")
            cmap = ListedColormap(sns.color_palette("tab10", len_mem).as_hex())
            para_vis_UMAP['para_plot']['cmap'] = cmap
        
        if annotate_df is not None:
            if not circle and annotate_kw_lst is not None:
                annotate_kw = annotate_kw_lst[i]
            elif circle:
                annotate_kw = None
                circle_kw_cur = copy.deepcopy(circle_kw)
                circle_kw_cur['annotate_kw']['xytext_lst'] = circle_kw_cur['annotate_kw']['xytext_lst'][i]
            fig_temp, vis_temp = vis_UMAP(adj_tr, UMAP_coord = cor, title = title_lst[i], feature = feature,\
                                                    annotate_kw = annotate_kw, circle = circle, circle_kw = circle_kw_cur, title_size =  title_size, annotate_df = annotate_df.iloc[i,], **para_vis_UMAP)
            
        else:
            fig_temp, vis_temp = vis_UMAP(adj_tr, UMAP_coord = cor, title = title_lst[i] , annotate_kw = None, feature = feature,\
                                                    title_size =  title_size, **para_vis_UMAP)
        fig_lst.append(fig_temp)
        vis_lst.append(vis_temp)
        
    return fig_lst, vis_lst


def add_annotation_fig_lst(fig_lst, annotate_kw_lst, df_annotate, circle = False,
             circle_kw = {'target_lst':None, 'annotate_kw':{'xytext' : (-20, -50), 'textcoords':'offset points',
                   'color':'black', 'size':10, 'arrowprops':dict(arrowstyle='-', facecolor='black', shrinkB=20 * 1.2, alpha = 0.5)}}):
    for i in range(len(fig_lst)):
        fig = fig_lst[i]
        ax = fig.get_axes()[0]
        x_lst, y_lst = df_annotate.iloc[i,]
        if not circle:
            annotate_kw = annotate_kw_lst[i]
        for j in range(len(x_lst)):
            text = x_lst[j]
            x, y = y_lst[j]
            if not circle:
                annotate_kw_cur = copy.deepcopy(annotate_kw)
                rotation = annotate_kw['rotation_lst'][j]
                annotate_kw_cur['rotation'] = rotation
                xytext = annotate_kw['xytext_lst'][j]
                annotate_kw_cur['xytext'] = xytext
                annotate_kw_cur.pop('rotation_lst')
                annotate_kw_cur.pop('xytext_lst')
                ax.annotate(text, xy=(x, y), **annotate_kw_cur)
            else:
                if (circle_kw['target_lst'] is not None) and (j in circle_kw['target_lst']):
                    ax.scatter(x, y, s=10, c=circle_kw['annotate_kw']['color'])
                    ax.plot(x, y, 'o', ms=30, mec=circle_kw['annotate_kw']['color'], mfc='none', mew=1)
                    current_circle_kw = copy.deepcopy(circle_kw['annotate_kw'])
                    mv = current_circle_kw['xytext_lst'][i][j]
                    current_circle_kw.pop('xytext_lst')
                    ax.annotate(text, xy = (x, y), xytext = mv,  **current_circle_kw)
                else:
                    ax.scatter(x, y, s=5, c=circle_kw['annotate_kw']['color'])
    return fig_lst

def vis_scatter(para_1, para_2, label_1, label_2, para_jointplot = {'kind':'hex', 'space':0.7, 'marginal_kws':dict(bins=30)}, xtick_integer = False, ytick_integer = False,\
               anno_font_size = 16, height = 8, ratio = 10):
    df = pd.DataFrame(data = {label_1: para_1, label_2 : para_2})
    with sns.axes_style('white'):
        sns_plot = sns.jointplot(x=label_1, y= label_2, data=df, 
                     **para_jointplot, height=height, ratio=ratio)
        sns_plot.set_axis_labels(label_1, label_2, fontsize= anno_font_size)
        if not xtick_integer:
            sns_plot.ax_joint.set_xticklabels([str(np.round(i,2)) for i in sns_plot.ax_joint.get_xticks()], fontsize = anno_font_size)
        else:
            sns_plot.ax_joint.set_xticklabels([str(np.int(i)) for i in sns_plot.ax_joint.get_xticks()], fontsize = anno_font_size)
        if not ytick_integer:
            sns_plot.ax_joint.set_yticklabels([str(np.round(i,2)) for i in sns_plot.ax_joint.get_yticks()], fontsize = anno_font_size)
        else:
            sns_plot.ax_joint.set_yticklabels([str(np.int(i)) for i in sns_plot.ax_joint.get_yticks()], fontsize = anno_font_size)
        sns_plot.figure.tight_layout()
    return sns_plot

    
def vis_surface(sample_size, gs_size, figsize = (12, 10), label = None, labelpad =8,  unit = 'second',\
                fontsize = 16, rotation = 60, dpi=300, alpha=.7, legend = True, legend_pos = (1.2, 1.0), elevation_rotation = [20, -60]):
    assert np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True) is not None, "The data path to res files is not right!"
    
    X, Y = np.meshgrid(sample_size, gs_size)
    num_method = len(np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[0])

    def fetch_time(x,y,i, unit = unit):
        temp = np.load(f"res_{x}_{y}.npy", allow_pickle = True)
        time = temp[0][i][1]
        if unit == 'minite':
            time = np.round(time/60, 3)
        elif unit == 'hour':
            time = np.round(time/(60*60), 3)
        return time

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= figsize, dpi=dpi)
    ### number of methods
    length = len(np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[1])
    if label  is None:
        label = [np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[0][i][0] for i in range(num_method)]
    for i in range(length):
        Z = np.array([fetch_time(x,y,i) for x in sample_size for y in gs_size]).reshape(len(sample_size),-1).transpose()
        surf = ax.plot_surface(X, Y, Z, label = label[i], alpha = alpha)
        surf._facecolors2d=surf._facecolor3d
        surf._edgecolors2d=surf._edgecolor3d

#     fig.suptitle("Running time comparison", fontsize = fontsize)
    ax.set_xlabel("sample size", fontsize=fontsize, rotation=rotation, labelpad = labelpad)
    ax.set_ylabel("number of gene sets", fontsize=fontsize, rotation=rotation, labelpad = labelpad)
    ax.set_zlabel(f'running time in {unit}s', fontsize=fontsize, rotation=rotation, labelpad = labelpad)
    ax.xaxis.set_tick_params(labelsize=fontsize-4)
    ax.yaxis.set_tick_params(labelsize=fontsize-4)
    ax.zaxis.set_tick_params(labelsize=fontsize-4)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    if legend:
        ax.legend(bbox_to_anchor= legend_pos, fontsize = fontsize-8)
    ax.view_init(*elevation_rotation)
    fig.tight_layout()
    return fig

def vis_surface_correlation(sample_size, gs_size, label = None, labelpad =8, reference = "modified RV", cor_mat_lst = None, figsize = (12, 10),\
                fontsize = 16, rotation = 60, dpi=300, alpha=.7, legend = True, legend_pos = (1.2, 1.0), elevation_rotation = [20, -60]):
    assert np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True) is not None, "The data path to res files is not right!"
    
    X, Y = np.meshgrid(sample_size, gs_size)
    num_method = len(np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[0])
    
    if cor_mat_lst is None:
        cor_mat_lst = []
        for ss in sample_size:
            for gs in gs_size:
                res = np.load(f"res_{ss}_{gs}.npy", allow_pickle = True)
                ref = np.load(f"res_{sample_size[-1]}_{gs}.npy", allow_pickle = True)
                cor_mat = np.array([(lambda i,j: scipy.stats.pearsonr(res[1][i][np.triu_indices_from(res[1][i], k = 1)],\
                                            ref[1][j][np.triu_indices_from(ref[1][j], k = 1)])[0])(i,j) for i in range(num_method)\
                  for j in range(num_method)]).reshape(num_method, num_method)
                cor_mat_lst.append(cor_mat)
    
    def fetch_value(x,y,i):
        temp = cor_mat_lst[x*len(gs_size) + y]
        if reference == "modified RV":
            return temp[i,1]
        elif reference == "Mantel":
            return temp[i,num_method-2]
        else:
            raise Exception("Reference is not correct!")
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize= figsize, dpi=dpi)
    if label  is None:
        label = [np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[0][i][0] for i in range(num_method)]
    ### number of methods
    length = len(np.load(f"res_{sample_size[0]}_{gs_size[0]}.npy", allow_pickle = True)[1])
    for i in range(length):
        Z = np.array([fetch_value(x,y,i) for x in range(len(sample_size)) for y in range(len(gs_size))]).reshape(len(sample_size),-1).transpose()
        surf = ax.plot_surface(X, Y, Z, label = label[i], alpha = alpha)
        surf._facecolors2d=surf._facecolor3d
        surf._edgecolors2d=surf._edgecolor3d

#     fig.suptitle(f"Compare with the {reference} coefficient", fontsize = fontsize)
    ax.set_xlabel("sample size", fontsize=fontsize, rotation=rotation, labelpad = labelpad)
    ax.set_ylabel("number of gene sets", fontsize=fontsize, rotation=rotation, labelpad = labelpad)
    ax.set_zlabel(f'Pearson correlation coefficient', fontsize=fontsize, rotation=rotation, labelpad = labelpad)
    ax.xaxis.set_tick_params(labelsize=fontsize-4)
    ax.yaxis.set_tick_params(labelsize=fontsize-4)
    ax.zaxis.set_tick_params(labelsize=fontsize-4)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    if legend:
        ax.legend(bbox_to_anchor= legend_pos, fontsize = fontsize-4)
    ax.view_init(*elevation_rotation)
    fig.tight_layout()
    return fig, cor_mat_lst


### https://stackoverflow.com/questions/72179957

def rho(x, y, ax=None, **kwargs):
    ax = plt.gca()
    _x = kwargs['data'].loc[:,x.name]
    _y = kwargs['data'].loc[:,y.name]
    r,p = spearmanr(_x,_y)
    if kwargs['label'] == 0:
        sns.regplot(data=kwargs['data'], x=x.name, y=y.name, lowess=True, scatter=False, color=kwargs['color'], line_kws={'linewidth':3})
        ax.annotate(f'$ρ = {r:.2f}$',
                xy=(.7, 0.95), xycoords=ax.transAxes, fontsize=16,
                color='darkred', backgroundcolor='#FFFFFF99')
    

def vis_attr_pair_plot(data, figsize = (19,19), legend_font_size = 25, title_font_size = 25, tick_label_font_size = 18, height = 5.5, palette = "tab10", kind = 'scatter', corner = True, markers = ['X', 'o', '.', '*', '^'] ):    
    plt.figure(figsize=figsize)
    plt.rc('legend', fontsize=legend_font_size, title_fontsize= title_font_size)
    plt.rc('axes', labelsize = tick_label_font_size )
    plt.rc('xtick', labelsize = tick_label_font_size)
    plt.rc('ytick', labelsize = tick_label_font_size)
    plt.rc('axes', labelpad = 20)
    fig = sns.pairplot(data, corner = corner, hue = 'multiplex community',\
                       kind = kind, palette = palette, height = height,\
                          markers = markers, plot_kws={'alpha': 0.3, "s":200 })
    fig.map_lower(rho, data= data, color='crimson')
    fig._legend.set_bbox_to_anchor((0.8, 0.85))
    fig._legend.set_title("Multiplex community")
    for l in fig._legend.legendHandles: 
        l.set_alpha(1)
        l._sizes = [300] 
    for ax in fig.axes:
        if ax[0].get_ylabel() == '':
            continue
        else:
            # rotate y axis labels
            ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation = 90, ha="right")
    
    plt.setp(fig._legend.get_texts(), fontsize=30) 
    plt.setp(fig._legend.get_title(), fontsize=34)
    return fig



    
    
    
    