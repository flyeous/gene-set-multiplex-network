## Delving into gene-set multiplex networks facilitated by a k-nearest neighbor-based measure of similarity (updating...)

Currently, we provide a demo example for explaining the essential steps of this study.

## :clipboard: Code environment

Test system: Ubuntu 18.04.6 LTS

`conda env create -f environment.yml`

You may need to install jupyter-notebook and ipykernel

```
pip install notebook
pip install ipykernel
```
and add python (in gs) into the kernels of the jupyter-notebook.

```
conda activate gs
python -m ipykernel install --user --name=gs
```

## :cactus: Essential steps

* [Fetch scRNA-seq datasets and gene sets](https://github.com/flyeous/gene_set_multiplex_network/blob/main/fetch_data.ipynb)

* [Build a multiplex network of gene sets](https://github.com/flyeous/gene_set_multiplex_network/blob/main/build_multiplex_network.ipynb)

* [Essential steps in analyzing the multiplex network of gene sets](https://github.com/flyeous/gene_set_multiplex_network/blob/main/downstream_analysis.ipynb)

* [Visualizations of multiplex networks](https://github.com/flyeous/gene_set_multiplex_network/blob/main/multiplex_visualization)

R >= 4.2.1 and installed package [grimon] are required (https://github.com/mkanai/grimon). 

## :e-mail: Contact 
* zheng.cheng.68e@st.kyoto-u.ac.jp