# Spectral Clustering with Refine
python scripts for my academic research#1.

For a high-level explanation, read the following our paper.
Yuta Yajima, Akihiro Inokuchi, [Refining Similarity Matrices for Clustering Attributed Networks Accurately](https://jsai.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=10714&item_no=1&page_id=13&block_id=23) (SIG-SAI 2020)

## Requirements
* Python3

## Pip install
* numpy
* scipy
* scikit-learn
* cvxopt

## How to run
For example, if you need to reproduce the experiment result described in our paper in a cora dataset,
```bash
python SClump.py cora 7
```
If you need to know how to set thparameters, 
```bash
python SClump.py --help
```
(The parameter names of the command line arguments are the same as the ones listed in our paper.)

# Guide to experimental replication
Experimental results described in our paper for each dataset could be reproduced by setting up the following.
Please specify the following parameters using the command line arguments.

| dataset | c | sigma | m | disable |
|:---:|:---:|:---:|:---:|:---:|
| cora | 7 | - | 80 | False |
| citeseer | 6 | - | 80 | True |
| football | 20 | 3 | 12 | False |
| politicsuk | 5 | 0.2 | 80 | True |
| olympics | 28 | 0.1 | 12 | False |

## Cite
Please cite our paper if you use this code in your own work:
