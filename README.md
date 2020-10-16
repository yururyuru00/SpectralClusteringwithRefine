# SpectralClusteringwithRefine
python scripts for my academic research#1

this is a python scripts for our academic research#1
plz cite https://jsai.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=10714&item_no=1&page_id=13&block_id=23

# How to Run
```bash
python SClump.py cora 7
```
If you need to know how to set the　parameters, 
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
