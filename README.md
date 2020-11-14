# SpectralClusteringwithRefine

this is a python scripts for our academic research#1


## Cite
Please cite our paper if you use this code in your own work: https://arxiv.org/abs/2010.06854
[Yuta Yajima and Akihiro Inokuchi 2020 JSAI]


## How to Run
```bash
python SClump.py cora 7
```
If you need to know how to set the parameters, 
```bash
python SClump.py --help
```
(The parameter names of the command line arguments are the same as the ones listed in our paper.)

## Guide to experimental replication
Experimental results described in our paper for each dataset could be reproduced by setting up the following.
Please specify the following parameters using the command line arguments.
| dataset | c | sigma | m | disable |
|:---:|:---:|:---:|:---:|:---:|
| cora | 7 | - | 80 | False |
| citeseer | 6 | - | 80 | True |
| football | 20 | 3 | 12 | False |
| politicsuk | 5 | 0.2 | 80 | True |
| olympics | 28 | 0.1 | 12 | False |

