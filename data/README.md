# Guide to experimental replication

Parameters could be specified by command line arguments. For each parameter, run below.
```bash
python SClump.py --help
```

Experimental results described in our paper for each dataset could be reproduced by setting up the following.
Please specify the following parameters using the command line arguments.

| dataset | c | sigma | m | disable |
|:---:|:---:|:---:|:---:|:---:|
| cora | 7 | - | 80 | False |
| citeseer | 6 | - | 80 | True |
| football | 20 | 3 | 12 | False |
| politicsuk | 5 | 0.2 | 80 | True |
| olympics | 28 | 0.1 | 12 | False |

