#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class P:
    #sp1
    sigma = 1
    #sp2
    sita = 3
    gamma = 0.6
    #refine
    alpha = 1.
    beta = 6.
    gamma2 = 0.5
    edge_size = 80
    mode = "norm"
    #dataset
    AN_data = 'cora'
    AN_type = 'd'
    clus_size = 7
    
    #experiment out file
    dir = 'D:/python/GCN/DeepGraphClustering/data/cora/'
    buff = "test"
    
    def read_setting(setting):
        P.sigma = float(setting[0])
        P.sita = int(setting[1])
        P.gamma = float(setting[2])
        P.alpha = float(setting[3])
        P.beta = float(setting[4])
        P.gamma2 = float(setting[5])
        P.edge_size = int(setting[6])
        P.mode = str(setting[7])
        P.AN_data = str(setting[8])
        P.AN_type = str(setting[9])
        P.clus_size = int(setting[10])
        P.buff = str(setting[11])

        
    def dump(path, usr_size, dim1_size):
        with open(path, "w") as w:
            w.write("sigma: {0}\n".format(P.sigma))
            w.write("sita: {0}\n".format(P.sita))
            w.write("gamma: {0}\n".format(P.gamma))
            w.write("alpha: {0}\n".format(P.alpha))
            w.write("beta: {0}\n".format(P.beta))
            w.write("gamma2: {0}\n".format(P.gamma2))
            w.write("edge_size: {0}\n".format(P.edge_size))
            w.write("mode: {0}\n".format(P.mode))
            w.write("usr_size: {0}\n".format(usr_size))
            w.write("dim1_size: {0}\n".format(dim1_size))
            w.write("clus_size: {0}\n".format(P.clus_size))

