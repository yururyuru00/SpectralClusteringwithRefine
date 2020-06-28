#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
from Parameters import P
import MatLib as ml
            
def read_attribute(path):
    with open(path, "r") as r:
        size = r.readline().rstrip().split(" ")
        usr_size = int(size[1])
        dim_size = int(size[0])
        X = np.zeros((usr_size, dim_size))
        dim_id = 0
        for line in r.readlines():
            dim, usr, val = split(line)
            X[usr][dim] = val
    return X

def split(line):
    line = line.split(" ")
    return int(line[0]), int(line[1]), float(line[2])

def read_sp1(path):
    def makeMat_sp1(val_list, mat):
        for i in range(len(val_list)):
            for j in range(i+1, len(val_list)):
                mat[i][j] = math.exp(-(val_list[i]-val_list[j])*(val_list[i]-val_list[j])                                      /(2*P.sigma*P.sigma))
                mat[j][i] = mat[i][j]
        for d in range(len(val_list)):
            mat[d][d] = 0.
            
    if(P.AN_type == 'h'):
        with open(path, "r") as r:
            size = r.readline().rstrip().split(" ")
            usr_size = int(size[1])
            dim1_size = int(size[0])
            mats = [np.zeros((usr_size, usr_size)) for i in range(dim1_size)]
            dim_id = 0
            val_list = np.zeros(int(size[1]))
            for line in r.readlines():
                dim, usr, val = split(line)
                if(dim != dim_id):
                    makeMat_sp1(val_list, mats[dim_id])
                    dim_id += 1
                val_list[usr] = val
            makeMat_sp1(val_list, mats[dim_id])
            for i in range(len(mats)):
                ml.normalized(mats[i])
        return mats
    
    else:
        with open(path, "r") as r:
            size = r.readline().rstrip().split(" ")
            usr_size = int(size[1])
            dim1_size = int(size[0])
            dim_id = 0
            X = np.zeros((usr_size, dim1_size))
            for line in r.readlines():
                dim, usr, val = split(line)
                X[usr][dim] = val
            mat = np.zeros((usr_size, usr_size))
            for i in range(len(mat)):
                for j in range(i+1, len(mat[0])):
                    mat[i][j] = np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
                    mat[j][i] = mat[i][j]
            for d in range(len(mat)):
                mat[d][d] = 0.
            ml.normalized(mat)
        return [mat]
            
def read_sp2(file, usr_size):
    gi = Graph(file, usr_size)
    s2a, s2b = gi.makeMat_sp2()
    ml.normalized(s2a)
    ml.normalized(s2b)
    return s2a, s2b

class Graph():
    def __init__(self, file, usr_size):
        self.N = np.array([set() for i in range(usr_size)])
        with open(file, "r") as r:
            for pair in r.readlines():
                pair = pair.split(" ")
                self.N[int(pair[0])].add(int(pair[1]))
        self.max_size = np.zeros(P.sita+1)
        self.R = np.array([[set() for i in range(usr_size)] for j in range(P.sita+1)])
        self.Ft = np.array([[set() for i in range(usr_size)] for j in range(P.sita+1)])    
    
    def makeMat_sp2(self):
        s2a = np.diag([0. for i in range(len(self.N))])
        s2b = np.diag([0. for i in range(len(self.N))])
        for i in range(len(s2a)):
            for j in self.N[i]:
                s2a[i][j] = P.gamma
                s2b[i][j] = P.gamma
            self.max_size[1] = max(self.max_size[1], len(self.N[i]))
        for v in range(len(self.N)):
            self.R[0][v].add(v)
            self.Ft[0][v].add(v)
            self.R[1][v].add(v)
            self.R[1][v] = self.R[1][v] | self.N[v]
            self.Ft[1][v] = self.N[v]
        for step in range(2, P.sita+1):
            for v in range(len(self.N)):
                self.R[step][v] = self.R[step-1][v].copy()
                for n in self.N[v]:
                    self.R[step][v] = self.R[step][v] | self.R[step-1][n]
                self.Ft[step][v] = self.R[step][v].copy() - self.R[step-1][v]
                self.max_size[step] = max(self.max_size[step], len(self.Ft[step][v]))
                for v_opp in self.Ft[step][v]:
                    path = 0
                    for v_opp_ft in self.Ft[step-1][v]:
                        if(v_opp in self.N[v_opp_ft]):
                            path += 1
                    s2a[v][v_opp], s2b[v][v_opp] = self.calc(step, path, v)
        return s2a, s2b
    
    def calc(self, step, path, v):
        if(len(self.Ft[step-1][v]) <= 1):
            rate1 = 0
        else:
            rate1 = (path-1) / (len(self.Ft[step-1][v])-1)
        rate2 = (path-1) / self.max_size[step-1]
        delta = math.pow(P.gamma, step-1) - math.pow(P.gamma, step)
        return math.pow(P.gamma, step) + delta * rate1,                math.pow(P.gamma, step) + delta * rate2


# In[ ]:




