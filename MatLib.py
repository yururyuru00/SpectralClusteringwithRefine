#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import ari
import math
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as clus
from sklearn.cluster import KMeans
from Parameters import P as P
from clustering_quolity import purity
import networkx as nx

def plot(vec, which):
    plt.figure(figsize=(36, 36))
    if(which=="scatter"):
        index = np.array([i for i in range(len(vec))])

        plt.tick_params(labelsize = 12) # (9)目盛線のラベルサイズ
        plt.tick_params(labelbottom=False, bottom=False)
        plt.yticks(np.arange(np.amax(vec)+1))
        plt.scatter(index, vec, s=150, c="b", marker="D", alpha=0.5) #(10)散布図の描画
        plt.grid(color="gray", axis="y")
        plt.rcParams['figure.figsize'] = (100, 50)
        plt.show()
    else:
        plt.xticks(np.arange(len(vec)+1))
        plt.plot(vec)
        
def plt_graph(graph, n_size, e_width, **kwargs):
    plt.figure(figsize=(144, 144))
    nodes = [[i+1, np.sum(graph[i])] for i in range(len(graph))]
    G = nx.Graph()
    G.add_nodes_from([(node[0], {"count":node[1]}) for node in nodes])
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            G.add_edge(i+1, j+1)
            G.edges[i+1,j+1]["weight"] = graph[i][j]
    if("pos" in kwargs):
        pos = kwargs["pos"]
    elif("k" in kwargs):
        pos = nx.spring_layout(G, k=0.1*kwargs["k"])
    else:
        pos = nx.spring_layout(G, k=0.1)
    node_size = [nodes[i][1]*n_size for i in range(len(graph))]
    nx.draw_networkx_nodes(G, pos, node_color="tan",alpha=0.6, 
                           node_size=node_size)
    if("label" in kwargs):
        labels = {}
        for i in range(len(graph)):
            labels[i+1] = str(i+1)
        nx.draw_networkx_labels(G, pos, labels, font_size=kwargs["label"])
    edge_width = [ d["weight"]*e_width for (u,v,d) in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="black", 
                           width=edge_width)
    plt.axis("off")
    #plt.savefig("C:/Users/yajima/python/data/cora_virtual.png")
    plt.show()
    
def heatmap(mat, vmin, vmax):
    plt.figure(figsize=(36, 36))
    sns.heatmap(mat, fmt='g', cmap='gray_r', vmin=vmin, vmax=vmax)
    
def heatmap_binary(mat, rate):
    binary_mat = np.copy(mat)
    k = int(len(binary_mat)*rate)
    for i in range(len(binary_mat)):
        vec = np.sort(binary_mat[i])
        bottom = vec[len(vec)-k]
        for j in range(len(binary_mat[i])):
            if(binary_mat[i][j] < bottom):
                binary_mat[i][j] = 0.0
            else:
                binary_mat[i][j] = 1.0
    plt.figure(figsize=(36, 36))
    sns.heatmap(binary_mat, fmt='g', cmap='gray_r', vmin=0, vmax=1)
            
def normalized(mat):
    for i in range(len(mat)):   
        sum_ = np.sum(mat[i])
        for j in range(len(mat[i])):
            if(sum_ == 0):
                mat[i][j] = 0.
            else:
                mat[i][j] = mat[i][j] / sum_
        
def makeLaplacian(mat):
    L = np.zeros((len(mat), len(mat[0])))
    for d in range(len(mat)):
        for i in range(len(mat)):
            L[d][d] += (mat[i][d] + mat[d][i])
        L[d][d] = L[d][d]/2.0
    for i in range(len(L)):
        for j in range(len(L[0])):
            L[i][j] -= (mat[i][j]+mat[j][i])/2.
    return L

def makeNormLaplacian(mat):
    L = makeLaplacian(mat)
    d = np.zeros(len(mat))
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            d[i] += (mat[i][j] + mat[j][i])
        d[i] = d[i] / 2.
    for i in range(len(L)):
        for j in range(len(L[i])):
            if(d[i] == 0 or d[j] == 0):
                L[i][j] = 0.
            else:
                L[i][j] = (1./np.sqrt(d[i])) * L[i][j] * (1./np.sqrt(d[j]))
    return L

def l2norm(vec):
    sum = 0.0
    for i in range(len(vec)):
        sum += vec[i]*vec[i]
    return math.sqrt(sum)

def frobenius_norm(mat):
    sum = 0.
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            sum += mat[i][j]*mat[i][j]
    return math.sqrt(sum)

def dot(A, B):
    sum = 0.
    for i in range(len(A)):
        for j in range(len(A[0])):
            sum += A[i][j]*B[i][j]
    return sum

def makeKnn(mat, k):
    knnmat = np.zeros((len(mat), len(mat[0])))
    for i in range(len(mat)):
        arg = np.argsort(-mat[i])
        for top in range(k):
            knnmat[i][arg[top]] = mat[i][arg[top]]
    return knnmat

def relu(val):
    if(val < 0.0):
        return 0.0
    else:
        return val
    
def sigmoid(val, a):
    return 1. / (1. + np.exp(-a*val))

def step(val):
    if(val < 0.):
        return 0.
    else:
        return 1.

def makeOnes(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if(mat[i][j] != 0.0):
                mat[i][j] = 1.0

def file2mat(path):
    with open(path, "r") as r:
        size = int(r.readline())
        mat = np.array([float(val) for vec in r.readlines() for val in vec.rstrip().split(" ")]).reshape((size, size))
    return mat

def mat2file(mat, path):
    with open(path, "w") as w:
        w.write(str(len(mat)) + "\n")
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                w.write(str(mat[i][j]) + " ")
            w.write("\n")

def vec2file(vec, path):
    with open(path, "w") as w:
        for i in vec:
            w.write("{} ".format(i))
            


# In[ ]:




