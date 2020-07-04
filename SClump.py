#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cvxopt
import MatLib as ml
import math
import sys
import os
import warnings
sys.path.append("D:\\python\\data2Mat")
import matplotlib.pyplot as plt
import seaborn as sns
from cvxopt import matrix
from scipy.sparse.linalg import eigsh
import scipy.optimize
from scipy import io, sparse
import txt2Mat
from clustering_quolity import purity
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as clus
import remaind
import networkx as nx
from Parameters import P as p


def spectral_clustering(S, labels):
    if(p.mode == "norm"):
        Ls = ml.makeNormLaplacian(S)
    else:
        Ls = ml.makeLaplacian(S)
    try:
        val, vec = eigsh(Ls, p.clus_size, which="SM")
    except scipy.sparse.linalg.ArpackNoConvergence:
        return 'nonconverge', 0, 0, 0, 0, 0, 0
    ari, nmi, pur, times = 0., 0., 0., 10
    for i in range(times):
        k_means = KMeans(n_clusters=p.clus_size, n_init=10, random_state=0, tol=0.0000001)
        k_means.fit(vec)
        ari += clus.adjusted_rand_score(labels, k_means.labels_)
        nmi += clus.adjusted_mutual_info_score(labels, k_means.labels_, "arithmetic")
        pur += purity(labels, k_means.labels_)
    return 'converge', Ls, val, vec, ari/times, nmi/times, pur/times
        
def make_buff(sp, S, W, lamb, ari, nmi, pur, val, vec, tri, labels, w, eigen0):
    root = p.dir + p.buff
    if(tri==3):
        top = np.argsort(-lamb)
        index = {0,1}
        for i in index:
            ml.mat2file(sp[top[i]], root + "/sp{}#{}.txt".format(top[i], i))
            ml.heatmap(sp[top[i]], 0, 1./len(S))
            plt.savefig(root + "/sp{}#{}.png".format(top[i], i))
    ml.mat2file(S, root + "/S{0}.txt".format(tri))
    ml.mat2file(W, root + "/W{0}.txt".format(tri))
    ml.vec2file(lamb, root + "/lamb{0}.txt".format(tri))
    ml.mat2file(vec, root + "/vec{0}.txt".format(tri))
    ml.vec2file(val, root + "/eigenval{}.txt".format(tri))
    ml.heatmap(S, 0, 1./p.edge_size)
    plt.savefig(root + "/S{0}.png".format(tri))
    ml.heatmap(W, 0, 1./p.edge_size)
    plt.savefig(root + "/W{0}.png".format(tri))
    fig = plt.figure(figsize=(60,60),dpi=200)
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        ax.plot(vec.T[i])
    fig.savefig(root + "/vec{0}.png".format(tri))  
    
def ini_lambda(sp):
    if(p.AN_type == 'h'):
        lamb = np.zeros(len(sp))
        with open(p.dir+'{}_attribute_ratio.txt'.format(p.AN_data), 'r') as r:
            ratio = r.readline().rstrip().split(' ')
            ratio = np.array([float(val) for val in ratio])
        ratio_sum = np.sum(ratio)
        for i in range(len(sp)-2):
            lamb[i] = 0.5 * (ratio[i]/ratio_sum)
        lamb[len(sp)-2], lamb[len(sp)-1] = 0.25, 0.25
        return lamb
    else:
        lamb = np.array([0.5,0.25,0.25])
        return lamb

def ini_W(sp, lamb):
    W = np.zeros((len(sp[0]), len(sp[0][0])))
    for i in range(len(sp)):
        W += sp[i]*lamb[i]
    return W
    
def update_S(S, W, vec):
    def func_lamb(x):
        sum_ = 0.0
        for j in range(len(ui)):
            sum_ += ml.relu(x - ui[j])
        return sum_/p.edge_size - x
        
    def func_prime_lamb(x):
        sum_ = 0.0
        for j in range(len(ui)):
            if(x - ui[j] >= 0.0):
                sum_ += 1.
        return sum_/p.edge_size - 1.

    print("\tupdate_S start: ")
    v_1 = np.ones(p.edge_size)
    for i in range(len(S)):
        arg = np.argsort(-S[i])
        tops = np.array([arg[j] for j in range(p.edge_size)])
        pi = np.array([(2.*W[i][top] - p.gamma2*ml.l2norm(vec[i]-vec[top])*ml.l2norm(vec[i]-vec[top])) 
                        / ((2.+2.*p.alpha)) for top in tops])
        ui = pi + 1./p.edge_size*v_1 - np.dot(v_1, pi)/p.edge_size*v_1
        opt_lamb = scipy.optimize.newton(func_lamb, 0., func_prime_lamb)
        S[i] = np.zeros(len(S[i]))
        for j in range(p.edge_size):
            S[i][tops[j]] = ml.relu(ui[j]-opt_lamb)

    print("")

def update_lamb(lamb, S, sp, Q):
    print("\tupdate_lambda start: ")
    Q = matrix(Q)
    q = matrix(np.array([-ml.dot(S, sp[k]) for k in range(len(lamb))]))
    G = matrix(np.diag([-1. for i in range(len(lamb))]))
    h = matrix(np.zeros(len(lamb)))
    A = matrix(np.array([[1. for i in range(len(lamb))]]))
    b = matrix(np.array([1.]))

    cvxopt.solvers.options['show_progress'] = False
    sol=cvxopt.solvers.qp(Q,q,G,h,A,b)
    index = 0
    for val in sol["x"]:
        lamb[index] = val
        index += 1
        
def gamma_tuning(val):
    eigen0 = 0
    for eig in val:
        if(eig < 1E-10):
            eigen0 += 1
    if(eigen0 < p.clus_size):
        p.gamma2 += (-(eigen0-1.)/(p.clus_size-1.) + 1.) * p.gamma2
    elif(eigen0 > p.clus_size):
        p.gamma2 = p.gamma2 / 2.
    print("\tnumOfComponents: {}\n\tgamma: {}".format(eigen0, p.gamma2))
    
    return eigen0

def main():
    #initialize
    plt.rcParams.update({'figure.max_open_warning': 0})
    np.random.seed(0)
    #os.mkdir(p.dir+p.buff)
    #w = open(p.dir+p.buff+"/result.txt", "w")
    
    features, edges, labels = txt2Mat.load_fromgen(p.dir, p.AN_data)
    sp1 = txt2Mat.make_sp1(features)
    sp2a, sp2b = txt2Mat.make_sp2(edges, len(sp1[0]))
    sp = sp1 + [sp2a, sp2b]
    lamb = ini_lambda(sp)
    Q = np.zeros((len(sp), len(sp)))
    for i in range(len(sp)):
        for j in range(i, len(sp)):
            sum = 0.
            for k in range(len(sp[0])):
                sum += np.dot(sp[i][k], sp[j][k])
            Q[i][j] = sum
            Q[j][i] = sum
    for d in range(len(Q)):
        Q[d][d] += p.beta
    W = ini_W(sp, lamb)
    S = np.copy(W)

    #update + SC
    for tri in range(2):
        state, Ls, val, vec, ari, nmi, pur = spectral_clustering(S, labels)
        if(state == 'converge'):
            eigen0 = gamma_tuning(val)
            print("tri: {}\n\tARI: {:.4f}\n\tNMI: {:.4f}\n\tPurity: {:.4f}".format(tri, ari, nmi, pur))
            #w.write("tri: {}\n\tARI: {:.4f}\n\tNMI: {:.4f}\n\tPurity: {:.4f}\n".format(tri, ari, nmi, pur))
            #w.write("\tnumOfComponents: {}\n".format(eigen0))
            if(eigen0 == p.clus_size):
                break
        else:
            print("=================================\narpack error\n===========================")
            #w.write("tri: {}\narpack no converge".format(tri))
            break
        #make_buff(sp, S, W, lamb, ari, nmi, pur, val, vec, tri, labels, w, eigen0)
        
        S_buff = np.copy(S)
        update_S(S, W, vec)
        print("\tS-S*: {0}".format(ml.frobenius_norm(S-S_buff)))
            
        update_lamb(lamb, S, sp, Q)
        W = ini_W(sp, lamb)
    
    #make_buff(sp, S, W, lamb, ari, nmi, pur, val, vec, tri, labels, w, eigen0)
    #p.dump(p.dir+p.buff+"/paramters.txt", len(S), len(sp))
    #w.close()
    
##############################
if __name__ == '__main__':
    main()


# In[ ]:


from Parameters import P
import SClump
import remaind


def execute():
    with open(P.dir + "experiment.csv") as r:
        for setting in r.readlines():
            setting = setting.rstrip().split(' ')
            P.read_setting(setting)
            main()
    remaind.run(1)


# In[ ]:




