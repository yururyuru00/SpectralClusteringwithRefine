import numpy as np
import matplotlib.pyplot as plt
import cvxopt, scipy.optimize
from tqdm import tqdm
from cvxopt import matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as clus

import MatLib as ml
import utilities
from Parameters import P as p


def spectral_clustering(S, labels, ite):
    if(p.mode == "norm"):
        Ls = ml.makeNormLaplacian(S)
    else:
        Ls = ml.makeLaplacian(S)
    try:
        eigen_val, eigen_vec = eigsh(Ls, p.clus_size, which="SM")
    except scipy.sparse.linalg.ArpackNoConvergence:
        return 'nonconverge'
    
    metrics = {'ari':0., 'nmi':0., 'purity':0.}
    for _ in range(10):
        k_means = KMeans(n_clusters=p.clus_size, n_init=10, tol=0.0000001)
        k_means.fit(eigen_vec)
        metrics['ari'] += clus.adjusted_rand_score(labels, k_means.labels_)
        metrics['nmi'] += clus.adjusted_mutual_info_score(labels, k_means.labels_, "arithmetic")
        metrics['purity'] += utilities.purity(labels, k_means.labels_)
    for key in metrics.keys(): metrics[key] = metrics[key] / 10.
    np.savetxt('./experiment/{0}_pred{1}'.format(p.AN_data, ite), k_means.labels_, fmt='%d')
    return 'converge', eigen_val, eigen_vec, metrics

def ini_Q(sp):
    obj_size, dim_size = len(sp[0]), len(sp)
    Q = np.zeros((dim_size, dim_size))

    pbar = tqdm(range(dim_size), ncols=100)
    pbar.set_description("\tinitialize Q start")
    for i in pbar:
        for j in range(i, dim_size):
            sum_ = 0.
            for k in range(obj_size):
                sum_ += np.dot(sp[i][k], sp[j][k])
            Q[i][j] = sum_
            Q[j][i] = sum_
    for d in range(len(Q)):
        Q[d][d] += p.beta
    return Q

def ini_lambda(sp):
    dim_size = len(sp)
    if(p.AN_type == 'h'):
        lamb = np.zeros(dim_size)
        with open('./data/{0}/{0}_pca_contribution_ratio.csv'.format(p.AN_data), 'r') as r:
            pca_contribution_ratio = r.readline().rstrip().split(' ')
            pca_contribution_ratio = np.array([float(val) for val in pca_contribution_ratio])
        ratio_sum = np.sum(pca_contribution_ratio)
        for i in range(dim_size-2):
            lamb[i] = 0.5 * (pca_contribution_ratio[i]/ratio_sum)
        lamb[dim_size-2], lamb[dim_size-1] = 0.25, 0.25
        return lamb
    else:
        lamb = np.array([0.5,0.25,0.25])
        return lamb

def ini_W(sp, lamb):
    obj_size, dim_size = len(sp[0]), len(sp)
    W = np.zeros((obj_size, obj_size))
    for i in range(dim_size):
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
    print("\tnum of components: {}\n\tgamma: {}".format(eigen0, p.gamma2))
    
    return eigen0

def main():
    #initialize
    plt.rcParams.update({'figure.max_open_warning': 0})
    np.random.seed(0)
    
    if(p.AN_data=='cora'): features, edges, labels = utilities.load_fromgen(p.AN_data)
    else: features, edges, labels = utilities.load_fromcsv(p.AN_data)
    sp1 = utilities.make_sp1(features)
    sp2a, sp2b = utilities.make_sp2(edges, len(sp1[0]))
    sp = sp1 + [sp2a, sp2b]

    Q = ini_Q(sp) #matrix Q is used for later process(update_lambda)
    lamb = ini_lambda(sp)
    W = ini_W(sp, lamb)
    S = np.copy(W)

    #Spectral Clustering + Refine(update_S)
    for tri in range(100):
        state, eigen_val, eigen_vec, metrics = spectral_clustering(S, labels, tri)
        if(state == 'converge'):
            print("tri: {}\n\tARI: {:.4f}\n\tNMI: {:.4f}\n\tPurity: {:.4f}" \
                        .format(tri, metrics['ari'], metrics['nmi'], metrics['purity']))
            eigen0 = gamma_tuning(eigen_val)
            if(eigen0 == p.clus_size): break
        else: print("===\narpack error\n==="); break

        update_S(S, W, eigen_vec)
        update_lamb(lamb, S, sp, Q)
        W = ini_W(sp, lamb)
    
#--------------------------------------------------
if __name__ == '__main__':
    main()