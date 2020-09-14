import argparse
import numpy as np
import cvxopt, scipy.optimize
from cvxopt import matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as clus

import MatLib as ml
import utilities
from Parameters import P


def spectral_clustering(S, labels, iter):
    if(P.disable_normalization == False): Ls = ml.makeNormLaplacian(S)
    else: Ls = ml.makeLaplacian(S)
    try: 
        eigen_val, eigen_vec = eigsh(Ls, P.cluster_size, which="SM")
    except scipy.sparse.linalg.ArpackNoConvergence:
        return 'nonconverge', -1, -1, -1
    
    metrics = {'ari':0., 'nmi':0., 'purity':0.}
    for _ in range(10):
        k_means = KMeans(n_clusters=P.cluster_size, n_init=10, tol=0.0000001)
        k_means.fit(eigen_vec)
        metrics['ari'] += clus.adjusted_rand_score(labels, k_means.labels_)
        metrics['nmi'] += clus.adjusted_mutual_info_score(labels, k_means.labels_, "arithmetic")
        metrics['purity'] += utilities.purity(labels, k_means.labels_)
    for key in metrics.keys(): metrics[key] = metrics[key] / 10.
    np.savetxt('./result/{}/pred_unnorm_{}.csv'.format(P.dataset, iter), k_means.labels_)
    return 'converge', eigen_val, eigen_vec, metrics

def ini_Q(sp):
    obj_size, dim_size = len(sp[0]), len(sp)
    Q = np.zeros((dim_size, dim_size))

    for i in range(dim_size):
        for j in range(i, dim_size):
            sum_ = 0.
            for k in range(obj_size):
                sum_ += np.dot(sp[i][k], sp[j][k])
            Q[i][j] = sum_
            Q[j][i] = sum_
    for d in range(len(Q)):
        Q[d][d] += P.beta
    return Q

def ini_lambda(sp):
    dim_size = len(sp)
    if(P.dataset_type == 'social_net'):
        lamb = np.zeros(dim_size)
        with open('./data/{0}/{0}_pca_contribution_ratio.csv'.format(P.dataset), 'r') as r:
            pca_contribution_ratio = r.readline().rstrip().split(' ')
            pca_contribution_ratio = np.array([float(val) for val in pca_contribution_ratio])
        ratio_sum = np.sum(pca_contribution_ratio)
        for i in range(dim_size-2):
            lamb[i] = 0.5 * (pca_contribution_ratio[i]/ratio_sum)
        lamb[dim_size-2], lamb[dim_size-1] = 0.25, 0.25
        return lamb
    elif(P.dataset_type == 'citation_net'):
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
        return sum_/P.number_of_updates - x
        
    def func_prime_lamb(x):
        sum_ = 0.0
        for j in range(len(ui)):
            if(x - ui[j] >= 0.0):
                sum_ += 1.
        return sum_/P.number_of_updates - 1.

    v_1 = np.ones(P.number_of_updates)
    for i in range(len(S)):
        arg = np.argsort(-S[i])
        tops = np.array([arg[j] for j in range(P.number_of_updates)])
        pi = np.array([(2.*W[i][top] - P.gamma*ml.l2norm(vec[i]-vec[top])*ml.l2norm(vec[i]-vec[top])) 
                        / ((2.+2.*P.alpha)) for top in tops])
        ui = pi + 1./P.number_of_updates*v_1 - np.dot(v_1, pi)/P.number_of_updates*v_1
        opt_lamb = scipy.optimize.newton(func_lamb, 0., func_prime_lamb)
        S[i] = np.zeros(len(S[i]))
        for j in range(P.number_of_updates):
            S[i][tops[j]] = ml.relu(ui[j]-opt_lamb)

def update_lamb(lamb, S, sp, Q):
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
        
def gamma_tuning(eigen_values):
    number_of_eigen0 = 0
    for eig in eigen_values:
        if(eig < 1E-10): number_of_eigen0 += 1
    if(number_of_eigen0 < P.cluster_size):
        P.gamma += (-(number_of_eigen0-1.)/(P.cluster_size-1.) + 1.) * P.gamma
    elif(number_of_eigen0 > P.cluster_size):
        P.gamma = P.gamma / 2.
    print("\tnum of components: {}\n".format(number_of_eigen0))
    
    return number_of_eigen0

def main():
    print('\ninitialize start ...\n')
    #set the hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='any of the following datasets {cora, citeseer, football, politicsuk, football}')
    parser.add_argument('c', type=int, help='number of clusters')
    parser.add_argument('--sigma', default=3., type=float, 
                                    help='parameter sigma used to generate similarity matrix S_A')
    parser.add_argument('--theta', default=3, type=int, 
                                    help='parameter theta used to generate similarity matrix S_L')
    parser.add_argument('--delta', default=0.6, type=float, 
                                    help='parameter theta used to generate similarity matrix S_L')
    parser.add_argument('--m', default=80, type=int, 
                                    help='parameter to determine how many elements in similarity matrices to update')
    parser.add_argument('--disable', action='store_true', default=False,
                                    help='Disables the normalization process of Laplacian matrices')
    setting = parser.parse_args()
    P.set(setting) #set the parameters to Parameter Class P 

    #initialize
    np.random.seed(0)
    if(P.dataset=='cora'): features, edges, labels = utilities.load_fromgen(P.dataset)
    else: features, edges, labels = utilities.load_fromcsv(P.dataset)
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
            number_of_eigen0 = gamma_tuning(eigen_val)
            if(number_of_eigen0 == P.cluster_size): break
        else: print("=========\narpack error\n========="); break

        update_S(S, W, eigen_vec)
        update_lamb(lamb, S, sp, Q)
        W = ini_W(sp, lamb)
    
#--------------------------------------------------
if __name__ == '__main__':
    main()