import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import cm
from sklearn.metrics import silhouette_samples

from Parameters import P
import MatLib as ml
            
def load_fromgen(dataset):
    idfeature_features_labels = np.genfromtxt("./data/{0}/{0}.content".format(dataset),
                                                dtype=np.dtype(str))
    features = np.array([[float(i) for i in vec] for vec in idfeature_features_labels[:, 1:-1]])
    labels = idfeature_features_labels[:, -1]
    clas_map = {clas : l for l, clas in enumerate(set(labels))}
    labels = np.array([l for l in map(clas_map.get, labels)])
    idfeature = np.array(idfeature_features_labels[:, 0], dtype=np.int32)
    idfeature_map = {j: i for i, j in enumerate(idfeature)}
    edges_unordered = np.genfromtxt("./data/{0}/{0}.cites".format(dataset), dtype=np.int32)
    edges = np.array(list(map(idfeature_map.get, edges_unordered.flatten())),
                                    dtype=np.int32).reshape(edges_unordered.shape)
    return features, edges, labels

def load_fromcsv(dataset):
    with open("./data/{0}/{0}_content.csv".format(dataset), 'r') as r:
        [dim_size, obj_size] = list(map(int, r.readline().rstrip().split(' ')))
    features = np.zeros((obj_size, dim_size), dtype=np.float)
    dim_idobjct_val = np.genfromtxt("./data/{0}/{0}_content.csv".format(dataset),
                                                            skip_header=1, dtype=np.dtype(int))
    for [dim_id, obj_id, val] in dim_idobjct_val:
        features[obj_id][dim_id] = float(val)
    edges = np.genfromtxt("./data/{0}/{0}_cites.csv".format(dataset),
                                            skip_header=0, usecols=[1,0], dtype=np.dtype(int))
    labels = np.genfromtxt("./data/{0}/{0}_label.csv".format(dataset),
                                            skip_header=0, dtype=np.dtype(int), delimiter=' ')
    return features, edges, labels

def make_sp1(features):
    def makeMat_sp1_i(feature, usr_size):
        mat = np.zeros((usr_size, usr_size))
        for i in range(usr_size):
            for j in range(i+1, usr_size):
                mat[i][j] = math.exp(-(feature[i]-feature[j])*(feature[i]-feature[j])                                      /(2*P.sigma*P.sigma))
                mat[j][i] = mat[i][j]
        for d in range(usr_size):
            mat[d][d] = 0.
        return mat
            
    usr_size, dim_size = len(features), len(features[0])
    if(P.AN_type == 'h'): #AN_type=human
        mats = [np.zeros((usr_size, usr_size)) for i in range(dim_size)]
        for i, feature in enumerate(features.T):
            mats[i] = makeMat_sp1_i(feature, usr_size)
            ml.normalized(mats[i])
        return mats
    
    else: #AN_type=document
        mat = np.zeros((usr_size, usr_size))
        for i in range(usr_size):
            for j in range(i+1, usr_size):
                mat[i][j] = np.dot(features[i], features[j]) / (np.linalg.norm(features[i])                                    * np.linalg.norm(features[j]))
                mat[j][i] = mat[i][j]
        for d in range(len(mat)):
            mat[d][d] = 0.
        ml.normalized(mat)
        return [mat]
            
def make_sp2(edges, usr_size):
    gi = Graph(edges, usr_size)
    s2a, s2b = gi.makeMat_sp2()
    ml.normalized(s2a)
    ml.normalized(s2b)
    return s2a, s2b

class Graph():
    def __init__(self, edges, usr_size):
        self.N = np.array([set() for i in range(usr_size)])
        for opponent, obj in edges:
            self.N[obj].add(opponent)
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

def plot_silhouette_score(X, pred_clusters, iterate):
    cluster_labels = np.unique(pred_clusters)
    n_clusters=cluster_labels.shape[0]
    if(P.AN_type=='h'):
        silhouette_vals = silhouette_samples(X, pred_clusters, metric='euclidean')
    else: 
        silhouette_vals = silhouette_samples(X, pred_clusters, metric='cosine')
    y_ax_lower, y_ax_upper, yticks= 0, 0, []

    fig = plt.figure(figsize=(17, 30))
    for i,c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[pred_clusters==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i)/n_clusters)
        plt.barh(range(y_ax_lower,y_ax_upper),
                         c_silhouette_vals,
                         height=1.0,
                         edgecolor='none',
                         color=color)
        yticks.append((y_ax_lower+y_ax_upper)/2)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,color="red",linestyle="--")
    plt.yticks(yticks,cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('silhouette coefficient')
    fig.savefig('./experiment/silhouette_{}_{}_{}.png'.format(P.AN_data, P.mode, iterate))

def purity(ans, estimated):
    usr_size = len(estimated)
    clusterA = np.amax(ans)+1
    clusterE = np.amax(estimated)+1
    table = np.zeros((clusterE, clusterA))
    for i in range(len(estimated)):
        table[estimated[i]][ans[i]] += 1
    sum = 0
    for k in range(len(table)):
        sum += np.amax(table[k])
    return sum/usr_size