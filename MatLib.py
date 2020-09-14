import math
import numpy as np
            
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