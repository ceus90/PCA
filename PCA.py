

import numpy as np
import logging
import sys

def cov(D):
    # calculate column means
    n = D.shape[0]
    mu = np.tile(np.mean(D, 0), (n, 1))
    Z = D - mu
    sigma = np.dot(Z.T, Z) / n
    return sigma
    
# 2a.  covariance
def q2a(D):
    # covariance manually
    covmx = cov(D)

    # using numpy function
    result = np.cov(D.T, bias=True)
    return '2a\n' + 'manual\n' + str(covmx) + '\nnumpy\n' + str(result) + '\n'

#2b. power-iteration method
def q2b(D):
    A = cov(D)
    k = 0
    p0 = np.array([1,1,1,1])
    p = [] # list of arrays
    p.append(p0)

    pdiff = 9999999
    epsilon = 1e-6
    while (pdiff > epsilon):
        k = k + 1
        p.append(np.dot(A.transpose(), p[k-1]))
        i = np.argmax(p[k])
        lmbda = p[k][i] / p[k-1][i]
        p[k]  = p[k] / p[k-1][i]
        pdiff = np.linalg.norm(p[k] - p[k-1])
 
    pnorm = p[k] / np.linalg.norm(p[k])
    result = np.linalg.eig(A)
    return '2b\n' + 'power iteration\n' + str(lmbda) + '\n' + str(pnorm) + '\nnumpy\n' + str(result) + '\n'

#2c. project on subspace
def q2c(D):
    A = cov(D)
    e_vals, e_vecs = np.linalg.eig(A)

    # 2 dominant eigenvectors
    eigvec1 = e_vecs[:, 0]
    eigvec2 = e_vecs[:, 1]
    eigvec3 = e_vecs[:, 2]
    eigvec4 = e_vecs[:, 3]
    
    proj1 = np.dot(A.T, eigvec1)
    proj2 = np.dot(A.T, eigvec2)
    projs = np.column_stack((proj1, proj2))
    
    result = np.trace(np.cov(projs.T, bias=True))
    return '2c\n' + str(result) + '\n'


#2d. print covariance matrix in eigen-decomposition form
def q2d(D):
    A = cov(D)
    e_vals, e_vecs = np.linalg.eig(A)
    U = e_vecs
    Lbda = np.diag(e_vals)
    
    # should matrch covariance matrix
    result = U.dot(Lbda.dot(U.T))
    return '2d\n' + str(U) + '\n' + str(Lbda) + '\n' + str(U.T) + '\n' + str(result)

#2e. principal components analysis
def pca(D, alpha=0.95):
    sigma = cov(D)
    
    e_vals, e_vecs = np.linalg.eig(sigma)
    
    fracvar = np.cumsum(e_vals) / sum(e_vals)
    for i in range(len(fracvar)):
        if fracvar[i] > alpha:
            break
        
    r = i + 1
    Ur = e_vecs[:, 0:r]
    Ar = Ur.T.dot(D.T)
    return r, e_vals, fracvar[i], Ur, Ar.T
    
def q2e(D):
    r, e_vals, propvar, Ur, Art = pca(D)

    # first 10 data points
    result = Art[0:10, :]
    return '2e\n' + str(result)

#2f. covariance of projected matrix
def q2f(D):
    sigma = cov(D)
    e_vals, e_vecs = np.linalg.eig(sigma)
    r, e_vals, propvar, Ur, Art = pca(D)    
    result1 = e_vals[0] + e_vals[1]

    Ur = e_vecs[:, 0:r]
    Ar = Ur.T.dot(D.T)
    result2 = np.trace(cov(Ar.T))
    return '2f\n' + 'sum of eigenvalues\n' + str(result1) + '\ncovariance\n' + str(result2)
    
def main():
    LOG_FILENAME = sys.argv[1]
    
    # setup logger
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

    # load data
    fname = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    
    iris_mx = np.genfromtxt(fname, delimiter=',', usecols=(0,1,2,3))
    
    logging.info(q2a(iris_mx))
    logging.info(q2b(iris_mx))
    logging.info(q2c(iris_mx))
    logging.info(q2d(iris_mx))
    logging.info(q2e(iris_mx))
    logging.info(q2f(iris_mx))
    

if __name__ == '__main__':
    main()