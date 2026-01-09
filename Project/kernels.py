
import numpy as np
from scipy.spatial.distance import cdist
from numpy import linalg as la
from numba import jit, prange

#@jit
def gaussian_kernel(X:np.ndarray,Y:np.ndarray, r)->np.ndarray:
    '''
    Computes K, i.e., the gaussian kernel between every data in X€R^(n*d) and in Y€R^(m*d). Data are rows, in both matrices. K will be of dimension n x m
    '''
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    #v = la.norm(X,axis=1)**2                                                            #a vectot whose entry are the squared norm of each row (i.e data) of X
    #w = la.norm(Y,axis=1)**2                                                            #a vectot whose entry are the squared norm of each row (i.e data) of Y
    v = np.einsum('ij,ij->i', X, X)
    w = np.einsum('ij,ij->i', Y, Y)
    #dist = -2*X@Y.T + np.outer(v,np.ones(Y.shape[0])) + np.outer(np.ones(X.shape[0]),w) #matrix version of ||xi-yj||^2 = -2<xi,yj> + <xi,xi> + <yj,yj>
    dist = -2*X@Y.T + v[:,None] + w[None,:] #matrix version of ||xi-yj||^2 = -2<xi,yj> + <xi,xi> + <yj,yj>
    dist = np.maximum(dist,0) #avoid having negative distances that are due to rounding errors
    return np.exp(-dist/(r**2))

@jit
def gaussian_kernel_scipy_version(X:np.ndarray,Y:np.ndarray, r)->np.ndarray:
    '''
    Computes K, i.e., the gaussian kernel between every data in X€R^(n*d) and in Y€R^(m*d). Data are rows, in both matrices. K will be of dimension n x m
    '''
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    dist = cdist(X, Y, metric='sqeuclidean') #diretly computes the euclidean distance
    return np.exp(-dist / (r**2))

@jit
def gaussian_kernel_numba(X,Y,r):
    """
    Computes the gaussian kernel between 1D vectors X and Y with bandwidth r
    """
    D = X.shape[0]
    dist_sq = 0.0
    for k in range(D):
        d = X[k] - Y[k]
        dist_sq += d * d
    dist_sq = np.maximum(dist_sq,0)
    return np.exp(-dist_sq/(r**2))

def linear_kernel(X:np.ndarray, Y:np.ndarray) -> np.ndarray:
    '''
    Computes the linear kernel between every data in X€R^(n*d) and in Y€R^(m*d). Data are rows, in both matrices. K will be of dimension n x m
    '''
    print("Don't know if linear kernel implementation works or not")
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    #v = la.norm(X,axis=1)**2                                                            #a vectot whose entry are the squared norm of each row (i.e data) of X
    #w = la.norm(Y,axis=1)**2                                                            #a vectot whose entry are the squared norm of each row (i.e data) of Y
    v = np.sum(X*X,axis = 1)
    w = np.sum(Y*Y,axis = 1)
    #dist = -2*X@Y.T + np.outer(v,np.ones(Y.shape[0])) + np.outer(np.ones(X.shape[0]),w) #matrix version of ||xi-yj||^2 = -2<xi,yj> + <xi,xi> + <yj,yj>
    dist = -2*X@Y.T + v[:,None] + w[None,:] #matrix version of ||xi-yj||^2 = -2<xi,yj> + <xi,xi> + <yj,yj>
    return dist
