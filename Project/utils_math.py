import numpy as np
from numpy import linalg as la
from scipy import linalg as spla

#wrapper for cho_factor & cho_solve
def cholesky_solve(A,b):
    c, low = spla.cho_factor(A)
    return spla.cho_solve((c, low), b)

#cholesky rank one update with expansion | NOT USED, numerically unstable (?)
def cholesky_rank1_expand(L, v, alpha):
    """
    Update L = chol(A) to obtain cholesky of [[A, v],[v.T, alpha]]
    Return the new row of such a new L
    """
    w = la.solve(L, v)

    ell_squared = alpha - np.inner(w, w)
    ell = np.sqrt(np.maximum(ell_squared, 1e-12))
    return np.append(w,ell)

#fit and nystrom fit
def full_fit(K_nn, reg_param, labels):
    '''
    The parameter labels is there beacuse in the criteria expression the Hat matrix A = K_nm(K_mnK_nm + n*lambda*K_mm)^(-1)K_mn  appears applied to the vector b instead to the usual vector of labels labels
    In order to use the fit-predict method in conjunction to mimic the expression Ab of the criteria, I let the user to pass whatever vector labels he prefers
    '''
    n = K_nn.shape[0]
    alpha = cholesky_solve(K_nn+n*reg_param*np.eye(n,n),labels) 
    return alpha

def Nystrom_fit(K_nm, K_mm, reg_param, labels):
    
    n = np.atleast_2d(K_nm).shape[0]
    m = np.atleast_2d(K_mm).shape[0]
    num_stability_correction = 1000*np.spacing(1)*np.eye(m,m)
    alpha = cholesky_solve(K_nm.T@K_nm + n*reg_param*K_mm + num_stability_correction,np.atleast_1d(K_nm.T@labels))
    return alpha

#predict
def predict(X_train, eval_point, alpha, kernel):
    return kernel(X_train,eval_point).T@alpha

#Generalized Cross Validation
def GCV(reg_param_range:np.ndarray, Q, Lambda, K_nn, labels) -> np.float64:
    '''
    Returns the best regularization parameter according to the GCV criteria 
    We are computing the regularization parameter that minimizes 1/n Σ [ (y_i-f(x_i)) / (1-tr(hat_matrix)/n) ]^2
    assuming we have access to the eigenvalue decomposition of the kernel matrix K_nn
    After some computations...
    '''
    n = K_nn.shape[0]
    labels_rotated = Q.T @ labels
    GCV_val = []
    for reg_param in reg_param_range:
        num = 1/n * np.sum((labels_rotated*n*reg_param/(Lambda + n*reg_param))**2)
        den = (1- np.sum(Lambda/(Lambda+n*reg_param))/n)**2 
        GCV_val.append(num/den)
    return reg_param_range[np.argmin(GCV_val)]

def risk(pred:np.ndarray,label:np.ndarray) -> np.float64:
    n = label.shape[0]
    return la.norm(pred-label)**2 /n