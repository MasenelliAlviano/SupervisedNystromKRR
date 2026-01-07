import numpy as np
from numpy import linalg as la
import scipy as sp
from scipy import linalg as spla
from KernelOperator_numba import NumbaKernelOperator

class FullKRR:
    def __init__(self, datapoints, labels, kernel, r):
        self.kernel = kernel
        self.X_train = datapoints
        self.labels_train = labels
        self.n = datapoints.shape[0]
        self.dtype = np.float64
        self.shape = (self.n,self.n)
        self.r = r
        self.operator = NumbaKernelOperator(self.X_train,self.r)
    
        self.alpha = None
        self.reg_param = None
        self.Lambda_k = None
        self.Q_k = None

    def _compute_Lambda_and_Q(self):
        self.Lambda_k, self.Q_k = sp.sparse.linalg.eigsh(self.operator,np.floor(np.sqrt(self.n)))
        #to order eigvalues and eigevectors
        idx = self.Lambda_k.argsort()[::-1]
        self.Lambda_k = self.Lambda_k[idx]
        self.Q_k = self.Q_k[:, idx]

    def GCV_approx(self, reg_param_range):
        '''
        Returns the best regularization parameter according to the GCV criteria 
        We are computing the regularization parameter that minimizes 1/n Σ [ (y_i-f(x_i)) / (1-tr(hat_matrix)/n) ]^2
        We compute the best k-eigenvalue decomposition of the kernel matrix K_nn up to k = sqrt(n)
        After some computations...
        '''
        self._compute_Lambda_and_Q()
        GCV_values = []
        k = self.Lambda_k.shape[0]

        labels_rotated = self.Q_k.T@self.labels_train
        
        labels_rotated_norm_sq = np.sum(labels_rotated**2)
        labels_norm_sq = np.sum(self.labels_train**2)
        labels_rotated_tail_norm_sq = max(0.0,labels_norm_sq-labels_rotated_norm_sq)

        for reg_param in reg_param_range:
            new_Lambda_k = self.n*reg_param/(self.Lambda_k+self.n*reg_param)

            num = np.sum( (new_Lambda_k*labels_rotated)**2 )/self.n + (labels_rotated_tail_norm_sq)/self.n
            den = (( np.sum(new_Lambda_k) + self.n-k)/self.n)**2 #/self.n
            GCV_values.append(num/den)
        self.reg_param = reg_param_range[np.argmin(GCV_values)]
        
    def fit(self):
        self.alpha = sp.sparse.linalg.cg(self.operator+self.n*self.reg_param*sp.sparse.linalg.eye(self.n), self.labels_train)
    
    def predict():
        pass
# def GCV(reg_param_range:np.ndarray, Q, Lambda, K_nn, labels) -> np.float64:
#     '''
#     Returns the best regularization parameter according to the GCV criteria 
#     We are computing the regularization parameter that minimizes 1/n Σ [ (y_i-f(x_i)) / (1-tr(hat_matrix)/n) ]^2
#     assuming we have access to the eigenvalue decomposition of the kernel matrix K_nn
#     After some computations...
#     '''
#     n = K_nn.shape[0]
#     labels_rotated = Q.T @ labels
#     GCV_val = []
#     for reg_param in reg_param_range:
#         num = 1/n * np.sum((labels_rotated*n*reg_param/(Lambda + n*reg_param))**2)
#         den = (1- np.sum(Lambda/(Lambda+n*reg_param))/n)**2 
#         GCV_val.append(num/den)
#     return reg_param_range[np.argmin(GCV_val)]

