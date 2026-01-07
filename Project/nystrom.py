#nystrom
import numpy as np
import sampling  #to fill in the dictionary of strategies
from utils_math import Nystrom_fit, predict
from numpy import linalg as la

class NystromKRR:
    def __init__(self, kernel, reg_param=1e-5, rng=None):
        #attributes
        self.kernel = kernel
        self.reg_param = reg_param
        self.rng = np.random.default_rng() if rng is None else rng
        
        self.alpha = None
        self.Nystrom_indexes = None
        self.X_train = None
        self.anchor_data = None

    def _compute_kernel(self, X1, X2):
        return self.kernel(X1, X2)

    def fit(self, X_train, labels, m, strategy='Uniform', **kwargs):
        
        self.X_train = X_train
        
        #collect the sampling strategy
        if strategy not in sampling.strategies:
            raise ValueError(f"Strategy '{strategy}' not found. Available stratgies: {list(sampling.strategies.keys())}")
        
        strategy_func = sampling.strategies[strategy]
        
        #use the strategy to get the Nystrom indexes
        self.Nystrom_indexes = strategy_func(
            X=self.X_train, 
            labels=labels, 
            m=m, 
            rng=self.rng,
            kernel=self.kernel, 
            reg_param=self.reg_param,     
            **kwargs                      
        )
        self.anchor_data = self.X_train[self.Nystrom_indexes]

        #the kernel matrix
        K_nm = self._compute_kernel(self.X_train, self.anchor_data)
        K_mm = self._compute_kernel(self.anchor_data, self.anchor_data)

        #the vector of parameters alpha
        self.alpha = Nystrom_fit(K_nm, K_mm, self.reg_param, labels)
        
        return self

    def predict(self, eval_point):
        if self.alpha is None:
            raise RuntimeError("You must fit the model before using it to predict")
            
        #K_test_m = self._compute_kernel(X_test, self.X_train[self.Nystrom_indexes])
        return predict(self.anchor_data, eval_point, self.alpha, self.kernel)
    
    def risk(self, pred:np.ndarray,label:np.ndarray) -> np.float64:
        n = label.shape[0]
        return la.norm(pred-label)**2 /n