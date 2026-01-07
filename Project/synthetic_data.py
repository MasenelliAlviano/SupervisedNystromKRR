import numpy as np
from numpy import linalg as la
from scipy import linalg as spla
import matplotlib.pyplot as plt

#1D
def sin(X):
    return np.sin(X)
def parabola(X):
    return X**2
def absolute_value(X):
    return np.abs(X)
def sin_relu(X):
    return np.sin(np.pi*X)+ np.maximum(X,0)
#2D
def radial_sin(X):
    return np.sin(np.sqrt(X[:,0]**2 + X[:,1]**2))
def inv_radial_sin(X):
    return 1/(1.25+np.sin((np.pi/2)*np.sqrt(X[:,0]**2 + X[:,1]**2)))
def paraboloid(X):
    return X[:,1]**2+X[:,0]**2
def model3(X):
    return X[:,1]*X[:,0]**2+X[:,0]
def model4(X):
    return np.tan(X[:,0])*X[:,1]
def norm_1(X):
    return np.abs(X[:,0]) + np.abs(X[:,1])
#kD
def radial_sin_kD(X):
    return np.sin(la.norm(X,axis = 1))
def inv_radial_sin_kD(X):
    return 1/(1.12+np.sin(la.norm(X,axis = 1)))
def norm_1_kD(X):
    return la.norm(X,axis = 1,ord = 1)
def random_function_generator_2D(kernel, p, rng, d, x_min, x_max):
    '''
    Samples and evaluates onto X a random function f from the guassian RKHS
    f(.) = sum_i=1^p alpha_i*k(x_i,.)
    where p is deterministic, alpha are standard gaussians, x_i are generate uniformly in the assigned domain
    '''
    alpha = rng.standard_normal(p)
    X_i = np.empty((p,d))
    for j in range(d):
        X_i[:,j] = rng.uniform(x_min,x_max,p)
    def random_function(X,Y):
        eval_data = np.column_stack((X, Y))
        return kernel(X_i,eval_data).T@alpha
    return random_function
def random_function_generator(kernel, p, rng, d, x_min, x_max):
    '''
    Samples and evaluates onto X a random function f from the guassian RKHS
    f(.) = sum_i=1^p alpha_i*k(x_i,.)
    where p is deterministic, alpha are standard gaussians, x_i are generate uniformly in the assigned domain
    '''
    alpha = rng.standard_normal(p)
    X_i = np.empty((p,d))
    for j in range(d):
        X_i[:,j] = rng.uniform(x_min,x_max,p)
    def random_function(X_eval):
        return kernel(X_i,X_eval).T@alpha
    return random_function
#remember to check why I do need two version based on the dimension.
#Most likely its a plotting issue

#generating data
#sigma = 0 if test datapoints are generated
#check dimension-issue with the 2D case
#1D has been twitcked so that the distribution fits visualization/exploration purposes 
def data_1D(number_of_data, d, model, sigma, rng, x_min, x_max):
    X = np.empty((number_of_data,d))                                                 #data matrix with data as rows
    X[:,0] = rng.uniform(x_min,x_max,number_of_data)
    #X[:8*number_of_data//10,0] = rng.normal(-10,3,size=(8*number_of_data//10,))#rng.normal(0,sigma**2,size=(number_of_data,))
    #X[8*number_of_data//10:,0] = rng.normal(10,3,size=(2*number_of_data//10,))
    #X[:,0] = np.linspace(x_min,x_max,n)
    labels = model(X).flatten() + rng.normal(0,sigma**2,size=(number_of_data,))
    return X,labels
def data_2D(number_of_data, d, model, sigma, rng, x_min, x_max, y_min, y_max):
    X = np.empty((number_of_data,d))                                                 #data matrix with data as rows
    X[:,0] = rng.uniform(x_min,x_max,number_of_data)
    X[:,1] = rng.uniform(y_min,y_max,number_of_data)
    #X[:,0] = np.linspace(x_min,x_max,number_of_data)
    #X[:,1] = np.linspace(y_min,y_max,number_of_data)
    labels = model(X) + rng.normal(0,sigma**2,size=(number_of_data,))
    return X,labels
def data_kD(number_of_data, d, model, sigma, rng, x_min, x_max):
    X = np.empty((number_of_data,d))
    for i in range(d):
        X[:,i] = rng.uniform(x_min,x_max,number_of_data)                #data matrix with data as rows
        labels = model(X) + rng.normal(0,sigma**2,size=(number_of_data,))
    return X,labels
