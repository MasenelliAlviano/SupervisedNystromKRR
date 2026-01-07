import numpy as np
from numpy import linalg as la
from scipy import linalg as spla
import matplotlib.pyplot as plt
import cProfile
import matplotlib.animation as animation
#plt.style.use('tableau-colorblind10')
import matplotlib as mpl
plt.rcParams['font.size'] = 16
from matplotlib.patches import Circle
from matplotlib.patches import Patch  # Usiamo Patch per una legenda "vuota"
import os
import imageio.v2 as imageio
#for animation
if not os.path.exists("frames"):
    os.makedirs("frames")
for name in ['EmpRisk','TestRisk','Experiments','SMAPE']:
    for name2 in ['1D','2D','Truedata','kD']:
        if not os.path.exists("plots/"+name2+"/"+name):
            os.makedirs("plots/"+name2+"/"+name)
#USEFUL FUNCTIONS

#models
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
def radial_sin(X,Y):
    return np.sin(np.sqrt(X**2 + Y**2))
def inv_radial_sin(X,Y):
    return 1/(1.25+np.sin((np.pi/2)*np.sqrt(X**2 + Y**2)))
def paraboloid(X,Y):
    return X**2+Y**2
def model3(X,Y):
    return Y*X**2+X
def model4(X,Y):
    return np.tan(X)*Y
def norm_1(X,Y):
    return np.abs(X) + np.abs(Y)
#kD
def radial_sin_kD(X):
    return np.sin(la.norm(X,axis = 1))
def inv_radial_sin_kD(X):
    return 1/(1.12+np.sin(la.norm(X,axis = 1)))
def norm_1_kD(X):
    return la.norm(X,axis = 1,ord = 1)
def random_function_generator_2D(p):
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
def random_function_generator(p):
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

#generating data
def data_1D(number_of_data):
    X = np.empty((number_of_data,d))                                                 #data matrix with data as rows
    X[:,0] = rng.uniform(x_min,x_max,number_of_data)
    X[:8*number_of_data//10,0] = rng.normal(-10,3,size=(8*number_of_data//10,))#rng.normal(0,sigma**2,size=(number_of_data,))
    X[8*number_of_data//10:,0] = rng.normal(10,3,size=(2*number_of_data//10,))
    #X[:,0] = np.linspace(x_min,x_max,n)
    if number_of_data == n:
        labels = model(X).flatten() + rng.normal(0,sigma**2,size=(number_of_data,))
    else: 
        labels = model(X).flatten()
    return X,labels
def data_2D(number_of_data):
    X = np.empty((number_of_data,d))                                                 #data matrix with data as rows
    X[:,0] = rng.uniform(x_min,x_max,number_of_data)
    X[:,1] = rng.uniform(y_min,y_max,number_of_data)
    #X[:,0] = np.linspace(x_min,x_max,number_of_data)
    #X[:,1] = np.linspace(y_min,y_max,number_of_data)
    if number_of_data == n:
        labels = model(X[:,0],X[:,1]) + rng.normal(0,sigma**2,size=(number_of_data,))
    else:
        labels = model(X[:,0],X[:,1])
    #labels = model(X) + rng.normal(0,sigma**2,size=(number_of_data,))
    return X,labels
def data_kD(number_of_data):
    X = np.empty((number_of_data,d))
    for i in range(d):
        X[:,i] = rng.uniform(x_min,x_max,number_of_data)                                                   #data matrix with data as rows
    if number_of_data == n:
        labels = model(X) + rng.normal(0,sigma**2,size=(number_of_data,))
    else:
        labels = model(X)
    return X,labels

#kernels
def gaussian_kernel(X:np.ndarray,Y:np.ndarray)->np.ndarray:
    '''
    Computes K, i.e., the gaussian kernel between every data in X€R^(n*d) and in Y€R^(m*d). Data are rows, in both matrices. K will be of dimension n x m
    '''
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    #v = la.norm(X,axis=1)**2                                                            #a vectot whose entry are the squared norm of each row (i.e data) of X
    #w = la.norm(Y,axis=1)**2                                                            #a vectot whose entry are the squared norm of each row (i.e data) of Y
    v = np.sum(X*X,axis = 1)
    w = np.sum(Y*Y,axis = 1)
    #dist = -2*X@Y.T + np.outer(v,np.ones(Y.shape[0])) + np.outer(np.ones(X.shape[0]),w) #matrix version of ||xi-yj||^2 = -2<xi,yj> + <xi,xi> + <yj,yj>
    dist = -2*X@Y.T + v[:,None] + w[None,:] #matrix version of ||xi-yj||^2 = -2<xi,yj> + <xi,xi> + <yj,yj>
    return np.exp(-dist/(r**2))
def linear_kernel(X:np.ndarray, Y:np.ndarray) -> np.ndarray:
    '''
    Computes the linear kernel between every data in X€R^(n*d) and in Y€R^(m*d). Data are rows, in both matrices. K will be of dimension n x m
    '''
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    #v = la.norm(X,axis=1)**2                                                            #a vectot whose entry are the squared norm of each row (i.e data) of X
    #w = la.norm(Y,axis=1)**2                                                            #a vectot whose entry are the squared norm of each row (i.e data) of Y
    v = np.sum(X*X,axis = 1)
    w = np.sum(Y*Y,axis = 1)
    #dist = -2*X@Y.T + np.outer(v,np.ones(Y.shape[0])) + np.outer(np.ones(X.shape[0]),w) #matrix version of ||xi-yj||^2 = -2<xi,yj> + <xi,xi> + <yj,yj>
    dist = -2*X@Y.T + v[:,None] + w[None,:] #matrix version of ||xi-yj||^2 = -2<xi,yj> + <xi,xi> + <yj,yj>
    return dist

#sampling strategies
def Uniform():
    return 1/n * np.ones(n)
def RLS(Q,Lambda,reg_param):
    '''
    exact_ridge_lev_score = diag((K+n*reg_param*I)^(-1))

    We have K = Q*Lambda*Q.T
    So we have K*(K+n*reg_param*I)^(-1) = Q*Lambda*(Lambda+n*reg_param*I)^(-1)*Q.T
    new_Lambda is just the inner matrix Lambda*(Lambda+n*reg_param*I)^(-1)

    '''
    new_Lambda = Lambda/(Lambda+n*reg_param)
    #scores = np.diag(Q@np.diag(new_Lambda)@(Q.T))
    scores = np.einsum('ij,j,ij->i', Q, new_Lambda, Q)
    return scores/np.sum(scores)
def exact_best_k_leverage_score(Q,rank):
    '''
    We have K = Q*Lambda*Q.T
    We compute squared norm of rows of Q up to column k = rank
    '''
    scores = la.norm(Q[:,0:rank],axis = 1)**2 
    return scores/np.sum(scores)
def RPCholesky(m):
    F = np.zeros(shape = (n,m))
    #d = np.array([kernel(X[i],X[i]) for i in range(n)]) #should be one good way of doing it, but since it does not work due to error in v = la.norm(X,axis=1)**2 when X is just one data
    d = np.diag(K_nn) #and since I have computed the full kernel matrix K, I will use this naive way
    Nystrom_indexes = []
    for i in range(m):
        if np.any(np.isnan(F)):
            print(f"The approximation seems good alredy, with {i} Nyström points")
            return np.reshape(np.array(Nystrom_indexes),(i,))
        sampled_pivot_index = np.random.choice(list(range(n)), size=1, p=d/np.sum(d))
        #sampled_pivot_index = np.atleast_1d(np.argmax(d))
        g = np.array([kernel(X,X[sampled_pivot_index])])
        g = np.reshape(g,(n,1))
        g = g - F[:,:i]@(F[sampled_pivot_index,:i].T)
        g = np.reshape(g,(n,))
        F[:,i] = g/np.sqrt(g[sampled_pivot_index])
        d = d - F[:,i]**2
        d = np.maximum(d,0)
        Nystrom_indexes.append(sampled_pivot_index)
    return np.reshape(np.array(Nystrom_indexes),(m,))#,F

#new selection strategies
def MER(m):
    selected_indexes = [] #initialized at -1 for convenience
    leftover_indexes = list(range(n))
    K_nm = np.zeros((n,m))
    criteria_vals_for_plot = []
    alpha_tilde_for_plot = []
    Nystrom_indexes_for_plot = []
    #selection of the first index | can be optimized vectorizing the fit & predict step and then usign np.argmin
    best_risk = np.inf
    best_index = -1
    criteria_vals = []
    for i in range(n):
        alpha_tilde = fit(kernel(X,X[i]),reg_param,labels,Nystrom = True,Nystrom_indexes=[i])
        approx_preds = predict(X[i],X,alpha_tilde)
        approx_empirical_risk = risk(approx_preds,labels)
        criteria_vals.append(approx_empirical_risk)
        if approx_empirical_risk < best_risk:
            best_index = i
            best_risk = approx_empirical_risk
    selected_indexes.append(best_index)
    leftover_indexes.remove(best_index)
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))
    #to plot/create a gif
    if d == 2:
        alpha_tilde = fit(K_nm[:,:1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
        criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = 0)
    if d == 1:
        alpha_tilde = fit(K_nm[:,:1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
        #criteria_and_plot_1D(selected_indexes,alpha_tilde,n*(np.array(criteria_vals)-risk(0,labels)))
        criteria_vals_for_plot.append(n*(np.array(criteria_vals)-risk(0,labels)))
        alpha_tilde_for_plot.append(alpha_tilde)
        Nystrom_indexes_for_plot.append(selected_indexes.copy())

    #selection of the rest m-1 indexes
    for l in range(1,m):            #K_nm[:,:l] is K_nl, K_nm[selected_indexes,:l] is K_ll 
        L_mm = la.cholesky(K_nm[selected_indexes,:l]+1000*np.spacing(1)*np.eye(len(selected_indexes)))       #choleksy of K_ll
        L_mm_tilde = la.cholesky(K_nm[:,:l].T@K_nm[:,:l]+n*reg_param*K_nm[selected_indexes,:l]+1000*np.spacing(1)*np.eye(len(selected_indexes)))     #cholesky of K_nlK_ln + nlambdaK_ll
        criteria_vals = criteria(K_nm[:,:l],L_mm_tilde,L_mm,selected_indexes,leftover_indexes)
        best_index = leftover_indexes[np.argmin(criteria_vals)]
        # if i == m-1:
        #     criteria_plot(i,criteria_vals,leftover_indexes,selected_indexes)
        selected_indexes.append(best_index)
        leftover_indexes.remove(best_index)
        K_nm[:,l] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
        if d == 2:
            # #to plot/create a gif
            alpha_tilde = fit(K_nm[:,:l+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
            criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = l)

        if d == 1 and (l == 1 or l == 2):
            # #to plot/create a gif
            alpha_tilde = fit(K_nm[:,:l+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
            #criteria_and_plot_1D(selected_indexes,alpha_tilde,criteria_vals)
            criteria_vals_for_plot.append(criteria_vals)
            alpha_tilde_for_plot.append(alpha_tilde)
            Nystrom_indexes_for_plot.append(selected_indexes.copy())

    if d == 5:
        global Nystrom_indexes_MER 
        Nystrom_indexes_MER = selected_indexes
        selection_true_data_plot_MER(np.concatenate([X, labels[:, None]], axis=1))
    # alpha_tilde = fit(K_nm[:,:l+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
    # criteria_and_plot_1D(selected_indexes,alpha_tilde,criteria_vals)
    if explanation_behavior:
        criteria_and_plot_1D_2ndVersion(Nystrom_indexes_for_plot,alpha_tilde_for_plot,criteria_vals_for_plot,'MER')
    return selected_indexes
def MER_batch(m,batch_size):
    selected_indexes = [] #initialized at -1 for convenience
    leftover_indexes = list(range(n))
    K_nm = np.zeros((n,m))

    #selection of the first index | can be optimized vectorizing the fit & predict step and then usign np.argmin
    best_risk = np.inf
    best_index = -1
    criteria_vals = []
    for i in range(n):
        alpha_tilde = fit(kernel(X,X[i]),reg_param,labels,Nystrom = True,Nystrom_indexes=[i])
        approx_preds = predict(X[i],X,alpha_tilde)
        approx_empirical_risk = risk(approx_preds,labels)
        criteria_vals.append(approx_empirical_risk)
        if approx_empirical_risk < best_risk:
            best_index = i
            best_risk = approx_empirical_risk
    selected_indexes.append(best_index)
    leftover_indexes.remove(best_index)
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))
    #to plot/create a gif
    # alpha_tilde = fit(K_nm[:,:1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
    # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = 0)

    #selection of the rest m-1 indexes
    for l in range(1,m):            #K_nm[:,:l] is K_nl, K_nm[selected_indexes,:l] is K_ll
        batch = np.random.choice(leftover_indexes,size = batch_size,p = 1/len(leftover_indexes)* np.ones(len(leftover_indexes)),replace = False)
        L_mm = la.cholesky(K_nm[selected_indexes,:l]+1000*np.spacing(1)*np.eye(len(selected_indexes)))       #choleksy of K_ll
        L_mm_tilde = la.cholesky(K_nm[:,:l].T@K_nm[:,:l]+n*reg_param*K_nm[selected_indexes,:l]+1000*np.spacing(1)*np.eye(len(selected_indexes)))     #cholesky of K_nlK_ln + nlambdaK_ll
        criteria_vals = criteria(K_nm[:,:l],L_mm_tilde,L_mm,selected_indexes,batch)
        best_index = batch[np.argmin(criteria_vals)]
        # if i == m-1:
        #     criteria_plot(i,criteria_vals,leftover_indexes,selected_indexes)
        selected_indexes.append(best_index)
        leftover_indexes.remove(best_index)
        K_nm[:,l] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
        # #to plot/create a gif
        # alpha_tilde = fit(K_nm[:,:l+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
        # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = i)   
    return selected_indexes
def MER_rank1update(m):         #not numerically stable
    selected_indexes = []
    leftover_indexes = list(range(n))
    K_nm = np.zeros((n,m))
    L_mm = np.empty((m,m))
    L_mm_tilde = np.empty((m,m))

    #selection of the first index | can be optimized vectorizing the fit & predict step and then usign np.argmin
    best_risk = np.inf
    best_index = -1
    criteria_vals = []
    for i in range(n):
        alpha_tilde = fit(kernel(X,X[i]),reg_param,labels,Nystrom = True,Nystrom_indexes=[i])
        approx_preds = predict(X[i],X,alpha_tilde)
        approx_empirical_risk = risk(approx_preds,labels)
        criteria_vals.append(approx_empirical_risk)
        if approx_empirical_risk < best_risk:
            best_index = i
            best_risk = approx_empirical_risk
    selected_indexes.append(best_index)
    leftover_indexes.remove(best_index)
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))
    L_mm[0,0] = la.cholesky(K_nm[selected_indexes,:1]+1000*np.spacing(1)*np.eye(len(selected_indexes)))[0,0]
    L_mm_tilde[0,0] =la.cholesky(K_nm[:,:1].T@K_nm[:,:1]+n*reg_param*K_nm[selected_indexes,:1])[0,0]
    A_T_time_a = K_nm[:,:i].T@K_nm[:,i]
    a_t_time_a = K_nm[:,i].T@K_nm[:,i]
    v = A_T_time_a+n*reg_param*K_nm[:-1,:i]
    alpha = a_t_time_a + n*reg_param*K_nm[-1,:i]
    L_mm_tilde[i,:i+1] = cholesky_rank1_expand(L_mm_tilde[:i,:i],v,alpha)
    #to plot/create a gif
    # alpha_tilde = fit(K_nm[:,:1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
    # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = 0)

    #selection of the rest m-1 indexes
    for i in range(1,m):
        A = K_nm[:,:i]@la.inv(K_nm[:,:i].T@K_nm[:,:i]+n*reg_param*K_nm[selected_indexes,:i])@K_nm[:,:i].T
        criteria_vals = criteria(A,K_nm[:,:i],L_mm[:i,:i],selected_indexes,leftover_indexes)
        best_index = leftover_indexes[np.argmin(criteria_vals)]
        # if i == m-1:
        #     criteria_plot(i,criteria_vals,leftover_indexes,selected_indexes)
        selected_indexes.append(best_index)
        leftover_indexes.remove(best_index)
        K_nm[:,i] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
        L_mm[i,:i+1] = cholesky_rank1_expand(L_mm[:i,:i],K_nm[selected_indexes[:-1],i],K_nm[selected_indexes[-1],i])
        # #to plot/create a gif
        # alpha_tilde = fit(K_nm[:,:i+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
        # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = i)   
    return selected_indexes
def MER_rank1update_batch(m,batch_size):        #not numerically stable
    selected_indexes = []
    leftover_indexes = list(range(n))
    K_nm = np.zeros((n,m))
    L_mm = np.empty((m,m))
    L_mm_tilde = np.empty((m,m))

    #selection of the first index | can be optimized vectorizing the fit & predict step and then usign np.argmin
    best_risk = np.inf
    best_index = -1
    criteria_vals = []
    for i in range(n):
        alpha_tilde = fit(kernel(X,X[i]),reg_param,labels,Nystrom = True,Nystrom_indexes=[i])
        approx_preds = predict(X[i],X,alpha_tilde)
        approx_empirical_risk = risk(approx_preds,labels)
        criteria_vals.append(approx_empirical_risk)
        if approx_empirical_risk < best_risk:
            best_index = i
            best_risk = approx_empirical_risk
    selected_indexes.append(best_index)
    leftover_indexes.remove(best_index)
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))
    L_mm[0,0] = la.cholesky(K_nm[selected_indexes,:1]+1000*np.spacing(1)*np.eye(len(selected_indexes)))[0,0]
    L_mm_tilde[0,0] =la.cholesky(K_nm[:,:1].T@K_nm[:,:1]+n*reg_param*K_nm[selected_indexes,:1])[0,0]
    #to plot/create a gif
    # alpha_tilde = fit(K_nm[:,:1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
    # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = 0)

    #selection of the rest m-1 indexes
    for i in range(1,m):
        batch = np.random.choice(leftover_indexes,size = batch_size,p = 1/len(leftover_indexes)* np.ones(len(leftover_indexes)),replace = False)
        criteria_vals = criteria(K_nm[:,:i],L_mm_tilde[:i,:i],L_mm[:i,:i],selected_indexes,batch)
        best_index = batch[np.argmin(criteria_vals)]
        # if i == m-1:
        #     criteria_plot(i,criteria_vals,leftover_indexes,selected_indexes)
        selected_indexes.append(best_index)
        leftover_indexes.remove(best_index)
        K_nm[:,i] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
        L_mm[i,:i+1] = cholesky_rank1_expand(L_mm[:i,:i],K_nm[selected_indexes[:-1],i],K_nm[selected_indexes[-1],i])
        A_T_time_a = K_nm[:,:i].T@K_nm[:,i]
        a_t_time_a = K_nm[:,i].T@K_nm[:,i]
        v = A_T_time_a+n*reg_param*K_nm[selected_indexes[:-1],i]
        alpha = a_t_time_a + n*reg_param*K_nm[selected_indexes[-1],i]
        L_mm_tilde[i,:i+1] = cholesky_rank1_expand(L_mm_tilde[:i,:i],v,alpha)
        # #to plot/create a gif
        # alpha_tilde = fit(K_nm[:,:i+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
        # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = i)   
    return selected_indexes
def MER_no_regularisation(m):
    selected_indexes = [] #initialized at -1 for convenience
    leftover_indexes = list(range(n))
    K_nm = np.zeros((n,m))

    #selection of the first index | can be optimized vectorizing the fit & predict step and then usign np.argmin
    best_risk = np.inf
    best_index = -1
    criteria_vals = []
    for i in range(n):
        alpha_tilde = fit_no_regularisation(kernel(X,X[i]),labels,Nystrom = True,Nystrom_indexes=[i])
        approx_preds = predict(X[i],X,alpha_tilde)
        approx_empirical_risk = risk(approx_preds,labels)
        criteria_vals.append(approx_empirical_risk)
        if approx_empirical_risk < best_risk:
            best_index = i
            best_risk = approx_empirical_risk
    selected_indexes.append(best_index)
    leftover_indexes.remove(best_index)
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))

    #selection of the rest m-1 indexes
    for l in range(1,m):            #K_nm[:,:l] is K_nl, K_nm[selected_indexes,:l] is K_ll 
        #L_mm = la.cholesky(K_nm[selected_indexes,:l]+1000*np.spacing(1)*np.eye(len(selected_indexes)))       #choleksy of K_ll
        L_mm_tilde = la.cholesky(K_nm[:,:l].T@K_nm[:,:l]+100*np.spacing(1)*np.eye(len(selected_indexes)))     #cholesky of K_nlK_ln + nlambdaK_ll
        criteria_vals = criteria_no_regularisation(L_mm_tilde,K_nm[:,:l],leftover_indexes)
        best_index = leftover_indexes[np.argmin(criteria_vals)]
        selected_indexes.append(best_index)
        leftover_indexes.remove(best_index)
        K_nm[:,l] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
    
    return selected_indexes
def MAR(m):
    selected_indexes = [] #initialized at -1 for convenience
    leftover_indexes = list(range(n))
    K_nm = np.zeros((n,m))
    criteria_vals_for_plot = []
    alpha_tilde_for_plot = []
    Nystrom_indexes_for_plot = []

    #selection of the first index | can be optimized vectorizing the fit & predict step and then usign np.argmin
    best_risk = np.inf
    best_index = -1
    criteria_vals = []
    best_index = np.argmax(np.abs(labels))
    # for i in range(n):
    #     alpha_tilde = fit(kernel(X,X[i]),reg_param,labels,Nystrom = True,Nystrom_indexes=[i])
    #     approx_preds = predict(X[i],X,alpha_tilde)
    #     approx_empirical_risk = risk(approx_preds,labels)
    #     criteria_vals.append(approx_empirical_risk)
    #     if approx_empirical_risk < best_risk:
    #         best_index = i
    #         best_risk = approx_empirical_risk
    selected_indexes.append(best_index)
    leftover_indexes.remove(best_index)
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))
    if d == 1:
        alpha_tilde = fit(K_nm[:,:1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
        #criteria_and_plot_1D(selected_indexes,alpha_tilde,n*(np.array(criteria_vals)-risk(0,labels)))
        criteria_vals_for_plot.append(np.abs(labels))
        alpha_tilde_for_plot.append(alpha_tilde)
        Nystrom_indexes_for_plot.append(selected_indexes.copy())
    #to plot/create a gif
    # alpha_tilde = fit(K_nm[:,:1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
    # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = 0)

    #selection of the rest m-1 indexes
    for i in range(1,m):
        preds = predict(X[selected_indexes],X[leftover_indexes],fit(K_nm[:,:i],reg_param,y = labels,Nystrom = True,Nystrom_indexes=selected_indexes))
        res = labels[leftover_indexes] - preds
        best_index = leftover_indexes[np.argmax(np.abs(res))]
        # if i == m-1:
        #     criteria_plot(i,criteria_vals,leftover_indexes,selected_indexes)
        selected_indexes.append(best_index)
        leftover_indexes.remove(best_index)
        K_nm[:,i] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
        # #to plot/create a gif
        # alpha_tilde = fit(K_nm[:,:i+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
        # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = i)
        if d == 1 and (i == 1 or i == 2):
            # #to plot/create a gif
            alpha_tilde = fit(K_nm[:,:i+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
            #criteria_and_plot_1D(selected_indexes,alpha_tilde,criteria_vals)
            criteria_vals_for_plot.append(np.abs(res))
            alpha_tilde_for_plot.append(alpha_tilde)
            Nystrom_indexes_for_plot.append(selected_indexes.copy())
    if d == 5:
        global Nystrom_indexes_MAR 
        Nystrom_indexes_MAR = selected_indexes
        selection_true_data_plot_MAR(np.concatenate([X, labels[:, None]], axis=1))
    if explanation_behavior:
        criteria_and_plot_1D_2ndVersion(Nystrom_indexes_for_plot,alpha_tilde_for_plot,criteria_vals_for_plot, 'MAR')
    return selected_indexes
def max_abs_argmax(m):
    Nystrom_indexes = np.argsort(np.absolute(labels.flatten()))[-m:]
    return Nystrom_indexes
def MS(m):
    abs_labels = np.absolute(labels.flatten())
    Nystrom_indexes = np.random.choice(list(range(n)), size=m, p=abs_labels/np.sum(abs_labels), replace=False)
    return Nystrom_indexes

#useful for my_strategies
def criteria(K_nm,L_mm_tilde,L_mm,Nystrom_indexes,new_indexes):
    '''
    Computes the criteria given the alredy selected indexes and all the remaining indexes.
    Since the A factor doesn't depend on the new point, as well as K_nm and L_mm (chol factor of K_mm), they are precomputed.
    '''
    
    K_nnew = kernel(X,X[new_indexes])
    b = K_nnew - K_nm@spla.cho_solve((L_mm,True),K_nnew[Nystrom_indexes,:])
    #b = K_nnew - K_nm@la.solve(L_mm.T,la.solve(L_mm,K_nnew[Nystrom_indexes,:]))
    c = np.diag(b)
    # res_y_neg = (A-np.eye(n))@labels                #(A-I)y
    # res_b_neg = (A-np.eye(n))@b                     #(A-I)b
    res_y_neg = A_time_x(K_nm,L_mm_tilde,labels)-labels                #(A-I)y
    res_b_neg = A_time_x(K_nm,L_mm_tilde,b)-b                      #(A-I)b
    #some_num = np.einsum('ik,jk->kij',res_b_neg,b) #(A-I)bb^t(A-I)y
    #some_num = some_num@res_y_neg
    #some_num = np.einsum('ik,jk,j->ki', res_b_neg, b, res_y_neg)
    some_num = ((b.T @ res_y_neg)[None, :] * res_b_neg).T
    #some_den = -b.T@res_b_neg
    some_den = np.einsum('ij,ji->i',-b.T,res_b_neg)
    #denominator = np.diag((some_den+n*reg_param*c))
    denominator = some_den+n*reg_param*c
    num_1 = (la.norm(some_num,axis = 1))**2
    num_2 = (some_num@res_y_neg)
    return 1/denominator * ( num_1 /denominator +2*num_2)
def A_time_x(K_nm,L_mm_tilde,x):
    return K_nm@spla.cho_solve((L_mm_tilde,True),K_nm.T@x)
def check_criteria_correctness(Nystrom_indexes):
    #predictions with m and m-1 Nyström points
    K_nm = kernel(X,X[Nystrom_indexes])   #Nystrom approx
    alpha_tilde_all = fit(K_nm,reg_param,labels,Nystrom=True,Nystrom_indexes = Nystrom_indexes)
    alpha_tilde_all_but_last = fit(K_nm[:,:-1],reg_param,labels,Nystrom=True,Nystrom_indexes = Nystrom_indexes[:-1])
    approx_empirical_risk_all = risk(predict(X[Nystrom_indexes],X,alpha_tilde_all),labels)
    approx_empirical_risk_all_but_last = risk(predict(X[Nystrom_indexes[:-1]],X,alpha_tilde_all_but_last),labels)
    
    #compute criteria using the m^th point as the new point
    K_nm = kernel(X,X[Nystrom_indexes[:-1]])
    L_mm = la.cholesky(K_nm[Nystrom_indexes[:-1],:])
    L_mm_tilde = la.cholesky(K_nm.T@K_nm+n*reg_param*K_nm[Nystrom_indexes[:-1],:])
    abs_error = np.abs(n*approx_empirical_risk_all-(n*approx_empirical_risk_all_but_last+criteria(K_nm,L_mm_tilde,L_mm,Nystrom_indexes[:-1],Nystrom_indexes[-1])))
    rel_error = np.abs(abs_error/np.abs(n*approx_empirical_risk_all))
    print(f"The absolut error between eps(Z+z) and eps(Z) + criteria is {abs_error}")
    print(f"while the relative error is{rel_error}")
def criteria_no_regularisation(L_mm,K_nm,new_indexes):
    K_nnew = kernel(X,X[new_indexes])
    w = K_nnew - A_time_x(K_nm,L_mm,K_nnew)
    w = w/la.norm(w,axis = 0)
    return -(labels@w)**2
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
#wrapper for cho_factor & cho_solve
def cholesky_solve(A,b):
    c, low = spla.cho_factor(A)
    return spla.cho_solve((c, low), b)

#sampling
def build_sampling_distributions():                                                            
    scores = {}                                                                                 #variuos type of sampling distributions
    scores['Uniform'] = Uniform()                                                                 #Uniform
    scores['RLS'] = RLS(Q,Lambda,reg_param)         #ridge leverage score
    scores['exact_best_k_leverage_score'] = exact_best_k_leverage_score(Q,rank_k)                 #best_k leverage score
    return scores
def sampling(strategy,m):
    '''
    Returns the sampled indexes, used to approximate the kernel matrix
    '''                                                     
    if strategy == 'RPCholesky':
        indexes = RPCholesky(m)
    elif strategy == 'MER':
        indexes = MER(m)
    elif strategy == 'MER_no_regularisation':
        indexes = MER_no_regularisation(m)
    elif strategy == 'MER_batch':
        indexes = MER_batch(m,batch_size)
    elif strategy == 'MER_rank1update':
        indexes = MER_rank1update(m)
    elif strategy == 'MER_rank1update_batch':
        indexes = MER_rank1update_batch(m,batch_size)
    elif strategy == 'max_abs_argmax':
        indexes = max_abs_argmax(m)
    elif strategy == 'MS':
        indexes = MS(m)
    elif strategy == 'MAR':
        indexes = MAR(m)
    else:
        indexes = np.random.choice(list(range(n)), size=m, p=scores[strategy], replace=False) #replace=True --> iid sampling, i.e. sampling with replacement
    return indexes

#fit & predict
def fit(Kernel_matrix,reg_param,y,Nystrom:bool = False, Nystrom_indexes = None):
    '''
    The parameter y is there beacuse in the criteria expression the Hat matrix A = K_nm(K_mnK_nm + n*lambda*K_mm)^(-1)K_mn  appears applied to the vector b instead to the usual vector of labels y
    In order to use the fit-predict method in conjunction to mimic the expression Ab of the criteria, I let the user to pass whatever vector y he prefers
    '''
    if Nystrom:
        #L = la.cholesky(Kernel_matrix.T@Kernel_matrix + n*reg_param*Kernel_matrix[Nystrom_indexes]+1000*np.spacing(1)*np.eye(len(Nystrom_indexes),len(Nystrom_indexes))) #Kernel_matrix[Nystrom_indexes,:]
        #alpha = la.solve(L.T,la.solve(L,np.atleast_1d(Kernel_matrix.T@y))) #np.atleast_1d(Kernel_matrix.T@y) so that even when Kernel_matrix is a column, I get a one dim vector and la.solve doesent complaint
        alpha = cholesky_solve(Kernel_matrix.T@Kernel_matrix + n*reg_param*Kernel_matrix[Nystrom_indexes]+1000*np.spacing(1)*np.eye(len(Nystrom_indexes),len(Nystrom_indexes)),np.atleast_1d(Kernel_matrix.T@y))
    else:
        alpha = cholesky_solve(Kernel_matrix+n*reg_param*np.eye(n,n),y)
        # L = la.cholesky(Kernel_matrix+n*reg_param*np.eye(n,n))
        # alpha = la.solve(L.T,la.solve(L,y)) 
    return alpha
def predict(Nystrom_data,eval_point,alpha):
    return kernel(Nystrom_data,eval_point).T@alpha
def fit_and_predict(Kernel_matrix,reg_param,y,Nystrom:bool = False, Nystrom_indexes = None,eval_point = None): #TO BE TESTED and USED
    '''
    Still not efficient for my strategy, since Cholesky factor L_mm is recomputed every time
    If eval_point is passed, the estimated function is evaluated there (eg. for plotting instances)
    If eval_point is not passed, the estimated function is evaluated onto the whole dataset, i.e., y^hat is computed
    '''
    alpha = fit(Kernel_matrix,reg_param,y,Nystrom = Nystrom, Nystrom_indexes = None)
    if eval_point is None:
        preds = Kernel_matrix.T@alpha
    else:
        preds = kernel(X[Nystrom_indexes], eval_point).T@alpha
    return preds


def fit_no_regularisation(Kernel_matrix,y,Nystrom:bool = False, Nystrom_indexes = None):
    '''
    The parameter y is there beacuse in the criteria expression the Hat matrix A = K_nm(K_mnK_nm + n*lambda*K_mm)^(-1)K_mn  appears applied to the vector b instead to the usual vector of labels y
    In order to use the fit-predict method in conjunction to mimic the expression Ab of the criteria, I let the user to pass whatever vector y he prefers
    '''
    if Nystrom:
        #L = la.cholesky(Kernel_matrix.T@Kernel_matrix + n*reg_param*Kernel_matrix[Nystrom_indexes]+1000*np.spacing(1)*np.eye(len(Nystrom_indexes),len(Nystrom_indexes))) #Kernel_matrix[Nystrom_indexes,:]
        #alpha = la.solve(L.T,la.solve(L,np.atleast_1d(Kernel_matrix.T@y))) #np.atleast_1d(Kernel_matrix.T@y) so that even when Kernel_matrix is a column, I get a one dim vector and la.solve doesent complaint
        alpha = cholesky_solve(Kernel_matrix.T@Kernel_matrix +1000*np.spacing(1)*np.eye(len(Nystrom_indexes),len(Nystrom_indexes)),np.atleast_1d(Kernel_matrix.T@y))
    else:
        alpha = cholesky_solve(Kernel_matrix,y)
        # L = la.cholesky(Kernel_matrix+n*reg_param*np.eye(n,n))
        # alpha = la.solve(L.T,la.solve(L,y)) 
    return alpha

#performances
def GCV(reg_param_range:np.ndarray) -> np.float64:
    '''
    We are computing the reuglarization parameter that minimizes 1/N Σ [ (y_i-f(x_i)) / (1-tr(hat_matrix)/N) ]^2
    '''
    GCV_val = []
    for param in reg_param_range:
        alpha = fit(K_nn,param,labels)
        predictions = predict(X,X,alpha)
        new_Lambda = Lambda/(Lambda+n*param)
        trace = np.sum(new_Lambda)
        GCV_val.append(1/n * np.sum( ((predictions-labels)/(1-trace/n))**2 ))
    return GCV_val
def risk(pred:np.ndarray,label:np.ndarray) -> np.float64:
    return la.norm(pred-label)**2 /n
def strategy_performance(strategy):
    '''
    Dataset is fixed, randomization only over the Nyström points
    '''
    for m in mrange:
        empirical_performance = 0
        test_perfomance = 0
        #In_sample_SMAPE_performance = 0
        #Out_of_sample_SMAPE_performance = 0
        #no need of repetitions if strategy is my strategy since, once the dataset is fixed, it is deterministic (not the batch one though)
        if (strategy == 'MER' or strategy == 'MER_batch'or strategy == 'MAR' or strategy == 'MER_no_regularisation'):
            print(strategy,m)
            Nystrom_indexes = sampling(strategy,m)
            K_nm = kernel(X,X[Nystrom_indexes])   #Nystrom approx
            alpha_tilde = fit(K_nm,reg_param,labels,Nystrom=True,Nystrom_indexes = Nystrom_indexes)
            approx_preds_empirical = predict(X[Nystrom_indexes],X,alpha_tilde)
            approx_preds_test = predict(X[Nystrom_indexes],X_test,alpha_tilde)
            empirical_performance = risk(approx_preds_empirical,labels)
            test_perfomance = risk(approx_preds_test,labels_test)
            approx_empirical_risk[strategy].append(empirical_performance)
            approx_test_risk[strategy].append(test_perfomance)
            #approx_in_sample_smape[strategy].append(SMAPE(approx_preds_empirical,labels))
            #approx_out_of_sample_smape[strategy].append(SMAPE(approx_preds_test,labels_test))
            if d == 2 and m == mrange[len(mrange)-1]:
                plot_2D(Nystrom_indexes=Nystrom_indexes,alpha_tilde = alpha_tilde,strategy = strategy)
            if d == 1 and m == mrange[len(mrange)-1]:
                plot_1D(Nystrom_indexes=Nystrom_indexes,alpha_tilde = alpha_tilde,strategy = strategy)
        else:
            for rep in range(randomization_repetitions):
                #print(strategy,m)
                Nystrom_indexes = sampling(strategy,m)
                K_nm = kernel(X,X[Nystrom_indexes])   #Nystrom approx
                #print(K_nm.shape,strategy,Nystrom_indexes.shape)
                alpha_tilde = fit(K_nm,reg_param,labels,Nystrom=True,Nystrom_indexes = Nystrom_indexes)
                approx_preds_empirical = predict(X[Nystrom_indexes],X,alpha_tilde)
                approx_preds_test = predict(X[Nystrom_indexes],X_test,alpha_tilde)
                empirical_performance += risk(approx_preds_empirical,labels)
                test_perfomance += risk(approx_preds_test,labels_test)
                #In_sample_SMAPE_performance += SMAPE(approx_preds_empirical,labels)
                #Out_of_sample_SMAPE_performance += SMAPE(approx_preds_test,labels_test)
            approx_empirical_risk[strategy].append(empirical_performance/randomization_repetitions)
            approx_test_risk[strategy].append(test_perfomance/randomization_repetitions)
            #approx_in_sample_smape[strategy].append(In_sample_SMAPE_performance/randomization_repetitions)
            #approx_out_of_sample_smape[strategy].append(Out_of_sample_SMAPE_performance/randomization_repetitions)
    return approx_empirical_risk,approx_test_risk#,approx_in_sample_smape,approx_out_of_sample_smape
def SMAPE(pred,label):
    num = np.abs(pred-label)
    den = 0.5*np.abs(pred)+0.5*np.abs(label)
    return np.sum(num/den)/len(pred)

#plotting performances & results
def plot_1D(Nystrom_indexes,alpha_tilde,strategy):
    xx = np.empty((n_eval,d))
    xx[:,0] = np.linspace(x_min-0.5,x_max+0.5,n_eval)   #evaluation/test points
    
    evaluations = predict(X,xx,alpha)
    approx_evaluations = predict(X[Nystrom_indexes],xx,alpha_tilde)
    
    number_of_axes = 1
    if sampling_strategy == 'RLS':
        number_of_axes = 2
    fig,ax = plt.subplots(1,number_of_axes,figsize = (8*number_of_axes,8),squeeze=False)
    ax = ax[0]

    #data & Nyström points
    ax[0].scatter(X,labels,color = 'orange',label = f'Data, n = {n}',alpha = 0.8, marker = 'x')
    ax[0].scatter(X[Nystrom_indexes],labels[Nystrom_indexes], label = f'Nyström points, m = {len(Nystrom_indexes)}',color = 'white',edgecolors = 'black')

    #true model
    ax[0].plot(xx,model(xx),label='True model',color = 'black')

    #fitted modelS
    ax[0].plot(xx,evaluations,label='Full model',color = 'red',linestyle = '--')
    ax[0].plot(xx,approx_evaluations,label='Nyström approximation',color = 'green',linestyle = '-.')

    ax[0].set_xlabel(f"$x$")
    ax[0].set_ylabel(f"$y$")
    ax[0].legend()

    if sampling_strategy == 'RLS':
        #sampling scores
        sorted_indexes = np.argsort(X,axis = 0)
        sorted_scores = scores[sampling_strategy][sorted_indexes]
        ax[1].scatter(X[sorted_indexes],sorted_scores)
        ax[1].set_title(sampling_strategy)

    # fig.text(0.5, 0.97, f"Model: {model.__name__}, strategy: {strategy}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
    #     ha='center',
    #     bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))

    plt.savefig(f"plots/1D/Experiments/{model.__name__}_{strategy}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2:.2f}.pdf")
    #plt.show()
def plot_2D(Nystrom_indexes,alpha_tilde,strategy):

    xx = np.empty((n_eval,d))
    xx[:,0] = np.linspace(x_min, x_max, n_eval)
    xx[:,1] = np.linspace(y_min, y_max, n_eval)
    XX, YY = np.meshgrid(xx[:,0], xx[:,1])
    
    XXX = np.array([XX.flatten(),YY.flatten()]).T #array version of meshgrid, vectorized for evaluation purposes
    
    evaluations = predict(X,XXX,alpha)
    approx_evaluations = predict(X[Nystrom_indexes],XXX,alpha_tilde)

    evaluations = evaluations.reshape(n_eval,n_eval)
    approx_evaluations = approx_evaluations.reshape(n_eval,n_eval)

    fig,ax = plt.subplots(1,3,figsize = (18,6.5),constrained_layout=True)
    ax = ax.flatten()

    #data & Nyström points
    vmin = np.min([np.min(model(XX.flatten(), YY.flatten())),np.min(evaluations),np.min(approx_evaluations)])
    vmax = np.max([np.max(model(XX.flatten(), YY.flatten())),np.max(evaluations),np.max(approx_evaluations)])

    ax[0].scatter(X[:,0],X[:,1],color = 'orange', label = f'Data, n = {n}',alpha = 0.8, marker = 'x')
    ax[0].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1], label = f"Nyström points, m = {len(Nystrom_indexes)}",color = 'white',edgecolors = 'black')
    ax[1].scatter(X[:,0],X[:,1],color = 'orange', label = f'Data, n = {n}',alpha = 0.8, marker = 'x')
    #ax[i].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1], label = f"Nyström points, m = {len(Nystrom_indexes)}",color = 'black')
    #ax[i].scatter(X[:,0],X[:,1],color = 'orange', label = 'data',alpha = 0.8, marker = 'x')
    #ax[1].scatter(X[:,0],X[:,1],color = 'orange', label = 'Data',alpha = 0.8, marker = 'x')
    ax[2].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1], label = f"Nyström points, m = {len(Nystrom_indexes)}",color = 'white',edgecolors = 'black')
    

    #true model
    img = ax[0].imshow(np.reshape(model(XX.flatten(), YY.flatten()),(n_eval,n_eval)), extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis',vmin = vmin, vmax = vmax)
    #img = ax[0].imshow(np.reshape(model(XXX),(200,200)), extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis') #works good for radial_sin_kD
    #fig.colorbar(img, ax = ax, label='True model values')
    ax[0].set_title('True underlying model')

    img = ax[1].imshow(evaluations, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis',vmin = vmin, vmax = vmax)
    #fig.colorbar(img, ax = ax, label='Fitted values')
    ax[1].set_title('Full estimated model')

    img = ax[2].imshow(approx_evaluations, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis',vmin = vmin, vmax = vmax)
    ax[2].set_title('Nyström approximation')

    fig.colorbar(img, ax = ax, fraction = 0.02,pad = 0.04,shrink = 0.87)
    for i in range(3):
        ax[i].set_xlabel(f"$x_1$")
        ax[i].set_ylabel(f"$x_2$")
        ax[i].legend(loc='upper left')
    
    # fig.text(0.5, 0.97, f"Model: {model.__name__}, strategy: {strategy}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
    #      ha='center',
    #      bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))

    plt.savefig(f"plots/2D/Experiments/{model.__name__}_{strategy}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2} .pdf")
def GCV_plot(): #Need to fix xticks
    fig,ax = plt.subplots(1,1,figsize = (8,8))
    
    ax.plot(reg_param_range,GCV_val)
    ax.set_title('Generalized Cross Validation values')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #print(f"the newticks are{newticks}")
    #print(f"while the old ones are{ticks}")
    ax.set_xticks(reg_param_range)
    formatted_labels = [f"{x:.2e}" for x in reg_param_range]
    ax.set_xticklabels(formatted_labels,rotation=45, ha='right')
    ax.axvline(reg_param,linestyle='--')
    ax.set_xlabel(r'$\\lambda$')
    plt.savefig('plots/GCV_plot')
    #plt.show()
def emp_risk_plot():
    fig,ax = plt.subplots(figsize = (8,8))
    for i, strategy in enumerate(sampling_strategies):
        ax.plot(mrange,approx_empirical_risk[strategy],label = strategy, marker = markers[i])
    ax.axhline(empirical_risk,label = 'Full model',color= 'black')
    #ax.axvline(degrees_of_freedom,color = 'black',linestyle = '--',label = 'df')
    ax.set_xscale('log')
    
    ax.set_xticks(mrange)
    ax.set_xticklabels(mrange.astype(str))
    
    
    #ax.set_yscale('log')
    ax.set_title(f'Empirical risk, n = {n}')
    ax.set_xlabel('Number of Nyström points')
    ax.set_ylabel('Empirical risk')
    ax.legend()
    if model is None:
        # fig.text(0.5, 0.97, f"Model: True data, r = {r}, $\\lambda$ = {reg_param:.2e}",
        #         ha='center',
        #         bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
        plt.savefig(f"plots/Truedata/EmpRisk/r = {r}_lambda = {reg_param:.2e}.pdf")
    else:
        # fig.text(0.5, 0.97, f"Model: {model.__name__}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
        #         ha='center',
        #         bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
        if d == 1:
            plt.savefig(f"plots/1D/EmpRisk/{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
        if d == 2:
            plt.savefig(f"plots/2D/EmpRisk/{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
        else:
            plt.savefig(f"plots/kD/EmpRisk/{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
def test_risk_plot():
    fig,ax = plt.subplots(figsize = (8,8))
    for i, strategy in enumerate(sampling_strategies):
        ax.plot(mrange,approx_test_risk[strategy],label = strategy, marker = markers[i])

    ax.axhline(test_risk,label = 'Full model',color = 'black')
    #ax.axvline(degrees_of_freedom,color = 'black',linestyle = '--',label = 'df')
    
    ax.set_xscale('log')
    #ax.set_yscale('log')

    ax.set_xticks(mrange)
    ax.set_xticklabels(mrange.astype(str))
    
    ax.set_title(f'Test risk, n = {n}, n_test = {n_test}')
    ax.set_xlabel('Number of Nyström points')
    ax.set_ylabel('Test risk')
    ax.legend()
    plt.tight_layout()
    if model is None:
        # fig.text(0.5, 0.97, f"Model: True data, r = {r}, $\\lambda$ = {reg_param:.2e}",
        #         ha='center',
        #         bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
        plt.savefig(f"plots/Truedata/TestRisk/r = {r}_lambda = {reg_param:.2e}.pdf")
    else:
        # fig.text(0.5, 0.97, f"Model: {model.__name__}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
        #         ha='center',
        #         bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
        if d == 1:
            plt.savefig(f"plots/1D/TestRisk/{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
        if d == 2:
            plt.savefig(f"plots/2D/TestRisk/{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
        else:
            plt.savefig(f"plots/kD/TestRisk/{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
def criteria_plot(l,criteria_vals,leftover_indexes,selected_indexes):
    fig,ax = plt.subplots(figsize = (7,8))
    
    ax.scatter(X[selected_indexes,0],X[selected_indexes,1],color = 'orange', label = 'Nystrom data',alpha = 0.8, marker = 'x')
    #ax.scatter(X[selected_indexes,0],X[selected_indexes,1], label = f"Nyström points, m = {len(selected_indexes)}",color = 'black')
    
    sc = ax.scatter(X[leftover_indexes,0],X[leftover_indexes,1],c = criteria_vals, label = 'Not selected data',cmap = 'viridis')
    fig.colorbar(sc, label = 'score', ax = ax)

    ax.set_title(f'Data Score of the {n-l} reamining points,{l} where alredy selected')
    ax.legend()
    plt.savefig('plots/Criteria')    
def criteria_and_plot_2D(Nystrom_indexes,alpha_tilde,criteria_vals = None,frame_id = 0):

    xx = np.empty((n_eval,d))
    xx[:,0] = np.linspace(x_min, x_max, n_eval)
    xx[:,1] = np.linspace(y_min, y_max, n_eval)
    XX, YY = np.meshgrid(xx[:,0], xx[:,1])
    
    XXX = np.array([XX.flatten(),YY.flatten()]).T #array version of meshgrid, vectorized for evaluation purposes
    
    #evaluations = predict(X,XXX,alpha)
    approx_evaluations = predict(X[Nystrom_indexes],XXX,alpha_tilde)

    #evaluations = evaluations.reshape(n_eval,n_eval)
    approx_evaluations = approx_evaluations.reshape(n_eval,n_eval)

    fig,ax = plt.subplots(1,3,figsize = (18,6),constrained_layout=True)
    ax = ax.flatten()
    
    vmin = np.min([np.min(model(XX.flatten(), YY.flatten())),np.min(approx_evaluations)])
    vmax = np.max([np.min(model(XX.flatten(), YY.flatten())),np.max(approx_evaluations)])

    shrink = 0.75

    #data & Nyström points
    ax[0].scatter(X[:,0],X[:,1],color = 'orange', label = f'Data, n = {n}',alpha = 0.8, marker = 'x')
    ax[0].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1], label = f"Nyström points, m = {len(Nystrom_indexes)}",color = 'white',edgecolors = 'black')
    ax[2].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1], label = f"Nyström points, m = {len(Nystrom_indexes)}",color = 'white',edgecolors = 'black')

    #true model
    img = ax[0].imshow(np.reshape(model(XX.flatten(), YY.flatten()),(n_eval,n_eval)), extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis',vmin = vmin, vmax = vmax)
    #img = ax[0].imshow(np.reshape(model(XXX),(200,200)), extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis') #works good for radial_sin_kD
    fig.colorbar(img, ax = ax[0], label='True model values', shrink = shrink,format='%.2f')
    ax[0].set_title('True model')

    leftover_indexes = np.setdiff1d(list(range(n)), Nystrom_indexes[:-1]).tolist()
    
    ax[1].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1],color = 'white', label = 'Nystrom data',alpha = 0.8,marker = 'x')
    
    circle = Circle((X[Nystrom_indexes[-1],0],X[Nystrom_indexes[-1],1]), radius=0.3, fill=False, edgecolor='red', linewidth=2)
    ax[1].add_patch(circle)

    #ax.scatter(X[selected_indexes,0],X[selected_indexes,1], label = f"Nyström points, m = {len(selected_indexes)}",color = 'black')
    sc = ax[1].scatter(X[leftover_indexes,0],X[leftover_indexes,1],c = criteria_vals, label = 'Leftovers data',cmap = 'cividis')
    fig.colorbar(sc, label = 'Score', ax = ax[1],shrink = shrink,format = '%.2f')

    ax[1].set_title(f'Data Score of the {len(criteria_vals)} reamining points, {n-len(criteria_vals)} where alredy selected')
    ax[1].set_xlim(xmin = x_min,xmax = x_max)
    ax[1].set_ylim(ymin = y_min,ymax = y_max)
    ax[1].set_aspect('equal')

    img = ax[2].imshow(approx_evaluations, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis',vmin = vmin, vmax = vmax)
    ax[2].set_title('Nystrom approximation')
    fig.colorbar(img, ax = ax[2], label='Approximated Fitted values',shrink = shrink)#, fraction = 0.1)#,pad = 0.04)
 
    for i in range(3):
        ax[i].legend(loc='upper left')

    #plt.savefig('criteria_and_2D')
        #if model is None:
            # fig.text(0.5, 0.97, f"Model: True data, r = {r}, $\\lambda$ = {reg_param:.2e}",
            #     ha='center',
            #     bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
        #else:
        # fig.text(0.5, 0.97, f"Model: {model.__name__}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
        #         ha='center',
        #         bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
    plt.savefig(f'frames/frame_{frame_id:03d}.pdf')  # Salva nella cartella "frames"
    plt.close()
def criteria_and_plot_1D(Nystrom_indexes,alpha_tilde,criteria_vals = None):
    xx = np.empty((n_eval,d))
    xx[:,0] = np.linspace(x_min-0.5,x_max+0.5,n_eval)   #evaluation/test points
    
    evaluations = predict(X,xx,alpha)
    approx_evaluations = predict(X[Nystrom_indexes],xx,alpha_tilde)
    
    number_of_axes = 2

    fig,ax = plt.subplots(1,number_of_axes,figsize = (16,8),squeeze=False)
    ax = ax[0]

    #data & Nyström points
    ax[0].scatter(X,labels,color = 'orange',label = f'Data, n = {n}',alpha = 0.8, marker = 'x')
    ax[0].scatter(X[Nystrom_indexes],labels[Nystrom_indexes], label = f'Nyström points, m = {len(Nystrom_indexes)}',color = 'white',edgecolors = 'black',s=80,linewidths = 2.5,marker='X')
    ax[1].scatter(X[Nystrom_indexes],np.zeros(len(Nystrom_indexes)), label = f'Nyström points, m = {len(Nystrom_indexes)}',color = 'white',edgecolors = 'black',s=80,linewidths = 2.5,marker='X')

    #true model
    #ax[0].plot(xx,model(xx),label='True model',color = 'black')

    #fitted modelS
    ax[0].plot(xx,evaluations,label='Full model',color = 'red')#,linestyle = '--')
    ax[0].plot(xx,approx_evaluations,label='Nyström approximation',color = 'green',linestyle = '--')

    ax[0].set_xlabel(f"$x$")
    ax[0].set_ylabel(f"$y$")
    ax[0].legend()

    leftover_indexes = np.setdiff1d(list(range(n)), Nystrom_indexes[:-1]).tolist()
    
    #sampling scores
    sorted_indexes = np.argsort(X[leftover_indexes],axis = 0).flatten()
    sorted_criteria = np.array(criteria_vals)[sorted_indexes]
    ax[1].plot(X[leftover_indexes][sorted_indexes],sorted_criteria)
    
    # ax[1].scatter(X[leftover_indexes],criteria_vals)
    
    ax[1].set_xlabel(f"$x$")
    ax[1].set_ylabel(f'$score(x,\\lambda)$')
    #ax[1].legend()
    
    ax[1].set_title(f'Score values')

    # fig.text(0.5, 0.97, f"Model: {model.__name__}, strategy: {strategy}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
    #     ha='center',
    #     bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))

    plt.savefig(f"plots/1D/Experiments/criteria_at_{len(Nystrom_indexes)}_{model.__name__}_{strategy}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
    #plt.show()
def criteria_and_plot_1D_2ndVersion(Nystrom_indexes,alpha_tilde,criteria_vals,strat_flag):
    xx = np.empty((n_eval,d))
    xx[:,0] = np.linspace(x_min-0.5,x_max+0.5,n_eval)   #evaluation/test points
    approx_evaluations = []
    for i in range(len(Nystrom_indexes)):
        approx_evaluations.append(predict(X[Nystrom_indexes[i]],xx,alpha_tilde[i])) 
    

    fig,ax = plt.subplots(1,2,figsize = (16,8),squeeze=False)
    ax = ax.flatten()
    #ax = ax[0]
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    #data & Nyström points
    for i in range(2):
        ax[i].scatter(X,labels,color = 'orange',label = f'Data, n = {n}',alpha = 0.8, marker = 'x')
        ax[i].scatter(X[Nystrom_indexes[i]],labels[Nystrom_indexes[i]], label = f'Nyström points',color = 'white',edgecolors = 'black',s=80,linewidths = 2.5,marker='X')#, m = {len(Nystrom_indexes)}
        #ax[i].plot(xx,evaluations,label='Full model',color = 'red',linestyle = linestyles[0])
        ax[i].plot(xx,approx_evaluations[i],label='Nyström approximation',linestyle =linestyles[1])#color = 'green'
        ax[i].set_xlabel(f"$x$")
        ax[0].set_ylabel(f"$y$")
        ax[i].set_title(f'Selection {i+1}')
    
    handles, legend_labels = ax[0].get_legend_handles_labels()

    #empirical_risk_sofar = risk(predict(X[Nystrom_indexes[i]],X,alpha_tilde[i]),labels)
    test_risk_sofar = risk(predict(X[Nystrom_indexes[1]],X_test,alpha_tilde[1]),labels_test)
    etichetta_custom = f'Test risk: {test_risk_sofar:.2f}'

    # Crea un "handle finto" solo per mostrare il testo nella legenda
    custom_patch = Patch(facecolor='none', edgecolor='none')  # Nessun colore

    # Aggiungilo alla legenda
    handles.append(custom_patch)
    legend_labels.append(etichetta_custom)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles, legend_labels,loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4)
    plt.savefig(f"plots/1D/Experiments/{strat_flag}_{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2:.2}.pdf")
    
    if strat_flag == 'MER':
        fig,ax = plt.subplots(1,2,figsize = (16,8),squeeze=False)
        ax = ax.flatten()

        for i in range(2):
            leftover_indexes = np.setdiff1d(list(range(n)), Nystrom_indexes[i][:-1]).tolist()
            ax[i].scatter(X[Nystrom_indexes[i]],np.zeros(len(Nystrom_indexes[:i+1])), label = f'Nyström points',color = 'white',edgecolors = 'black',s=80,linewidths = 2.5,marker='X')#, m = {len(Nystrom_indexes)}
            sorted_indexes = np.argsort(X[leftover_indexes],axis = 0).flatten()
            sorted_criteria = np.array(np.minimum(criteria_vals[i],0))[sorted_indexes]
            ax[i].plot(X[leftover_indexes][sorted_indexes],sorted_criteria)
            ax[i].set_ylim(1.05*np.min(criteria_vals[0]),-0.05*np.min(criteria_vals[0]))
            # ax[1].scatter(X[leftover_indexes],criteria_vals)
            
            ax[i].set_xlabel(f"$x$")
            ax[0].set_ylabel(r'$\Delta(z_{m+1})$', fontsize=20)
            #ax[1].legend()
            
            ax[i].set_title(f'Selection {i+1}')
        handles, legend_labels = ax[0].get_legend_handles_labels()
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.legend(handles, legend_labels,loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4)
        plt.savefig(f"plots/1D/Experiments/criteria_{strat_flag}_{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2:.2e}.pdf")
        

    
    #plt.show()
def typical_function_in_RKHS_plot():
    xxs = np.empty((n_eval,d))
    xxs[:,0] = np.linspace(-6,6,n_eval)   #evaluation/test points

    single_center = np.empty((1,d))
    single_center[:,0] = 0
    single_alpha = np.array([1])
    
    centers = np.empty((10,d))
    #alphas =  rng.standard_normal(10)
    alphas =  np.array([0.5, 1, -1.5, -1, 4, -2, 0.5, -1, 0.5, 2])
    for i in range(10):
        centers[i,:] = i-5

    function_1 = kernel(single_center,xxs).T@single_alpha
    lin_comb_functions = kernel(centers,xxs).T@alphas
    y_lim = []
    y_lim.append(np.min(lin_comb_functions))
    y_lim.append(np.max(lin_comb_functions))
    number_of_axes = 2

    fig,ax = plt.subplots(1,number_of_axes,figsize = (8*number_of_axes,8),squeeze=False)
    ax = ax.flatten()

    ax[0].plot(xxs,function_1,color = 'black')
    ax[1].plot(xxs,lin_comb_functions,color = 'black')
    #fitted modelS
    #ax[0].plot(xx,evaluations,label='Full model',color = 'red',linestyle = '--')
    #ax[0].plot(xx,approx_evaluations,label='Nyström approximation',color = 'green',linestyle = '-.')

    ax[0].set_xlabel(f"$x$")
    ax[0].set_ylabel(f"$y$")
    ax[0].set_title('Example of atom')
    ax[0].set_ylim(y_lim)

    ax[1].set_xlabel(f"$x$")
    #ax[1].set_ylabel(f"$y$")
    ax[1].set_title('Example of function')


    # fig.text(0.5, 0.97, f"Model: {model.__name__}, strategy: {strategy}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
    #     ha='center',
    #     bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))

    plt.savefig(f"plots/1D/Experiments/Typical_functions.pdf")
    #plt.show()
def Signal_plot():
    xx = np.empty((n_eval,d))
    xx[:,0] = np.linspace(x_min-0.5,x_max+0.5,n_eval)   #evaluation/test points
    
    
    number_of_axes = 1
    fig,ax = plt.subplots(1,number_of_axes,figsize = (8*number_of_axes,8),squeeze=False)
    ax = ax[0]

    #data & Nyström points
    ax[0].scatter(X,labels,color = 'orange',label = f'Data, n = {n}',alpha = 0.8, marker = 'x')
    #ax[0].scatter(X[Nystrom_indexes],labels[Nystrom_indexes], label = f'Nyström points, m = {len(Nystrom_indexes)}',color = 'white',edgecolors = 'black')

    #true model
    ax[0].plot(xx,model(xx),label='True model',color = 'black')

    #fitted modelS
    #ax[0].plot(xx,evaluations,label='Full model',color = 'red',linestyle = '--')
    #ax[0].plot(xx,approx_evaluations,label='Nyström approximation',color = 'green',linestyle = '-.')

    ax[0].set_xlabel(f"$x$")
    ax[0].set_ylabel(f"$y$")
    ax[0].set_title('Example of signal and data')
    ax[0].legend()

    # fig.text(0.5, 0.97, f"Model: {model.__name__}, strategy: {strategy}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
    #     ha='center',
    #     bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))

    plt.savefig(f"plots/1D/Experiments/Signal_plot.pdf")
    #plt.show()
def bad_approx_vs_good_approx_plot(alpha):
    xx = np.empty((n_eval,d))
    xx[:,0] = np.linspace(x_min-0.5,x_max+0.5,n_eval)   #evaluation/test points
    
    sorted_indexes = np.argsort(X,axis = 0).flatten()
    bad_nystrom_indexes = sorted_indexes[0:200:14]
    #bad_nystrom_indexes = np.random.choice(list(range(n)), size=10, p=1/n * np.ones(n), replace=False)
    good_nystrom_indexes = MER(15)
    K_nm_bad = kernel(X,X[bad_nystrom_indexes])
    K_nm_good = kernel(X,X[good_nystrom_indexes])
    bad_alpha = fit(K_nm_bad,reg_param,labels,Nystrom=True,Nystrom_indexes = bad_nystrom_indexes)
    good_alpha = fit(K_nm_good,reg_param,labels,Nystrom=True,Nystrom_indexes = good_nystrom_indexes)
    bad_predictions = predict(X[bad_nystrom_indexes],xx,bad_alpha)
    good_predictions = predict(X[good_nystrom_indexes],xx,good_alpha)
    full_predictions = predict(X,xx,alpha)

    number_of_axes = 2
    fig,ax = plt.subplots(1,number_of_axes,figsize = (8*number_of_axes,8),squeeze=False)
    ax = ax.flatten()

    #data & Nyström points
    ax[0].scatter(X,labels,color = 'orange',label = f'Data, n = {n}',alpha = 0.8, marker = 'x')
    ax[1].scatter(X,labels,color = 'orange',label = f'Data, n = {n}',alpha = 0.8, marker = 'x')

    ax[0].scatter(X[bad_nystrom_indexes],labels[bad_nystrom_indexes], label = f'Nyström points, m = {len(bad_nystrom_indexes)}',color = 'white',edgecolors = 'black',s=80,linewidths = 2.5,marker='X')
    ax[1].scatter(X[good_nystrom_indexes],labels[good_nystrom_indexes], label = f'Nyström points, m = {len(good_nystrom_indexes)}',color = 'white',edgecolors = 'black',s=80,linewidths = 2.5,marker='X')

    #ax[0].scatter(X[Nystrom_indexes],labels[Nystrom_indexes], label = f'Nyström points, m = {len(Nystrom_indexes)}',color = 'white',edgecolors = 'black')

    #true model
    ax[0].plot(xx,bad_predictions,color = 'red')
    ax[0].plot(xx,full_predictions,label = 'Full model',color = 'black',linestyle = '--')
    #ax[0].plot(xx,model(xx),label='True model',color = 'blue')
    ax[1].plot(xx,good_predictions,color = 'green')
    ax[1].plot(xx,full_predictions,color = 'black',linestyle = '--')

    #fitted modelS
    #ax[0].plot(xx,evaluations,label='Full model',color = 'red',linestyle = '--')
    #ax[0].plot(xx,approx_evaluations,label='Nyström approximation',color = 'green',linestyle = '-.')

    ax[0].set_xlabel(f"$x$")
    ax[0].set_ylabel(f"$y$")
    ax[0].set_title('Example of bad approximation')

    ax[1].set_xlabel(f"$x$")
    #ax[1].set_ylabel(f"$y$")
    ax[1].set_title('Example of good approximation')
    
    handles, legend_labels = ax[0].get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles, legend_labels,loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4)
    
    plt.savefig(f"plots/1D/Experiments/bad_and_good_approximation.pdf")
def marginal_plot(x):
    fig,ax  = plt.subplots(2,3,figsize = (9,7))
    ax = ax.flatten()
    for i in range(6):
        if i == 5:
            ax[i].hist(x[:,i], bins=30, color='blue', edgecolor='black',density = True,alpha = 0.8,range = (0,1))
            ax[i].set_xlabel(f"$y$")
        else:
            ax[i].hist(x[:,i], bins=30, color='orange', edgecolor='black',density = True,alpha = 0.8,range = (0,1))
            ax[i].set_xlabel(f"$x_{i+1}$")
        ax[i].set_ylabel('Density')
    fig.suptitle('Marginal distributions')
    plt.tight_layout()
    plt.savefig('plots/Truedata/Marginal_distributions.pdf')
def joint_plot(x):
    fig,ax  = plt.subplots(2,3,figsize = (9,7))
    ax = ax.flatten()
    ax[5].remove()
    for i in range(5):
        ax[i].scatter(x[:,i],x[:,5],s = 5)
        ax[i].set_xlim(0,1)
        ax[i].set_ylim(0,1)
        ax[i].set_xlabel(f"$x_{i+1}$")
        ax[0].set_ylabel(f"$y$")
        ax[3].set_ylabel(f"$y$")
    fig.suptitle('Joint distributions')
    plt.tight_layout()
    plt.savefig('plots/Truedata/Joint_distributions.pdf')
def CVplot():
    xx = np.empty((n_eval,d))
    xx[:,0] = np.linspace(x_min-0.5,x_max+0.5,n_eval)   #evaluation/test points
    reg_range = [0.000001,0.01,0.1,1]
    CVPredictions = np.empty((len(reg_range),n_eval))
    
    for i, reg_param in enumerate(reg_range):
        alpha = fit(K_nn,reg_param,labels)
        CVPredictions[i] = predict(X,xx,alpha)
        
    #number_of_axes = len(reg_range)

    fig,ax = plt.subplots(2,2,figsize = (8,8),squeeze=False)
    
    ax = ax.flatten()


    #data & Nyström points
    for i in range(len(reg_range)):
        # ax[i].scatter(X,labels,color = colors[0],label = f'Data, n = {n}',alpha = 0.8, marker = 'x')

        # ax[i].plot(xx,model(xx),label='True model',color = colors[1])
        # ax[i].plot(xx,CVPredictions[i],label='Fitted model',color = colors[2], linestyle = '--')
        ax[i].scatter(X,labels,color = 'orange',label = f'Data, n = {n}',alpha = 0.8, marker = 'x')

        ax[i].plot(xx,model(xx),label='True model',color = 'black')
        ax[i].plot(xx,CVPredictions[i],label='Fitted model', color = 'red',linestyle = '--')
        
        ax[i].set_xlabel(f"$x$")
        ax[i].set_ylabel(f"$y$")
        ax[i].set_title(f'$\\lambda$ = {reg_range[i]:.0e}')
        #ax[i].legend()

    # fig.text(0.5, 0.91, f"$\\sigma^2$ = {sigma**2}",
    #     ha='center',
    #     bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
    handles, legend_labels = ax[0].get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles, legend_labels,loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)
    #plt.tight_layout()

    plt.savefig(f"plots/CVPlot/{model.__name__}_r = {r}_n = {n}_noise = {sigma**2}.pdf")
def selection_true_data_plot_MAR(x):
    fig,ax  = plt.subplots(2,3,figsize = (9,7))
    ax = ax.flatten()
    ax[5].remove()
    for i in range(5):
        ax[i].scatter(x[:,i],x[:,5],s = 5)
        ax[i].scatter(x[Nystrom_indexes_MAR,i],x[Nystrom_indexes_MAR,5],label = f'Nyström points, m = {len(Nystrom_indexes_MAR)}',color = 'white',edgecolors = 'black',s=80,linewidths = 2.5,marker='X')
        ax[i].set_xlim(0,1)
        ax[i].set_ylim(0,1)
        ax[i].set_xlabel(f"$x_{i+1}$",fontsize = 14)
        ax[0].set_ylabel(f"$y$",fontsize = 14)
        ax[3].set_ylabel(f"$y$",fontsize = 14)

    handles, legend_labels = ax[0].get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles, legend_labels,loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4)
    #fig.suptitle('Real data')
    plt.savefig(f'plots/Truedata/Experiments/Selections_MAR_m = {len(Nystrom_indexes_MAR)}.pdf')
    plt.close()
def selection_true_data_plot_MER(x):
    fig,ax  = plt.subplots(2,3,figsize = (9,7))
    ax = ax.flatten()
    ax[5].remove()
    for i in range(5):
        ax[i].scatter(x[:,i],x[:,5],s = 5)
        ax[i].scatter(x[Nystrom_indexes_MER,i],x[Nystrom_indexes_MER,5],label = f'Nyström points, m = {len(Nystrom_indexes_MER)}',color = 'white',edgecolors = 'black',s=80,linewidths = 2.5,marker='X')
        ax[i].set_xlim(0,1)
        ax[i].set_ylim(0,1)
        ax[i].set_xlabel(f"$x_{i+1}$",fontsize = 14)
        ax[0].set_ylabel(f"$y$",fontsize = 14)
        ax[3].set_ylabel(f"$y$",fontsize = 14)

    handles, legend_labels = ax[0].get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles, legend_labels,loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4)
    #fig.suptitle('Real data')
    plt.savefig(f'plots/Truedata/Experiments/Selections_MER_m = {len(Nystrom_indexes_MER)}.pdf')
    plt.close()
def CVplotSingle(reg_paramter_cvplot):
    xx = np.empty((n_eval,d))
    xx[:,0] = np.linspace(x_min-0.5,x_max+0.5,n_eval)   #evaluation/test points
    
    alpha = fit(K_nn,reg_paramter_cvplot,labels)
    predictions = predict(X,xx,alpha)
        
    #number_of_axes = len(reg_range)

    fig,ax = plt.subplots(1,1,figsize = (8,8),squeeze=False)
    
    ax = ax[0,0]

    #data & Nyström points

    # ax[i].scatter(X,labels,color = colors[0],label = f'Data, n = {n}',alpha = 0.8, marker = 'x')

    # ax[i].plot(xx,model(xx),label='True model',color = colors[1])
    # ax[i].plot(xx,CVPredictions[i],label='Fitted model',color = colors[2], linestyle = '--')
    ax.scatter(X,labels,color = 'orange',label = f'Data, n = {n}',alpha = 0.8, marker = 'x')

    ax.plot(xx,model(xx),label='True model',color = 'black')
    ax.plot(xx,predictions,label='Fitted model', color = 'red',linestyle = '--')
    
    ax.set_xlabel(f"$x$")
    ax.set_ylabel(f"$y$")
    ax.set_title(f'$\\lambda$ = {reg_paramter_cvplot:.0e}')
    #ax[i].legend()

    # fig.text(0.5, 0.91, f"$\\sigma^2$ = {sigma**2}",
    #     ha='center',
    #     bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
    handles, legend_labels = ax.get_legend_handles_labels()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.legend(handles, legend_labels,loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)
    #plt.tight_layout()

    plt.savefig(f"plots/CVPlot/{model.__name__}_r = {r}_n = {n}_noise = {sigma**2:.2f}_reg_param = {reg_paramter_cvplot}.pdf")
def In_Sample_SMAPE_plot():
    fig,ax = plt.subplots(figsize = (8,8))
    for i,strategy in enumerate(sampling_strategies):
        ax.plot(mrange,approx_in_sample_smape[strategy],label = strategy, marker = markers[i])
    #ax.axhline(empirical_risk,label = 'Full model',color= 'black')
    #ax.axvline(degrees_of_freedom,color = 'black',linestyle = '--',label = 'df')
    ax.set_xscale('log')
    
    ax.set_xticks(mrange)
    ax.set_xticklabels(mrange.astype(str))
    
    
    #ax.set_yscale('log')
    ax.set_title(f'In sample SMAPE, n = {n}')
    ax.set_xlabel('Number of Nyström points')
    ax.set_ylabel('SMAPE')
    ax.legend()
    if model is None:
        # fig.text(0.5, 0.97, f"Model: True data, r = {r}, $\\lambda$ = {reg_param:.2e}",
        #         ha='center',
        #         bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
        plt.savefig(f"plots/Truedata/SMAPE/in_r = {r}.pdf")
    else:
        # fig.text(0.5, 0.97, f"Model: {model.__name__}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
        #         ha='center',
        #         bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
        if d == 1:
            plt.savefig(f"plots/1D/SMAPE/in_{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
        else:
            plt.savefig(f"plots/2D/SMAPE/in_{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
def Out_of_Sample_SMAPE_plot():
    fig,ax = plt.subplots(figsize = (8,8))
    for i,strategy in enumerate(sampling_strategies):
        ax.plot(mrange,approx_out_of_sample_smape[strategy],label = strategy, marker = markers[i])
    #ax.axhline(empirical_risk,label = 'Full model',color= 'black')
    #ax.axvline(degrees_of_freedom,color = 'black',linestyle = '--',label = 'df')
    ax.set_xscale('log')
    
    ax.set_xticks(mrange)
    ax.set_xticklabels(mrange.astype(str))
    
    
    #ax.set_yscale('log')
    ax.set_title(f'Out-of sample SMAPE, n = {n_test}')
    ax.set_xlabel('Number of Nyström points')
    ax.set_ylabel('SMAPE')
    ax.legend()
    if model is None:
        # fig.text(0.5, 0.97, f"Model: True data, r = {r}, $\\lambda$ = {reg_param:.2e}",
        #         ha='center',
        #         bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
        plt.savefig(f"plots/Truedata/SMAPE/out_r = {r}.pdf")
    else:
        # fig.text(0.5, 0.97, f"Model: {model.__name__}, r = {r}, $\\lambda$ = {reg_param:.2e}, $\\sigma^2$ = {sigma**2}",
        #         ha='center',
        #         bbox=dict(facecolor='white', boxstyle='round', edgecolor='black'))
        if d == 1:
            plt.savefig(f"plots/1D/SMAPE/out_{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")
        else:
            plt.savefig(f"plots/2D/SMAPE/out_{model.__name__}_r = {r}_n = {n}_lambda = {reg_param:.2e}_noise = {sigma**2}.pdf")

#INITIALIZATION AND DEFINITION OF GENERAL STRUCTURES, VARIABLES AND PARAMETERS 
real_simulation = True         #True | False
single_simulation = False      #True | False
Exploraiton_plots = False               #True | False
explanation_behavior = False
rng = np.random.default_rng(19) #18 for good exploration plotting, 14,15 good candidate for criteria, 19 is perfect


#regression
reg_param = np.nan                   #regularization parameter
reg_param_range = np.logspace(-12,-1,20) 

#sampling                           #CHANGE HERE THE SAMPLING STRATEGY
sampling_strategy = 'MER_no_regularisation'       #'Uniform' | 'RLS' | 'exact_best_k_leverage_score' | 'RPCholesky' | 'MER' | ','max_abs_argmax''
sampling_strategies = ['RPCholesky','MER_batch','MS','MAR','MER','MER_no_regularisation']#,'max_abs_argmax']#,'Uniform','RLS']#,'MER_null_reg_batch']#,','max_abs_argmax'']#'exact_best_k_leverage_score', ||,'MER_vec']
m = 30                          #number of subsamples in Nystrom approx for one shot simulation
mrange = np.logspace(0,6,7,True,2,'int') #n//10      #number of subsamples in Nystrom approx for iterative simulations
rank_k = 30                         #rank for the best_k approximation in best_k leverage scores = 60
batch_size = 100                     #at each iteration, the amount of points used to look for the next Nystrom in MER

#performances
approx_empirical_risk = {}              #dictionary for empirical risk attending different selection strategies
approx_test_risk = {}                   #dictionary for test risk attending different selection strategies
approx_in_sample_smape = {}
approx_out_of_sample_smape = {}
for strategy in sampling_strategies:    #constructor of such dictionaries
    approx_empirical_risk[strategy] = []
    approx_test_risk[strategy] = []
    approx_in_sample_smape[strategy] = []
    approx_out_of_sample_smape[strategy] = []
randomization_repetitions = 10           #number of repetitions of simulations for each Nystrom selection strategy

#plots
n_eval = 200                        #number of evaluation points, for plotting models
# colors = ['#ff7f00', '#377eb8', '#e41a1c', '#4daf4a',
#                   '#f781bf', '#a65628', '#984ea3',
#                   '#999999','#dede00']
#colors = ['#006BA4', '#FF800E', '#595959', '#ABABAB', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
markers = ['o','v','s','P','*','h','D','d']#,'','','']
#DATA GENERATION
if real_simulation:
    kernel = gaussian_kernel
    r = 0.1                                    #gaussian bandwidth
    data = np.loadtxt('dataset',delimiter=',')
    tot_data = data.shape[0]
    n = int(tot_data*0.1)#8000
    d = 5
    n_test = tot_data-n
    model = None

    training_indexes = np.random.choice(list(range(tot_data)), size=n, p=1/tot_data * np.ones(tot_data),replace=False)
    test_indexes = np.setdiff1d(list(range(tot_data)),training_indexes)
    X = data[training_indexes,:5]
    labels = data[training_indexes,6]
    X_test = data[test_indexes,:5]
    labels_test = data[test_indexes,6]
    for i in range(5):
        print(np.max(X[:,i]),np.min(X[:,i]))
    print(np.max(labels),np.min(labels))
    marginal_plot(np.concatenate([X, labels[:, None]], axis=1))
    joint_plot(np.concatenate([X, labels[:, None]], axis=1))
    
else:
    kernel = gaussian_kernel
    r = 1                                #gaussian bandwidth
    r_random = 0                         #gaussian bandwidth for random RKHS functions, ot used yet
    #dataset
    n = 200                               #dataset size
    n_test = 10*n                       #testset size
    d = 1                               #data dimensionality
    sigma = np.sqrt(0.8)                         #noise level in data measurement
    p = 100                              #number of samples in the random function model case

    x_min = -20
    x_max = 20
    y_min = -6
    y_max = 6
    
    #1D case
    if d == 1:                                  #CHANGE THE MODEL HERE
        model = random_function_generator(p)                        #CHANGE THE MODEL HERE      
        X,labels = data_1D(number_of_data = n)
        X_test,labels_test = data_1D(number_of_data = n_test)

    #2D case
    if d == 2:                                  #CHANGE THE MODEL HERE
        model = random_function_generator_2D(p)                         #CHANGE THE MODEL HERE
        X,labels = data_2D(number_of_data = n)
        X_test,labels_test = data_2D(number_of_data = n_test)
        #labels[:25] += 20

    #kD case
    if d > 2:                                    #CHANGE THE MODEL HERE
        model = random_function_generator(p)                        #CHANGE THE MODEL HERE
        X,labels = data_kD(number_of_data=n)
        X_test,labels_test = data_kD(number_of_data = n_test)
        print(np.max(labels),np.min(labels),np.max(labels_test),np.min(labels_test))

#FULL MODEL

#kernel matrix
K_nn = kernel(X,X)                                 #full kernel matrix

#eigenvalue decomposition of the kernel matrix, useful both for Sampling Scores and for Generalized cross validation
Lambda,Q = la.eigh(K_nn)
Lambda = np.flip(Lambda)                        #Lambda is a vector containing the eigenvalues in descending order
Q = np.flip(Q,axis = 1)                         #Q contains the corresponding eigenvectors by column

#Cross validation (on the full model) for determining the regularization parameter lambda
GCV_val = GCV(reg_param_range)
reg_param = reg_param_range[np.argmin(GCV_val)]

#degrees of freedom fo full model
degrees_of_freedom = np.sum(Lambda/(Lambda+n*reg_param))    

#fit, predict & performance
alpha = fit(K_nn,reg_param,labels)
empirical_risk = risk(predict(X,X,alpha),labels)
test_risk = risk(predict(X,X_test,alpha),labels_test)

if Exploraiton_plots == True:
    Signal_plot()
    typical_function_in_RKHS_plot()
    bad_approx_vs_good_approx_plot(alpha)
    reg_range = [0.000001,0.01,0.1,1]
    for i in reg_range:
        CVplotSingle(i)

#NYSTROM APPROX
scores = build_sampling_distributions() #computes the scores for Uniform, best_k and ridge leverage scores. It utilizes the eigen decomposition of K
                                        #In the following simulations, what changes is just the number of sampled points from this distributions, not the distributions themselves    

#single strategy and single m value
if single_simulation:
    Nystrom_indexes = sampling(sampling_strategy,m)
    #check_criteria_correctness(Nystrom_indexes)
    #cProfile.run('sampling(sampling_strategy,m)')

    K_nm = kernel(X,X[Nystrom_indexes])   #Nystrom approx
    alpha_tilde = fit(K_nm,reg_param,labels,Nystrom=True,Nystrom_indexes = Nystrom_indexes)

    approx_empirical_risk = risk(predict(X[Nystrom_indexes],X,alpha_tilde),labels)
    approx_test_risk = risk(predict(X[Nystrom_indexes],X_test,alpha_tilde),labels_test)
    
    print(f"The empirical risk is {empirical_risk}")
    print(f"The empirical risk using approximated model is {approx_empirical_risk}")
    abs_err_emp = np.abs(approx_empirical_risk-empirical_risk)
    rel_err_emp = abs_err_emp/np.abs(empirical_risk)
    print(f"The absolute empirical risk error is {abs_err_emp}")
    print(f"The relative empirical risk error is {rel_err_emp}")
    print("\n")
    print(f"The test risk is {test_risk}")
    print(f"The test risk using approximated model is {approx_test_risk}")
    abs_err_test = np.abs(approx_test_risk-test_risk)
    rel_err_test = abs_err_test/np.abs(test_risk)
    print(f"The absolute test risk error is {abs_err_test}")
    print(f"The relative test risk error is {rel_err_test}")
    plot_1D(Nystrom_indexes,alpha_tilde,strategy)
    #1D
    # if d == 1:
    #     plot_1D(Nystrom_indexes=Nystrom_indexes,alpha_tilde = alpha_tilde,strategy = sampling_strategy)

    # #2D
    # if d == 2:
    #     plot_2D(Nystrom_indexes=Nystrom_indexes,alpha_tilde = alpha_tilde,strategy = sampling_strategy)
    #     frames = []
    #     for i in range(m):
    #         filename = f"frames/frame_{i:03d}.pdf"
    #         frames.append(imageio.imread(filename))

    #     imageio.mimsave(f"criteria_{sampling_strategy}_r = {r}_noise = {sigma**2}.gif", frames, fps=0.3)  # 5 frame al secondo

else:
    #all strategies for multiple values of m, randomization_repetitions times
    for strategy in sampling_strategies:
        strategy_performance(strategy)

    emp_risk_plot()
    test_risk_plot()
    
    #In_Sample_SMAPE_plot()
    #Out_of_Sample_SMAPE_plot()

#PLOTTING RESULTS

#GCV_plot()

#--------------------------------------------------------------------------------------------------------------------------------------------------

#play with lambda and noise level to make the method fail --> we expect not being robust to noise level
# plot the criteria (2D) to check the presence of local behaviour 
# 
# fatti una cultura on sequential least squares 

#------------------------------------------------------------------------------------------------------------------------------------------
# 3) use cholesky_solve whenever is possible
#------------------------------------------------------------------------------------------------------------------------------------------


# n  = 1000

# m = 1 --> 0.915       
# m = 2 --> 1.544       
# m = 4 --> 2.222       
# m = 8 --> 4.956       
# m = 16 --> 7.674      
# m = 32 --> 17.327     
# m = 64 --> 34.050     
# m = 128 --> 68.701    
# m = 256 --> 136.002   

# m = 30

# n = 100 --> 0.348
# n = 200 --> 0.717
# n = 400 --> 2.823
# n = 800 --> 9.378
# n = 1600 --> 35.046
# n = 2000 --> 53.249
# n = 2200 --> 71.079

#------------------------------------------------------------------------------------------------------------------------------------------
# TODO
# noise level = really small 0.01 really large 5
# bandwidth = really small 0.01 really large 10  (half size of domain)

# nei emp e test plot aumento mrange finche non raggiungi le full model performances

# all axes clearly labeled
# color scheme consistent
# label of curve on the right (intstead)
# use different marker and dashed lines for different strategies 
# Test against random RKHS Functions
# investigate what makes differ my strat and residual strat --> Outliers? in the real dataset

# generate training and test data from guassian (or mixture) distribution while keeping the Uniform for the random RKHS anchor points

# DONE
# Keep track of the trace of the hat matrix as a guideline for m
# remove max_abs_argmax beacuse its stupid
# use unicode for symbols in textboxes over figures
# salvare in pdf e non in png tutto
# pallini bianchi al posto che neri, magari con bordo nero
# Fare e mandare index
# implement random RKHS Functions
# plot marginal distributions of label and every regressor
# not introduce noise in the test labels
# marginal scatter regressors againts y

# implement deterministic and sampled strategies (?)

#------------------------------------------------------------------------------------------------------------------------------------------
#OLD FUNCTIONS VERSIONS REPO

# criteria_vals = []
# not_yet_tried_indexes = np.setdiff1d(np.arange(n), Nystrom_indexes).tolist()
# for i in not_yet_tried_indexes:
#     criteria_vals.append(criteria(X,X[Nystrom_indexes],X[i]))
# fig,ax = plt.subplots(figsize = (8,8))
# ax.plot(not_yet_tried_indexes,criteria_vals,marker = 'o')
# plt.show()

#def criteria_plot(d):
#    if d == 2:
#    else:


# def plot_2D(Nystrom_indexes,alpha_tilde,l = None,criteria_vals = None):

#     xx = np.empty((n_eval,d))
#     xx[:,0] = np.linspace(x_min, x_max, n_eval)
#     xx[:,1] = np.linspace(y_min, y_max, n_eval)
#     XX, YY = np.meshgrid(xx[:,0], xx[:,1])
    
#     XXX = np.array([XX.flatten(),YY.flatten()]).T #array version of meshgrid, vectorized for evaluation purposes
    
#     evaluations = predict(X,XXX,alpha)
#     approx_evaluations = predict(X[Nystrom_indexes],XXX,alpha_tilde)

#     evaluations = evaluations.reshape(n_eval,n_eval)
#     approx_evaluations = approx_evaluations.reshape(n_eval,n_eval)

#     fig,ax = plt.subplots(3,2,figsize = (20,20),constrained_layout=True)
#     ax = ax.flatten()

#     # for i in range(6):
#     #     ax[i].set_aspect('equal')            #'equal' | 'auto'
    

#     #data & Nyström points
    
#     ax[0].scatter(X[:,0],X[:,1],color = 'orange', label = 'data',alpha = 0.8, marker = 'x')
#     ax[0].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1], label = f"Nyström points, m = {len(Nystrom_indexes)}",color = 'black')
#     ax[1].scatter(X[:,0],X[:,1],color = 'orange', label = 'data',alpha = 0.8, marker = 'x')
#     #ax[i].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1], label = f"Nyström points, m = {len(Nystrom_indexes)}",color = 'black')
#     #ax[i].scatter(X[:,0],X[:,1],color = 'orange', label = 'data',alpha = 0.8, marker = 'x')
#     ax[2].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1], label = f"Nyström points, m = {len(Nystrom_indexes)}",color = 'black')


#     #true model
#     img = ax[0].imshow(model(XX, YY), extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
#     #img = ax[0].imshow(np.reshape(model(XXX),(200,200)), extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis') #works good for radial_sin_kD
#     fig.colorbar(img, ax = ax[0], label='True model values')
#     ax[0].set_title('True model')

#     img = ax[1].imshow(evaluations, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
#     fig.colorbar(img, ax = ax[1], label='Fitted values')
#     ax[1].set_title('Estimated Full model')

#     img = ax[2].imshow(approx_evaluations, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
#     fig.colorbar(img, ax = ax[2], label='Approximated Fitted values')
#     ax[2].set_title('Estimated Nystrom approximation model')

#     #img = ax[3].imshow(evaluations - approx_evaluations, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
#     #fig.colorbar(img, ax = ax[3])
#     #ax[3].set_title('Full & Nystrom prediction difference')

#     if strategy == 'MER':   
#         leftover_indexes = np.setdiff1d(list(range(n)), Nystrom_indexes).tolist()
        
#         ax[3].scatter(X[Nystrom_indexes,0],X[Nystrom_indexes,1],color = 'orange', label = 'Nystrom data',alpha = 0.8, marker = 'x')
#         #ax.scatter(X[selected_indexes,0],X[selected_indexes,1], label = f"Nyström points, m = {len(selected_indexes)}",color = 'black')
        
#         sc = ax[3].scatter(X[leftover_indexes,0],X[leftover_indexes,1],c = criteria_vals, label = 'Not selected data',cmap = 'viridis')
#         fig.colorbar(sc, label = 'score', ax = ax[3])

#         ax[3].set_title(f'Data Score of the {n-len(Nystrom_indexes)} reamining points,{len(Nystrom_indexes)} where alredy selected')
#         ax[3].legend()
#         #plt.savefig('Criteria')
        
#     elif not(sampling_strategy == 'RPCholesky'):# or sampling_strategy == 'MER'):# or sampling_strategy == 'max_abs'):
#         sc = ax[3].scatter(X[:,0],X[:,1],c = scores[sampling_strategy], label = 'data',cmap = 'viridis')
#         ax[3].set_title('Data Score')
#         fig.colorbar(sc, label = 'score', ax = ax[4])

#         ax[4].scatter(np.arange(n),scores[sampling_strategy])
#         ax[4].set_title(sampling_strategy)
    
#     else:
#         ax[3].remove()
#         ax[4].remove()

#     for i in range(6):
#         ax[i].legend()
    
    
#     #plt.show()
#     plt.savefig('2D_plot')

# def criteria_first_version(data,Nystrom_data,new_point):
#     K_nm = kernel(data,Nystrom_data)
#     A = K_nm@la.inv(K_nm.T@K_nm+n*reg_param*kernel(Nystrom_data,Nystrom_data))@K_nm.T
#     b = b_func(data,Nystrom_data,new_point)
#     c = c_func(Nystrom_data,new_point)
#     denominator = (b.T@(np.eye(n)-A)@b+n*reg_param*c)[0,0]        #[0,0] to extract the element, so that the denominator is a float (scalar)
#     return 1/denominator * ( (la.norm( (A-np.eye(n))@b@b.T@(A-np.eye(n))@labels ))**2 /denominator +2*(labels.T@(A-np.eye(n))@(A-np.eye(n))@b@b.T@(A-np.eye(n))@labels))

# def MER_non_vec(m):
#     selected_indexes = [] #initialized at -1 for convenience
#     leftover_indexes = list(range(n))
#     K_nm = np.zeros((n,m))

#     #selection of the first index | can be optimized vectorizing the fit & predict step and then usign np.argmin
#     best_risk = np.inf
#     best_index = -1
#     criteria_vals = []
#     for i in range(n):
#         alpha_tilde = fit(kernel(X,X[i]),reg_param,labels,Nystrom = True,Nystrom_indexes=[i])
#         approx_preds = predict(X[i],X,alpha_tilde)
#         approx_empirical_risk = risk(approx_preds,labels)
#         criteria_vals.append(approx_empirical_risk)
#         if approx_empirical_risk < best_risk:
#             best_index = i
#             best_risk = approx_empirical_risk
#     selected_indexes.append(best_index)
#     leftover_indexes.remove(best_index)
#     K_nm[:,0] = np.reshape(kernel(X,X[selected_indexyes[-1]]),(n,))
#     #to plot/create a gif
#     alpha_tilde = fit(K_nm[:,:1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
#     criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = 0)

#     #selection of the rest m-1 indexes
#     for i in range(1,m):
#         best_criteria = np.inf
#         best_index = -1
#         L_mm = la.cholesky(K_nm[selected_indexes,:i]+100*np.spacing(1)*np.eye(len(selected_indexes)))
#         A = K_nm[:,:i]@la.inv(K_nm[:,:i].T@K_nm[:,:i]+n*reg_param*K_nm[selected_indexes,:i])@K_nm[:,:i].T
#         criteria_vals = []
#         for index in leftover_indexes:
#             criteria_val = criteria_non_vec(A,K_nm[:,:i],L_mm,selected_indexes,index)
#             criteria_vals.append(criteria_val)
#             if criteria_val < best_criteria:
#                 best_index = index
#                 best_criteria = criteria_val
#         # if i == m-1:
#         #     criteria_plot(i,criteria_vals,leftover_indexes,selected_indexes)
#         selected_indexes.append(best_index)
#         leftover_indexes.remove(best_index)
#         K_nm[:,i] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
#         #to plot/create a gif
#         # alpha_tilde = fit(K_nm[:,:i+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
#         # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = i)
#     return selected_indexes

# def criteria_non_vec(A,K_nm,L_mm,Nystrom_indexes,new_index):                #I AM JUST TRANSPOSING, NOT CONJ_TRANSPOSING
#     '''
#     Computes the criteria given the alredy selected indexes and the new point(index).
#     Since the A factor doesn't depend on the new point, as well as K_nm and L_mm, they are precomputed.
#     L_mm is the cholesky factor of the matrix K_mm
#     '''
#     K_nnew = kernel(X,X[new_index])
#     b = K_nnew - K_nm@la.solve(L_mm.T,la.solve(L_mm,K_nnew[Nystrom_indexes,:]))
#     c = K_nnew[new_index,:] - K_nnew[Nystrom_indexes,:].T@la.solve(L_mm.T,la.solve(L_mm,K_nnew[Nystrom_indexes,:]))
#     res_y_neg = (A-np.eye(n))@labels                #(A-I)y
#     res_b_neg = (A-np.eye(n))@b                     #(A-I)b
#     some_num = res_b_neg@b.T@res_y_neg              #(A-I)bb^t(A-I)y
#     some_den = -b.T@res_b_neg
#     #denominator = (b.T@(np.eye(n)-A)@b+n*reg_param*c)[0,0] #[0,0] to extract the element, so that the denominator is a float (scalar)
#     denominator = (some_den+n*reg_param*c)[0,0]
#     #num_1 = (la.norm( (A-np.eye(n))@b@b.T@(A-np.eye(n))@labels ))**2
#     num_1 = (la.norm( some_num ))**2
#     num_2 = (res_y_neg.T@some_num)
#     return 1/denominator * ( num_1 /denominator +2*num_2)

# 0.341 2.254 17.263

# def MER(m):
#     selected_indexes = [] #initialized at -1 for convenience
#     leftover_indexes = list(range(n))
#     K_nm = np.zeros((n,m))

#     #selection of the first index | can be optimized vectorizing the fit & predict step and then usign np.argmin
#     best_risk = np.inf
#     best_index = -1
#     criteria_vals = []
#     for i in range(n):
#         alpha_tilde = fit(kernel(X,X[i]),reg_param,labels,Nystrom = True,Nystrom_indexes=[i])
#         approx_preds = predict(X[i],X,alpha_tilde)
#         approx_empirical_risk = risk(approx_preds,labels)
#         criteria_vals.append(approx_empirical_risk)
#         if approx_empirical_risk < best_risk:
#             best_index = i
#             best_risk = approx_empirical_risk
#     selected_indexes.append(best_index)
#     leftover_indexes.remove(best_index)
#     K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))
#     #to plot/create a gif
#     # alpha_tilde = fit(K_nm[:,:1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
#     # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = 0)

#     #selection of the rest m-1 indexes
#     for i in range(1,m):
#         L_mm = la.cholesky(K_nm[selected_indexes,:i]+100*np.spacing(1)*np.eye(len(selected_indexes)))
#         A = K_nm[:,:i]@la.inv(K_nm[:,:i].T@K_nm[:,:i]+n*reg_param*K_nm[selected_indexes,:i])@K_nm[:,:i].T
#         criteria_vals = criteria(A,K_nm[:,:i],L_mm,selected_indexes,leftover_indexes)
#         best_index = leftover_indexes[np.argmin(criteria_vals)]
#         # if i == m-1:
#         #     criteria_plot(i,criteria_vals,leftover_indexes,selected_indexes)
#         selected_indexes.append(best_index)
#         leftover_indexes.remove(best_index)
#         K_nm[:,i] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
#         # #to plot/create a gif
#         # alpha_tilde = fit(K_nm[:,:i+1],reg_param,labels,Nystrom = True,Nystrom_indexes=selected_indexes)
#         # criteria_and_plot_2D(selected_indexes,alpha_tilde,criteria_vals,frame_id = i)   
#     return selected_indexes
# def criteria(A,K_nm,L_mm,Nystrom_indexes,new_indexes):                #I AM JUST TRANSPOSING, NOT CONJ_TRANSPOSING
#     '''
#     Computes the criteria given the alredy selected indexes and all the remaining indexes.
#     Since the A factor doesn't depend on the new point, as well as K_nm and L_mm (chol factor of K_mm), they are precomputed.
#     '''
#     K_nnew = kernel(X,X[new_indexes])
#     b = K_nnew - K_nm@la.solve(L_mm.T,la.solve(L_mm,K_nnew[Nystrom_indexes,:]))
#     c = np.diag(b)
#     res_y_neg = (A-np.eye(n))@labels                #(A-I)y
#     res_b_neg = (A-np.eye(n))@b                     #(A-I)b
#     some_num = np.einsum('ik,jk->kij',res_b_neg,b)@res_y_neg #(A-I)bb^t(A-I)y
#     #some_den = -b.T@res_b_neg
#     some_den = np.einsum('ij,ji->i',-b.T,res_b_neg)
#     #denominator = np.diag((some_den+n*reg_param*c))
#     denominator = some_den+n*reg_param*c
#     num_1 = (la.norm(some_num,axis = 1))**2
#     num_2 = (some_num@res_y_neg)
#     return 1/denominator * ( num_1 /denominator +2*num_2)
# def check_criteria_correctness(Nystrom_indexes):
#     #predictions with m and m-1 Nyström points
#     K_nm = kernel(X,X[Nystrom_indexes])   #Nystrom approx
#     alpha_tilde_all = fit(K_nm,reg_param,labels,Nystrom=True,Nystrom_indexes = Nystrom_indexes)
#     alpha_tilde_all_but_last = fit(K_nm[:,:-1],reg_param,labels,Nystrom=True,Nystrom_indexes = Nystrom_indexes[:-1])
#     approx_empirical_risk_all = risk(predict(X[Nystrom_indexes],X,alpha_tilde_all),labels)
#     approx_empirical_risk_all_but_last = risk(predict(X[Nystrom_indexes[:-1]],X,alpha_tilde_all_but_last),labels)
    
#     #compute criteria using the m^th point as the new point
#     K_nm = kernel(X,X[Nystrom_indexes[:-1]])
#     L_mm = la.cholesky(K_nm[Nystrom_indexes[:-1],:])
#     A = K_nm@la.inv(K_nm.T@K_nm+n*reg_param*K_nm[Nystrom_indexes[:-1],:])@K_nm.T
#     print(f"The difference between eps(Z+z) and eps(Z) + criteria is {n*approx_empirical_risk_all-(n*approx_empirical_risk_all_but_last+criteria(A,K_nm,L_mm,Nystrom_indexes[:-1],Nystrom_indexes[-1]))}")
