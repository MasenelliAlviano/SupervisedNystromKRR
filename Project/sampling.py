#sampling
import numpy as np
from numpy import linalg as la
from scipy import linalg as spla
from utils_math import Nystrom_fit, predict, risk, cholesky_solve

#dictionary book
strategies = {}

#recorder
def register_strategy(name):
    def decorator(func):
        strategies[name] = func
        return func
    return decorator

@register_strategy('Uniform')
def uniform(X, m, rng, **kwargs):
    n = X.shape[0]
    indexes = rng.choice(n, m, replace = False) 
    return indexes

@register_strategy('RLS')
def RLS(Q,Lambda,reg_param, m, rng, **kwargs):
    n = Lambda.shape[0]
    new_Lambda = Lambda/(Lambda+n*reg_param)
    scores = np.einsum('ij,j,ij->i',Q,new_Lambda,Q)
    scores = scores/np.sum(scores)
    indexes = rng.choice(n, m, p = scores, replace = False)
    return indexes

@register_strategy('RPCholesky')
def RPCholesky(X, m, K_nn, kernel, rng, **kwargs):
    n = K_nn.shape[0]
    F = np.zeros(shape = (n,m))
    diagonal = np.array([kernel(X[i],X[i]).item() for i in range(n)]) #should be one good way of doing it, but since it does not work due to error in v = la.norm(X,axis=1)**2 when X is just one data
    #diagonal = np.diag(K_nn) #and since I have computed the full kernel matrix K, I will use this naive way
    Nystrom_indexes = []
    for i in range(m):
        if np.any(np.isnan(F)):
            print(f"The approximation seems good alredy, with {i} Nyström points")
            return np.reshape(np.array(Nystrom_indexes),(i,))
        
        sampled_pivot_index = rng.choice(n, 1, p=diagonal/np.sum(diagonal))
        #sampled_pivot_index = np.atleast_1d(np.argmax(d))
        g = np.array([kernel(X,X[sampled_pivot_index])])
        g = np.reshape(g,(n,1))
        g = g - F[:,:i]@(F[sampled_pivot_index,:i].T)
        g = np.reshape(g,(n,))
        F[:,i] = g/np.sqrt(g[sampled_pivot_index])
        diagonal = diagonal - F[:,i]**2
        diagonal = np.maximum(diagonal,0)
        Nystrom_indexes.append(sampled_pivot_index)
    return np.reshape(np.array(Nystrom_indexes),(m,))#,F

@register_strategy('MS')
def MS(labels, m, rng, **kwargs):
    n = labels.shape[0]
    abs_labels = np.absolute(labels.flatten())
    scores = abs_labels/np.sum(abs_labels)
    indexes = rng.choice(n, m, p =  scores, replace = False)
    return indexes

@register_strategy('MAR')
def MAR(X,labels, m, kernel, reg_param, **kwargs):
    n = labels.shape[0]
    
    selected_indexes = []
    leftover_indexes = list(range(n))
    K_nm = np.zeros((n,m))

    best_index = -1

    #selection of the first index 
    best_index = np.argmax(np.abs(labels))
    selected_indexes.append(best_index)
    leftover_indexes.remove(best_index)
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))

    #selection of the rest m-1 indexes
    for i in range(1,m):
        alpha_tilde = Nystrom_fit(K_nm[:,:i], K_nm[selected_indexes,:i], reg_param, labels)
        preds = predict(X[selected_indexes],X[leftover_indexes],alpha_tilde, kernel)
        res = labels[leftover_indexes] - preds
        best_index = leftover_indexes[np.argmax(np.abs(res))]
        selected_indexes.append(best_index)
        leftover_indexes.remove(best_index)
        K_nm[:,i] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
    return selected_indexes

#@register_strategy('MER')
def MER(X, labels, m, kernel, reg_param, **kwargs):
    n = labels.shape[0]

    selected_indexes = []
    leftover_indexes = list(range(n))
    
    K_nm = np.zeros((n,m))

    #selection of the first index | might be optimized vectorizing the fit & predict step and then usign np.argmin
    best_risk = np.inf
    best_index = -1
    criteria_vals = []
    for i in range(n):
        alpha_tilde = Nystrom_fit(kernel(X,X[i]),kernel(X[i],X[i]),reg_param,labels)
        approx_preds = predict(X[i],X,alpha_tilde,kernel)
        approx_empirical_risk = risk(approx_preds,labels)
        if approx_empirical_risk < best_risk:
            best_index = i
            best_risk = approx_empirical_risk
    selected_indexes.append(best_index)
    leftover_indexes.remove(best_index)
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))

    #selection of the rest m-1 indexes
    for l in range(1,m):            #K_nm[:,:l] is K_nl, K_nm[selected_indexes,:l] is K_ll
        num_stability_correction = 1000*np.spacing(1)*np.eye(l,l)
        L_mm = la.cholesky(K_nm[selected_indexes,:l]+num_stability_correction)       #choleksy of K_ll
        L_mm_tilde = la.cholesky(K_nm[:,:l].T@K_nm[:,:l]+n*reg_param*K_nm[selected_indexes,:l]+num_stability_correction)     #cholesky of K_nlK_ln + nlambdaK_ll
        criteria_vals = _criteria(K_nm[:,:l],L_mm_tilde,L_mm,selected_indexes,leftover_indexes,kernel, labels, reg_param, n, X)
        best_index = leftover_indexes[np.argmin(criteria_vals)]
        selected_indexes.append(best_index)
        leftover_indexes.remove(best_index)
        K_nm[:,l] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
    return selected_indexes

@register_strategy('MER batch')
def MER_batch(X, labels, m, kernel, reg_param, MER_batch_size, **kwargs):
    n = labels.shape[0]
    selected_indexes = []
    leftover_indexes = list(range(n))
    K_nm = np.zeros((n,m))

    #selection of the first index | can be optimized vectorizing the fit & predict step and then usign np.argmin
    best_risk = np.inf
    best_index = -1
    criteria_vals = []
    batch = np.random.choice(leftover_indexes,size = MER_batch_size,p = 1/len(leftover_indexes)* np.ones(len(leftover_indexes)),replace = False)
    for i in batch:
        alpha_tilde = Nystrom_fit(kernel(X,X[i]),kernel(X[i],X[i]),reg_param,labels)
        approx_preds = predict(X[i],X,alpha_tilde,kernel)
        approx_empirical_risk = risk(approx_preds,labels)
        if approx_empirical_risk < best_risk:
            best_index = i
            best_risk = approx_empirical_risk
    selected_indexes.append(best_index)
    leftover_indexes.remove(best_index)
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))

    #selection of the rest m-1 indexes
    for l in range(1,m):            #K_nm[:,:l] is K_nl, K_nm[selected_indexes,:l] is K_ll
        batch = np.random.choice(leftover_indexes,size = MER_batch_size,p = 1/len(leftover_indexes)* np.ones(len(leftover_indexes)),replace = False)
        num_stability_correction = 1000*np.spacing(1)*np.eye(l,l)
        L_mm = la.cholesky(K_nm[selected_indexes,:l]+num_stability_correction)       #choleksy of K_ll
        L_mm_tilde = la.cholesky(K_nm[:,:l].T@K_nm[:,:l]+n*reg_param*K_nm[selected_indexes,:l]+num_stability_correction)     #cholesky of K_nlK_ln + nlambdaK_ll
        criteria_vals = _criteria(K_nm[:,:l],L_mm_tilde,L_mm,selected_indexes,batch,kernel, labels, reg_param, n, X)
        best_index = batch[np.argmin(criteria_vals)]
        selected_indexes.append(best_index)
        leftover_indexes.remove(best_index)
        K_nm[:,l] = np.reshape(kernel(X,X[selected_indexes[-1]]), (n,))
    return selected_indexes

def _criteria(K_nm,L_mm_tilde,L_mm,Nystrom_indexes,new_indexes, kernel, labels, reg_param, n,X):
    '''
    Computes the criteria given the alredy selected indexes and all the remaining indexes.
    Since the A factor doesn't depend on the new point, as well as K_nm and L_mm (chol factor of K_mm), they are precomputed.
    '''
    
    K_nnew = kernel(X,X[new_indexes])
    b = K_nnew - K_nm@spla.cho_solve((L_mm,True),K_nnew[Nystrom_indexes,:])
    c = np.diag(b)
    res_y_neg = _A_time_x(K_nm,L_mm_tilde,labels)-labels
    res_b_neg = _A_time_x(K_nm,L_mm_tilde,b)-b          
    some_num = ((b.T @ res_y_neg)[None, :] * res_b_neg).T
    some_den = np.einsum('ij,ji->i',-b.T,res_b_neg)
    denominator = some_den+n*reg_param*c
    num_1 = (la.norm(some_num,axis = 1))**2
    num_2 = (some_num@res_y_neg)
    return 1/denominator * ( num_1 /denominator +2*num_2)

def _A_time_x(K_nm,L_mm_tilde,x):
    return K_nm@spla.cho_solve((L_mm_tilde,True),K_nm.T@x)

@register_strategy('Blended_MP')
def Blended_MP(X, labels, m, kernel, reg_param, initial_guess, kappa, eta, tau, **kwargs):
    n = labels.shape[0]

    all_indexes = list(range(n))

    selected_indexes = [initial_guess]
    leftover_indexes = list(range(n))
    leftover_indexes.remove(initial_guess)
    number_of_active_atoms = len(selected_indexes)

    K_nm = np.zeros((n,m))
    K_nm[:,0] = np.reshape(kernel(X,X[selected_indexes[-1]]),(n,))
    alpha = np.zeros((m,))

    alpha[0] = Nystrom_fit(K_nm[:,0],K_nm[0,0],reg_param,labels)

    inner_product = inner_prod(selected_indexes,all_indexes,kernel,reg_param,n,labels,K_nm[:,0],X,alpha[0])
    phi = -np.max(np.abs(inner_product))/tau
    
    for i in range(1,m):
    #while number_of_active_atoms != m:
        #print(f"La stima del gap vale{phi}")
        #print(f"Abbiamo selezionato {number_of_active_atoms}")

        inner_product = inner_prod(selected_indexes,
                                   selected_indexes,
                                   kernel,
                                   reg_param,
                                   n,
                                   labels,
                                   K_nm[:,:number_of_active_atoms],
                                   X,
                                   alpha[:number_of_active_atoms]
                                   )
        if -np.max(np.abs(inner_product)) <= phi/eta:
            beta = opt_cost_function_over_projected_grad(alpha[:number_of_active_atoms],
                                                         K_nm[:,:number_of_active_atoms],
                                                         labels,
                                                         n,
                                                         reg_param,
                                                         selected_indexes
                                                         )
            num_stability_correction = 1000*np.spacing(1)*np.eye(number_of_active_atoms,number_of_active_atoms)
            correction = cholesky_solve(K_nm[selected_indexes,:number_of_active_atoms]+num_stability_correction,weights_vector(alpha[:number_of_active_atoms],K_nm[:,:number_of_active_atoms],labels,n*reg_param,selected_indexes))
            alpha[:number_of_active_atoms] += beta*correction
        else:
            inner_product = inner_prod(selected_indexes,
                                       leftover_indexes,
                                       kernel,
                                       reg_param,
                                       n,
                                       labels,
                                       K_nm[:,:number_of_active_atoms],
                                       X,
                                       alpha[:number_of_active_atoms]
                                       )
            if -np.max(np.abs(inner_product)) <= phi/kappa:
                best_index = leftover_indexes[np.argmax(np.abs(inner_product))]
                K_nm[:,number_of_active_atoms] = np.reshape(kernel(X,X[best_index]),(n,))

                beta = opt_cost_function_over_new_atom(alpha[:number_of_active_atoms],K_nm[:,:number_of_active_atoms],labels,n,reg_param,selected_indexes,best_index,K_nm[:,number_of_active_atoms])
                alpha[number_of_active_atoms] = beta
                
                selected_indexes.append(best_index)
                leftover_indexes.remove(best_index)
                number_of_active_atoms += 1
            else:
                phi = phi/tau
    print(f"m vale {m} e ho selezionato {number_of_active_atoms} atomi")
    return selected_indexes



def inner_prod(selected_indexes, new_indexes, kernel, reg_param, n, labels, K_nm, X, alpha):
        """

        A function to ease the computation of expressions of the form <∇ε(f_m),v> for atom v in some set
        In formulas, I compute K_n(x).T @ (K_nm @ alpha - y) + n* reg_param * K_m(x).T @ alpha 
        where x identifies v
        should work with multiple vectors v --> already Vectorized 

        """
        w_total = np.atleast_1d(alpha)@np.atleast_2d(K_nm.T)
        w_total -= labels
        w_total[selected_indexes] += alpha*n*reg_param
        return w_total@kernel(X,X[new_indexes])

#a function to compute the expression K_mn @ (y-K_nm @ alpha) + reg_constant*K_mm @ alpha
def weights_vector(alpha,K_nm,labels,reg_constant,selected_indexes):
    weights = labels - K_nm@alpha
    term_1 = K_nm.T@weights
    term_2 = reg_constant*K_nm[selected_indexes,:]@alpha
    return term_1+term_2

def opt_cost_function_over_projected_grad(alpha,K_nm,labels,n,reg_param,selected_indexes):
    num_stability_correction = 1000*np.spacing(1)*np.eye(len(selected_indexes),len(selected_indexes))
    weights = weights_vector(alpha,K_nm,labels,n*reg_param,selected_indexes)

    weigths_with_negative_reg_constant = weights_vector(alpha,K_nm,labels,-n*reg_param,selected_indexes)
    K_mm_inverse_times_weights = cholesky_solve(K_nm[selected_indexes,:]+num_stability_correction,weights)
    num = weigths_with_negative_reg_constant@K_mm_inverse_times_weights

    squared_norm_proj_grad = weights@K_mm_inverse_times_weights
    squared_norm_proj_grad_evaluated = np.sum((K_nm@K_mm_inverse_times_weights)**2) 
    den = squared_norm_proj_grad_evaluated + n*reg_param*squared_norm_proj_grad 
    return num/den

def opt_cost_function_over_new_atom(alpha,K_nm,labels,n,reg_param,selected_indexes,new_index,K_nnew):
    weights = labels - K_nm@alpha
    weights[selected_indexes] -= n*reg_param*alpha
    num = weights@K_nnew 
    den = np.sum(K_nnew**2) + n*reg_param*K_nnew[new_index]
    return num/den





#num_stability_correction = 1000*np.spacing(1)*np.eye(l,l)
#L_mm = la.cholesky(K_nm[selected_indexes,:l]+num_stability_correction)       #choleksy of K_ll
#L_mm_tilde = la.cholesky(K_nm[:,:l].T@K_nm[:,:l]+n*reg_param*K_nm[selected_indexes,:l]+1000*np.spacing(1)*np.eye(len(selected_indexes)))     #cholesky of K_nlK_ln + nlambdaK_ll
