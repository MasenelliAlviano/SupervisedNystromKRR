import numpy as np
import pandas as pd
from numpy import linalg as la
from scipy import linalg as spla
from functools import partial
import os
from pathlib import Path

import nystrom as ny
import sampling         #to populate the register of strategies
import plotting
import synthetic_data
import kernels as krnls
import utils_math

base_dir = Path(__file__).parent
plots_dir = base_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

def experiment_risk_vs_m():
    results_log = []
    for m in mrange:
        for strategy in strategies:
            for rep in range(randomization_repetitions):

                #set up and fit
                model = ny.NystromKRR(kernel=kernel, reg_param=reg_param, rng=rng)
                model.fit(X_train=X_train, 
                        labels=labels_train, 
                        m=m,
                        strategy=strategy,
                        Q=Q,
                        Lambda=Lambda,
                        K_nn=K_nn,
                        MER_batch_size = MER_batch_size,
                        initial_guess = initial_guess,
                        kappa = kappa,
                        tau = tau,
                        eta = eta
                        )
                
                #compute risks
                train_risk = model.risk(model.predict(X_train),labels_train)
                test_risk = model.risk(model.predict(X_test),labels_test)
                
                #saving results
                record = {
                    'strategy': strategy,
                    'tot_iterations': m,
                    'number_of_nystrom_points': model.alpha.size,
                    'repetition': rep,
                    'train_risk': train_risk,
                    'test_risk': test_risk
                }
                results_log.append(record)

    return pd.DataFrame(results_log)
def single_experiment_1D(strat = "Blended_MP"):
    n_eval = 1000
    X_grid = np.linspace(x_min-0.5, x_max+0.5, n_eval).reshape(-1, 1)

    true_predictions_grid = ground_truth(X_grid)
    fullKRR_predictions_grid = utils_math.predict(X_train = X_train,eval_point=X_grid, alpha=alpha, kernel=kernel)

    m = 30
    strat = strat # o 'Uniform', 'MER'

    # 3. Fit del Modello Nystrom
    model = ny.NystromKRR(kernel=kernel, reg_param=reg_param, rng=rng)
    model.fit(X_train = X_train, labels = labels_train, m=m, strategy=strat,initial_guess=initial_guess, kappa = kappa, eta = eta,tau = tau) # Passa eventuali kwargs se servono

    # 4. Predizione Nystrom sulla griglia
    Nystrom_predictions_grid = model.predict(X_grid)

    # 5. CHIAMATA AL PLOT
    # Prepariamo i dati dei punti selezionati
    Nystrom_idxs = model.Nystrom_indexes
    X_nystrom = X_train[Nystrom_idxs]
    y_nystrom = labels_train[Nystrom_idxs]

    # Costruzione nome file dinamico
    filename = f"{plots_dir}/1D/Experiments/Fit_{strat}_m={m}_r={r}.pdf"

    plotting.plot_1D_fullKRR_vs_Nystrom(
        X_train=X_train, 
        y_train=labels_train,
        X_nystrom=X_nystrom,
        y_nystrom=y_nystrom,
        X_grid=X_grid,
        y_true_grid=true_predictions_grid,
        y_full_grid=fullKRR_predictions_grid,
        y_approx_grid=Nystrom_predictions_grid,
        strategy_name=strat,
        scores=None, # Passa None se non è RLS
        save_path=filename
    )

r = 0.1
kernel = partial(krnls.gaussian_kernel,r = r)#_scipy_version, r = r)
rng = np.random.default_rng(1)


#dataset
n = 300                             #dataset size
n_test = 10*n                       #testset size
d = 1                               #data dimensionality
sigma = np.sqrt(0.2)                         #noise level in data measurement
p = 50                              #number of samples in the random function model case

x_min = -np.pi
x_max = np.pi
y_min = -6
y_max = 6
ground_truth = synthetic_data.random_function_generator(kernel=kernel, p=p, rng=rng, d = d, x_min=x_min,x_max=x_max) #CHANGE THE MODEL HERE
#ground_truth = synthetic_data.sin
#1D case
if d == 1:      
    X_train,labels_train = synthetic_data.data_1D(number_of_data = n, d=d,model=ground_truth,sigma=sigma, rng=rng,x_min=x_min,x_max=x_max)
    X_test,labels_test = synthetic_data.data_1D(number_of_data = n_test,d=d,model=ground_truth,sigma=0, rng=rng,x_min = x_min ,x_max=x_max)

#2D case
if d == 2:                                        
    X_train,labels_train = synthetic_data.data_2D(number_of_data = n, d=d,model=ground_truth,sigma=sigma, rng=rng,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max)
    X_test,labels_test = synthetic_data.data_2D(number_of_data = n_test,d=d,model=ground_truth,sigma=0, rng=rng,x_min = x_min ,x_max=x_max,y_min=y_min,y_max=y_max)

#kD case
if d > 2:                                         
    X_train,labels_train = synthetic_data.data_kD(number_of_data = n, d=d,model=ground_truth,sigma=sigma, rng=rng,x_min=x_min,x_max=x_max)
    X_test,labels_test = synthetic_data.data_kD(number_of_data = n_test,d=d,model=ground_truth,sigma=0, rng=rng,x_min = x_min ,x_max=x_max)


#FULL MODEL

#kernel matrix
K_nn = kernel(X_train,X_train)                                 #full kernel matrix

#eigenvalue decomposition of the kernel matrix, useful both for Sampling Scores and for Generalized cross validation
Lambda,Q = la.eigh(K_nn)
Lambda = np.flip(Lambda)                        #Lambda is a vector containing the eigenvalues in descending order
Q = np.flip(Q,axis = 1)                         #Q contains the corresponding eigenvectors by column

#Cross validation (on the full model) for determining the regularization parameter lambda
reg_param = np.nan
reg_param_range = np.logspace(-12,-1,20)
reg_param = utils_math.GCV(reg_param_range, Q=Q, Lambda = Lambda, K_nn=K_nn, labels=labels_train)

#degrees of freedom fo full model
degrees_of_freedom = np.sum(Lambda/(Lambda+n*reg_param))    

#fit, predict & performance
alpha = utils_math.full_fit(K_nn, reg_param, labels=labels_train)
train_risk_full_KRR = utils_math.risk(utils_math.predict(X_train = X_train,eval_point=X_train, alpha=alpha, kernel=kernel), labels_train)
test_risk_full_KRR = utils_math.risk(utils_math.predict(X_train = X_train, eval_point=X_test, alpha=alpha, kernel=kernel), labels_test)

print(f"Empirical risk of the full model is {train_risk_full_KRR}")
print(f"Test risk of the full model is {test_risk_full_KRR}")

#NYSTROM MODELS & SIMULATIONS

#MAIN EXPERIMENT
#general parameters
mrange = np.logspace(0,6,7,True,2,'int')
strategies = sampling.strategies.keys()      #extracting the actual functions

#additional variables for specific strategies
MER_batch_size = 100                #in case of MER_batch
#initial_guess = rng.integers(n)     
initial_guess = np.argmax(np.abs(labels_train))
kappa = 2
tau = 2
eta = 5

#simulation variables & parameters
randomization_repetitions = 1

df_results = experiment_risk_vs_m()

plotting.plot_risk_vs_iterations(df_results, metric = 'train_risk', 
                        title = 'Training risk evolution',
                        baseline = train_risk_full_KRR,
                        save_path = plots_dir / "1D" / "EmpRisk")

plotting.plot_risk_vs_iterations(df_results,
                        metric = 'test_risk',
                        title = 'Test risk evolution',
                        baseline = test_risk_full_KRR,
                        save_path = plots_dir / "1D" / "TestRisk")

plotting.plot_risk_vs_number_of_nystrom_points(df_results, metric = 'train_risk', 
                        title = 'Training risk evolution',
                        baseline = train_risk_full_KRR,
                        save_path = plots_dir / "1D" / "EmpRisk")

plotting.plot_risk_vs_number_of_nystrom_points(df_results,
                        metric = 'test_risk',
                        title = 'Test risk evolution',
                        baseline = test_risk_full_KRR,
                        save_path = plots_dir / "1D" / "TestRisk")



#In case of 1D simulation, here is a new single experiment with visualization
if d == 1:
    single_experiment_1D("Blended_MP")
