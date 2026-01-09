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
                regularization_cost = reg_param*model.alpha@model.predict(model.anchor_data)
                cost_function = train_risk + regularization_cost
                #saving results
                record = {
                    'strategy': strategy,
                    'tot_iterations': m,
                    'number_of_nystrom_points': model.alpha.size,
                    'repetition': rep,
                    'train_risk': train_risk,
                    'test_risk': test_risk,
                    'regularization_cost': regularization_cost,
                    'cost_function': cost_function
                }
                results_log.append(record)

    return pd.DataFrame(results_log)

r = 0.1
kernel = partial(krnls.gaussian_kernel, r = r)
rng = np.random.default_rng(1)


#dataset    

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, 'dataset')

data = np.loadtxt(dataset_path, delimiter=',')
tot_data = data.shape[0]
n = int(tot_data*0.1)
d = 5
n_test = tot_data-n
model = None

training_indexes = np.random.choice(list(range(tot_data)), size=n, p=1/tot_data * np.ones(tot_data),replace=False)
test_indexes = np.setdiff1d(list(range(tot_data)),training_indexes)
X_train = data[training_indexes,:5]
labels_train = data[training_indexes,6]
X_test = data[test_indexes,:5]
labels_test = data[test_indexes,6]
print("Datapoints:")
for i in range(5):
    print(f"Dimension | largest | smallest: {i+1} ,  {np.max(X_train[:,i])} , {np.min(X_train[:,i])}")
print("Measurement:")
print(f"largest | smallest: {np.max(labels_train)} , {np.min(labels_train)}")

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
cost_function_full_KRR = train_risk_full_KRR + reg_param*alpha@utils_math.predict(X_train = X_train,eval_point=X_train, alpha=alpha, kernel=kernel)


print(f"Empirical risk of the full model is {train_risk_full_KRR}")
print(f"Test risk of the full model is {test_risk_full_KRR}")

#NYSTROM MODELS & SIMULATIONS

#MAIN EXPERIMENT
#general parameters
mrange = np.logspace(0,8,9,True,2,'int')
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

plotting.plot_risk_vs_iterations(df_results,
                        metric = 'train_risk',
                        title = 'Training risk evolution',
                        baseline = train_risk_full_KRR,
                        save_path = plots_dir/ "Truedata" / "EmpRisk")

plotting.plot_risk_vs_iterations(df_results,
                        metric = 'test_risk',
                        title = 'Test risk evolution',
                        baseline = test_risk_full_KRR,
                        save_path = plots_dir / "Truedata" / "TestRisk")

plotting.plot_risk_vs_number_of_nystrom_points(df_results,
                        metric = 'train_risk',
                        title = 'Training risk evolution',
                        baseline = train_risk_full_KRR,
                        save_path = plots_dir/ "Truedata" / "EmpRisk")

plotting.plot_risk_vs_number_of_nystrom_points(df_results,
                        metric = 'test_risk',
                        title = 'Test risk evolution',
                        baseline = test_risk_full_KRR,
                        save_path = plots_dir / "Truedata" / "TestRisk")

plotting.plot_single_strategy_cost_function(df = df_results,
                                            title = "Cost function, risk and regularization terms decay",
                                            baseline= cost_function_full_KRR,
                                            save_path = plots_dir / "Truedata" / "Experiments"
                                            )


plotting.plot_single_strategy_cost_function_separate_plots(df = df_results,
                                            title = "Cost function, risk and regularization terms decay",
                                            baseline= cost_function_full_KRR,
                                            save_path = plots_dir / "Truedata" / "Experiments"
                                            )

