import os
from iminuit import Minuit
import numpy as np
import MGLensing
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import multiprocessing
from datetime import timedelta



MGLtest = MGLensing.MGL("ini_files/config_files/config_minimise.yaml")
best_fit = {}

def log_probability_function(bias_array):
    pars = {'b2L_1': bias_array[0], 'b2L_2': bias_array[1], 'b2L_3': bias_array[2], 'b2L_4': bias_array[3], 'b2L_5': bias_array[4],
            'bs2L_1': bias_array[5], 'bs2L_2': bias_array[6], 'bs2L_3': bias_array[7], 'bs2L_4': bias_array[8], 'bs2L_5': bias_array[9],
            'blaplL_1': bias_array[10], 'blaplL_2': bias_array[11], 'blaplL_3': bias_array[12], 'blaplL_4': bias_array[13], 'blaplL_5': bias_array[14]
            }
    param_dic = pars | MGLtest.params_fixed
    like = MGLtest.Like.compute(param_dic)
    print(like)
    best_fit = np.copy(pars)
    return like



par_guess_array = np.array([
    0.1, 0.1, 0.1, 0.1, 0.1,
    -0.1, -0.1, -0.1, -0.1, -0.1, 
    0.2, 0.2, 0.2, 0.2, 0.2
    ])
ndim = len(par_guess_array)

nll = lambda *args: -log_probability_function(*args)
initial = par_guess_array + 0.01 * np.random.randn(ndim)

min_obj= Minuit(nll,initial,
name=('b2L_1', 'b2L_2', 'b2L_3', 'b2L_4', 'b2L_5',
'bs2L_1', 'bs2L_2', 'bs2L_3', 'bs2L_4', 'bs2L_5',
'blaplL_1', 'blaplL_2', 'blaplL_3', 'blaplL_4', 'blaplL_5'
) )
min_obj.errordef = Minuit.LIKELIHOOD

#priors
for i in range(5):
     min_obj.limits["b2L_"+str(i+1)] = (-2., 2.)
     min_obj.limits["bs2L_"+str(i+1)] = (-2., 2.)
     min_obj.limits["blaplL_"+str(i+1)] = (-2., 2.)

#min_obj.tol = 0.001
print(min_obj.migrad(ncall=10000, iterate=100))
print(min_obj.params)
print(best_fit)








