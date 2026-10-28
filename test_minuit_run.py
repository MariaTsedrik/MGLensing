import os
from iminuit import Minuit
import numpy as np
import MGLensing
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import multiprocessing
from datetime import timedelta



MGLtest = MGLensing.MGL("ini_files/config_files/config_minimise_fig4.yaml")
best_fit = {}

def log_probability_function(bias_array):
    pars = {
        'Omega_m': bias_array[0], 'sigma8_cb': bias_array[1], 'log10Mc_bc': bias_array[2],
        'b1L_1': bias_array[3], 'b1L_2': bias_array[4], 'b1L_3': bias_array[5], 'b1L_4': bias_array[6], 'b1L_5': bias_array[7],
        'b2L_1': bias_array[8], 'b2L_2': bias_array[9], 'b2L_3': bias_array[10], 'b2L_4': bias_array[11], 'b2L_5': bias_array[12],
            'bs2L_1': bias_array[13], 'bs2L_2': bias_array[14], 'bs2L_3': bias_array[15], 'bs2L_4': bias_array[16], 'bs2L_5': bias_array[17],
            'blaplL_1': bias_array[18], 'blaplL_2': bias_array[19], 'blaplL_3': bias_array[20], 'blaplL_4': bias_array[21], 'blaplL_5': bias_array[22]
            }
    param_dic = pars | MGLtest.params_fixed
    like = MGLtest.Like.compute(param_dic)
    print(like)
    best_fit = np.copy(pars)
    return like



par_guess_array = np.array([
    0.31, 0.83, 13.8,
0.187684, 0.312375, 0.45121, 0.60626, 0.779385,
    0.1, 0.1, 0.1, 0.1, 0.1,
    -0.1, -0.1, -0.1, -0.1, -0.1, 
    0.2, 0.2, 0.2, 0.2, 0.2
    ])
ndim = len(par_guess_array)

nll = lambda *args: -log_probability_function(*args)
initial = par_guess_array + 0.01 * np.random.randn(ndim)

min_obj= Minuit(nll,initial,
name=(
 'Omega_m', 'sigma8_cb', 'log10Mc_bc',
'b1L_1', 'b1L_2', 'b1L_3', 'b1L_4', 'b1L_5',
'b2L_1', 'b2L_2', 'b2L_3', 'b2L_4', 'b2L_5',
'bs2L_1', 'bs2L_2', 'bs2L_3', 'bs2L_4', 'bs2L_5',
'blaplL_1', 'blaplL_2', 'blaplL_3', 'blaplL_4', 'blaplL_5'
) )
min_obj.errordef = Minuit.LIKELIHOOD

#priors
for i in range(5):
    min_obj.limits["b1L_"+str(i+1)] = (0., 2.)
    min_obj.limits["b2L_"+str(i+1)] = (-2., 2.)
    min_obj.limits["bs2L_"+str(i+1)] = (-2., 2.)
    min_obj.limits["blaplL_"+str(i+1)] = (-2., 2.)

#min_obj.tol = 0.001
print(min_obj.migrad(ncall=10000, iterate=100))
print(min_obj.params)
print(best_fit)








