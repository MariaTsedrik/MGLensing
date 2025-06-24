import os
os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OMP_PLACES"] = "threads"
from iminuit import Minuit
import numpy as np
import MGLensing
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import multiprocessing
from datetime import timedelta



MGLtest = MGLensing.MGL("config.yaml")


def log_probability_function(pars):
    param_dic = pars | MGLtest.params_fixed
    return MGLtest.Like.compute(param_dic)



par_fid_array = np.array([
    2.06, 2.13, 2.07, 2.29, 0., 0., 0., 0., 0., 0., 0., 0.,...
    ])
ndim = len(par_fid_array)

nll = lambda *args: -log_probability_function(*args)
initial = par_fid_array + 0.1 * np.random.randn(ndim)

min_obj= Minuit(nll,initial)
min_obj.errordef = Minuit.LIKELIHOOD

#priors
#min_obj.limits['b1_nz1'] = (0,4)
#min_obj.limits['b1_nz3'] = (0,4)
#min_obj.limits['b1_sz1'] = (0,4)
#min_obj.limits['b1_sz3'] = (0,4)


#min_obj.tol = 0.001
print(min_obj.migrad(ncall=10000, iterate=100))
print(min_obj.params)









