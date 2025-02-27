import numpy as np
import MGLensing
import matplotlib.pyplot as plt
from nautilus import Prior, Sampler
from scipy.stats import norm

MGLtest = MGLensing.MGL("config.yaml")
#MGLtest.test()


#print(MGLtest.params_fixed)
#print(MGLtest.params_model)
print(MGLtest.params_fiducial)

def log_probability_function(pars):
    param_dic = pars | MGLtest.params_fixed
    return MGLtest.Like.loglikelihood_det_3x2pt(param_dic, MGLtest.theo_model_dic)

Omm_arr = np.linspace(0.28, 0.34, 10)
pars = MGLtest.params_fiducial
for Omm in Omm_arr:
    pars['Omega_m'] = Omm
    print(log_probability_function(pars))