import numpy as np
import MGLensing
import time
import matplotlib.pyplot as plt
import os
from nautilus import Prior, Sampler
from scipy.stats import norm
import multiprocessing
from datetime import timedelta
os.environ["OMP_NUM_THREADS"] = "1"


MGLtest = MGLensing.MGL("config.yaml")

def log_probability_function_3x2pt(pars):
    param_dic = pars | MGLtest.params_fixed
    return MGLtest.Like.loglikelihood_det_3x2pt(param_dic, MGLtest.theo_model_dic)
def log_probability_function_wl(pars):
    param_dic = pars | MGLtest.params_fixed
    return MGLtest.Like.loglikelihood_det_wl(param_dic, MGLtest.theo_model_dic)
def log_probability_function_gc(pars):
    param_dic = pars | MGLtest.params_fixed
    return MGLtest.Like.loglikelihood_det_gc(param_dic, MGLtest.theo_model_dic)

prior = Prior()
for par_i in MGLtest.params_model:
    if MGLtest.params_priors[par_i]['type'] == 'G':
        prior.add_parameter(par_i, dist=norm(loc=MGLtest.params_priors[par_i]['p1'] , scale=MGLtest.params_priors[par_i]['p2']))
    elif MGLtest.params_priors[par_i]['type'] == 'U':
        prior.add_parameter(par_i, dist=(MGLtest.params_priors[par_i]['p1'] , MGLtest.params_priors[par_i]['p2']))

if MGLtest.probe == 'WL':
    log_probability_function = log_probability_function_wl
elif MGLtest.probe == 'GC':
    log_probability_function = log_probability_function_gc    
elif MGLtest.probe == '3x2pt':
    log_probability_function = log_probability_function_3x2pt
else:
    raise ValueError('Check the probe name!')      

def main():
    sampler = Sampler(prior, log_probability_function, 
                                filepath='chains/hdf5/'+MGLtest.hdf5_name+'.hdf5', resume=MGLtest.mcmc_resume, n_live=MGLtest.mcmc_nlive, pool=MGLtest.mcmc_pool)
    start = time.time()
    sampler.run(verbose=MGLtest.mcmc_verbose, discard_exploration=True)
    log_z = sampler.evidence()
    points, log_w, log_l = sampler.posterior()
    finish = time.time()
    chain_time = finish-start

    np.savetxt("chains/chain_"+MGLtest.chain_name+".txt", np.c_[points, log_w, log_l], header=MGLtest.gen_output_header(), footer='log_Z = {log_z};  chain_time = {chain_time} (--> {chain_time_hms} hh:mm:ss)'.format(log_z=log_z, chain_time=chain_time, chain_time_hms=timedelta(seconds=chain_time)))


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all pools are properly closed
        multiprocessing.active_children()








