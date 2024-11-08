from scipy.integrate import trapz,simpson, quad
from scipy import interpolate as itp
from scipy.interpolate import interp1d, RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.special import erf
import numpy as np
import MGrowth as mg
from multiprocessing import Pool
import os
import baccoemu as bacco
import pyhmcode 
import cosmopower as cp
from cosmopower import cosmopower_NN
from nautilus import Prior, Sampler
from scipy.stats import norm
import BCemu
import math
from compute_power_spectra import *
from setup_and_priors import *
from compute_Cell  import *
from compute_baryonic_boost  import *
import time

################################################
#               MODEL                         #
################################################
ModelParsCosmo = ['Omega_m', 'Ombh2', 'h', 'log10As', 'ns', 'gamma', 'q1'] #always write Mnu in the end
ModelParsIA = ['aIA', 'etaIA']
ModelParsBC = ['log10Mc']#['log10Mc', 'thej', 'mu', 'gamma_bc', 'delta', 'eta', 'deta']
Model = 'GQ' #HMcode, nDGP, GQ, GzQ, MuSpline
Flag_Baryons_Model = True #False or True
Flag_Delta_z_model = False
Mnu_model = 0.0#0.06#0.0
Flag_Fix_Bias = False
Flag_Fix_IA = False

hdf5_name = 'GQ_3x2pt_broadMc'
description = 'GQ on GQ + broad log10Mc + vary cosmology'
chain_name = "GQfid_GQModel_lmax3k1k_PlanckBBN_nautilus3k_3x2ptprobe_broadMc"


################################################
#               MOCK DATA                      #
################################################
###change here###
Model_fid = 'GQ' #HMcode, nDGP, GQ, GzQ
Flag_Baryons_fid = True #False or True
Flag_Probes = '3x2pt' #'GC', '3x2pt', 'WL'
#################
log10As_fid = 3.044
h_fid = 0.68
ns_fid = 0.97
Omega_m_fid = 0.3085640138408304
Omega_b_fid = 0.04904844290657439
Mnu_fid = 0.06#0.0
w0_fid = -1.#-0.9#-0.45 DESI+CMB 
wa_fid = 0.#-0.1#-1.79 DESI+CMB 
aIA_fid = 1.72
etaIA_fid = -0.41
betaIA_fid = 0.0
Omega_nu_fid = Mnu_fid/(93.14*h_fid**2)
fb_fid = Omega_b_fid/(Omega_m_fid-Omega_nu_fid)  
Cosmo_fid = {'ns': ns_fid, 'As': np.exp(log10As_fid)*1e-10, 'log10As':log10As_fid,
             'h': h_fid, 'Omega_b': Omega_b_fid, 'Ombh2': Omega_b_fid*h_fid**2,                                                                                            
             'Omega_m': Omega_m_fid, 'Mnu': Mnu_fid, 'Omega_nu': Omega_nu_fid, 'w0':w0_fid, 'wa':wa_fid, 'tau':0.09,
             'gamma': 0.4, 'gamma0':0.55, 'gamma1':0., 'q1':0.76, 
             'fb': fb_fid, 
            'log10omegarc': np.log10(0.25)} 

nu_Mc_fid = 0.0
log10Mc_fid = 13.32
thej_fid = 4.235
mu_fid = 0.93 
gamma_bc_fid = 2.25
delta_fid = 6.40
eta_fid = 0.15
deta_fid = 0.14
Baryons_fid = {'log10Mc': log10Mc_fid, 'thej': thej_fid, 'mu': mu_fid, 'gamma_bc': gamma_bc_fid, 'delta': delta_fid,
               'eta': eta_fid, 'deta': deta_fid}   
b0 = 0.68
bias_array_fid = np.array([b0*(1.+z_bin_center_i) for z_bin_center_i in z_bin_center ])
print('Probe: ', Flag_Probes)
print('start computing mock data')
if Flag_Probes=='3x2pt':
    Cov_observ, Cov_observ_high = compute_covariance_WL_3x2pt(Cosmo_fid, Baryons_fid, aIA_fid, etaIA_fid, betaIA=0., bias_array=bias_array_fid, Flag_Model=Model_fid, Flag_Baryons=Flag_Baryons_fid, Flag_Probes=Flag_Probes)
    d_obs = np.linalg.det(Cov_observ) 
    d_obs_high = np.linalg.det(Cov_observ_high) 
elif Flag_Probes=='WL':
    Cov_observ = compute_covariance_WL_3x2pt(Cosmo_fid, Baryons_fid, aIA_fid, etaIA_fid, betaIA=0., bias_array=bias_array_fid, Flag_Model=Model_fid, Flag_Baryons=Flag_Baryons_fid, Flag_Probes=Flag_Probes)
    d_obs = np.linalg.det(Cov_observ) 
    ells_one_probe = ells_WL
elif Flag_Probes=='GC':   
    Cov_observ = compute_covariance_WL_3x2pt(Cosmo_fid, Baryons_fid, aIA_fid, etaIA_fid, betaIA=0., bias_array=bias_array_fid, Flag_Model=Model_fid, Flag_Baryons=Flag_Baryons_fid, Flag_Probes=Flag_Probes)
    d_obs = np.linalg.det(Cov_observ) 
    ells_one_probe = ells_GC 
print('finish computing mock data')

################################################
#               LIKELIHOOD                     #
################################################
def loglikelihood_det(param_dict):
    Cosmo = {}
    Baryons = Baryons_fid #copy and then change only the params that are sampled
    if  Model == 'MuSpline':
        Cosmo = Cosmo_fid 
    for key in ModelParsCosmo:
        Cosmo[key] = param_dict[key]
    if Flag_Baryons_Model == True:    
        for key in ModelParsBC:
            Baryons[key] = param_dict[key]
    if Flag_Probes=='WL':
        bias_array = np.ones(nbin)
    elif  Flag_Fix_Bias == True:
        bias_array = bias_array_fid   
    else:
        bias_array = np.array([param_dict['b'+str(i+1)] for i in range(nbin)])    
    if Flag_Fix_IA == True:
        param_dict['aIA'] = aIA_fid
        param_dict['etaIA'] = etaIA_fid    
    #if Flag_Delta_z_model == True:
    #    Delta_z_array = np.array([param_dict['Delta_z'+str(i+1)] for i in range(nbin)])   
    #else:
    #    Delta_z_array =  None
    Delta_z_array =  None    
    h = Cosmo['h']    
    Omega_b = Cosmo['Ombh2']/h**2
    Omega_nu = Cosmo['Mnu']/(93.14*h**2) if 'Mnu' in ModelParsCosmo else Mnu_model/(93.14*h**2)
    Omega_m = Cosmo['Omega_m']
    Cosmo['Omega_b'] = Omega_b #power spectrum models expect Omega_b and Omega_m
    Cosmo['Omega_nu'] = Omega_nu
    Cosmo['As'] = np.exp(Cosmo['log10As'])*1e-10
    fb = Omega_b/(Omega_m-Omega_nu)  
    Cosmo['fb'] = fb  
    if Flag_Baryons_Model == True and (fb < 0.1 or fb > 0.25): #the baryon fraction is out of bounds
        return -1e10
    
    chi2 = 0. 
    if Flag_Probes=='3x2pt':
        Cov_theory, Cov_theory_high = compute_covariance_WL_3x2pt(Cosmo, Baryons, param_dict['aIA'], param_dict['etaIA'], betaIA=0., bias_array=bias_array, Delta_z=Delta_z_array, Flag_Model=Model, Flag_Baryons=Flag_Baryons_Model, Flag_Probes=Flag_Probes)
        d_the = np.linalg.det(Cov_theory)
        d_mix = np.zeros_like(d_the)
        d_obs = np.linalg.det(Cov_observ) 
        d_obs_high = np.linalg.det(Cov_observ_high) 
        for i in range(2*nbin):
            newCov = Cov_theory.copy()
            newCov[:, i] = Cov_observ[:, :, i]
            d_mix += np.linalg.det(newCov)

        d_the_high = np.linalg.det(Cov_theory_high)
        d_mix_high = np.zeros_like(d_the_high)
        for i in range(nbin):
            newCov = Cov_theory_high.copy()
            newCov[:, i] = Cov_observ_high[:, :, i]
            d_mix_high += np.linalg.det(newCov)
        d_the = np.concatenate([d_the,d_the_high])
        d_obs = np.concatenate([d_obs,d_obs_high])
        d_mix = np.concatenate([d_mix,d_mix_high])
        chi2 += np.sum((2*ells_WL+1)*fsky*((d_mix/d_the)+np.log(d_the/d_obs)))
    else:
        Cov_theory = compute_covariance_WL_3x2pt(Cosmo, Baryons, param_dict['aIA'], param_dict['etaIA'], betaIA=0., bias_array=bias_array, Delta_z=Delta_z_array, Flag_Model=Model, Flag_Baryons=Flag_Baryons_Model, Flag_Probes=Flag_Probes)
        d_the = np.linalg.det(Cov_theory)
        d_mix = np.zeros_like(d_the)
        for i in range(nbin):
            newCov = np.copy(Cov_theory)
            d_obs = np.linalg.det(Cov_observ) 
            newCov[:, i] = Cov_observ[:, :, i]
            d_mix += np.linalg.det(newCov)
        chi2 += np.sum((2*ells_one_probe+1)*fsky*((d_mix/d_the)+np.log(d_the/d_obs)))
    return -0.5*chi2




################################################
#               MCMC (nautilus)                #
################################################
os.environ["OMP_NUM_THREADS"] = '1'
if Model == 'HMcode':
    emu_name = 'HMcode'
else:
    emu_name = 'MGemus'    
prior = Prior()
for par_i in ModelParsCosmo:
    if MasterPriors[emu_name][par_i]['type'] == 'G':
        prior.add_parameter(par_i, dist=norm(loc=MasterPriors[emu_name][par_i]['p1'] , scale=MasterPriors[emu_name][par_i]['p2']))
    else:
        prior.add_parameter(par_i, dist=(MasterPriors[emu_name][par_i]['p1'] , MasterPriors[emu_name][par_i]['p2']))
if Flag_Probes != 'GC' and Flag_Fix_IA==False:
    for par_i in ModelParsIA:
        if MasterPriors['IA'][par_i]['type'] == 'G':
            prior.add_parameter(par_i, dist=norm(loc=MasterPriors['IA'][par_i]['p1'] , scale=MasterPriors['IA'][par_i]['p2']))
        else:
            prior.add_parameter(par_i, dist=(MasterPriors['IA'][par_i]['p1'] , MasterPriors['IA'][par_i]['p2']))
if Flag_Baryons_Model == True:
    for par_i in ModelParsBC:
        if MasterPriors['BCemu'][par_i]['type'] == 'G':
            prior.add_parameter(par_i, dist=norm(loc=MasterPriors['BCemu'][par_i]['p1'] , scale=MasterPriors['BCemu'][par_i]['p2']))
        else:
            prior.add_parameter(par_i, dist=(MasterPriors['BCemu'][par_i]['p1'] , MasterPriors['BCemu'][par_i]['p2']))
if Flag_Probes != 'WL' and Flag_Fix_Bias==False:
    for i in range(nbin):
        if MasterPriors['b']['type'] == 'G':
            prior.add_parameter('b'+str(i+1), dist=norm(loc=MasterPriors['b']['p1'] , scale=MasterPriors['b']['p2']))
        else:
            prior.add_parameter('b'+str(i+1), dist=(MasterPriors['b']['p1'] , MasterPriors['b']['p2']))
#if Flag_Delta_z_model == True:
#    for i in range(nbin):
#        if MasterPriors['Delta_z']['type'] == 'G':
#            prior.add_parameter('Delta_z'+str(i+1), dist=norm(loc=MasterPriors['Delta_z']['p1'] , scale=MasterPriors['Delta_z']['p2']))
#        else:
#            prior.add_parameter('Delta_z'+str(i+1), dist=(MasterPriors['Delta_z']['p1'] , MasterPriors['Delta_z']['p2']))


start = time.time()
sampler = Sampler(prior, loglikelihood_det, filepath='hdf5/'+hdf5_name+'.hdf5', n_live=3000, pool=14)
sampler.run(verbose=True, discard_exploration=True)#, n_eff=30000)
log_z = sampler.evidence()


points, log_w, log_l = sampler.posterior()
finish = time.time()
print('time for mcmc')
print(description, ' : ', finish-start)
np.savetxt("chains/"+chain_name+".txt", np.c_[points, log_w, log_l])

