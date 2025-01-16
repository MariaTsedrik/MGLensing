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
import matplotlib.pyplot as plt
################################################
#               MODEL                         #
################################################
ModelParsCosmo = ['Omega_m', 'Ombh2', 'h', 'log10As', 'ns'] #always write Mnu in the end
ModelParsIA = ['aIA', 'etaIA']
ModelParsBC = ['log10Mc']#['log10Mc', 'thej', 'mu', 'gamma_bc', 'delta', 'eta', 'deta']
Model = 'HMcode' #HMcode, nDGP, GQ, GzQ, MuSpline
Flag_Baryons_Model = False #False or True
Flag_Delta_z_model = False
Mnu_model = 0.0#0.06#0.0
Flag_Fix_Bias = False
Flag_Fix_IA = False
Flag_test = True
Flag_plot = False
bias_model = 'binned_quadratic' #'binned_constant', 'binned_quadratic', 'interpld'
hdf5_name = 'test'
description = 'LCDM on LCDM + no bar + vary cosmology'
#chain_name = "GQfid_GQModel_lmax3k1k_PlanckBBN_nautilus3k_3x2ptprobe_broadMc"
chain_name = "test"

################################################
#               MOCK DATA                      #
################################################
###change here###
Model_fid = 'HMcode' #HMcode, nDGP, GQ, GzQ
Flag_Baryons_fid = False #False or True
Flag_Probes = '3x2pt' #'GC', '3x2pt', 'WL'
bias_model_fid = 'binned_quadratic' #'binned_constant', 'binned_quadratic', 'interpld'
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
            'log10omegarc': np.log10(0.25),
            'aIA': aIA_fid, 'etaIA':etaIA_fid} 

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
bias2_array_fid = bias_array_fid*8.
print('Probe: ', Flag_Probes)
print('start computing mock data')

if Flag_Probes=='3x2pt':
    Cov_observ, Cov_observ_high = compute_covariance_WL_3x2pt(Cosmo_fid, Baryons_fid, aIA_fid, etaIA_fid, betaIA=0., bias_array=bias_array_fid, Flag_Model=Model_fid, Flag_Baryons=Flag_Baryons_fid, Flag_Probes=Flag_Probes, bias_model=bias_model_fid)
    d_obs = np.linalg.det(Cov_observ) 
    d_obs_high = np.linalg.det(Cov_observ_high) 
elif Flag_Probes=='WL':
    Cov_observ = compute_covariance_WL_3x2pt(Cosmo_fid, Baryons_fid, aIA_fid, etaIA_fid, betaIA=0., bias_array=bias_array_fid, Flag_Model=Model_fid, Flag_Baryons=Flag_Baryons_fid, Flag_Probes=Flag_Probes, bias_model=bias_model_fid)
    d_obs = np.linalg.det(Cov_observ) 
    ells_one_probe = ells_WL
elif Flag_Probes=='GC':   
    Cov_observ = compute_covariance_WL_3x2pt(Cosmo_fid, Baryons_fid, aIA_fid, etaIA_fid, betaIA=0., bias_array=bias_array_fid, Flag_Model=Model_fid, Flag_Baryons=Flag_Baryons_fid, Flag_Probes=Flag_Probes, bias_model=bias_model_fid)
    d_obs = np.linalg.det(Cov_observ) 
    ells_one_probe = ells_GC 
print('finish computing mock data')

################################################
#               TESTING                        #
################################################
if Flag_plot == True:
    bias_model_test = 'binned_constant'
    Cov_theory, Cov_theory_high, dic_Pk = compute_covariance_WL_3x2pt(Cosmo_fid, Baryons_fid, aIA_fid, etaIA_fid, betaIA=0., bias_array=bias_array_fid, Delta_z=None, Flag_Model=Model_fid, Flag_Baryons=Flag_Baryons_fid, Flag_Probes=Flag_Probes, Flag_savePk=True, bias_model=bias_model_test)
    pk_m_l = dic_Pk['Pmm'] 
    pk_m_bar = dic_Pk['Pmm_baryons']
    ell =  dic_Pk['ell']
    k_ell =  dic_Pk['k']
    rcom = dic_Pk['r_com']
    Cl_GG =  dic_Pk['Cl_GG']
    print('shapes:')
    print('pk_m_l: ', pk_m_l.shape )
    print('ell: ', ell.shape )
    print('k_ell: ', k_ell.shape)
    print('rcom: ', rcom.shape)
    print('Cl_GG: ', Cl_GG.shape)

    z_int_pick = [10, 100, 195]
    def k2ell(x, ind=z_int_pick[0]):
        return x*rcom[ind]-0.5
    def ell2k(x, ind=z_int_pick[0]):
        return (x+0.5)/rcom[ind]
    def k2ell1(x, ind=z_int_pick[1]):
        return x*rcom[ind]-0.5
    def ell2k1(x, ind=z_int_pick[1]):
        return (x+0.5)/rcom[ind]
    def k2ell2(x, ind=z_int_pick[2]):
        return x*rcom[ind]-0.5
    def ell2k2(x, ind=z_int_pick[2]):
        return (x+0.5)/rcom[ind]

    f_arr = [k2ell, k2ell1, k2ell2]
    f_inv_arr = [ell2k, ell2k1, ell2k2]

    fig, ax = plt.subplots(1,3, figsize=(12,5))
    for i in range(3):
        #ax[i].axvline(x=50., color='k')
        ax[i].scatter(k_ell[:, z_int_pick[i]], pk_m_l[:, z_int_pick[i]])
        ax[i].legend(title='$z=$'+str(round(zz_integr[z_int_pick[i]],2)), loc='lower left')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel("$k$ [$h$/Mpc]")
        secax = ax[i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
        secax.set_xlabel('$\ell$')
    #for i in range(3):
        #print(z_int_pick[i])
        #f1 = lambda x: k2ell(x, z_int_pick[i])
        #f2 = lambda x: ell2k(x, z_int_pick[i])
        #ax[i].secondary_xaxis('top', functions=(f1, f2))
        #ax[i].secondary_xaxis('top', functions=(lambda xx: k2ell(xx, z_int_pick[i]), lambda yy: ell2k(yy, z_int_pick[i])))
    #    secax = ax[i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
    #    secax.set_xlabel('$\ell$')
    ax[0].set_ylabel("$P^{\\rm NL}(k(\ell, z), z)$")
    plt.savefig('./figs/power_spectra.png')


    bias_array = bias_array_fid
    if bias_model_test == 'binned_constant' :
        bias = bias_array #(10,)
        W_G = bias[None,:] #z_int, i-bin
    # - case where the bias is a single function b(z) for all bins
    elif bias_model_test == 'interpld':
        W_G = np.zeros((zbin_integr, nbin), 'float64')
        biasfunc = interp1d(z_bin_center, bias_array, bounds_error=False, fill_value="extrapolate")
        W_G =  biasfunc(zz_integr)[:,None] #(200 z_int, i-bin)
    Pk_gg = W_G[None,:,:,None] * W_G[None,: , None, :] * pk_m_l[:,:,None,None]   #ell, z_int, i-bin, j-bin
    print('Pk_gg shape: ', Pk_gg.shape)  
    #bias_model_test == 'quadratic'
    bias0 = bias_array
    bias2 = bias2_array_fid
    W_G_b2 = bias[None, None,:]+bias2[None, None, :]*k_ell[:, :, None]**2
    Pk_gg_b2 = W_G_b2[:,:,:,None] * W_G_b2[:,: , None, :] * pk_m_l[:,:,None,None]   #ell, z_int, i-bin, j-bin

    bins=['1-3', '3-3', '3-5']
    ibin = [0, 2, 2]
    jbin = [2, 2, 4]
    fig, ax = plt.subplots(1,3, figsize=(12,5))
    for i in range(3):
        ax[i].scatter(k_ell[:, 100], Pk_gg[:, 100, ibin[i], jbin[i]], label='$b_i b_j P_{\\rm NL}$: $b_i$='+str(round(bias_array[ibin[i]],2))+', $b_j$='+str(round(bias_array[jbin[i]],2)))
        ax[i].scatter(k_ell[:, 100], pk_m_l[:, 100], label='$P_{\\rm NL}$')
        ax[i].legend(title='$z=$'+str(round(zz_integr[100],2))+' in bin '+bins[i], loc='lower left')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel("$k$ [$h$/Mpc]")
    ax[0].set_ylabel("$P^{\\rm NL}(k(\ell, z), z)$")
    plt.savefig('./figs/galaxy_power_spectra_linear_bias.png')


    bins=['1-3', '3-3', '3-5']
    ibin = [0, 2, 2]
    jbin = [2, 2, 4]
    fig, ax = plt.subplots(1,3, figsize=(12,5))
    for i in range(3):
        ax[i].scatter(k_ell[:, 100], pk_m_l[:, 100], label='$P_{\\rm NL}$')
        ax[i].scatter(k_ell[:, 100], Pk_gg[:, 100, ibin[i], jbin[i]], label='$b_i b_j P_{\\rm NL}$: $b_i$='+str(round(bias_array[ibin[i]],2))+', $b_j$='+str(round(bias_array[jbin[i]],2)))
        ax[i].scatter(k_ell[:, 100], Pk_gg_b2[:, 100, ibin[i], jbin[i]], label='$b_i = b_i + b_{i,2} k^2$: $b_{i,2}$='+str(round(bias2[ibin[i]],2))+', $b_{j,2}$='+str(round(bias2[jbin[i]],2)))
        ax[i].legend(title='$z=$'+str(round(zz_integr[100],2))+' in bin '+bins[i], loc='lower left')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel("$k$ [$h$/Mpc]")
    ax[0].set_ylabel("$P^{\\rm NL}(k(\ell, z), z)$")
    plt.savefig('./figs/galaxy_power_spectra_quadratic_bias.png')

    bias_model_test_v2 = 'binned_quadratic'
    Cov_theory, Cov_theory_high, dic_Pk_quadr_bias = compute_covariance_WL_3x2pt(Cosmo_fid, Baryons_fid, aIA_fid, etaIA_fid, betaIA=0., bias_array=bias_array_fid, bias2_array=bias2_array_fid, Delta_z=None, Flag_Model=Model_fid, Flag_Baryons=Flag_Baryons_fid, Flag_Probes=Flag_Probes, Flag_savePk=True, bias_model=bias_model_test_v2)
    Cl_GG_quadr_bias = dic_Pk_quadr_bias['Cl_GG']
    Cl_LG_quadr_bias = dic_Pk_quadr_bias['Cl_LG']
    Cl_GG_lin_bias = dic_Pk['Cl_GG']
    Cl_LG_lin_bias = dic_Pk['Cl_LG']
    #for ii in range(10):
    #    print(Cl_LG_quadr_bias[:, ii, ii])
    #print(Cl_GG_lin_bias.any()<0)

    fig, ax = plt.subplots(figsize=(28, 24), nrows = 10, ncols=10, sharex=True, facecolor='w')
    for i in range(10):
        for j in range(10):
            if i<j:
                ax[i, j].axis('off')
            else:
                if i==0 and j==0:
                    ax[i, j].loglog(l_GC, Cl_GG_quadr_bias[:, i, j], color='tab:blue', label='$b_i=b_i+b_{i,2}k^2$')
                    ax[i, j].loglog(l_GC, Cl_GG_lin_bias[:, i, j], color='tab:orange', label='$b_i=b_i$')
                else:    
                    ax[i, j].loglog(l_GC, Cl_GG_quadr_bias[:, i, j], color='tab:blue')
                    ax[i, j].loglog(l_GC, Cl_GG_lin_bias[:, i, j], color='tab:orange')
                ax[i, j].legend(loc='lower left', title_fontsize=16, title='bin '+str(i+1)+'-'+str(j+1))
    for i in range(10):
        ax[9][i].set_xlabel('$\ell$')
        
    ax[5][0].set_ylabel('$C^{\\rm GG}_{\ell}$')
    plt.tight_layout()
    plt.savefig('./figs/Cell_GG.png')

    fig, ax = plt.subplots(figsize=(28, 24), nrows = 10, ncols=10, sharex=True, facecolor='w')
    for i in range(10):
        for j in range(10):
            if i<j:
                ax[i, j].axis('off')
            else:
                if i==0 and j==0:
                    ax[i, j].loglog(l_GC, Cl_LG_quadr_bias[:, i, j], color='tab:blue', label='$b_i=b_i+b_{i,2}k^2$')
                    ax[i, j].loglog(l_GC, Cl_LG_lin_bias[:, i, j], color='tab:orange', label='$b_i=b_i$')
                else:    
                    ax[i, j].loglog(l_GC, Cl_LG_quadr_bias[:, i, j], color='tab:blue')
                    ax[i, j].loglog(l_GC, Cl_LG_lin_bias[:, i, j], color='tab:orange')
                ax[i, j].legend(loc='lower left', title_fontsize=16, title='bin '+str(i+1)+'-'+str(j+1))
    for i in range(10):
        ax[9][i].set_xlabel('$\ell$')
        
    ax[5][0].set_ylabel('$C^{\\rm LG}_{\ell}$')
    plt.tight_layout()
    plt.savefig('./figs/Cell_LG.png')

################################################
#               LIKELIHOOD                     #
################################################
def loglikelihood_det(param_dict):
    Cosmo = {}
    #print('inside likelihood')
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
        bias2_array = np.zeros(nbin)
    elif  Flag_Fix_Bias == True:
        bias_array = bias_array_fid   
        bias2_array = bias2_array_fid
    else:
        bias_array = np.array([param_dict['b'+str(i+1)] for i in range(nbin)]) 
        if  bias_model=='binned_quadratic':
            bias2_array = np.array([param_dict['b2_'+str(i+1)] for i in range(nbin)]) 
        else:
            bias2_array = np.zeros(nbin)



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
    #print(h, Omega_b, Omega_nu, Omega_m, fb)
    if Flag_Baryons_Model == True and (fb < 0.1 or fb > 0.25): #the baryon fraction is out of bounds
        return -1e10
    
    chi2 = 0. 
    if Flag_Probes=='3x2pt':
        Cov_theory, Cov_theory_high = compute_covariance_WL_3x2pt(Cosmo, Baryons, param_dict['aIA'], param_dict['etaIA'], betaIA=0., bias_array=bias_array, bias2_array=bias2_array, Delta_z=Delta_z_array, Flag_Model=Model, Flag_Baryons=Flag_Baryons_Model, Flag_Probes=Flag_Probes, bias_model=bias_model)
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
        #print('here')
        Cov_theory = compute_covariance_WL_3x2pt(Cosmo, Baryons, param_dict['aIA'], param_dict['etaIA'], betaIA=0., bias_array=bias_array, bias2_array=bias2_array, Delta_z=Delta_z_array, Flag_Model=Model, Flag_Baryons=Flag_Baryons_Model, Flag_Probes=Flag_Probes, bias_model=bias_model)
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
prior_dic={}
for par_i in ModelParsCosmo:
    if MasterPriors[emu_name][par_i]['type'] == 'G':
        prior.add_parameter(par_i, dist=norm(loc=MasterPriors[emu_name][par_i]['p1'] , scale=MasterPriors[emu_name][par_i]['p2']))
        prior_dic[par_i] = 'N('+str(MasterPriors[emu_name][par_i]['p1'])+', '+str(MasterPriors[emu_name][par_i]['p2'])+')'
    else:
        prior.add_parameter(par_i, dist=(MasterPriors[emu_name][par_i]['p1'] , MasterPriors[emu_name][par_i]['p2']))
        prior_dic[par_i] = '['+str(MasterPriors[emu_name][par_i]['p1'])+', '+str(MasterPriors[emu_name][par_i]['p2'])+']'
if Flag_Probes != 'GC' and Flag_Fix_IA==False:
    for par_i in ModelParsIA:
        if MasterPriors['IA'][par_i]['type'] == 'G':
            prior.add_parameter(par_i, dist=norm(loc=MasterPriors['IA'][par_i]['p1'] , scale=MasterPriors['IA'][par_i]['p2']))
            prior_dic[par_i] = 'N('+str(MasterPriors['IA'][par_i]['p1'])+', '+str(MasterPriors['IA'][par_i]['p2'])+')'
        else:
            prior.add_parameter(par_i, dist=(MasterPriors['IA'][par_i]['p1'] , MasterPriors['IA'][par_i]['p2']))
            prior_dic[par_i] = '['+str(MasterPriors['IA'][par_i]['p1'])+', '+str(MasterPriors['IA'][par_i]['p2'])+']'
if Flag_Baryons_Model == True:
    for par_i in ModelParsBC:
        if MasterPriors['BCemu'][par_i]['type'] == 'G':
            prior.add_parameter(par_i, dist=norm(loc=MasterPriors['BCemu'][par_i]['p1'] , scale=MasterPriors['BCemu'][par_i]['p2']))
            prior_dic[par_i] = 'N('+str(MasterPriors['BCemu'][par_i]['p1'])+', '+str(MasterPriors['BCemu'][par_i]['p2'])+')'
        else:
            prior.add_parameter(par_i, dist=(MasterPriors['BCemu'][par_i]['p1'] , MasterPriors['BCemu'][par_i]['p2']))
            prior_dic[par_i] = '['+str(MasterPriors['BCemu'][par_i]['p1'])+', '+str(MasterPriors['BCemu'][par_i]['p2'])+']'
if Flag_Probes != 'WL' and Flag_Fix_Bias==False:
    for i in range(nbin):
        if MasterPriors['b']['type'] == 'G':
            prior.add_parameter('b'+str(i+1), dist=norm(loc=MasterPriors['b']['p1'] , scale=MasterPriors['b']['p2']))
            prior_dic['b'+str(i+1)] = 'N('+str(MasterPriors['b']['p1'])+', '+str(MasterPriors['b']['p2'])+')'
        else:
            prior.add_parameter('b'+str(i+1), dist=(MasterPriors['b']['p1'] , MasterPriors['b']['p2']))
            prior_dic['b'+str(i+1)] = '['+str(MasterPriors['b']['p1'])+', '+str(MasterPriors['b']['p2'])+']'
        if  bias_model == 'binned_quadratic':
            if MasterPriors['b']['type'] == 'G':
                prior.add_parameter('b2_'+str(i+1), dist=norm(loc=MasterPriors['b2']['p1'] , scale=MasterPriors['b2']['p2']))
                prior_dic['b2_'+str(i+1)] = 'N('+str(MasterPriors['b2']['p1'])+', '+str(MasterPriors['b2']['p2'])+')'
            else:
                prior.add_parameter('b2_'+str(i+1), dist=(MasterPriors['b2']['p1'] , MasterPriors['b2']['p2']))
                prior_dic['b2_'+str(i+1)] = '['+str(MasterPriors['b2']['p1'])+', '+str(MasterPriors['b2']['p2'])+']'

#if Flag_Delta_z_model == True:
#    for i in range(nbin):
#        if MasterPriors['Delta_z']['type'] == 'G':
#            prior.add_parameter('Delta_z'+str(i+1), dist=norm(loc=MasterPriors['Delta_z']['p1'] , scale=MasterPriors['Delta_z']['p2']))
#        else:
#            prior.add_parameter('Delta_z'+str(i+1), dist=(MasterPriors['Delta_z']['p1'] , MasterPriors['Delta_z']['p2']))

output_header = f'Mock data is computed with {Model_fid} for {Cosmo_fid}' 
if Flag_Probes == '3x2pt' or Flag_Probes == 'GC':
    output_header += f'\n with bias_model_fid = {bias_model_fid}: bias_array_fid = {bias_array_fid}'
    if bias_model_fid == 'binned_quadratic':
        output_header += f' and bias2_array_fid = {bias2_array_fid}'
if Flag_Baryons_fid==True:
    output_header += f'\n and BCemu model with {Baryons_fid}' 
output_header += f'\n --------- \n'     
output_header += f'Model = {Model} with Euclid-like {Flag_Probes} probe: \n ModelParsCosmo = {ModelParsCosmo},  ModelParsIA = {ModelParsIA}'
if Flag_Baryons_Model==True:
    output_header += f'ModelParsBC = {ModelParsBC}'
if ('Mnu' in ModelParsCosmo)==False:
    output_header += f'\n fixed neutrinos with Mnu = {Mnu_model}'
if Flag_Probes == '3x2pt' or Flag_Probes == 'GC':
    output_header += f'  with bias_model = {bias_model}'     
output_header += f'\n --------- \n'  
output_header += f'Priors: \n {prior_dic}'  
output_header += f'\n --------- \n'  
output_header += f'points: ModelParsCosmo, ModelParsIA, (ModelParsBC), (b1...b10, b2_1...b2_10); log_w, log_l'  

if Flag_test==False:
    print('prior dimensionality: ', prior.dimensionality())
    start = time.time()
    sampler = Sampler(prior, loglikelihood_det, filepath='hdf5/'+hdf5_name+'.hdf5', n_live=3000, pool=14)
    sampler.run(verbose=True, discard_exploration=True, n_eff=30000)
    log_z = sampler.evidence()


    points, log_w, log_l = sampler.posterior()
    finish = time.time()
    print('time for mcmc')
    print(description, ' : ', finish-start)
    np.savetxt("chains/"+chain_name+".txt", np.c_[points, log_w, log_l], header=output_header)

np.savetxt("output_test.txt", np.zeros(4), header=output_header)
