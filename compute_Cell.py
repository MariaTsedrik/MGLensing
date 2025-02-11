from scipy.integrate import trapz,simpson, quad
from scipy import interpolate as itp
from scipy.interpolate import interp1d, RectBivariateSpline, InterpolatedUnivariateSpline, CubicSpline
from scipy.special import erf
import numpy as np
import MGrowth as mg
from multiprocessing import Pool
import os
import baccoemu as bacco
#import pyhmcode 
import cosmopower as cp
from cosmopower import cosmopower_NN
from scipy.stats import norm
import BCemu
import math
from compute_power_spectra import *
from setup_and_priors import *
from compute_baryonic_boost  import *

################################################
#               COSMOLOGY                      #
################################################
###compute expansion and comoving distance###
def Ez_rz(CosmoDict, zz):
    omega0 = CosmoDict['Omega_m']
    w0 = CosmoDict['w0'] if 'w0' in CosmoDict else -1.
    wa = CosmoDict['wa'] if 'wa' in CosmoDict else 0.
    H0_h_c = 1./2997.92458 #=100/c in Mpc/h
    omegaL_func = lambda z: (1.-omega0) * pow(1.+z, 3.*(1.+w0+wa)) * np.exp(-3.*wa*z/(1.+z))
    E_z_func = lambda z: np.sqrt(omega0*pow(1.+z, 3) + omegaL_func(z))
    E_z_grid = np.array([E_z_func(zz_i) for zz_i in zz])
    r_z_int = lambda z: 1./np.sqrt(omega0*pow(1.+z, 3) + omegaL_func(z))
    r_z_func = lambda z_in: quad(r_z_int, 0, z_in)[0]
    r_z_grid = np.array([r_z_func(zz_i) for zz_i in zz])/H0_h_c #Mpc/h
    return E_z_grid, r_z_grid

################################################
#               MODELLING                      #
################################################
H0_h_c = 1./2997.92458
a_mu_interp = np.linspace(1e-3, 1, 256)
# Constants
nnode = 10
zstart, ztanh = 3.0, 4.0
astart, atanh, aend = 1.0 / (1.0 + zstart), 1.0 / (1.0 + ztanh), 1.0
# Arrays
a_arr = np.zeros(2 * nnode, dtype=np.float64)
mu_arr = np.zeros(2 * nnode, dtype=np.float64)
for i in range(1, nnode + 1):
        a_arr[i - 1] = atanh * float(i - 1) / float(nnode - 1)
        a_arr[nnode + i - 1] = astart + (1.0 - astart) * float(i - 1) / float(nnode - 1)
a_arr[0]=1e-3


def mu_recon(a, mu11):
    for i in range(nnode+1):
        mu_arr[nnode-1+i]=mu11[i]
    for j in range(nnode-1):
        mu_arr[j] = (mu_arr[nnode-1] - 1.0) / 2.0 * (1.0 + np.tanh((a_arr[j] - atanh / 2.0) / 0.04)) + 1.0
    # Spline interpolation
    spline_mu = CubicSpline(a_arr, mu_arr)
    return(spline_mu(a))

def compute_WL(Cosmo, Baryons, aIA, etaIA, betaIA=0., Flag_Model=False, Flag_Baryons=False):
    Ez, rz = Ez_rz(Cosmo, zz_integr)
    k =(l_WL[:,None]+0.5)/rz
    #Omega_nu = Cosmo['Mnu']/(93.14*Cosmo['h']**2)
    #Omega_cdmb = Cosmo['Omega_m']-Omega_nu
    Omega_m =  Cosmo['Omega_m']
    background ={'Omega_m': Omega_m,#Omega_cdmb,
            'h' : Cosmo['h'],
            'w0': Cosmo['w0'] if 'w0' in Cosmo else -1.,
            'wa': Cosmo['wa'] if 'wa' in Cosmo else 0.,
            'a_arr': np.hstack((aa_integr, 1.))}

    pk_m_l  = np.zeros((lbin, zbin_integr), 'float64')
    index_pknn = np.array(np.where((k > k_min_h_by_Mpc) & (k<k_max_h_by_Mpc))).transpose()
    if Flag_Model=='HMcode':
        Pk_l_interp = call_cambhmcode(Cosmo) 
        cosmo1 = mg.w0waCDM(background)   
        Da, _ = cosmo1.growth_parameters() 
    elif Flag_Model=='nDGP':  
        Pk_l_interp = call_hmcode_with_boost_nDGP(Cosmo)
        cosmo1 = mg.nDGP(background)   
        Da, _ = cosmo1.growth_parameters(omegarc=10**Cosmo['log10omegarc'])   
    elif Flag_Model=='GQ':
        Pk_l_interp = call_hmcode_with_boost_gamma(Cosmo) 
        cosmo1 = mg.Linder_gamma(background)   
        Da, _ = cosmo1.growth_parameters(gamma=Cosmo['gamma']) 
    elif Flag_Model=='GzQ':      
        Pk_l_interp = call_hmcode_with_boost_gamma_z(Cosmo)  
        cosmo1 = mg.Linder_gamma_a(background)   
        Da, _ = cosmo1.growth_parameters(gamma0=Cosmo['gamma0'], gamma1=Cosmo['gamma1']) 
    elif Flag_Model=='MuSpline':
        Pk_l_interp = call_hmcode_with_boost_mu_spline(Cosmo)
        cosmo1 = mg.mu_a(background) 
        mu11_arr = np.array([Cosmo['mu1'], Cosmo['mu2'], Cosmo['mu3'], Cosmo['mu4'], Cosmo['mu5'], Cosmo['mu6'], Cosmo['mu7'], Cosmo['mu8'], Cosmo['mu9'], Cosmo['mu10'], Cosmo['mu11']])
        mu_all = mu_recon(a_mu_interp, mu11_arr[::-1])
        mu_interpolator = interp1d(a_mu_interp, mu_all, bounds_error=False,
                        kind='cubic',
                        fill_value=(mu_all[0], mu_all[-1])) 
        Da, _ = cosmo1.growth_parameters(mu_interp=mu_interpolator)

    for index_l, index_z in index_pknn:
        pk_m_l[index_l, index_z] = Pk_l_interp(zz_integr[index_z], k[index_l,index_z])
    Pk = pk_m_l
    
    ###Add baryonic feedback 
    if Flag_Baryons == True:
        boost_BCemu = np.zeros((lbin, zbin_integr), 'float64')
        boost_BCemu_interp = call_BCemu(Cosmo, Baryons)
        for index_l, index_z in index_pknn:
            boost_BCemu[index_l, index_z] = boost_BCemu_interp(min(zz_integr[index_z], 2), k[index_l,index_z])
        Pk *= boost_BCemu

    ###Window functions W_L(l,z,bin) in units of [W] = h/Mpc

    integrand = 3./2.*H0_h_c**2. * Omega_m * rz[None,:,None]*(1.+zz_integr[None,:,None])*eta_z.T[:,None,:]*(1.-rz[None,:,None]/rz[None,None,:])
    W_gamma  = np.trapz(np.triu(integrand),zz_integr,axis=-1).T

    ######zNLA############
    # Compute contribution from IA (Intrinsic Alignement)
    # - compute window function W_IA
    W_IA = eta_z * Ez[:,None] * H0_h_c
    # - IA contribution depends on a few parameters assigned here
    # fiducial values {a, eta, beta} = {1.72, -0.41, 0.0}
    C_IA = 0.0134
    A_IA = aIA
    eta_IA = etaIA
    beta_IA = betaIA

    # - compute functions F_IA(z) and D(z)
    F_IA = (1.+zz_integr)**eta_IA * (lum_func(zz_integr))**beta_IA


    Dz = Da[::-1] #should be normalised to z=0
    Dz = Dz[1:]/Dz[0]
    Dz= Dz[None,:]
    W_L = W_gamma[None,:,:] - A_IA*C_IA*Omega_m*F_IA[None,:,None]/Dz[:,:,None] * W_IA[None,:,:]

    Cl_LL_int = W_L[:,:,:,None] * W_L[:,:,None,:] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
    Cl_LL     = trapz(Cl_LL_int,zz_integr,axis=1)[:nell_WL,:,:]
    noise = {
    'LL': rms_shear**2./n_bar,
    'LG': 0.,
    'GL': 0.,
    'GG': 1./n_bar}
    for i in range(nbin):
        Cl_LL[:,i,i] += noise['LL']   
    return Cl_LL


def compute_covariance(Cosmo, Baryons, aIA, etaIA, betaIA=0., Flag_Model='HMcode', Flag_Baryons=False):
    spline_LL = np.empty((nbin, nbin),dtype=(list,3))
    Cl_LL = compute_WL(Cosmo, Baryons, aIA, etaIA, betaIA, Flag_Model, Flag_Baryons)
    for Bin1 in range(nbin):
        for Bin2 in range(nbin):
            spline_LL[Bin1,Bin2] = list(itp.splrep(l_WL[:], Cl_LL[:,Bin1,Bin2]))

    Cov_theory = np.zeros((len(ells_WL), nbin, nbin), 'float64')       
    for Bin1 in range(nbin):
        for Bin2 in range(nbin):  
            Cov_theory[:,Bin1,Bin2] = itp.splev(ells_WL[:], spline_LL[Bin1,Bin2]) 
    return  Cov_theory   


def compute_covariance_WL_3x2pt(Cosmo, Baryons, aIA, etaIA, betaIA=0., bias_array=np.ones(nbin), bias2_array=np.zeros(nbin), Delta_z=None, Flag_Model=False, Flag_Baryons=False, Flag_Probes='WL', Flag_savePk=False, bias_model='interpld'):
    Ez, rz = Ez_rz(Cosmo, zz_integr)
    k =(l_WL[:,None]+0.5)/rz

    Omega_m =  Cosmo['Omega_m']
    background ={'Omega_m': Omega_m,#Omega_cdmb,
            'h' : Cosmo['h'],
            'w0': Cosmo['w0'] if 'w0' in Cosmo else -1.,
            'wa': Cosmo['wa'] if 'wa' in Cosmo else 0.,
            'a_arr': np.hstack((aa_integr, 1.))}

    pk_m_l  = np.zeros((lbin, zbin_integr), 'float64')
    index_pknn = np.array(np.where((k > k_min_h_by_Mpc) & (k<k_max_h_by_Mpc))).transpose()
    if Flag_Model=='HMcode':
        Pk_l_interp = call_cambhmcode(Cosmo) 
        cosmo1 = mg.w0waCDM(background)   
        Da, _ = cosmo1.growth_parameters() 
    elif Flag_Model=='nDGP':  
        Pk_l_interp = call_hmcode_with_boost_nDGP(Cosmo)
        cosmo1 = mg.nDGP(background)   
        Da, _ = cosmo1.growth_parameters(omegarc=10**Cosmo['log10omegarc'])   
    elif Flag_Model=='GQ':
        Pk_l_interp = call_hmcode_with_boost_gamma(Cosmo) 
        cosmo1 = mg.Linder_gamma(background)   
        Da, _ = cosmo1.growth_parameters(gamma=Cosmo['gamma']) 
    elif Flag_Model=='GzQ':      
        Pk_l_interp = call_hmcode_with_boost_gamma_z(Cosmo)  
        cosmo1 = mg.Linder_gamma_a(background)   
        Da, _ = cosmo1.growth_parameters(gamma0=Cosmo['gamma0'], gamma1=Cosmo['gamma1']) 
    elif Flag_Model=='MuSpline':
        Pk_l_interp = call_hmcode_with_boost_mu_spline(Cosmo)
        cosmo1 = mg.mu_a(background) 
        mu11_arr = np.array([Cosmo['mu1'], Cosmo['mu2'], Cosmo['mu3'], Cosmo['mu4'], Cosmo['mu5'], Cosmo['mu6'], Cosmo['mu7'], Cosmo['mu8'], Cosmo['mu9'], Cosmo['mu10'], Cosmo['mu11']])
        mu_all = mu_recon(a_mu_interp, mu11_arr[::-1])
        mu_interpolator = interp1d(a_mu_interp, mu_all, bounds_error=False,
                        kind='cubic',
                        fill_value=(mu_all[0], mu_all[-1])) 
        Da, _ = cosmo1.growth_parameters(mu_interp=mu_interpolator)

    for index_l, index_z in index_pknn:
        pk_m_l[index_l, index_z] = Pk_l_interp(zz_integr[index_z], k[index_l,index_z])
    Pk = pk_m_l     
    
    ###Add baryonic feedback 
    if Flag_Baryons == True:
        boost_BCemu = np.zeros((lbin, zbin_integr), 'float64')
        boost_BCemu_interp = call_BCemu(Cosmo, Baryons)
        for index_l, index_z in index_pknn:
            boost_BCemu[index_l, index_z] = boost_BCemu_interp(min(zz_integr[index_z], 2), k[index_l,index_z])
        Pk *= boost_BCemu

    dic_Pk = {}
    if Flag_savePk==True:
        dic_Pk['Pmm']=pk_m_l 
        dic_Pk['Pmm_baryons']=Pk
        dic_Pk['ell']=l_WL 
        dic_Pk['r_com']=rz 
        dic_Pk['k']=k

    noise = {
    'LL': rms_shear**2./n_bar,
    'LG': 0.,
    'GL': 0.,
    'GG': 1./n_bar}

    if Delta_z!=None:
        for Bin in range(nbin):
            for nz in range(zbin_integr):
                z = zz_integr[nz]-Delta_z[Bin]
                photoerror_z[nz,Bin] = photo_z_distribution(z,Bin+1)
                eta_z[nz, Bin] = photoerror_z[nz,Bin] * galaxy_distribution(z)
        for Bin in range(nbin):
            eta_z[:,Bin] /= trapz(eta_z[:,Bin],zz_integr[:]) 

    ###Window functions W_L(l,z,bin) in units of [W] = h/Mpc
    if Flag_Probes=='WL' or Flag_Probes=='3x2pt':
        integrand = 3./2.*H0_h_c**2. * Omega_m * rz[None,:,None]*(1.+zz_integr[None,:,None])*eta_z.T[:,None,:]*(1.-rz[None,:,None]/rz[None,None,:])
        W_gamma  = np.trapz(np.triu(integrand),zz_integr,axis=-1).T
        # Compute contribution from IA (Intrinsic Alignement)
        # - compute window function W_IA
        W_IA = eta_z * Ez[:,None] * H0_h_c

        # - IA contribution depends on a few parameters assigned here
        # fiducial values {a, eta, beta} = {1.72, -0.41, 0.0}
        C_IA = 0.0134
        A_IA = aIA
        eta_IA = etaIA
        beta_IA = betaIA

        # - compute functions F_IA(z) and D(z)
        F_IA = (1.+zz_integr)**eta_IA * (lum_func(zz_integr))**beta_IA

    
        Dz = Da[::-1] #should be normalised to z=0
        Dz = Dz[1:]/Dz[0]
        Dz= Dz[None,:]
        W_L = W_gamma[None,:,:] - A_IA*C_IA*Omega_m*F_IA[None,:,None]/Dz[:,:,None] * W_IA[None,:,:]

        Cl_LL_int = W_L[:,:,:,None] * W_L[:,:,None,:] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        Cl_LL     = trapz(Cl_LL_int,zz_integr,axis=1)[:nell_WL,:,:]
        for i in range(nbin):
            Cl_LL[:,i,i] += noise['LL']

        spline_LL = np.empty((nbin, nbin),dtype=(list,3))
        for Bin1 in range(nbin):
            for Bin2 in range(nbin):
                spline_LL[Bin1,Bin2] = list(itp.splrep(
                    l_WL[:], Cl_LL[:,Bin1,Bin2]))    


    if Flag_Probes=='GC' or Flag_Probes=='3x2pt':
        # Compute window function W_G(z) of galaxy clustering for each bin:
        bias = np.zeros((nbin),'float64')
        # - case where there is one constant bias value b_i for each bin i
        if bias_model == 'binned_constant':
            bias = bias_array
            W_G = np.zeros((zbin_integr, nbin), 'float64')
            W_G = bias[None,:] * Ez[:,None] * H0_h_c * eta_z
            Cl_GG_int = W_G[None,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
            Cl_LG_int = W_L[:,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        
        # - case where the bias is a single function b(z) for all bins
        elif bias_model == 'interpld':
            W_G = np.zeros((zbin_integr, nbin), 'float64')
            biasfunc = interp1d(z_bin_center, bias_array, bounds_error=False, fill_value="extrapolate")
            W_G =  (Ez * H0_h_c * biasfunc(zz_integr))[:,None] * eta_z    
            Cl_GG_int = W_G[None,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
            Cl_LG_int = W_L[:,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        
        elif bias_model == 'binned_quadratic':
            bias = bias_array
            bias2 = bias2_array
            W_G = np.zeros((lbin, zbin_integr, nbin), 'float64')
            W_G = ( bias[None, None,:] + bias2[None, None, :]*k[:, :, None]**2 ) * Ez[None, :,None] * H0_h_c * eta_z[None, :, :]
            Cl_GG_int = W_G[:,:,:,None] * W_G[:,: , None, :] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
            Cl_LG_int = W_L[:,:,:,None] * W_G[:,: , None, :] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        

        Cl_GG     = trapz(Cl_GG_int,zz_integr,axis=1)[:nell_GC,:,:]
        for i in range(nbin):
            Cl_GG[:,i,i] += noise['GG']
        
        spline_GG = np.empty((nbin, nbin), dtype=(list,3))
        for Bin1 in range(nbin):
            for Bin2 in range(nbin):
                spline_GG[Bin1,Bin2] = list(itp.splrep(
                    l_GC[:], Cl_GG[:,Bin1,Bin2]))

    if Flag_Probes=='3x2pt':
        #Cl_LG_int = W_L[:,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        Cl_LG     = trapz(Cl_LG_int,zz_integr,axis=1)[:nell_XC,:,:]
        Cl_GL     = np.transpose(Cl_LG,(0,2,1))

        spline_LG = np.empty((nbin, nbin), dtype=(list,3))
        spline_GL = np.empty((nbin, nbin), dtype=(list,3))
        for Bin1 in range(nbin):
            for Bin2 in range(nbin):
                spline_LG[Bin1,Bin2] = list(itp.splrep(
                    l_XC[:], Cl_LG[:,Bin1,Bin2]))
                spline_GL[Bin1,Bin2] = list(itp.splrep(
                    l_XC[:], Cl_GL[:,Bin1,Bin2]))
      
    ######################################################
    if Flag_Probes=='3x2pt':
        Cov_theory = np.zeros((ell_jump, 2*nbin, 2*nbin), 'float64')
        Cov_theory_high = np.zeros(((len(ells_WL)-ell_jump), nbin, nbin), 'float64')    
        for Bin1 in range(nbin):
            for Bin2 in range(nbin):
                Cov_theory[:,Bin1,Bin2] = itp.splev(
                    ells_GC[:], spline_LL[Bin1,Bin2])
                Cov_theory[:,nbin+Bin1,Bin2] = itp.splev(
                    ells_GC[:], spline_GL[Bin1,Bin2])
                Cov_theory[:,Bin1,nbin+Bin2] = itp.splev(
                    ells_GC[:], spline_LG[Bin1,Bin2])
                Cov_theory[:,nbin+Bin1,nbin+Bin2] = itp.splev(
                    ells_GC[:], spline_GG[Bin1,Bin2])
                Cov_theory_high[:,Bin1,Bin2] = itp.splev(
                    ells_WL[ell_jump:], spline_LL[Bin1,Bin2])
        if Flag_savePk==True:
            dic_Pk['Cl_LL'] = Cl_LL
            dic_Pk['Cl_GG'] = Cl_GG
            dic_Pk['Cl_LG'] = Cl_LG
            return Cov_theory, Cov_theory_high, dic_Pk
        else:         
            return Cov_theory, Cov_theory_high
    
    elif Flag_Probes=='WL':
        Cov_theory = np.zeros((len(ells_WL), nbin, nbin), 'float64')
        for Bin1 in range(nbin):
            for Bin2 in range(nbin):  
                Cov_theory[:,Bin1,Bin2] = itp.splev(ells_WL[:], spline_LL[Bin1,Bin2]) 
        return Cov_theory  
          
    elif Flag_Probes=='GC':
        Cov_theory = np.zeros((len(ells_GC), nbin, nbin), 'float64')
        for Bin1 in range(nbin):
            for Bin2 in range(nbin):  
                Cov_theory[:,Bin1,Bin2] = itp.splev(ells_GC[:], spline_GG[Bin1,Bin2]) 
        return Cov_theory        
        
        