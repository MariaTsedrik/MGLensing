import BCemu
import math
from scipy.integrate import trapz,simpson, quad
from scipy import interpolate as itp
from scipy.interpolate import interp1d, RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.special import erf
import numpy as np
from setup_and_priors import *


###for baryonic feedback###
BCemu_k_bins = 200
BCemu_z_bins = 20

bfcemu = BCemu.BCM_7param(verbose=False)
#bfcemu = BCemu.BCM_3param(verbose=False)

kmin_bcemu = 0.0342
kmax_bcemu = 12.51
k_bfc = np.logspace(np.log10(kmin_bcemu), np.log10(kmax_bcemu), BCemu_k_bins)
z_bfc = np.linspace(0., min(2, zz_integr[-1]), BCemu_z_bins)

def call_BCemu(CosmoDict, BarDict):
    log10Mc = BarDict['log10Mc']
    thej = BarDict['thej']
    mu = BarDict['mu']
    gamma_bc = BarDict['gamma_bc']
    delta = BarDict['delta']
    eta = BarDict['eta']
    deta = BarDict['deta']
    bcemu_dict ={
           'log10Mc' : log10Mc,
           'mu'     : mu,
           'thej'   : thej,  
           'gamma'  : gamma_bc,
           'delta'  : delta, 
           'eta'    : eta, 
           'deta'   : deta, 
            }

    fb = CosmoDict['fb']
    ###last element extrapolation###
    Boost = [bfcemu.get_boost(z_i,bcemu_dict,k_bfc,fb) for z_i in z_bfc]
    Boost_itp = [interp1d(k_bfc, Boost[i], bounds_error=False,
                kind='cubic',
                fill_value=(Boost[i][0], Boost[i][-1])) for i in range(BCemu_z_bins)]
    Boost_k = np.array([Boost_itp[z](kL) for z in range(BCemu_z_bins)], dtype=np.float64)
    BFC_interpolator = RectBivariateSpline(z_bfc, kL, Boost_k)
    return BFC_interpolator   