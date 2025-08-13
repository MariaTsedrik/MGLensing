import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import MGLensing
import matplotlib.pyplot as plt
from numpy import array 
# Ensure the directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('figs/modelling', exist_ok=True)

from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
#Is this for a presentation?
isPrezi=False
SMALL_SIZE = 14
MEDIUM_SIZE = 20#22
BIGGER_SIZE = 28
VERY_SMALL= 14
if isPrezi:
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
else:
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('legend', fontsize=VERY_SMALL)

NL_MODEL_HMCODE = 0
NL_MODEL_BACCO = 1

NO_BARYONS = 0
BARYONS_HMCODE = 1
BARYONS_BCEMU = 2
BARYONS_BACCO = 3

BIAS_LIN = 0
BIAS_B1B2 = 1
BIAS_HEFT = 2
BIAS_HEFT_PNL = 3
BIAS_HEFT_PNL_BAR = 4

print('Ensure that likelihood type is binned in config.yaml')

MGL = MGLensing.MGL("config.yaml")
zz = MGL.Survey.zz_integr
nbin = MGL.Survey.nbin_l

l_wl_max, l_gc_max = MGL.Survey.ells_wl_max, MGL.Survey.ells_gc_max
l_wl, l_gc, l_xc = MGL.Survey.l_wl, MGL.Survey.l_gc, MGL.Survey.l_xc

print('l_gc_max: ', l_gc_max)

import matplotlib as mpl
import seaborn as sns
palette = sns.color_palette("viridis", as_cmap=True)

def plot_cells(type,  cl_gg_list, cl_xc_list, cl_wl_list, show=True):
    types = ['GG', 'XC', 'WL']
    ells = [l_gc, l_xc, l_wl]
    lmax = [l_gc[-1], l_xc[-1], l_wl[-1]]
    lmax_ij = [l_gc_max, l_gc_max, l_wl_max]
    cls = [cl_gg_list, cl_xc_list, cl_wl_list]
    errs = [err_cl_gg, err_cl_lg, err_cl_ll]
    
    c = [-0.5, 0., 0.5]
    
    fig, ax = plt.subplots(figsize=(8, 8), nrows=nbin, ncols=nbin, sharex=True, sharey=True,facecolor='w')
    
    for ind in range(len(cls[type])):
        for i in range(nbin):
            for j in range(nbin):
                if i < j:
                    ax[i, j].axis('off')
                else:    
                    ax[i, j].semilogx(ells[type], cls[type][ind][:, i, j]/cls[type][0][:, i, j], linestyle='--' if (ind == 2 or ind == 4) else '-')
    
    for i in range(nbin):
        for j in range(nbin):
                if i < j:
                    ax[i, j].axis('off')
                else:
                    err_prop = errs[type][:, i, j]/cls[type][0][:, i, j]
                    ax[i, j].fill_between(ells[type], np.ones(len(ells[type]))+err_prop, np.ones(len(ells[type]))-err_prop, color='tab:pink', alpha=0.1)
                    ax[i, j].axvspan(xmin=lmax_ij[type][i, j], xmax=lmax[type], color='grey', alpha=0.3)
                    ax[i, j].set_ylim(0.8, 1.01)
    for j in range(nbin):
        ax[-1, j].set_xlabel(r'$\ell$', fontsize=SMALL_SIZE)
    ax[2, 0].set_ylabel(r'$C_\ell^{\rm ' + types[type] + '}$ ratio', fontsize=SMALL_SIZE)
    fig.legend([#'heft no bar', 
        #'heft $\\times \\sqrt{S}$', 'heft $\\times 1$', 
        'without baryons', 'with baryons'
        #'heft+$P_{\\rm NL }\\times \\sqrt{S}$', 'heft+$P_{\\rm NL }$', 
        ], loc='upper right', ncol=1, bbox_to_anchor=(1, 0.93), bbox_transform=fig.transFigure, fontsize=20, frameon=False)    
    plt.tight_layout()
    plt.show() if show else plt.savefig('figs/modelling/c_ell/baryons/c_ells_' + types[type] + '_' + MGL.Survey.survey_name  + '_baryons_vs_errors_extreme.png', bbox_inches='tight')



params = {
    'Omega_m' :  0.315,
    'Omega_c' :  0.315-0.05,
    'Omega_cb' :  0.315,
    'Omega_nu':  0.,
    'As'      :  np.exp(3.07)*1.e-10,
    'sigma8_cb':  0.83,
    'Omega_b' :  0.05,
    'ns'      :  0.96,
    'h'       :  0.67,
    'Mnu'     :  0.0,
    'w0'      :  -1.0,
    'wa'      :  0.0,
    'a1_IA': 0.16,
    'eta1_IA': 1.66,
    'beta_IA': 0.,

    'log10T_AGN': 7.9,
    'log10Mc_bc': 15., #14.5, #15., #13.8,
    'eta_bc': 0.69, #-0.3,
    'beta_bc': 0.69, #-0.22,
    'log10Mz0_bc': 9., #10.5,
    'thetaout_bc': 0., #0.25,
    'thetainn_bc': -0.523, #-0.86,
    'log10Minn_bc': 9., #12.4,

    'M_c'           :  15.,
    'eta'           : 0.69,
    'beta'          : 0.69,
    'M1_z0_cen'     : 9.,
    'theta_out'     : 0.,
    'theta_inn'     : -0.523,
    'M_inn'         : 9,

    'omega_cold'    :  0.315,
    'sigma8_cold'   :  0.83,
    'omega_baryon'  :  0.05,
    'ns'            :  0.96,
    'hubble'        :  0.67,
    'neutrino_mass' :  0.0,
    'w0'            : -1.0,
    'wa'            :  0.0,
    'expfactor'     :  [1./(1.+1.5), 1./2, 1],
}

b0 = 0.68
bias1_arr = np.array([b0*(1.+z_bin_center_i) for z_bin_center_i in MGL.Survey.z_bin_center_l ])
bias2_arr = bias1_arr*2. 
print('bias1_arr , bias2_arr: ', bias1_arr , bias2_arr)
bias1_arr = np.array([1.239, 1.378, 1.525, 1.677, 1.832])
biasL1_arr = bias1_arr-1
#Lagrangian co-evolution 
#b2_arr = np.array([-0.258, -0.062, 0.107, 0.267, 0.462])
#biasL2_arr = b2_arr-8./21*biasL1_arr
#b2_arr = np.zeros(nbin)
#biasL2_arr = (0.9*biasL1_arr**2+0.5)-8./21*biasL1_arr
#biasL2_arr = np.zeros(nbin)
biasL2_arr = np.array([0.46036128, 0.4845956, 0.5480625, 0.65459134, 0.80604922])
#local-in-matter-density (LIMD) Lagrangian bias:
biasLs2_arr = np.zeros(nbin)
biasLlapl_arr = np.zeros(nbin) 
for bin_i in range(nbin):
    params[f'b1_{bin_i+1}']=biasL1_arr[bin_i]+1.
    params[f'b1L_{bin_i+1}']=biasL1_arr[bin_i]
    params[f'b2L_{bin_i+1}']=biasL2_arr[bin_i]
    params[f'bs2L_{bin_i+1}']=biasLs2_arr[bin_i]
    params[f'blaplL_{bin_i+1}']=biasLlapl_arr[bin_i]

err_cl_ll, err_cl_gg, err_cl_lg  = MGL.get_errorbars(params)

model={
    #'bacco_nobar_heft': {'nl_model': NL_MODEL_BACCO, 'bias_model': 2, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    #'bacco_bar_heft_sqrt': {'nl_model': NL_MODEL_BACCO, 'bias_model': 2, 'ia_model': 0, 'baryon_model': BARYONS_BACCO, 'cross_sqrt_baryon':True, 'photoz_err_model': 0.},
    #'bacco_bar_heft_nosqrt': {'nl_model': NL_MODEL_BACCO, 'bias_model': 2, 'ia_model': 0, 'baryon_model': BARYONS_BACCO, 'cross_sqrt_baryon':False, 'photoz_err_model': 0.},
    #'bacco_bar_heftpnls': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_HEFT_PNL, 'ia_model': 0, 'baryon_model': BARYONS_BACCO, 'cross_sqrt_baryon':False, 'photoz_err_model': 0.},
    #'bacco_bar_heftpnl': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_HEFT_PNL_BAR, 'ia_model': 0, 'baryon_model': BARYONS_BACCO, 'cross_sqrt_baryon':False, 'photoz_err_model': 0.},

    'bacco_bar_heft_nosqrt': {'nl_model': NL_MODEL_BACCO, 'bias_model': 2, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'cross_sqrt_baryon':False, 'photoz_err_model': 0.},
    'bacco_bar_heft_sqrt': {'nl_model': NL_MODEL_BACCO, 'bias_model': 2, 'ia_model': 0, 'baryon_model': BARYONS_BACCO, 'cross_sqrt_baryon':True, 'photoz_err_model': 0.},
    
    }


cls_list = []
cls_xc_list = []
cls_wl_list = []

for key_i in model.keys():
    wl, gg, xc, _ = MGL.get_c_ells(params, model[key_i])
    cls_list.append(gg)
    cls_xc_list.append(xc)
    cls_wl_list.append(wl)

plot_cells(2, cls_list, cls_xc_list, cls_wl_list, False)
"""

import baccoemu 

emulator = baccoemu.Matter_powerspectrum()
k = np.logspace(-2, np.log10(0.7), 10, endpoint=True)
k, S = emulator.get_baryonic_boost(k=k, **params)
print(np.sqrt(S))
"""