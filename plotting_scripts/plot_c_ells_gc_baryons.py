import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import MGLensing
import matplotlib.pyplot as plt
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

print('Ensure that likelihood type is binned in config.yaml')

MGL = MGLensing.MGL("config.yaml")
zz = MGL.Survey.zz_integr
nbin = MGL.Survey.nbin_l

l_wl_max, l_gc_max = MGL.Survey.ells_wl_max, MGL.Survey.ells_gc_max
l_wl, l_gc, l_xc = MGL.Survey.l_wl, MGL.Survey.l_gc, MGL.Survey.l_xc

print('l_gc_max: ', l_gc_max)

def plot_cells_ratio(type,  cl_gg_list, show=True, names="", annotation=""):
    types = ['GG']
    ells = [l_gc]
    lmax = [l_gc[-1]]
    lmax_ij = [l_gc_max]
    cls = [cl_gg_list]
    errs = [err_cl_gg]
    fig, ax = plt.subplots(figsize=(8, 8), nrows=nbin, ncols=nbin, sharex=True, sharey=True, facecolor='w')
    for ind, name in enumerate(names):
        for i in range(nbin):
            for j in range(nbin):
                if i < j:
                    ax[i, j].axis('off')
                else:
                    err_prop = errs[type][:, i, j]*cls[type][ind][:, i, j]/cls[type][0][:, i, j]*(1./cls[type][ind][:, i, j])
                    ax[i, j].semilogx(ells[type], cls[type][ind+1][:, i, j]/cls[type][0][:, i, j], label=name if i == 4 and j == 0 else "")
                    ax[i, j].axvspan(xmin=lmax_ij[type][i, j], xmax=lmax[type], color='grey', alpha=0.1)
                    ax[i, j].fill_between(ells[type], np.ones(len(ells[type]))+err_prop, np.ones(len(ells[type]))-err_prop, color='tab:pink', alpha=0.1)
                    ax[i, j].legend(loc='lower left', title_fontsize=10, title='bin ' + str(i + 1) + '-' + str(j + 1))
    for i in range(nbin):
        ax[nbin - 1][i].set_xlabel('$\ell$')
        ax[nbin - 1][i].set_ylim(0.9, 1.05)
    ax[int(nbin / 2)][0].set_ylabel('$C^{\\rm ' + types[type] + '}_{\ell}$ ratio')
    ax[1, 1].annotate(annotation, (1.1, 0.05), xycoords='axes fraction', clip_on=False)
    plt.tight_layout()
    plt.show() if show else plt.savefig('figs/modelling/c_ells_' + types[type] + '_' + MGL.Survey.survey_name + '_ratio_bs2.pdf', bbox_inches='tight')

import matplotlib as mpl
import seaborn as sns
palette = sns.color_palette("viridis", as_cmap=True)
def plot_cells_ratio_diag(type,  cl_gg_list, b_arr, label_bias="", show=True):
    types = ['GG']
    ells = [l_gc]
    lmax = [l_gc[-1]]
    lmax_ij = [l_gc_max]
    cls = [cl_gg_list]
    errs = [err_cl_gg]
    
    c = [-0.5, 0., 0.5]
    
    fig, ax = plt.subplots(figsize=(10, 4), nrows=1, ncols=nbin, sharex=True, sharey=True, facecolor='w')
    
    norm = mpl.colors.Normalize(vmin=b_arr[0], vmax=b_arr[-1])
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=palette)#mpl.cm.jet)
    cmap.set_array([])
    for ind in range(len(cls[type])-2):
        print(ind)
        for i in range(nbin):
            j=i
            #err_prop = errs[type][:, i, j]*cls[type][ind][:, i, j]/cls[type][0][:, i, j]*(1./cls[type][ind][:, i, j])
            ax[i].semilogx(ells[type], cls[type][ind+2][:, i, j]/cls[type][0][:, i, j], c=cmap.to_rgba(b_arr[ind]))
            #ax[i].fill_between(ells[type], np.ones(len(ells[type]))+err_prop, np.ones(len(ells[type]))-err_prop, color='tab:pink', alpha=0.1)
            ax[i].legend(loc='lower left', title_fontsize=10, title='bin ' + str(i + 1) + '-' + str(j + 1))
    for i in range(nbin):
        j=i
        err_prop = errs[type][:, i, j]*cls[type][2][:, i, j]/cls[type][0][:, i, j]*(1./cls[type][ind][:, i, j])
        ax[i].fill_between(ells[type], np.ones(len(ells[type]))+err_prop, np.ones(len(ells[type]))-err_prop, color='tab:pink', alpha=0.3)
            
        ax[i].semilogx(ells[type], cls[type][1][:, i, j]/cls[type][0][:, i, j], color='tab:red')
        ax[i].semilogx(ells[type], cls[type][0][:, i, j]/cls[type][0][:, i, j], color='k')
        ax[i].set_xlabel('$\ell$')
        ax[i].set_ylim(0.9, 1.05)
        ax[i].axvspan(xmin=lmax_ij[type][i, j], xmax=lmax[type], color='grey', alpha=0.3)
    ax[0].set_ylabel('$C^{\\rm ' + types[type] + '}_{\ell}$ ratio')
    #plt.tight_layout()
    fig.colorbar(cmap, ax=ax, label=label_bias)   
    plt.show() if show else plt.savefig('figs/modelling/c_ell/baryons/c_ells_' + types[type] + '_' + MGL.Survey.survey_name  + '_ratio_bLaplace.pdf', bbox_inches='tight')



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
    'log10Mc_bc': 14.5, #15., #13.8,
    'eta_bc': -0.3,
    'beta_bc': -0.22,
    'log10Mz0_bc': 10.5,
    'thetaout_bc': 0.25,
    'thetainn_bc': -0.86,
    'log10Minn_bc': 12.4,
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
b2_arr = np.zeros(nbin)
#biasL2_arr = (0.9*biasL1_arr**2+0.5)-8./21*biasL1_arr
biasL2_arr = np.zeros(nbin)
#local-in-matter-density (LIMD) Lagrangian bias:
biasLs2_arr = np.zeros(nbin)
biasLlapl_arr = np.zeros(nbin) 
for bin_i in range(nbin):
    params[f'b1_{bin_i+1}']=bias1_arr[bin_i]
    params[f'b2_{bin_i+1}']=bias2_arr[bin_i]
    params[f'b1L_{bin_i+1}']=biasL1_arr[bin_i]
    params[f'b2L_{bin_i+1}']=biasL2_arr[bin_i]
    params[f'bs2L_{bin_i+1}']=biasLs2_arr[bin_i]
    params[f'blaplL_{bin_i+1}']=biasLlapl_arr[bin_i]

err_cl_ll, err_cl_gg, err_cl_lg  = MGL.get_errorbars(params)

model={
    #'bacco_bar': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': BARYONS_BACCO, 'photoz_err_model': 0.},
    #'bacco_nobar': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    'bacco_bar': {'nl_model': NL_MODEL_BACCO, 'bias_model': 4, 'ia_model': 0, 'baryon_model': BARYONS_BACCO, 'photoz_err_model': 0.},   # ref
    'bacco_nobar': {'nl_model': NL_MODEL_BACCO, 'bias_model': 3, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},   # den
    'bacco_nobar_heft': {'nl_model': NL_MODEL_BACCO, 'bias_model': 3, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.}
    }

cls_fid = MGL.get_c_ells(params, model['bacco_nobar'])[1]
cls_fid_bar = MGL.get_c_ells(params, model['bacco_bar'])[1]
cls_list = []
cls_list.append(cls_fid)
cls_list.append(cls_fid_bar)
n_points = 30
vary_bias = np.linspace(0., 0.5, n_points)
for b_i in vary_bias:
    params_new = params.copy()
    #local-in-matter-density (LIMD) Lagrangian bias:
    biasLs2_arr = np.zeros(nbin)
    #biasLs2_arr = b_i*np.ones(nbin)
    #biasLlapl_arr = np.zeros(nbin) 
    biasLlapl_arr = b_i*np.ones(nbin)
    for bin_i in range(nbin):
        params_new[f'bs2L_{bin_i+1}']=biasLs2_arr[bin_i]
        params_new[f'blaplL_{bin_i+1}']=biasLlapl_arr[bin_i]
    cls_list.append(MGL.get_c_ells(params_new, model['bacco_nobar_heft'])[1])
"""
vary_Mc = np.linspace(9., 15., n_points)
for Mc_i in vary_Mc:
    params_new = params.copy()
    params_new['log10Mc_bc'] = Mc_i
    cls_list.append(MGL.get_c_ells(params_new, model['bacco_bar'])[1])

print(cls_list)
print(np.array(cls_list).shape)

plot_cells_ratio(0, cls_list, True, names=[str(vary_Mc[i]) for i in range(n_points)])
"""

plot_cells_ratio_diag(0, cls_list, vary_bias, "$b_{\\nabla}$", False)
