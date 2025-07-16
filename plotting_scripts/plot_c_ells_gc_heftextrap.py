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

print('Ensure that likelihood type is binned in config.yaml')

MGL = MGLensing.MGL("config.yaml")
zz = MGL.Survey.zz_integr
nbin = MGL.Survey.nbin

l_wl_max, l_gc_max = MGL.Survey.ells_wl_max, MGL.Survey.ells_gc_max
l_wl, l_gc, l_xc = MGL.Survey.l_wl, MGL.Survey.l_gc, MGL.Survey.l_xc

print('l_gc_max: ', l_gc_max)

import matplotlib as mpl
import seaborn as sns
palette = sns.color_palette("viridis", as_cmap=True)
def plot_cells_diag(type,  cl_gg_list, show=True):
    types = ['GG']
    ells = [l_gc]
    lmax = [l_gc[-1]]
    lmax_ij = [l_gc_max]
    cls = [cl_gg_list]
    errs = [err_cl_gg]
    
    c = [-0.5, 0., 0.5]
    
    fig, ax = plt.subplots(figsize=(14, 4), nrows=1, ncols=nbin, sharex=True, sharey=True, facecolor='w')
    
    for ind in range(len(cls[type])-1):
    #for ind in range(len(cls[type])):    
        print(ind)
        for i in range(nbin):
            j=i
            #err_prop = errs[type][:, i, j]*cls[type][ind][:, i, j]/cls[type][0][:, i, j]*(1./cls[type][ind][:, i, j])
            #ax[i].semilogx(ells[type], cls[type][ind+1][:, i, j]/cls[type][0][:, i, j], label=str(ind+1) if i==0 else "")
            ax[i].loglog(ells[type], cls[type][ind][:, i, j], label=str(ind+1) if i==0 else "")
            ax[i].loglog(ells[type], cls_heft_extrap[ind][:, i, j], linestyle='--')
            #ax[i].fill_between(ells[type], np.ones(len(ells[type]))+err_prop, np.ones(len(ells[type]))-err_prop, color='tab:pink', alpha=0.1)
    for i in range(nbin):
        j=i
        ax[i].loglog(ells[type], cls[type][-1][:, i, j], label="chain-breaker" if i==0 else "", color='tab:red')
        #err_prop = errs[type][:, i, j]*cls[type][1][:, i, j]/cls[type][0][:, i, j]*(1./cls[type][ind][:, i, j])
        #ax[i].fill_between(ells[type], np.ones(len(ells[type]))+err_prop, np.ones(len(ells[type]))-err_prop, color='tab:pink', alpha=0.3)
        #ax[i].semilogx(ells[type], cls[type][0][:, i, j]/cls[type][0][:, i, j], color='k')
        ax[i].set_xlabel('$\ell$')
        #ax[i].set_ylim(-10., 10.)
        ax[i].axvspan(xmin=lmax_ij[type][i, j], xmax=lmax[type], color='grey', alpha=0.3)
        ax[i].legend(loc='lower left', title_fontsize=10, title='bin ' + str(i + 1) + '-' + str(j + 1))
    ax[0].set_ylabel('$C^{\\rm ' + types[type] + '}_{\ell}$ ratio')
    plt.tight_layout()
    plt.show() if show else plt.savefig('figs/modelling/c_ell/baryons/c_ells_' + types[type] + '_' + MGL.Survey.survey_name  + '_heftextrap.pdf', bbox_inches='tight')



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
#b2_arr = np.zeros(nbin)
#biasL2_arr = (0.9*biasL1_arr**2+0.5)-8./21*biasL1_arr
#biasL2_arr = np.zeros(nbin)
biasL2_arr = np.array([0.46036128, 0.4845956, 0.5480625, 0.65459134, 0.80604922])
#local-in-matter-density (LIMD) Lagrangian bias:
biasLs2_arr = np.zeros(nbin)
biasLlapl_arr = np.zeros(nbin) 
for bin_i in range(nbin):
    params[f'b1L_{bin_i+1}']=biasL1_arr[bin_i]
    params[f'b2L_{bin_i+1}']=biasL2_arr[bin_i]
    params[f'bs2L_{bin_i+1}']=biasLs2_arr[bin_i]
    params[f'blaplL_{bin_i+1}']=biasLlapl_arr[bin_i]

err_cl_ll, err_cl_gg, err_cl_lg  = MGL.get_errorbars(params)

model={
    'bacco_nobar_heft': {'nl_model': NL_MODEL_BACCO, 'bias_model': 2, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.}
    }

cls_fid = MGL.get_c_ells(params, model['bacco_nobar_heft'])[1]
cls_list = []
cls_list.append(cls_fid)
n_points = 30

dic_chain = {1: {'sigma8_cb': array(0.75158667), 'b1L_1': array(1.4204324), 'b1L_2': array(0.69431766), 'b1L_3': array(1.93543141), 'b1L_4': array(0.86214115), 'b1L_5': array(0.71999603), 'b2L_1': array(1.18935584), 'b2L_2': array(1.89727849), 'b2L_3': array(1.4566858), 'b2L_4': array(1.94297669), 'b2L_5': array(0.99156927), 'bs2L_1': array(-0.72449865), 'bs2L_2': array(-0.220814), 'bs2L_3': array(0.47991525), 'bs2L_4': array(-0.67869858), 'bs2L_5': array(-1.60589988), 'blaplL_1': array(-0.04573221), 'blaplL_2': array(-0.09292113), 'blaplL_3': array(-0.05539138), 'blaplL_4': array(-2.46917485), 'blaplL_5': array(-0.39294152)},
2: {'sigma8_cb': array(0.8157392), 'b1L_1': array(0.88110474), 'b1L_2': array(0.50743258), 'b1L_3': array(0.67104359), 'b1L_4': array(0.34854277), 'b1L_5': array(0.53201865), 'b2L_1': array(0.14258059), 'b2L_2': array(1.41303339), 'b2L_3': array(1.7898494), 'b2L_4': array(0.23250804), 'b2L_5': array(0.68605245), 'bs2L_1': array(0.88746551), 'bs2L_2': array(1.98819247), 'bs2L_3': array(0.3455334), 'bs2L_4': array(-1.2339117), 'bs2L_5': array(-1.22533472), 'blaplL_1': array(1.13986773), 'blaplL_2': array(1.55075696), 'blaplL_3': array(-1.64024556), 'blaplL_4': array(-2.7260251), 'blaplL_5': array(2.06980876)},
3: {'sigma8_cb': array(0.75163809), 'b1L_1': array(0.62192446), 'b1L_2': array(0.05222978), 'b1L_3': array(1.65391816), 'b1L_4': array(1.86393378), 'b1L_5': array(1.67251058), 'b2L_1': array(0.54619858), 'b2L_2': array(0.25123856), 'b2L_3': array(1.04329668), 'b2L_4': array(0.7250786), 'b2L_5': array(0.14166094), 'bs2L_1': array(1.04436402), 'bs2L_2': array(-0.1682304), 'bs2L_3': array(-1.23745627), 'bs2L_4': array(-1.85399507), 'bs2L_5': array(-0.57511573), 'blaplL_1': array(0.00866568), 'blaplL_2': array(1.64409929), 'blaplL_3': array(1.73834292), 'blaplL_4': array(-1.44594222), 'blaplL_5': array(0.83159384)},
4: {'sigma8_cb': array(0.85706924), 'b1L_1': array(0.51557843), 'b1L_2': array(1.89900537), 'b1L_3': array(1.2639658), 'b1L_4': array(1.96969807), 'b1L_5': array(1.40263976), 'b2L_1': array(0.17546532), 'b2L_2': array(0.52064275), 'b2L_3': array(0.79827402), 'b2L_4': array(0.26856086), 'b2L_5': array(1.72246076), 'bs2L_1': array(-1.24604264), 'bs2L_2': array(1.63017568), 'bs2L_3': array(0.92908943), 'bs2L_4': array(-0.20059683), 'bs2L_5': array(-1.17711128), 'blaplL_1': array(0.86883157), 'blaplL_2': array(0.63523221), 'blaplL_3': array(0.49228919), 'blaplL_4': array(-0.33442824), 'blaplL_5': array(-0.68624083)},
5: {'sigma8_cb': array(0.73312383), 'b1L_1': array(0.43744311), 'b1L_2': array(1.1536681), 'b1L_3': array(0.35764457), 'b1L_4': array(0.91031508), 'b1L_5': array(0.14932075), 'b2L_1': array(-0.9902653), 'b2L_2': array(1.89699217), 'b2L_3': array(-0.25573775), 'b2L_4': array(-1.72634931), 'b2L_5': array(-0.28628294), 'bs2L_1': array(-0.6763561), 'bs2L_2': array(-0.96775029), 'bs2L_3': array(-0.33880783), 'bs2L_4': array(1.55526373), 'bs2L_5': array(-0.5608721), 'blaplL_1': array(1.72486442), 'blaplL_2': array(-1.63392224), 'blaplL_3': array(-1.9114334), 'blaplL_4': array(-2.0310951), 'blaplL_5': array(1.73162285)}, 
}
for i in range(5):
    params_new = params.copy()
    for par_i in dic_chain[i+1]:
        params_new[par_i] = dic_chain[i+1][par_i]
    print('params_new: ', params_new)
    cls_list.append(MGL.get_c_ells(params_new, model['bacco_nobar_heft'])[1])

#np.savez('cls_heft_extrap.npz', cls_list=cls_list)
cls_heft_extrap = np.load('cls_heft_extrap.npz')['cls_list']
plot_cells_diag(0, cls_list, True)