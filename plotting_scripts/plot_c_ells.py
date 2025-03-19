import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import MGLensing
import matplotlib.pyplot as plt
# Ensure the directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('figs/modelling', exist_ok=True)
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

def plot_n_of_z(show=True):
    etaz = [MGL.Survey.eta_z_s, MGL.Survey.eta_z_l]
    titles = ['sources', 'lenses']
    fig, ax = plt.subplots(1, 2, figsize=(8, 5), sharex=True, sharey=True)
    for k in range(2):
        for i in range(etaz[k].shape[1]):
            ax[k].plot(zz, etaz[k][:, i], label='bin ' + str(i + 1))
        ax[k].set_xlabel("$z$")
        ax[k].legend(loc='upper right', title=titles[k])
    ax[0].set_ylabel("$\\eta(z)$")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('figs/modelling/etas_of_z_' + MGL.Survey.survey_name + '.png')

def plot_cells_ratio(type, cl_ll_list, cl_lg_list, cl_gg_list, show=True, names="", annotation=""):
    types = ['LL', 'LG', 'GG']
    ells = [l_wl, l_xc, l_gc]
    lmax = [l_wl[-1], l_xc[-1], l_gc[-1]]
    lmax_ij = [l_wl_max, l_gc_max, l_gc_max]
    cls = [cl_ll_list, cl_lg_list, cl_gg_list]
    errs = [err_cl_ll, err_cl_lg, err_cl_gg]
    fig, ax = plt.subplots(figsize=(8, 8), nrows=nbin, ncols=nbin, sharex=True, facecolor='w')
    for ind, name in enumerate(names):
        for i in range(nbin):
            for j in range(nbin):
                if i < j:
                    ax[i, j].axis('off')
                else:
                    err_prop = errs[type][:, i, j]*cls[type][ind][:, i, j]/cls[type][0][:, i, j]*(1./cls[type][ind][:, i, j])
                    ax[i, j].semilogx(ells[type], cls[type][ind][:, i, j]/cls[type][0][:, i, j], label=name if i == 0 and j == 0 else "")
                    ax[i, j].axvspan(xmin=lmax_ij[type][i, j], xmax=lmax[type], color='grey', alpha=0.1)
                    ax[i, j].fill_between(ells[type], np.ones(len(ells[type]))+err_prop, np.ones(len(ells[type]))-err_prop, color='tab:pink', alpha=0.1)
                    ax[i, j].legend(loc='upper left', title_fontsize=10, title='bin ' + str(i + 1) + '-' + str(j + 1))
    for i in range(nbin):
        ax[nbin - 1][i].set_xlabel('$\ell$')
    ax[int(nbin / 2)][0].set_ylabel('$C^{\\rm ' + types[type] + '}_{\ell}$ ratio')
    ax[1, 1].annotate(annotation, (1.1, 0.05), xycoords='axes fraction', clip_on=False)
    plt.tight_layout()
    plt.show() if show else plt.savefig('figs/modelling/c_ells_' + types[type] + '_' + MGL.Survey.survey_name + '_ratio.png')

def plot_cells(type, cl_ll_list, cl_lg_list, cl_gg_list, show=True, names="", annotation=""):
    types = ['LL', 'LG', 'GG']
    ells = [l_wl, l_xc, l_gc]
    lmax = [l_wl[-1], l_xc[-1], l_gc[-1]]
    lmax_ij = [l_wl_max, l_gc_max, l_gc_max]
    cls = [cl_ll_list, cl_lg_list, cl_gg_list]
    errs = [err_cl_ll, err_cl_lg, err_cl_gg]
    fig, ax = plt.subplots(figsize=(8, 8), nrows=nbin, ncols=nbin, sharex=True, sharey=True, facecolor='w')
    for ind, name in enumerate(names):
        for i in range(nbin):
            for j in range(nbin):
                if i < j:
                    ax[i, j].axis('off')
                else:
                    ax[i, j].loglog(ells[type], cls[type][ind][:, i, j], label=name if i == 0 and j == 0 else "")
                    ax[i, j].axvspan(xmin=lmax_ij[type][i, j], xmax=lmax[type], color='grey', alpha=0.1)
                    #ax[i, j].errorbar(ells[type], cls[type][ind][:, i, j], yerr=errs[type][:, i, j])
                    ax[i, j].legend(loc='lower left', title_fontsize=10, title='bin ' + str(i + 1) + '-' + str(j + 1))
    for i in range(nbin):
        ax[nbin - 1][i].set_xlabel('$\ell$')
    ax[int(nbin / 2)][0].set_ylabel('$C^{\\rm ' + types[type] + '}_{\ell}$')
    ax[1, 1].annotate(annotation, (1.1, 0.05), xycoords='axes fraction', clip_on=False)
    plt.tight_layout()
    plt.show() if show else plt.savefig('figs/modelling/c_ells_' + types[type] + '_' + MGL.Survey.survey_name + '.png')


params = {
    'Omega_m' :  0.315,
    'Omega_c' :  0.315-0.05,
    'Omega_cb' :  0.315,
    'Omega_nu':  0.,
    'As'      :  np.exp(3.07)*1.e-10,
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
    'log10Mc_bc': 13.8,
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
b2_arr = np.array([-0.258, -0.062, 0.107, 0.267, 0.462])
#biasL2_arr = b2_arr-8./21*biasL1_arr
biasL2_arr = (0.9*biasL1_arr**2+0.5)-8./21*biasL1_arr
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

models = {
    'hmcode': {'nl_model': NL_MODEL_HMCODE, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    'bacco': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.}
    }

cls_dic = {}
for key, model in models.items():
    cls_dic[key] = MGL.get_c_ells(params, model)

cl_ll_hm, cl_gg_hm, cl_lg_hm  = cls_dic['hmcode'][:-1]
cl_ll_bacco, cl_gg_bacco, cl_lg_bacco = cls_dic['bacco'][:-1]


for type_i, cl_hm, cl_bacco in zip([0, 1, 2], [cl_ll_hm, cl_lg_hm, cl_gg_hm], [cl_ll_bacco, cl_lg_bacco, cl_gg_bacco]):
    plot_cells(type_i, [cl_hm, cl_bacco], [cl_hm, cl_bacco], [cl_hm, cl_bacco], False, names=['hmcode', 'bacco'])

for type_i, cl_hm, cl_bacco in zip([0, 1, 2], [cl_ll_hm, cl_lg_hm, cl_gg_hm], [cl_ll_bacco, cl_lg_bacco, cl_gg_bacco]):
    plot_cells_ratio(type_i, [cl_hm, cl_bacco], [cl_hm, cl_bacco], [cl_hm, cl_bacco], False, names=['hmcode', 'bacco'])