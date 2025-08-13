import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import MGLensing
import matplotlib.pyplot as plt
# Ensure the directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('figs/modelling/c_ell', exist_ok=True)
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

folder = os.path.dirname(os.path.abspath(__file__))+'/../'
os.chdir(folder)
MGL = MGLensing.MGL("config_lssty10.yaml")
zz = MGL.Survey.zz_integr
nbin_s, nbin_l = MGL.Survey.nbin_s, MGL.Survey.nbin_l

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
        # run the script from the MGlensing folder: python plotting_scripts/plot_c_ells.py
        plt.savefig('figs/modelling/c_ell/etas_of_z_' + MGL.Survey.survey_name + '.png')

plot_n_of_z()


def plot_cells_ratio(type, cl_ll_list, cl_lg_list, cl_gg_list, show=True, names="", annotation=""):
    types = ['LL', 'XC', 'GG']
    ells = [l_wl, l_xc, l_gc]
    nbin_i = [nbin_s, nbin_l, nbin_l]
    nbin_j = [nbin_s, nbin_s, nbin_l]
    lmax = [l_wl[-1], l_xc[-1], l_gc[-1]]
    lmax_ij = [l_wl_max, l_gc_max, l_gc_max]
    cls = [cl_ll_list, cl_lg_list, cl_gg_list]
    errs = [err_cl_ll, err_cl_xc, err_cl_gg]
    fig, ax = plt.subplots(figsize=(12, 12), nrows=nbin_i[type], ncols=nbin_j[type], sharex=True, sharey='row', facecolor='w')
    for ind, name in enumerate(names):
        for i in range(nbin_i[type]):
            for j in range(nbin_j[type]):
                if i < j:
                    ax[i, j].axis('off')
                else:
                    err_prop = errs[type][:, i, j]*cls[type][ind][:, i, j]/cls[type][0][:, i, j]*(1./cls[type][ind][:, i, j])
                    ax[i, j].semilogx(ells[type], cls[type][ind][:, i, j]/cls[type][0][:, i, j], label=name if i == 0 and j == 0 else "")
                    ax[i, j].axvspan(xmin=lmax_ij[type][i, j], xmax=lmax[type], color='grey', alpha=0.1)
                    ax[i, j].fill_between(ells[type], np.ones(len(ells[type]))+err_prop, np.ones(len(ells[type]))-err_prop, color='tab:pink', alpha=0.1)
                    ax[i, j].legend(loc='upper left', title_fontsize=10, title='bin ' + str(i + 1) + '-' + str(j + 1))
    for j in range(nbin_j[type]):
        ax[nbin_i[type] - 1][j].set_xlabel('$\ell$')
    ax[int(nbin_i[type] / 2)][0].set_ylabel('$C^{\\rm ' + types[type] + '}_{\ell}$ ratio')
    ax[1, 1].annotate(annotation, (1.1, 0.05), xycoords='axes fraction', clip_on=False)
    plt.tight_layout()
    # run the script from the MGlensing folder: python plotting_scripts/plot_c_ells.py
    
    plt.show() if show else plt.savefig('figs/modelling/c_ell/c_ells_' + types[type] + '_' + MGL.Survey.survey_name + '_ratio.png')


def plot_cells(type, cl_ll_list, cl_lg_list, cl_gg_list, show=True, names="", annotation=""):
    types = ['LL', 'XC', 'GG']
    ells = [l_wl, l_xc, l_gc]
    nbin_i = [nbin_s, nbin_l, nbin_l]
    nbin_j = [nbin_s, nbin_s, nbin_l]
    lmax = [l_wl[-1], l_xc[-1], l_gc[-1]]
    lmax_ij = [l_wl_max, l_gc_max, l_gc_max]
    cls = [cl_ll_list, cl_lg_list, cl_gg_list]
    #errs = [err_cl_ll, err_cl_xc, err_cl_gg]
    fig, ax = plt.subplots(figsize=(12, 12), nrows=nbin_i[type], ncols=nbin_j[type], sharex=True, sharey=True, facecolor='w')
    for ind, name in enumerate(names):
        for i in range(nbin_i[type]):
            for j in range(nbin_j[type]):
                if i < j:
                    ax[i, j].axis('off')
                else:
                    ax[i, j].loglog(ells[type], cls[type][ind][:, i, j], label=name if i == 0 and j == 0 else "")
                    ax[i, j].axvspan(xmin=lmax_ij[type][i, j], xmax=lmax[type], color='grey', alpha=0.1)
                    #ax[i, j].errorbar(ells[type], cls[type][ind][:, i, j], yerr=errs[type][:, i, j])
                    ax[i, j].legend(loc='lower left', title_fontsize=10, title='bin ' + str(i + 1) + '-' + str(j + 1))
    for j in range(nbin_j[type]):
        ax[nbin_i[type] - 1][j].set_xlabel('$\ell$')
    ax[int(nbin_i[type] / 2)][0].set_ylabel('$C^{\\rm ' + types[type] + '}_{\ell}$')
    ax[1, 1].annotate(annotation, (1.1, 0.05), xycoords='axes fraction', clip_on=False)
    plt.tight_layout()
    # run the script from the MGlensing folder: python plotting_scripts/plot_c_ells.py
    plt.show() if show else plt.savefig('figs/modelling/c_ell/c_ells_' + types[type] + '_' + MGL.Survey.survey_name + '.png')


def plot_deltaCl(ell_list, cl_list_1, cl_list_2, cl_list_3, errors, show=True, label="", annotation=""):
    ''' 
    Plot the difference between the two Cls divided by the errors (Cl1-Cl2)/sigma.
    cl_list_1, cl_list_2: list of Cls to be compared, each one contains [LL, LG, GG].
    errors: list of errors for Cl.
    '''
    types = ['LL', 'XC', 'GG']
    lmax = [ell_list[0][-1], ell_list[1][-1], ell_list[2][-1]]
    lmax_ij = [l_wl_max, l_gc_max, l_gc_max]
    nbin_i = [nbin_s, nbin_l, nbin_l]
    nbin_j = [nbin_s, nbin_s, nbin_l]
    for type_idx, type_name in enumerate(types):
        fig, axes = plt.subplots(figsize=(12, 12), nrows=nbin_i[type_idx], ncols=nbin_j[type_idx], sharex=True, sharey=True, facecolor='w')
        for i in range(nbin_i[type_idx]):
            for j in range(nbin_j[type_idx]):
                if i < j:
                    axes[i, j].axis('off')
                else:
                    # Calculate delta Cl
                    delta_cl12 = (cl_list_2[type_idx][:, i, j] - cl_list_1[type_idx][:, i, j]) / errors[type_idx][:, i, j]
                    delta_cl13 = (cl_list_3[type_idx][:, i, j] - cl_list_1[type_idx][:, i, j]) / errors[type_idx][:, i, j]
                    # Plot delta Cl
                    axes[i, j].semilogx(ell_list[type_idx], delta_cl12, label=label if i == 0 and j == 0 else "")
                    axes[i, j].semilogx(ell_list[type_idx], delta_cl13, label=label if i == 0 and j == 0 else "")
                    axes[i, j].axhline(y=0, color='gray', linestyle='--')
                    # Plot scale cuts
                    axes[i,j].text(.4, .9, 'bin %s, %s'%(str(i+1),(j+1)), fontsize="medium", horizontalalignment='center', transform=axes[i,j].transAxes)  
                    axes[i,j].axvspan(xmin=lmax_ij[type_idx][i, j], xmax=lmax[type_idx], color='grey', alpha=0.1)
                    axes[i,j].set_ylim(-10, 15)
        for j in range(nbin_j[type_idx]):
            axes[nbin_i[type_idx] - 1][j].set_xlabel('$\ell$')
        axes[int(nbin_i[type_idx] / 2)][0].set_ylabel(f'$\\Delta C^{{\\rm {type_name}}}_{{\\ell}} / \\sigma$')
        axes[1, 1].annotate(annotation, (1.1, 0.05), xycoords='axes fraction', clip_on=False)
        fig.legend(['b1b2-HEFT', 'b1beb2-HEFT'], loc='upper right', ncol=1, bbox_to_anchor=(1, 0.93), bbox_transform=fig.transFigure, fontsize=15)
        fig.suptitle('kmax=0.7') #  kmax = config.yaml')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Reduce space between subplots
        plt.tight_layout()
        if show:
            plt.show()
        else:
            # run the script from the MGlensing folder: python plotting_scripts/plot_c_ells.py
            plt.savefig(f'figs/modelling/c_ell/deltaCl_{type_name}_{MGL.Survey.survey_name}_with_b_extra.png')




params = {
    'Omega_m' :  0.315,
    'Omega_c' :  0.315-0.05,
    'Omega_cb' :  0.315,
    'Omega_nu':  0.,
    'As'      :  np.exp(3.07)*1.e-10,
    'sigma8_cb': 0.83,
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
# bias1_arr = np.array([1.239, 1.378, 1.525, 1.677, 1.832])
biasL1_arr = bias1_arr-1
#Lagrangian co-evolution 
b2_arr = np.array([-0.258, -0.062, 0.107, 0.267, 0.462])
#biasL2_arr = b2_arr-8./21*biasL1_arr
biasL2_arr = (0.9*biasL1_arr**2+0.5)-8./21*biasL1_arr
#local-in-matter-density (LIMD) Lagrangian bias:
biasLs2_arr = np.zeros(nbin_l)
biasLlapl_arr = np.zeros(nbin_l) 
for bin_i in range(nbin_l):
    params[f'b1_{bin_i+1}']=bias1_arr[bin_i]
    params[f'be_{bin_i+1}']=0.
    params[f'b2_{bin_i+1}']=bias2_arr[bin_i]
    params[f'b1L_{bin_i+1}']=biasL1_arr[bin_i]
    params[f'b2L_{bin_i+1}']=biasL2_arr[bin_i]
    params[f'bs2L_{bin_i+1}']=biasLs2_arr[bin_i]
    params[f'blaplL_{bin_i+1}']=biasLlapl_arr[bin_i]

err_cl_ll, err_cl_gg, err_cl_xc  = MGL.get_errorbars(params)
print(err_cl_ll.shape, err_cl_gg.shape, err_cl_xc.shape)
models = {
    'hmcode': {'nl_model': NL_MODEL_HMCODE, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    'bacco': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.}
    }

cls_dic = {}
for key, model in models.items():
    cls_dic[key] = MGL.get_c_ells(params, model)

cl_ll_hm, cl_gg_hm, cl_lg_hm,  cl_gl_hm  = cls_dic['hmcode']
cl_ll_bacco, cl_gg_bacco, cl_lg_bacco, cl_gl_bacco = cls_dic['bacco']

for type_i, cl_hm, cl_bacco in zip([0, 1, 2], [cl_ll_hm, cl_gl_hm, cl_gg_hm], [cl_ll_bacco, cl_gl_bacco, cl_gg_bacco]):
    plot_cells(type_i, [cl_hm, cl_bacco], [cl_hm, cl_bacco], [cl_hm, cl_bacco], False, names=['hmcode', 'bacco'])

for type_i, cl_hm, cl_bacco in zip([0, 1, 2], [cl_ll_hm, cl_gl_hm, cl_gg_hm], [cl_ll_bacco, cl_gl_bacco, cl_gg_bacco]):
    plot_cells_ratio(type_i, [cl_hm, cl_bacco], [cl_hm, cl_bacco], [cl_hm, cl_bacco], True, names=['hmcode', 'bacco'])





cl_ll_b1b2, cl_gg_b1b2, cl_lg_b1b2, cl_gl_b1b2 = MGL.get_c_ells(params, {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_B1B2, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.})
cl_ll_heft, cl_gg_heft, cl_lg_heft, cl_gl_heft = MGL.get_c_ells(params, {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_HEFT, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.})
cl_heft_list = [cl_ll_heft, cl_gl_heft, cl_gg_heft]
cl_b1b2_list = [cl_ll_b1b2, cl_gl_b1b2, cl_gg_b1b2]
heft_err_list = [err_cl_ll, err_cl_xc, err_cl_gg]

b_extra_arr = bias1_arr*0.5
for bin_i in range(nbin_l):
    params[f'be_{bin_i+1}']=b_extra_arr[bin_i]

cl_ll_b1beb2, cl_gg_b1beb2, cl_lg_b1beb2, cl_gl_b1beb2 = MGL.get_c_ells(params, {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_B1B2, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.})
cl_b1beb2_list = [cl_ll_b1beb2, cl_gl_b1beb2, cl_gg_b1beb2]

print(bias1_arr)
print(bias2_arr)
print(b_extra_arr)

plot_deltaCl([l_wl, l_xc, l_gc], cl_heft_list, cl_b1b2_list, cl_b1beb2_list, heft_err_list, True)
