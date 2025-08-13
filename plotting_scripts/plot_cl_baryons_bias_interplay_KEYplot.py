import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
import numpy as np
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import MGLensing
os.chdir(parent_dir)

my_colors = ['#035177', '#8ECAE6', '#FFDD03', '#ED91DC', '#F58300', '#78206E']
my_colors = [ '#18698F', '#7AC011', '#FF7083', '#99D4E5']


def plot_compare_Cls(ell, denominator_cl, reference_cl, cl_list, err_cl, param_values, par_name='', savename='', nbin=5, save_dir=''):
    '''
    Plot comparison of Cls with and without baryons, varying one parameter.
    Inputs:
        ell : multipoles corresponding to all Cls (array)
        denominator_cl : Cl denominator for all Cls (array(nell, nbin, nbin))
        reference_cl : Cl with baryons and b1*Pnl*S + h.o. heft = 0 (array(nell, nbin, nbin))
        cl_list : list of Cls with b1*Pnl + h.o. heft != 0 without baryons, varying one parameter (array(nvalues, nell, nbin, nbin))
        err_cl : Cl error (array(nell, nbin, nbin))
        param_values : values of the varied parameter (array(nvalues))
        param_name : name of the varied parameter (str)
        savename : save plot name (str)
        nbin : number of redshift bins (int)
        save_dir : directory to save the plot (str)
    '''
    cmap = LinearSegmentedColormap.from_list("", ["#7AC011","#18698F"])
    # normalize parameter values for the colormap
    norm = Normalize(vmin=min(param_values), vmax=max(param_values))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  
    ell_cuts = [kmax07]

    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, nbin + 1, width_ratios=[1] * nbin + [0.05], wspace=0.1)
    ax = [fig.add_subplot(gs[i]) for i in range(nbin)]
    for i in range(nbin):
        ax[i].hlines(1., min(ell), max(ell), 'k', linestyle='--')
        ax[i].plot(ell, reference_cl[:,i,i]/denominator_cl[:,i,i], linewidth=3, color='#FF7083', zorder=10) 
        for c in range(len(cl_list)):
            color = cmap(norm(param_values[c]))
            ax[i].plot(ell, cl_list[c][:,i,i]/denominator_cl[:,i,i], linewidth=2, color=color, zorder=0)
        
        err = err_cl[:,i,i]*reference_cl[:,i,i]/denominator_cl[:,i,i]*(1./reference_cl[:,i,i])
        ax[i].fill_between(ell, np.ones(len(ell))+err, np.ones(len(ell))-err, color='#99D4E5', alpha=0.2, zorder=0)
        for cuts in ell_cuts:
            ax[i].axvspan(5000, cuts[i], color='gray', alpha=0.1)
            
        ax[i].set_xscale('log')
        ax[i].set_xlim(min(ell), max(ell))
        ax[i].set_ylim(0.9, 1.05)
        ax[i].tick_params(axis='both', which='major', labelsize=15) 
        if i > 0:
            ax[i].tick_params(labelleft=False)   
        ax[i].set_xlabel(r'$\ell$', fontsize=18)
    ax[0].set_ylabel(r'$C_\ell$ ratio', fontsize=18) 
    # add colorbar
    cbar = plt.colorbar(sm, cax=fig.add_subplot(gs[-1]), orientation='vertical')
    cbar.set_label(par_name, fontsize=18)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(save_dir + savename + '.png', bbox_inches='tight', pad_inches=0.2)
    # plt.show()


def get_cells_bias_dependence(params_dic, model_dic, par_name, par_values, Nell, nbin=5):
    '''
    Get Cls for a given parameter varying its value.
    Inputs:
        params_dic : dictionary of parameters (dict)
        model_dic : dictionary of model parameters (dict)
        par_name : name of the parameter to vary (str)
        par_values : values of the parameter to vary (array)
        Nell : number of multipoles (int)
        nbin : number of redshift bins (int)
    Outputs:
        cl_ll : lensing-lensing Cls for all parameter values 
        cl_gg : galaxy-galaxy Cls for all parameter values 
        cl_lg : lensing-galaxy Cls for all parameter values 
        cl_gl : galaxy-lensing Cls for all parameter values 
    '''
    npars = par_values.shape[0]
    cl_ll = np.zeros((npars, Nell, nbin, nbin))
    cl_gg = np.zeros((npars, Nell, nbin, nbin))
    cl_lg = np.zeros((npars, Nell, nbin, nbin))
    cl_gl = np.zeros((npars, Nell, nbin, nbin))
    for i in range(npars):
        for z in range(nbin):
            params_dic[par_name+'_'+str(z+1)] = par_values[i]
        cl_ll[i], cl_gg[i], cl_lg[i], cl_gl[i] = MGLtest.get_c_ells(params_dic, model_dic)
    return cl_ll, cl_gg, cl_lg, cl_gl



MGLtest = MGLensing.MGL("config.yaml")

kmax07 = [482, 616, 716, 780, 818]
kmax04 = [273, 352, 409, 445, 467]
kmax03 = [206, 264, 306, 334, 350]
kmax01 = [68, 88, 102, 111, 116]

MGL = MGLensing.MGL("config.yaml")
# MGL Cells 
ell_g = MGL.Survey.l_gc
zz_integr = MGL.Survey.zz_integr
nbin = MGL.Survey.nbin_l

# b_i = 0 (i > 1)
params = {'Omega_m' : 0.31,
    'Omega_c'   : 0.31-0.05,
    'Omega_cb'  : 0.31,
    'sigma8_cb' : 0.83,
    'As'      : np.exp(3.07)*1.e-10,
    'Omega_b'   : 0.05,
    'ns'        : 0.96,
    'h'         : 0.68,
    'Mnu'       : 0.0,
    'w0'        : -1.0,
    'wa'        : 0.0, 
    'b1_1': 1.239,
    'b1_2': 1.378,
    'b1_3': 1.525,
    'b1_4': 1.677,
    'b1_5': 1.832,
    'b1L_1': 0.187684,
    'b1L_2': 0.312375,
    'b1L_3': 0.45121,
    'b1L_4': 0.60626,
    'b1L_5': 0.779385,
    'b2L_1': 0., #-0.1322075,
    'b2L_2': 0., #-0.0345375,
    'b2L_3': 0., #0.0579305,
    'b2L_4': 0., #0.133662,
    'b2L_5': 0., #0.232387,
    'bs2L_1': 0., #0.1127485,
    'bs2L_2': 0., #0.0036835,
    'bs2L_3': 0., #-0.1261985,
    'bs2L_4': 0., #-0.285537,
    'bs2L_5': 0., #-0.406101,
    'blaplL_1': 0., #-0.100126,
    'blaplL_2': 0., #-0.1563475,
    'blaplL_3': 0., #-0.115281,
    'blaplL_4': 0., #0.1059295,
    'blaplL_5': 0., #0.361450,
    'a1_IA'   : 0.67,
    'eta1_IA' : 1.66,
    'beta_IA' : 0.,
    'log10Mc_bc': 13.8,
    'eta_bc': -0.3,
    'beta_bc': -0.22 ,
    'log10Mz0_bc': 10.5,
    'thetaout_bc': 0.25,
    'thetainn_bc': -0.86,
    'log10Minn_bc': 12.4,
}     
# model for "reference" line
model = {
    'nl_model' : 1,
    'bias_model' : 4,
    'baryon_model' : 3,
    'ia_model' : 0,
    'photoz_err_model' : 0
}
cl_ll_ref, cl_gg_ref, cl_lg_ref, cl_gl_ref = MGL.get_c_ells(params, model)
err_cl_ll, err_cl_gg, err_cl_lg  = MGL.get_errorbars(params)


###
model['bias_model'] = 3
model['baryon_model'] = 0
cl_ll_den, cl_gg_den, cl_lg_den, cl_gl_den = MGL.get_c_ells(params, model)


###
npoints = 15
print('plotting for bs2...')
bs2_values = np.linspace(-0.2, 0.5, npoints)
cl_ll_list, cl_gg_list, cl_lg_list, cl_gl_list = get_cells_bias_dependence(params, model, 'bs2L', bs2_values, len(ell_g))
save_dir = 'figs/other/'

plot_compare_Cls(ell_g, cl_gg_den, cl_gg_ref, cl_gg_list, err_cl_gg, bs2_values, r'$b_{\rm s^2}$', savename='cl_gg_keyplot1_bs2xxx', save_dir=save_dir)


###
print('plottig for blap...')
for i in range(nbin):
    params['bs2L_'+str(i+1)] = 0
blapl_values = np.linspace(-0.2, 0.5, npoints)
cl_ll_list, cl_gg_list, cl_lg_list, cl_gl_list = get_cells_bias_dependence(params, model, 'blaplL', blapl_values, len(ell_g))

plot_compare_Cls(ell_g, cl_gg_den, cl_gg_ref, cl_gg_list, err_cl_gg, blapl_values, r'$b_{\nabla^2}$', savename='cl_gg_keyplot1_blaplxxx', save_dir=save_dir)
