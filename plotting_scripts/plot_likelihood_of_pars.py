import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import MGLensing
import yaml
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
VERY_SMALL= 14
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('font', size=SMALL_SIZE)
plt.rc('legend', fontsize=VERY_SMALL)
# Ensure the directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('figs/modelling', exist_ok=True)

with open("plotting_scripts/params_names.yaml", "r") as file:
    params_names = yaml.safe_load(file)
NL_MODEL_BACCO = 1
BARYONS_BACCO = 3
BIAS_HEFT = 2
params_fid = {
    'Omega_m' :  0.31,
    'Omega_b' :  0.05,
    'sigma8_cb': 0.83, 
    'ns'      :  0.97,
    'h'       :  0.68,
    'Mnu'     :  0.06,
    'w0'      :  -1.0,
    'wa'      :  0.0,
    'a1_IA': 0.16,
    'eta1_IA': 1.66,
    'beta_IA': 0.,

    'log10Mc_bc': 13.8,
    'eta_bc': -0.3,
    'beta_bc': -0.22,
    'log10Mz0_bc': 10.5,
    'thetaout_bc': 0.25,
    'thetainn_bc': -0.86,
    'log10Minn_bc': 12.4,
}

print('Specifiy likelihood, data and theoretical model in config.yaml')
MGL = MGLensing.MGL("config.yaml")
zz = MGL.Survey.zz_integr
nbin = MGL.Survey.nbin

bias1_arr = np.array([1.239, 1.378, 1.525, 1.677, 1.832])
biasL1_arr = bias1_arr-1
biasL2_arr = [0.46036128, 0.4845956, 0.5480625, 0.65459134, 0.80604922] #(0.9*biasL1_arr**2+0.5)-8./21*biasL1_arr
#local-in-matter-density (LIMD) Lagrangian bias:
biasLs2_arr = np.zeros(nbin)
biasLlapl_arr = np.zeros(nbin) 
for bin_i in range(nbin):
    params_fid[f'b1L_{bin_i+1}']=biasL1_arr[bin_i]
    params_fid[f'b2L_{bin_i+1}']=biasL2_arr[bin_i]
    params_fid[f'bs2L_{bin_i+1}']=biasLs2_arr[bin_i]
    params_fid[f'blaplL_{bin_i+1}']=biasLlapl_arr[bin_i]

ranges_all = {
        'Omega_cb':     {'p1': 0.23,         'p2': 0.4},    
        'Omega_b':      {'p1':0.04,          'p2': 0.06},  
        'Omega_m':     {'p1': 0.25,         'p2': 0.37}, 
        'h':            {'p1': 0.6,          'p2': 0.8},      
        'sigma8_cb':    {'p1': 0.73,         'p2': 0.9},
        'ns':           {'p1': 0.92,         'p2': 1.01}, 
        'Mnu':          {'p1': 0.0,          'p2': 0.4},
        'w0':           {'p1': -1.15,        'p2': -0.85},    
        'wa':           {'p1': -0.3,         'p2': 0.3}, 
        'log10Mc_bc':   {'p1': 9.,           'p2': 15.}, 
        'eta_bc':       {'p1': -0.69,        'p2': 0.69},
        'beta_bc':      {'p1': -1.,          'p2': 0.69},
        'log10Mz0_bc':  {'p1': 9.,           'p2': 13.},
        'thetaout_bc':  {'p1': 0.,           'p2': 0.47},
        'thetainn_bc':  {'p1': -2.,          'p2': -0.523},
        'log10Minn_bc': {'p1': 9.,           'p2': 13.5},
        'bias':          {'p1': -3.,           'p2': 3.},
        'a1_IA': {'p1':-5., 'p2':5.},
        'eta1_IA': {'p1':-5., 'p2':5.},
        }
fid_like = MGL.get_loglike(params_fid.copy())
print('fiducial like: ', fid_like)

loglike = {}
vary_pars_dic = {}
n_points = 20
#cosmo_pars = ['Omega_m', 'sigma8_cb', 'Omega_b', 'h', 'ns', 'Mnu', 'w0', 'wa', 'a1_IA', 'eta1_IA']

#b1L = np.array([f'b1L_{bin_i+1}' for bin_i in range(nbin)])
#b2L = np.array([f'b2L_{bin_i+1}' for bin_i in range(nbin)])
#bs2L = np.array([f'bs2L_{bin_i+1}' for bin_i in range(nbin)])
#blL = np.array([f'blaplL_{bin_i+1}' for bin_i in range(nbin)])
#cosmo_pars = np.concatenate((b1L, b2L, bs2L, blL))

cosmo_pars = ['log10Mc_bc', 'eta_bc', 'beta_bc', 'log10Mz0_bc', 'thetaout_bc',  'thetainn_bc', 'log10Minn_bc']
params_new = {}
for pars in cosmo_pars:
    vary_pars = np.linspace(ranges_all[pars]['p1'], ranges_all[pars]['p2'], n_points)
    #vary_pars = np.linspace(ranges_all['bias']['p1'], ranges_all['bias']['p2'], n_points)
    loglike_list = []
    for i in range(n_points):
        params_new = params_fid.copy()
        params_new[pars] = vary_pars[i]
        loglike_list.append(MGL.get_loglike(params_new))
    loglike[pars] = np.array(loglike_list)
    vary_pars_dic[pars] = vary_pars


labels = [params_names[p] for p in cosmo_pars]

nrows = 2
fig, ax = plt.subplots(figsize=(15, 8), nrows=nrows, #(15, 8) #(15, 12)
                       ncols=5, #sharey='row', 
                       facecolor='w')
count = 0
for i in range(nrows):
    for j in range(5):
        if count<len(labels):
            ax[i][j].axvline(params_fid[cosmo_pars[count]])
            #ax[i][j].scatter(vary_pars_dic[cosmo_pars[count]], abs((loglike[cosmo_pars[count]]-fid_like)/(vary_pars_dic[cosmo_pars[count]]-params_fid[cosmo_pars[count]])))
            ax[i][j].plot(vary_pars_dic[cosmo_pars[count]], loglike[cosmo_pars[count]])
            ax[i][j].set_xlabel('$'+labels[count]+'$') 
            #ax[i][j].set_yscale('log')
            count+=1
    ax[i][0].set_ylabel('$\log{\mathcal{L}}$')        
#fig.text(0.06, 0.5, '$|(\log{\mathcal{L}}-\log{\mathcal{L}_{\\rm fid}})/(\\theta-\\theta_{\\rm fid})|$', ha='center', va='center', rotation='vertical', fontsize=20)
plt.tight_layout()
plt.show()

