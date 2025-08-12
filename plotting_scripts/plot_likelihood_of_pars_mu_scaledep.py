import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(folder)
import numpy as np
import MGLensing
import yaml
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False)
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

params_fid = {
    'Omega_m' :  0.31,
    'Omega_b' :  0.05,
    'log10As': 3.044, 
    'ns'      :  0.97,
    'h'       :  0.68,
    'Mnu'     :  0.0,
    'w0'      :  -1.0,
    'wa'      :  0.0,
    'a1_IA': 1.72,
    'eta1_IA': -0.41,
    'beta_IA': 0.,

'mu0': 0.5,
'c1': 2.,
'lambda': 0.1,


}

print('Specifiy likelihood, data and theoretical model in config.yaml')
MGL = MGLensing.MGL("config_euclid.yaml")
zz = MGL.Survey.zz_integr
nbin = MGL.Survey.nbin_s

bias1_arr = np.array([1.239, 1.378, 1.525, 1.677, 1.832])
for bin_i in range(nbin):
    params_fid[f'b1_{bin_i+1}']=bias1_arr[bin_i]


ranges_all = {
        'Omega_cb':     {'p1': 0.23,         'p2': 0.4},    
        'Omega_b':      {'p1':0.04,          'p2': 0.06},  
        'Omega_m':      {'p1': 0.25,         'p2': 0.37}, 
        'h':            {'p1': 0.6,          'p2': 0.8},      
        'log10As':    {'p1': 2.,         'p2': 4.},
        'ns':           {'p1': 0.92,         'p2': 1.01}, 
        'Mnu':          {'p1': 0.0,          'p2': 0.4},
        'mu0' :          {'p1': -0.99,          'p2': 3.},
        'c1' :          {'p1': -5.,          'p2': 5.},
        'lambda':       {'p1': 0.0,         'p2': 5.},  
        #'w0':           {'p1': -1.15,        'p2': -0.85},    
        #'wa':           {'p1': -0.3,         'p2': 0.3}, 
        'bias':          {'p1': -3.,           'p2': 3.},
        'a1_IA': {'p1':-5., 'p2':5.},
        'eta1_IA': {'p1':-5., 'p2':5.},
        }
fid_like = MGL.get_loglike(params_fid.copy())
print('fiducial like: ', fid_like)

loglike = {}
vary_pars_dic = {}
n_points = 20
cosmo_pars = ['Omega_m', 'log10As', 'Omega_b', 'h', 'ns', 'Mnu', 'mu0', 'c1', 'lambda']

params_new = {}
print('computing likelihoods...')
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
#plt.show()
plt.savefig('figs/modelling/likelihood_of_pars_mu_scaledep.png', bbox_inches='tight')
