import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder+'/../chains/')

# check FoM and FoB examples here: https://arxiv.org/pdf/1810.10104
# my_colors = ['#FAC13B', '#48849C', '#F5998E', '#BCDCDA', '#A32F28']
my_colors = ['#FF7083', '#7AC011', '#18698F', "#82CFE5", '#FFDD03']
# my_colors = ['#8ECAE6', '#F58300', '#78206E', '#ED91DC', '#035177', '#FFDD03']


def read_last_header_line(file_path):  # from plot_posterior.py
    '''
    file_path: path to the chain file
    ---
    returns: list of parameter names
    '''
    last_header = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                last_header = line.strip('#').strip()  
            else:
                break  
    if last_header:
        return last_header.split() 
    else:
        return []    

def get_chain(file_path):
    '''
    file_path: path to the chain file
    ---
    returns: dictionary with parameter names and corresponding values
    '''
    chain = np.genfromtxt(file_path)
    chain_pars = read_last_header_line(file_path)
    chain_pars = chain_pars[:-2]
    chains_info = {}
    chains_info['chain'] = chain
    chains_info['pars'] = chain_pars
    chains_info['weights'] = np.exp(chain[:,-2])
    return chains_info

def get_FoM(file_path, pars_name='all'):
    '''
    file_path: path to the chain file
    pars_name: list of parameter names to calculate FoM for. If 'all', FoM is calculated for all parameters
    ---
    returns: FoM value
    '''
    chain = get_chain(file_path)
    # build covariance of given parameters
    if pars_name == 'all':
        chain_selected_pars = chain['chain'][:, :-2]        
    else:
        chain_selected_pars = chain['chain'][:, [chain['pars'].index(p) for p in pars_name]]
    cov = np.cov(chain_selected_pars, rowvar=False, aweights=chain['weights'])
    # calculate FoM
    if len(chain_selected_pars[0]) == 1:
        fom = 1 / np.sqrt(cov)
    else:
        fom = 1 / np.sqrt(np.linalg.det(cov))
    return fom

def get_FoB(file_path, fiducials, pars_name='all'):
    '''
    file_path: path to the chain file
    fiducials : dictionary with fiducial values of parameters
    pars_name: list of parameter names to calculate FoB for. If 'all', FoB is calculated for all parameters
    ---
    returns: FoB value
    '''
    chain = get_chain(file_path)
    # build covariance of given parameters
    if pars_name == 'all':
        chain_selected_pars = chain['chain'][:, :-2]     
        fiducials_selected_pars = np.array([fiducials[p] for p in chain['pars']])   
    else:
        chain_selected_pars = chain['chain'][:, [chain['pars'].index(p) for p in pars_name]]
        fiducials_selected_pars = np.array([fiducials[p] for p in pars_name])
    cov = np.cov(chain_selected_pars, rowvar=False, aweights=chain['weights'])
    # calculate difference with fiducial values
    delta_fid = np.zeros(len(fiducials_selected_pars))
    for i in range(len(chain_selected_pars[0])):
        delta_fid[i] = np.average(chain_selected_pars[:,i], weights=chain['weights']) - fiducials_selected_pars[i]
    # calculate FoB
    if len(chain_selected_pars[0]) == 1:
        fob = np.abs(delta_fid) / np.sqrt(cov)
    else:
        fob = np.sqrt(np.linalg.multi_dot([delta_fid, np.linalg.inv(cov), delta_fid.T]))
    return fob


fiducials = {
'Omega_m': 0.31,
'Ombh2': 0.02268,
'Omega_b': 0.04904844290657439,
'h': 0.68,
'sigma8_cb': 0.83,
'ns': 0.97,
'Mnu': 0.06,
'w0': -1.,
'wa': 0.0,
'a1_IA': 0.16,
'eta1_IA': 1.66,
'beta_IA': 0.,
'b1_1': 1.239,
'b1_2': 1.378,
'b1_3': 1.525,
'b1_4': 1.677,
'b1_5': 1.832,
'b2_1': 1.74148,
'b2_2': 2.27732,
'b2_3': 2.58944,
'b2_4': 2.95324,
'b2_5': 3.96032,
'b1L_1': 0.239,
'b1L_2': 0.378,
'b1L_3': 0.525,
'b1L_4': 0.677,
'b1L_5': 0.832,
'b2L_1': 0.46036128,
'b2L_2': 0.4845956,
'b2L_3': 0.5480625,
'b2L_4': 0.65459134,
'b2L_5': 0.80604922,
'bs2L_1': 0.0,
'bs2L_2': 0.0,
'bs2L_3': 0.0,
'bs2L_4': 0.0,
'bs2L_5': 0.0,
'blaplL_1': 0.0,
'blaplL_2': 0.0,
'blaplL_3': 0.0,
'blaplL_4': 0.0,
'blaplL_5': 0.0,
'log10Mc_bc': 13.8
}

# chain_files = [['chain_HEFTdata_b1model_kmax0.1_noSystematics-fcfs_03Apr.txt', 'chain_HEFTdata_b1model_kmax0.3_noSystematics_28Apr.txt', 'chain_HEFTdata_b1model_kmax0.5_noSystematics_29Apr.txt', 'chain_HEFTdata_b1model_kmax0.7_noSystematics_28Apr.txt'],  #'chain_HEFTdata_b1model_kmax0.2_noSystematics_21Apr.txt',
#                ['chain_heft_heft_kmax0.1_fixed_bs2_bLap_toZero_15May.txt', 'chain_heft_heft_kmax0.3_fixed_bs2_bLap_toZero_13May.txt', 'chain_heft_heft_kmax0.4_fixed_bs2_bLap_toZero_15May.txt', 'chain_heft_heft_kmax0.5_fixed_bs2_bLap_toZero_15May.txt', 'chain_heft_heft_kmax0.7_fixed_bs2_bLap_toZero_13May.txt'],
#                ['chain_HEFTdata_HEFTmodel_kmax0.1_noSystematics-fcfs_03Apr.txt', 'chain_HEFTdata_HEFTmodel_kmax0.3_noSystematics_28Apr.txt', 'chain_HEFTdata_HEFTmodel_kmax0.5_noSystematics_28Apr.txt', 'chain_HEFTdata_HEFTmodel_kmax0.7_noSystematics-fcfs_07Apr.txt']]
# # with SRD bias fiducials           
# chain_files = [['chain_heft_b1_kmax0.1_SRD_23Jun.txt', 'chain_heft_b1_kmax0.3_SRD_23Jun.txt', 'chain_heft_b1_kmax0.5_SRD_23Jun.txt', 'chain_heft_b1_kmax0.7_SRD_20Jun.txt'],
#                    ['chain_heft_heft_kmax0.1_SRD_fixing_bs2_bLap_toZero_19Jun.txt', 'chain_heft_heft_kmax0.3_SRD_fixing_bs2_bLap_toZero_19Jun.txt', 'chain_heft_heft_kmax0.5_SRD_fixing_bs2_bLap_toZero_19Jun.txt', 'chain_heft_heft_kmax0.7_SRD_fixing_bs2_bLap_toZero_19Jun.txt'],
#                    ['chain_heft_heft_kmax0.1_SRD_20Jun.txt', 'chain_heft_heft_kmax0.3_SRD_20Jun.txt', 'chain_heft_heft_kmax0.5_SRD_20Jun.txt', 'chain_heft_heft_kmax0.7_SRD_20Jun.txt']]
# # fixed bias varying baryons 
# chain_files = [['chain_heft_heft_kmax0.1_fixbias_7baryons_23Jun.txt', 'chain_heft_heft_kmax0.3_fixbias_7baryons_23Jun.txt', 'chain_heft_heft_kmax0.5_fixbias_7baryons_23Jun.txt', 'chain_heft_heft_kmax0.7_fixbias_7baryons_23Jun.txt'],
#                ['chain_heft_heft_kmax0.1_fixbias_logMcEtaBeta_23Jun.txt', 'chain_heft_heft_kmax0.3_fixbias_logMcEtaBeta_23Jun.txt', 'chain_heft_heft_kmax0.5_fixbias_logMcEtaBeta_23Jun.txt', 'chain_heft_heft_kmax0.7_fixbias_logMcEtaBeta_23Jun.txt'],
#                ['chain_heft_heft_kmax0.1_fixbias_logMc_23Jun.txt', 'chain_heft_heft_kmax0.3_fixbias_logMc_23Jun.txt', 'chain_heft_heft_kmax0.5_fixbias_logMc_23Jun.txt', 'chain_heft_heft_kmax0.7_fixbias_logMc_23Jun.txt']]
# new extrapolation with SRD bias fiducials and negative b2 prior - b1, fix h.o., full heft
# chain_files = [['chain_heft_b1_kmax0.1_SRD_21Jul.txt', 'chain_heft_b1_kmax0.3_SRD_21Jul.txt', 'chain_heft_b1_kmax0.5_SRD_21Jul.txt', 'chain_heft_b1_kmax0.7_SRD_21Jul.txt'],
#                ['chain_heft_heft_kmax0.1_SRD_fix_bs2_blap_toZero_negativeb2prior_21Jul.txt', 'chain_heft_heft_kmax0.3_SRD_fix_bs2_blap_toZero_negativeb2prior_21Jul.txt', 'chain_heft_heft_kmax0.5_SRD_fix_bs2_blap_toZero_negativeb2prior_21Jul.txt', 'chain_heft_heft_kmax0.7_SRD_fix_bs2_blap_toZero_negativeb2prior_21Jul.txt'],
#                ['chain_heft_heft_kmax0.1_SRD_negativeb2prior_21Jul.txt', 'chain_heft_heft_kmax0.3_SRD_negativeb2prior_21Jul.txt', 'chain_heft_heft_kmax0.5_SRD_negativeb2prior_21Jul.txt', 'chain_heft_heft_kmax0.7_SRD_negativeb2prior_21Jul.txt']]
# new extrapolation with negative prior on b2 and free logMC
chain_files = [['chain_heft_b1_kmax0.1_SRD_negativeb2prior_logMc_31Jul.txt', 'chain_heft_b1_kmax0.3_SRD_negativeb2prior_logMc_31Jul.txt', 'chain_heft_b1_kmax0.5_SRD_negativeb2prior_logMc_31Jul.txt', 'chain_heft_b1_kmax0.7_SRD_negativeb2prior_logMc_31Jul.txt'],
               ['chain_heft_heft_kmax0.1_SRD_negativeb2prior_fix_bs2_blap_logMc_28Jul.txt', 'chain_heft_heft_kmax0.3_SRD_negativeb2prior_fix_bs2_blap_logMc_28Jul.txt', 'chain_heft_heft_kmax0.5_SRD_negativeb2prior_fix_bs2_blap_logMc_28Jul.txt', 'chain_heft_heft_kmax0.7_SRD_negativeb2prior_fix_bs2_blap_logMc_28Jul.txt'],
               ['chain_heft_heft_kmax0.1_SRD_negativeb2prior_logMc_28Jul.txt', 'chain_heft_heft_kmax0.3_SRD_negativeb2prior_logMc_28Jul.txt', 'chain_heft_heft_kmax0.5_SRD_negativeb2prior_logMc_28Jul.txt', 'chain_heft_heft_kmax0.7_SRD_negativeb2prior_logMc_25Jul.txt'],]

neutrino_chains = [['chain_heft_heft_kmax0.1_SRD_negativeb2prior_fix_bs2_blap_with_logMc_Mnu_w0wa_29Jul.txt', 'chain_heft_heft_kmax0.3_SRD_negativeb2prior_fix_bs2_blap_with_logMc_Mnu_w0wa_29Jul.txt', 'chain_heft_heft_kmax0.5_SRD_negativeb2prior_fix_bs2_blap_with_logMc_Mnu_w0wa_29Jul.txt', 'chain_heft_heft_kmax0.7_SRD_negativeb2prior_fix_bs2_blap_with_logMc_Mnu_w0wa_28Jul.txt'],]

name_save = 'b1_fix_bs2_blap_logMc'

print('------ Omegam, sigma8 -------')
fom_Omegam_sigma8 = np.zeros((len(chain_files), len(chain_files[0])))
fob_Omegam_sigma8 = np.zeros((len(chain_files), len(chain_files[0])))
# calculate FoM for Omega_m and sigma8  
for i in range(len(chain_files)):
    for j in range(len(chain_files[i])):
        fom_Omegam_sigma8[i,j] = get_FoM(chain_files[i][j], ['Omega_m', 'sigma8_cb'])
        fob_Omegam_sigma8[i,j] = get_FoB(chain_files[i][j], {'Omega_m':0.31, 'sigma8_cb':0.83}, ['Omega_m', 'sigma8_cb'])
    print('FoM: ', fom_Omegam_sigma8[i])
    print('FoB: ', fob_Omegam_sigma8[i])

print('-------- neutrinos ----------')
fom_neutrinos = np.zeros((len(neutrino_chains), len(neutrino_chains[0])))
fob_neutrinos = np.zeros((len(neutrino_chains), len(neutrino_chains[0])))
# calculate FoM for Omega_m and sigma8  
for i in range(len(neutrino_chains)):
    for j in range(len(neutrino_chains[i])):
        fom_neutrinos[i,j] = get_FoM(neutrino_chains[i][j], ['Mnu'])
        fob_neutrinos[i,j] = get_FoB(neutrino_chains[i][j], {'Mnu':0.06}, ['Mnu'])
    print('FoM: ', fom_neutrinos[i])
    print('FoB: ', fob_neutrinos[i])


fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
ax[0].plot([0.1, 0.3, 0.5, 0.7], fom_neutrinos[0], 'o-', color=my_colors[0])
ax[0].set_ylabel(r'FoM $(M_\nu)$')
ax[1].plot([0.1, 0.3, 0.5, 0.7], fob_neutrinos[0], 'o-', color=my_colors[0], label='logMc, w0wa, heft with fixed bs2 and blap') 
ax[1].axhspan(0, 1., facecolor='grey', alpha=0.1)
ax[1].axhspan(0, 2., facecolor='grey', alpha=0.1)
ax[1].set_xlim(0.09, 0.71)
ax[1].set_ylim(-0.1, 3)
ax[1].set_xlabel(r'$k_{\rm max}\, h/$Mpc')
ax[1].set_ylabel(r'FoB $(M_\nu)$')
ax[1].legend(loc='upper left')
fig.tight_layout()
plt.savefig(folder + '/../figs/other/FoM_FoB_neutrinos.png', bbox_inches='tight', pad_inches=0.2)


fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
ax[0].plot([0.1, 0.3, 0.5, 0.7], fom_Omegam_sigma8[0], 'o-', color=my_colors[0])
ax[0].plot([0.1, 0.3, 0.5, 0.7], fom_Omegam_sigma8[1], 'o-', color=my_colors[1])
ax[0].plot([0.1, 0.3, 0.5, 0.7], fom_Omegam_sigma8[2], 'o-', color=my_colors[2])
ax[0].set_ylabel(r'FoM $(\Omega_m,\sigma_8)$', fontsize=15)
# Set scientific notation for ax[0] y-axis
ax[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax[0].tick_params(labelsize=15)
#
# ax[1].grid(color='gray', alpha=0.1)
ax[1].plot([0.1, 0.3, 0.5, 0.7], fob_Omegam_sigma8[0], 'o-', color=my_colors[0], label=r'Linear bias')
ax[1].plot([0.1, 0.3, 0.5, 0.7], fob_Omegam_sigma8[1], 'o-', color=my_colors[1], label=r'HEFT with fixed $b_{s^2}=b_{\nabla^2}=0$')
ax[1].plot([0.1, 0.3, 0.5, 0.7], fob_Omegam_sigma8[2], 'o-', color=my_colors[2], label=r'HEFT')
ax[1].axhspan(0, 1.52, facecolor='gray', alpha=0.1)
ax[1].axhspan(0, 2.49, facecolor='gray', alpha=0.1)
ax[1].set_xlim(0.09, 0.71)
ax[1].set_xlabel(r'$k_{\rm max}\, [h/$Mpc]', fontsize=15)
ax[1].set_ylabel(r'FoB $(\Omega_m,\sigma_8)$', fontsize=15)
ax[1].tick_params(labelsize=15)
ax[1].legend(loc='upper left', fontsize=13)
fig.suptitle(r'Baryonic feedback (free log$M_{\rm C}$)', fontsize=15)
fig.tight_layout()
plt.savefig(folder + '/../figs/other/FoM_FoB_' + name_save + '.png', bbox_inches='tight', pad_inches=0.2)



# ------------ #
# FoM steepness
# ------------ #
plt.figure(figsize=(7,4))
plt.plot([0.1, 0.3, 0.5, 0.7], fom_Omegam_sigma8[0]/np.linalg.norm(fom_Omegam_sigma8[0]), 'o-', color=my_colors[1], label='Omegam, sigma8 (free logMc, fix h.o. bias)')
plt.plot([0.1, 0.3, 0.5, 0.7], fom_neutrinos[0]/np.linalg.norm(fom_neutrinos[0]), 'o-', color=my_colors[2], label='Mnu (free logMc, w0wa, fix h.o. bias)')
plt.ylabel(r'normalized FoM')
plt.xlabel(r'$k_{\rm max}$')
plt.legend()
plt.savefig(folder + '/../figs/other/FoM_omegam_sigma8_vs_Mnu_steepness.png', bbox_inches='tight', pad_inches=0.2)