import os
import matplotlib.pyplot as plt
import numpy as np
folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder+'/../chains/')

# check FoM and FoB examples here: https://arxiv.org/pdf/1810.10104
# my_colors = ['#FAC13B', '#48849C', '#F5998E', '#BCDCDA', '#A32F28']
my_colors = ['#8ECAE6', '#F58300', '#78206E', '#ED91DC', '#035177', '#FFDD03']


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

chain_files = [['chain_HEFTdata_b1model_kmax0.1_noSystematics-fcfs_03Apr.txt', 'chain_HEFTdata_b1model_kmax0.3_noSystematics_28Apr.txt', 'chain_HEFTdata_b1model_kmax0.5_noSystematics_29Apr.txt', 'chain_HEFTdata_b1model_kmax0.7_noSystematics_28Apr.txt'],  #'chain_HEFTdata_b1model_kmax0.2_noSystematics_21Apr.txt',
               ['chain_heft_heft_kmax0.1_fixed_bs2_bLap_toZero_15May.txt', 'chain_heft_heft_kmax0.3_fixed_bs2_bLap_toZero_13May.txt', 'chain_heft_heft_kmax0.4_fixed_bs2_bLap_toZero_15May.txt', 'chain_heft_heft_kmax0.5_fixed_bs2_bLap_toZero_15May.txt', 'chain_heft_heft_kmax0.7_fixed_bs2_bLap_toZero_13May.txt'],
               ['chain_HEFTdata_HEFTmodel_kmax0.1_noSystematics-fcfs_03Apr.txt', 'chain_HEFTdata_HEFTmodel_kmax0.3_noSystematics_28Apr.txt', 'chain_HEFTdata_HEFTmodel_kmax0.5_noSystematics_28Apr.txt', 'chain_HEFTdata_HEFTmodel_kmax0.7_noSystematics-fcfs_07Apr.txt']]
# with SRD bias fiducials           
chain_files = [['chain_heft_b1_kmax0.1_SRD_23Jun.txt', 'chain_heft_b1_kmax0.3_SRD_23Jun.txt', 'chain_heft_b1_kmax0.5_SRD_23Jun.txt', 'chain_heft_b1_kmax0.7_SRD_20Jun.txt'],
                   ['chain_heft_heft_kmax0.1_SRD_fixing_bs2_bLap_toZero_19Jun.txt', 'chain_heft_heft_kmax0.3_SRD_fixing_bs2_bLap_toZero_19Jun.txt', 'chain_heft_heft_kmax0.5_SRD_fixing_bs2_bLap_toZero_19Jun.txt', 'chain_heft_heft_kmax0.7_SRD_fixing_bs2_bLap_toZero_19Jun.txt'],
                   ['chain_heft_heft_kmax0.1_SRD_20Jun.txt', 'chain_heft_heft_kmax0.3_SRD_20Jun.txt', 'chain_heft_heft_kmax0.5_SRD_20Jun.txt', 'chain_heft_heft_kmax0.7_SRD_20Jun.txt']]
# fixed bias varying baryons 
chain_files = [['chain_heft_heft_kmax0.1_fixbias_7baryons_23Jun.txt', 'chain_heft_heft_kmax0.3_fixbias_7baryons_23Jun.txt', 'chain_heft_heft_kmax0.5_fixbias_7baryons_23Jun.txt', 'chain_heft_heft_kmax0.7_fixbias_7baryons_23Jun.txt'],
               ['chain_heft_heft_kmax0.1_fixbias_logMcEtaBeta_23Jun.txt', 'chain_heft_heft_kmax0.3_fixbias_logMcEtaBeta_23Jun.txt', 'chain_heft_heft_kmax0.5_fixbias_logMcEtaBeta_23Jun.txt', 'chain_heft_heft_kmax0.7_fixbias_logMcEtaBeta_23Jun.txt'],
               ['chain_heft_heft_kmax0.1_fixbias_logMc_23Jun.txt', 'chain_heft_heft_kmax0.3_fixbias_logMc_23Jun.txt', 'chain_heft_heft_kmax0.5_fixbias_logMc_23Jun.txt', 'chain_heft_heft_kmax0.7_fixbias_logMc_23Jun.txt']]

name_save = 'baryons_fixed_bias'

fom_Omegam_sigma8 = np.zeros((len(chain_files), len(chain_files[0])))
fob_Omegam_sigma8 = np.zeros((len(chain_files), len(chain_files[0])))
# calculate FoM for Omega_m and sigma8  
for i in range(len(chain_files)):
    for j in range(len(chain_files[i])):
        fom_Omegam_sigma8[i,j] = get_FoM(chain_files[i][j], ['Omega_m', 'sigma8_cb'])
        fob_Omegam_sigma8[i,j] = get_FoB(chain_files[i][j], {'Omega_m':0.31, 'sigma8_cb':0.83}, ['Omega_m', 'sigma8_cb'])
    print(fom_Omegam_sigma8[i])


fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
ax[0].plot([0.1, 0.3, 0.5, 0.7], fom_Omegam_sigma8[0], 'o-', color=my_colors[0])
ax[0].plot([0.1, 0.3, 0.5, 0.7], fom_Omegam_sigma8[1], 'o-', color=my_colors[1])
ax[0].plot([0.1, 0.3, 0.5, 0.7], fom_Omegam_sigma8[2], 'o-', color=my_colors[2])
# ax[0].set_ylim(1.6e5, 3.3e5)
ax[0].set_ylabel(r'FoM $(\Omega_m,\sigma_8)$ ')
# ax[0].set_yscale('log')
#
ax[1].plot([0.1, 0.3, 0.5, 0.7], fob_Omegam_sigma8[0], 'o-', color=my_colors[0], label='all 7 parameters')  #label=r'HEFT data, linear bias model')
ax[1].plot([0.1, 0.3, 0.5, 0.7], fob_Omegam_sigma8[1], 'o-', color=my_colors[1], label=r'log$M_C$, log$\eta$, log$\beta$')  #label=r'HEFT data, HEFT model, fixed $b_{s^2}=0$ and $b_{\nabla^2}=0$')
ax[1].plot([0.1, 0.3, 0.5, 0.7], fob_Omegam_sigma8[2], 'o-', color=my_colors[2], label=r'log$M_C$')  #label=r'HEFT data, HEFT model')
ax[1].axhspan(0, 1.52, facecolor='grey', alpha=0.1)
ax[1].axhspan(0, 2.49, facecolor='grey', alpha=0.1)
ax[1].set_xlim(0.09, 0.71)
# ax[1].set_ylim(-0.2, 15)
ax[1].set_xlabel(r'$k_{\rm max}$')
ax[1].set_ylabel(r'FoB ($N_\sigma$)')
ax[1].legend(loc='upper left')
# ax[1].set_xticks(range(len(chain_files[0])), ['0.1', '0.7'])
fig.tight_layout()
# fig.suptitle(r'$\Omega_m$ and $\sigma_8$ FoM and FoB')
# plt.subplots_adjust(hspace=0)
plt.savefig('/home/s2561233/Documents/lss/nonlinear-bias-3x2-MG/new-MGlensing/MGlensing/figs/other/FoM_FoB_' + name_save + '.png', transparent=False)
