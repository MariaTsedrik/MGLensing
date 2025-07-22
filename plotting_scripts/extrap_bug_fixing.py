import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import MGLensing
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import baccoemu 


NL_MODEL_HMCODE = 0
NL_MODEL_BACCO = 1

NO_BARYONS = 0
BARYONS_HMCODE = 1
BARYONS_BCEMU = 2
BARYONS_BACCO = 3

BIAS_LIN = 0
BIAS_B1B2 = 1
BIAS_HEFT = 2

folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder)
MGL = MGLensing.MGL("config.yaml")
zz = MGL.Survey.zz_integr
nbin = MGL.Survey.nbin

l_wl_max, l_gc_max = MGL.Survey.ells_wl_max, MGL.Survey.ells_gc_max
l_wl, l_gc, l_xc = MGL.Survey.l_wl, MGL.Survey.l_gc, MGL.Survey.l_xc

def plot_cells(type, cl_ll_list, cl_lg_list, cl_gg_list, show=True, names="", annotation=""):
    types = ['LL', 'LG', 'GG']
    ells = [l_wl, l_xc, l_gc]
    lmax = [l_wl[-1], l_xc[-1], l_gc[-1]]
    lmax_ij = [l_wl_max, l_gc_max, l_gc_max]
    cls = [cl_ll_list, cl_lg_list, cl_gg_list]
    #errs = [err_cl_ll, err_cl_lg, err_cl_gg]
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
                    ax[i, j].legend(loc='upper left', title_fontsize=10, title='bin ' + str(i + 1) + '-' + str(j + 1))
    for i in range(nbin):
        ax[nbin - 1][i].set_xlabel('$\ell$')
    ax[int(nbin / 2)][0].set_ylabel('$C^{\\rm ' + types[type] + '}_{\ell}$')
    ax[1, 1].annotate(annotation, (1.1, 0.05), xycoords='axes fraction', clip_on=False)
    plt.tight_layout()
    plt.show() if show else plt.savefig('figs/modelling/c_ell/c_ells_' + types[type] + '_' + MGL.Survey.survey_name + '.png')

def plot_pmm(z_int_pick, pmm_list, labels, show=True, name=""):
    fig, ax = plt.subplots(2, 3, figsize=(15, 10), facecolor='w')
    for i in range(3):
        for j, pmm in enumerate(pmm_list):
            ax[0][i].loglog(k_ell[:, z_int_pick[i]], pmm[:, z_int_pick[i]], label=labels[j] if i == 0 else '')
            ax[1][i].semilogx(k_ell[:, z_int_pick[i]], pmm[:, z_int_pick[i]]/pmm_list[0][:, z_int_pick[i]])
        ax[0][i].legend(title=f'$z={zz[z_int_pick[i]]:.2f}$', loc='upper right', frameon=True)
        ax[1][i].set_xlabel("$k$ [$h$/Mpc]")
        secax = ax[0][i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
        secax.set_xlabel('$\ell$')
    ax[0][0].set_ylabel("$P^{\\rm mm}(k(\ell, z), z)$")
    ax[1][0].set_ylabel("$P^{\\rm mm}/P_{\\rm ref}^{\\rm mm}$")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f'figs/modelling/power_spectra/pmm_of_k_{name}.png')

def plot_pgg(z_int_pick, bini, binj, pgg_list, labels, show=True, name=""):
    fig, ax = plt.subplots(2, 3, figsize=(12, 7), sharex=True,  facecolor='w')
    for i in range(3):
        for j, pgg in enumerate(pgg_list):
            ax[0][i].loglog(k_ell[:, z_int_pick[i]], pgg[:, z_int_pick[i], bini-1, binj-1], label=labels[j] if i == 0 else '', linestyle='--' if j==2 else '-')
            #interp = interp1d(k, pgg_bacco[2-i, :, bini-1, binj-1], bounds_error=False, fill_value='extrapolate')
            ax[0][i].loglog(k, pgg_bacco[2-i, :, bini-1, binj-1], label='bacco' if i == 0 else '', linestyle='--' if j==2 else '-')
            #ax[0][i].loglog(k_ell[:, z_int_pick[i]], interp(k_ell[:, z_int_pick[i]]), linestyle='--' if j==2 else '-')
            ax[0][i].loglog(k, pgg_bacco_no_b2i_plus_b2j[2-i, :, bini-1, binj-1], label='bacco$-(b_2^i+b_2^j)P_{\\rm dmd2}$' if i == 0 else '', linestyle='--' if j==2 else '-')
            for ind in range(len(heft_terms_labels)):
                ax[1][i].semilogx(k, heft_terms[ind, 2-i, :],  label=heft_terms_labels[ind] if i == 0 else '', linestyle='--' if j==2 else '-')

        ax[0][i].legend(title=f'$z={zz[z_int_pick[i]]:.2f}$ in bin {bini}-{binj}', loc='lower left', frameon=True)
        ax[1][i].set_xlabel("$k$ [$h$/Mpc]")
        secax = ax[0][i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
        secax.set_xlabel('$\ell$')
    ax[1][0].legend(loc='upper right', frameon=True)    
    ax[0][0].set_ylabel("$P^{\\rm gg}(k(\ell, z), z)$")
    ax[1][0].set_ylabel("$P^{\\rm gg}_{ij, \\rm heft}$")
    ax[0][0].set_xlim(0.005, 1.)
    for i in range(3):
        #ax[1][i].set_ylim(0.8, 1.2)
        ax[0][i].set_ylim(10**(-12), 10**6)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f'figs/modelling/power_spectra/pgg_of_k_{name}.png')



params = {'sigma8_cb': 0.77599533, 
 'b1L_1': 1.44123401, 
 'b1L_2': 0.17638873, 
 'b1L_3': 0.59984704, 
 'b1L_4': 1.94382065, 
 'b1L_5': 1.57190398, 
 'b2L_1': -1.78099796, 
 'b2L_2': -1.23844968, 
 'b2L_3': -1.35003984, 
 'b2L_4': -1.08479841, 
 'b2L_5': -0.08253111, 
 'bs2L_1': 0., #-0.20254223, 
 'bs2L_2': 0., #1.39277313, 
 'bs2L_3': 0., #0.1529742, 
 'bs2L_4': 0., #-0.43038101, 
 'bs2L_5': 0., #0.07061175, 
 'blaplL_1': 0., #-0.99755141, 
 'blaplL_2': 0., #-1.95117061, 
 'blaplL_3': 0., #1.8938422, 
 'blaplL_4': 0., #2.71822122, 
 'blaplL_5': 0., #-0.08841235, 
 'Omega_cb': 0.3086053923645651,
 'Omega_m': 0.31, 'Omega_b': 0.05, 'h': 0.68, 'ns': 0.97, 'Mnu': 0.06, 'w0': -1.0, 'wa': 0.0, 'a1_IA': 0.16, 'eta1_IA': 1.66, 'beta_IA': 0.0}

models = {
    'bacco': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_HEFT, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.}
    }

#cl_ll_bacco, cl_gg_bacco, cl_lg_bacco = MGL.get_c_ells(params, models['bacco'])[:-1]
#plot_cells(1, [cl_ll_bacco], [cl_lg_bacco], [cl_gg_bacco], show=True, names=['Bacco'], annotation='Bacco')

_, rcom = MGL.get_expansion_and_rcom(params)
# pick 3 redshifts
z_int_pick = [60, 100, 140] #goes from 0 to 199
def k2ell(x, ind=z_int_pick[0]):
    return x*rcom[ind]-0.5
def ell2k(x, ind=z_int_pick[0]):
    return (x+0.5)/rcom[ind]
def k2ell1(x, ind=z_int_pick[1]):
    return x*rcom[ind]-0.5
def ell2k1(x, ind=z_int_pick[1]):
    return (x+0.5)/rcom[ind]
def k2ell2(x, ind=z_int_pick[2]):
    return x*rcom[ind]-0.5
def ell2k2(x, ind=z_int_pick[2]):
    return (x+0.5)/rcom[ind]
f_arr = [k2ell, k2ell1, k2ell2]
f_inv_arr = [ell2k, ell2k1, ell2k2]
# pick the bins
bin_i=5
bin_j=3



heftemulator = baccoemu.Lbias_expansion()
params_bacco = {
                'ns'            :  params['ns'],
                'hubble'        :  params['h'],
                'sigma8_cold'   :  params['sigma8_cb'],
                'omega_baryon'  :  params['Omega_b'],
                'omega_cold'    :  params['Omega_cb'], 
                'neutrino_mass' :  0., 
                'w0'            :  -1.,
                'wa'            :  0.,
                'expfactor'     :   np.array([1./(1.+zz[z_int_pick_i]) for z_int_pick_i in z_int_pick])[::-1]
            }
k, pnn = heftemulator.get_nonlinear_pnn(**params_bacco)

p_dmdm = pnn[0]        
p_dmd1 = pnn[1]
p_dmd2 = pnn[2]
p_dms2 = pnn[3]      
p_dmk2 = pnn[4]     
p_d1d1 = pnn[5]    
p_d1d2 = pnn[6]      
p_d1s2 = pnn[7]    
p_d1k2 = pnn[8]       
p_d2d2 = pnn[9]       
p_d2s2 = pnn[10] 
p_d2k2 = pnn[11]
p_s2s2 = pnn[12]
p_s2k2 = pnn[13] 
p_k2k2 = pnn[14] 

heft_terms = np.array([p_dmdm, p_dmd1, p_d1d1, p_dmd2, p_d1d2, p_d2d2])
print(heft_terms.shape)
heft_terms_labels =['dmdm', 'dmd1', 'd1d1', 'dmd2', 'd1d2', 'd2d2']
bL1 = np.array([params['b1L_'+str(i+1)] for i in range(nbin)])
bL2 = np.array([params['b2L_'+str(i+1)] for i in range(nbin)])
bs2 = np.array([params['bs2L_'+str(i+1)] for i in range(nbin)])
blapl = np.array([params['blaplL_'+str(i+1)] for i in range(nbin)])   
pgg_bacco = (p_dmdm[:,:,None,None]  +
                (bL1[None,None,:,None]+bL1[None,None, None, :]) * p_dmd1[:,:,None,None] +
                (bL1[None,None, :,None]*bL1[None,None, None, :]) * p_d1d1[:,:,None,None] +
                (bL2[None,None, :,None] + bL2[None,None, None, :]) * p_dmd2[:,:,None,None] +
                (bs2[None,None, :,None] + bs2[None,None, None, :]) * p_dms2[:,:,None,None] +
                (bL1[None,None, :,None]*bL2[None,None, None, :] + bL1[None,None, None, :]*bL2[None,None, :,None]) * p_d1d2[:,:,None,None] +
                (bL1[None,None, :,None]*bs2[None,None, None, :] + bL1[None,None, None, :]*bs2[None,None, :,None]) * p_d1s2[:,:,None,None] +
                (bL2[None,None, :,None]*bL2[None,None, None, :]) * p_d2d2[:,:,None,None] +
                (bL2[None,None, :,None]*bs2[None,None, None, :] + bL2[None,None, None, :]*bs2[None,None, :,None]) * p_d2s2[:,:,None,None] +
                (bs2[None,None, :,None]*bs2[None,None, None, :])* p_s2s2[:,:,None,None] +
                (blapl[None,None, :,None] + blapl[None,None, None, :]) * p_dmk2[:,:,None,None] +
                (bL1[None,None, None, :] * blapl[None,None, :,None] + bL1[None,None, :,None] * blapl[None,None, None, :]) * p_d1k2[:,:,None,None] +
                (bL2[None,None, None, :] * blapl[None,None, :,None] + bL2[None,None, :,None] * blapl[None,None, None, :]) * p_d2k2[:,:,None,None] +
                (bs2[None,None, None, :] * blapl[None,None, :,None] + bs2[None,None, :,None] * blapl[None,None, None, :]) * p_s2k2[:,:,None,None] +
                (blapl[None,None, :,None] * blapl[None,None, None, :]) * p_k2k2[:,:,None,None])
pgg_bacco_no_b2i_plus_b2j = pgg_bacco - (bL2[None,None, :,None] + bL2[None,None, None, :]) * p_dmd2[:,:,None,None]

print(pgg_bacco.shape, k.shape)
k_ell, _, p_mg_heft, p_gg_heft = MGL.get_power_spectra(params, models['bacco'])
plot_pgg(z_int_pick, bin_i, bin_j, [p_gg_heft], ['mgl'], show=True, name='')


print(pgg_bacco[2, -10:, bin_i-1, bin_j-1])