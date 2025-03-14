import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import MGLensing
import matplotlib.pyplot as plt
# Ensure the directories exist
os.makedirs('figs', exist_ok=True)
os.makedirs('figs/modelling', exist_ok=True)


def plot_pk_fields(pmm, pgm, pgg, z_int_pick, bini, binj, show=True, name=""):
    fig, ax = plt.subplots(1, 3, figsize=(12, 5),  facecolor='w')
    labels = ['matter', f'galaxy bin {bini}-{binj}', f'cross bin {bini}', f'cross bin {binj}']
    for i in range(3):
        for j, data in enumerate([pmm, pgg[:, :, bini-1, binj-1], pgm[:, :, bini-1], pgm[:, :, binj-1]]):
            ax[i].loglog(k_ell[:, z_int_pick[i]], data[:, z_int_pick[i]], label=labels[j] if i == 0 else None)
        ax[i].legend(title=f'$z={zz[z_int_pick[i]]:.2f}$', loc='lower left')
        ax[i].set_xlabel("$k$ [$h$/Mpc]")
        secax = ax[i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
        secax.set_xlabel('$\ell$')
    ax[0].set_ylabel("$P^{\\rm NL}(k(\ell, z), z)$")
    plt.tight_layout()
    plt.show() if show else plt.savefig(f'figs/modelling/p_of_k_{name}.png')

def plot_pmm(z_int_pick, pmm_list, labels, show=True, name=""):
    fig, ax = plt.subplots(2, 3, figsize=(12, 5),  facecolor='w')
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
    plt.show() if show else plt.savefig(f'figs/modelling/pmm_of_k_{name}.png')

def plot_pgg(z_int_pick, bini, binj, pgg_list, labels, show=True, name=""):
    fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex=True,  facecolor='w')
    for i in range(3):
        for j, pgg in enumerate(pgg_list):
            ax[0][i].loglog(k_ell[:, z_int_pick[i]], pgg[:, z_int_pick[i], bini-1, binj-1], label=labels[j] if i == 0 else '')
            ax[1][i].semilogx(k_ell[:, z_int_pick[i]], pgg[:, z_int_pick[i], bini-1, binj-1]/pgg_list[0][:, z_int_pick[i], bini-1, binj-1])
        ax[0][i].legend(title=f'$z={zz[z_int_pick[i]]:.2f}$ in bin {bini}-{binj}', loc='lower left', frameon=True)
        ax[1][i].set_xlabel("$k$ [$h$/Mpc]")
        secax = ax[0][i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
        secax.set_xlabel('$\ell$')
    ax[0][0].set_ylabel("$P^{\\rm gg}(k(\ell, z), z)$")
    ax[1][0].set_ylabel("$P^{\\rm gg}/P_{\\rm ref}^{\\rm gg}$")
    ax[0][0].set_xlim(0.005, 1.)
    for i in range(3):
        ax[1][i].set_ylim(0.8, 1.2)
    plt.tight_layout()
    plt.show() if show else plt.savefig(f'figs/modelling/pgg_of_k_{name}.png')

NL_MODEL_HMCODE = 0
NL_MODEL_BACCO = 1

NO_BARYONS = 0
BARYONS_HMCODE = 1
BARYONS_BCEMU = 2
BARYONS_BACCO = 3

BIAS_LIN = 0
BIAS_B1B2 = 1
BIAS_HEFT = 2

MGL = MGLensing.MGL("config.yaml")
zz = MGL.Survey.zz_integr
nbin = MGL.Survey.nbin

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

_, rcom = MGL.get_expansion_and_rcom(params)
# pick 3 redshifts
z_int_pick = [30, 100, 180] #goes from 0 to 199
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
bin_i=2
bin_j=3

models = {
    'hmcode': {'nl_model': NL_MODEL_HMCODE, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    'bacco': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    'hmcode_b1b2': {'nl_model': NL_MODEL_HMCODE, 'bias_model': BIAS_B1B2, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    'hmcode_bar': {'nl_model': NL_MODEL_HMCODE, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': BARYONS_HMCODE, 'photoz_err_model': 0.},
    'bacco_bar': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': BARYONS_BACCO, 'photoz_err_model': 0.},
    'bacco_heft': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_HEFT, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    'bacco_b1b2': {'nl_model': NL_MODEL_BACCO, 'bias_model': BIAS_B1B2, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.}
}

spectra = {}
for key, model in models.items():
    spectra[key] = MGL.get_power_spectra(params, model)

k_ell = spectra['hmcode'][0]
pmm_hm, pmg_hm, pgg_hm = spectra['hmcode'][1:]
pmm_hm_b, pmg_hm_b, pgg_hm_b = spectra['hmcode_bar'][1:]
pmg_hm_b1b2, pgg_hm_b1b2 = spectra['hmcode_b1b2'][2:]
pmm_bc, pmg_bc, pgg_bc = spectra['bacco'][1:]
pmm_bc_b, pmg_bc_b, pgg_bc_b = spectra['bacco_bar'][1:]
pmg_bc_heft, pgg_bc_heft = spectra['bacco_heft'][2:]
pmg_bc_b1b2, pgg_bc_b1b2 = spectra['bacco_b1b2'][2:]

plot_pmm(z_int_pick, [pmm_hm, pmm_hm_b, pmm_bc, pmm_bc_b], ['hmcode', 'hmcode+Tagn', 'bacco', 'bacco+bfc'],  show=True)
plot_pgg(z_int_pick, bin_i, bin_j, [pgg_bc, pgg_bc_b1b2, pgg_bc_heft], ['$b_1$', '$b_1+b_2 k^2$', 'heft'],  show=True)
plot_pk_fields(pmm_bc, pmg_bc, pgg_bc, z_int_pick, bin_i, bin_j, True)
plot_pk_fields(pmm_hm, pmg_hm, pgg_hm, z_int_pick, bin_i, bin_j, True)
