import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as itp
import baccoemu
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import MGLensing
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

NL_MODEL_HMCODE = 0
NL_MODEL_BACCO = 1

NO_BARYONS = 0
BARYONS_HMCODE = 1
BARYONS_BCEMU = 2
BARYONS_BACCO = 3

BIAS_LIN = 0
BIAS_B1B2 = 1
BIAS_HEFT = 2


def plot_cl_mgl_vs_ccl(ell, lmax, cl_mgl, cl_mgl_lin, cl_ccl, cl_ccl_lin, err_cl, title, show,  filename, annotation_text=""):
    fig, ax = plt.subplots(nbin, nbin, figsize=(10, 10), sharey='row', sharex=True, gridspec_kw={'wspace': 0.1, 'hspace': 0})
    for i in range(nbin):
        for j in range(nbin):
            if i >= j:
                ax[i,j].loglog(ell, abs((cl_mgl[:,i,j]-cl_ccl[i, j, :])/err_cl[:, i, j]), linewidth=2)
                ax[i,j].loglog(ell, abs((cl_mgl_lin[:,i,j]-cl_ccl_lin[i, j, :])/err_cl[:, i, j]), linewidth=2)
                ax[i, j].set_ylim(1e-4, 5.)
                ax[i,j].text(.4, .7, 'bin %s, %s'%(str(i+1),(j+1)), fontsize="medium", horizontalalignment='center', transform=ax[i,j].transAxes)  if i<2 else ax[i,j].text(.4, .2, 'bin %s, %s'%(str(i+1),(j+1)), fontsize="medium", horizontalalignment='center', transform=ax[i,j].transAxes)
                ax[i, j].axhline(y=1, linestyle='-.', linewidth=2, color='k')
                #ax[i,j].semilogx(ell, (cl_mgl[:,i,j]-cl_ccl[i, j, :])/err_cl[:, i, j], linewidth=2)
                #ax[i,j].semilogx(ell, (cl_mgl_lin[:,i,j]-cl_ccl_lin[i, j, :])/err_cl[:, i, j], linewidth=2)
                #ax[i, j].fill_between(ell, 1, -1, color='tab:pink', alpha=0.1)
                #ax[i, j].set_ylim(-2.4, 2.4)
                #ax[i,j].text(.4, .8, 'bin %s, %s'%(str(i+1),(j+1)), fontsize="medium", horizontalalignment='center', transform=ax[i,j].transAxes)  
                #ax[i, j].axhline(0, color='grey', alpha=0.1)  
                ax[i, j].axvspan(xmin=lmax[i, j], xmax=ell[-1], color='grey', alpha=0.1)
            else:
                ax[i,j].axis('off')       
    fig.text(0.06, 0.5, r'$\Delta C_\ell/\sigma_\ell$', ha='center', va='center', rotation='vertical', fontsize=20)
    fig.text(0.5, 0.04, r'$\ell$', ha='center', va='center', fontsize=20)   
    fig.suptitle(MGL.Survey.survey_name + ' - ' + title, fontsize=20)
    fig.legend(['MGL-CCL nonlinear', 'MGL-CCL linear'], loc='upper right', ncol=1, bbox_to_anchor=(1, 0.93), bbox_transform=fig.transFigure, fontsize=20, frameon=False)
    fig.text(0.8, 0.65, annotation_text, ha='center', va='center', fontsize=12)   
    plt.show() if show else plt.savefig(f'figs/modelling/cls_{filename}.png')
                
def plot_pmm(z_int_pick, pmm_mgl, pmm_mgl_zlin, pmm_ccl, pmm_bacco, show=True, name=""):
    fig, ax = plt.subplots(2, 3, figsize=(14, 6), sharex='col', gridspec_kw={'height_ratios': [2, 1], 'wspace': 0.3, 'hspace': 0})
    for i in range(3):
        k_ = k_ell[:, z_int_pick[i]]
        idx_k = (k_ >= 1e-3) & (k_ < 4.9)
        k_bacco = k_[idx_k]
        ax[0][i].loglog(k_, pmm_mgl[:, z_int_pick[i]], label='MGL: hmcode' if i == 0 else '')
        ax[0][i].loglog(k_, pmm_mgl_zlin[:, z_int_pick[i]], label='MGL: bacco linear' if i == 0 else '')
        ax[0][i].loglog(k_, pmm_ccl[i, :] * params['h']**3, label='CCL: halofit' if i == 0 else '')
        if pmm_bacco[i] is not None:
            ax[0][i].loglog(k_bacco, pmm_bacco[i][0], label='BACCO' if i == 0 else '', color='k', linestyle='--')
        ax[1][i].semilogx(k_, pmm_mgl[:, z_int_pick[i]] / (pmm_ccl[i, :] * params['h']**3))
        ax[1][i].semilogx(k_, pmm_mgl_zlin[:, z_int_pick[i]] / (pmm_ccl[i, :] * params['h']**3))
        if pmm_bacco[i] is not None:
            pmm_interp = itp.interp1d(k_, pmm_ccl[i, :] * params['h']**3, kind='linear')
            ax[1][i].semilogx(k_bacco, pmm_bacco[i][0] / pmm_interp(k_bacco), color='k', linestyle='--')
        ax[0][i].legend(title=f'$z={zz[z_int_pick[i]]:.2f}$', loc='lower left', frameon=False)
        ax[1][i].set_xlabel("$k$ [$h$/Mpc]")
        ax[1][i].set_ylim(0.8, 1.2)
        secax = ax[0][i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
        secax.set_xlabel('$\ell$')
    ax[0][0].set_ylabel("$P^{\\rm mm}(k(\ell, z), z)$ [Mpc$^3$/$h^3$]")
    ax[1][0].set_ylabel("$P^{\\rm mm}_{\\rm MGL}/P_{\\rm CCL}^{\\rm mm}$")
    plt.tight_layout()
    plt.show() if show else plt.savefig(f'figs/modelling/pmm_of_k_{name}.png')

# pick 3 redshifts
#z_int_pick = [60, 105, 130] #goes from 0 to 199
z_int_pick = [110, 150, 190] 
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

# ------------------------- #
# MGL initialization 
# ------------------------- #
MGL = MGLensing.MGL("config.yaml")
zz = MGL.Survey.zz_integr
nbin = MGL.Survey.nbin


l_wl_max, l_gc_max = MGL.Survey.ells_wl_max, MGL.Survey.ells_gc_max
l_wl, l_gc, l_xc = MGL.Survey.l_wl, MGL.Survey.l_gc, MGL.Survey.l_xc

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
}
_, rcom = MGL.get_expansion_and_rcom(params)
b0 = 0.68
bias1_arr = np.array([b0*(1.+z_bin_center_i) for z_bin_center_i in MGL.Survey.z_bin_center_l ])
for bin_i in range(nbin):
    params[f'b1_{bin_i+1}']=bias1_arr[bin_i]
# ------------------------- #
# nonlinear power spectrum
# ------------------------- #
models = {
    'bacco_zextr_lin': {'nl_model': NL_MODEL_BACCO, 'bacco_option': 'z_extrap_linear', 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    'bacco': {'nl_model': NL_MODEL_BACCO, 'bacco_option': 'z_extrap_hmcode', 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.},
    'bacco_lin': {'nl_model': NL_MODEL_BACCO, 'bacco_option': 'linear', 'bias_model': BIAS_LIN, 'ia_model': 0, 'baryon_model': NO_BARYONS, 'photoz_err_model': 0.}
    }

spectra = {}
for key, model in models.items():
    spectra[key] = MGL.get_power_spectra(params, model)

# non-linear matter power spectrum from mgl
k_ell = spectra['bacco'][0]
pmm_mgl_lin, pmg_mgl_lin, pgg_mgl_lin = spectra['bacco_lin'][1:]
pmm_mgl_zlin, pmg_mgl_zlin, pgg_mgl_zlin = spectra['bacco_zextr_lin'][1:]
pmm_mgl, pmg_mgl, pgg_mgl = spectra['bacco'][1:]


# non-linear matter power spectrum from ccl
bemu_nl = ccl.BaccoemuNonlinear()
cosmo_nl = ccl.Cosmology(Omega_c=params['Omega_c'], Omega_b=params['Omega_b'], h=params['h'], n_s=params['ns'], A_s=params['As'],
                      m_nu=params['Mnu'], transfer_function='boltzmann_camb',
                      matter_power_spectrum=bemu_nl)
pmm_ccl = np.array([ccl.nonlin_matter_power(cosmo_nl, k=k_ell[:, z_int_pick[i]]*params['h'], a=1/(1+zz[z_int_pick[i]])) for i in range(3)])
bemu_lin =ccl.BaccoemuLinear()
cosmo_lin = ccl.Cosmology(Omega_c=params['Omega_c'], Omega_b=params['Omega_b'], h=params['h'], n_s=params['ns'], A_s=params['As'],
                      m_nu=params['Mnu'], transfer_function='boltzmann_camb',
                      matter_power_spectrum=bemu_lin)
# non-linear matter power spectrum from bacco directly
baccoemulator = baccoemu.Matter_powerspectrum()
pnl_bacco = []
for i in range(3):
    a_bacco = 1 / (1 + zz[z_int_pick[i]])
    if a_bacco >= 0.4:
        params_bacco = {
            'ns': params['ns'],
            'hubble': params['h'],
            'sigma8_cold': params['sigma8_cb'],
            'omega_baryon': params['Omega_b'],
            'omega_cold': params['Omega_cb'],
            'neutrino_mass': 0.,
            'w0': -1,
            'wa': 0.,
            'expfactor': a_bacco
        }
        k_bacco = k_ell[:, z_int_pick[i]][(k_ell[:, z_int_pick[i]] >= 1e-3) & (k_ell[:, z_int_pick[i]] < 4.9)]
        pnl_bacco.append(baccoemulator.get_nonlinear_pk(k=k_bacco, cold=False, **params_bacco)[1:])
    else:
        pnl_bacco.append(None)
# ---- # 
# plot
# ---- #        
#plot_pmm(z_int_pick, pmm_mgl, pmm_mgl_zlin, pmm_ccl, pnl_bacco, show=False, name="mgl_vs_ccl_highz")


err_cl_ll, err_cl_gg, err_cl_lg  = MGL.get_errorbars(params)
cls_dic = {}
for key, model in models.items():
    cls_dic[key] = MGL.get_c_ells(params, model)

cl_ll_mgl_lin, cl_gg_mgl_lin, cl_lg_mgl_lin = cls_dic['bacco_lin'][:-1]
cl_ll_mgl, cl_gg_mgl, cl_lg_mgl = cls_dic['bacco'][:-1]

# tracers
lens = []
lens_lin = []
bias_ia = params['a1_IA']*(1.+zz)**params['eta1_IA']
cluster = []
cluster_lin = []
for i in range(nbin):
    lens.append(ccl.WeakLensingTracer(cosmo_nl, dndz=(zz, MGL.Survey.eta_z_s[:,i]),ia_bias=(zz,bias_ia))) #CCL automatically normalizes dNdz
    lens_lin.append(ccl.WeakLensingTracer(cosmo_lin, dndz=(zz, MGL.Survey.eta_z_s[:,i]), ia_bias=(zz,bias_ia))) #CCL automatically normalizes dNdz
    bias_gal = bias1_arr[i]*np.ones(len(zz))
    cluster.append(ccl.NumberCountsTracer(cosmo_nl, has_rsd=False, dndz=(zz, MGL.Survey.eta_z_l[:,i]), bias=(zz, bias_gal)))
    cluster_lin.append(ccl.NumberCountsTracer(cosmo_lin, has_rsd=False, dndz=(zz, MGL.Survey.eta_z_l[:,i]), bias=(zz, bias_gal)))

cl_ll_ccl_nl = np.zeros((nbin, nbin, len(l_wl)))
cl_ll_ccl_lin = np.zeros((nbin, nbin, len(l_wl)))
cl_gg_ccl_nl = np.zeros((nbin, nbin, len(l_gc)))
cl_gg_ccl_lin = np.zeros((nbin, nbin, len(l_gc)))
cl_lg_ccl_nl = np.zeros((nbin, nbin, len(l_xc)))
cl_lg_ccl_lin = np.zeros((nbin, nbin, len(l_xc)))
# calculate CCL Cls
for i in range(nbin):
    for j in range(nbin):
        cl_ll_ccl_nl[i,j] = ccl.angular_cl(cosmo_nl, lens[i], lens[j], l_wl) 
        cl_ll_ccl_lin[i,j] = ccl.angular_cl(cosmo_lin, lens_lin[i], lens_lin[j], l_wl) 
        cl_gg_ccl_nl[i,j] = ccl.angular_cl(cosmo_nl, cluster[i], cluster[j], l_gc) 
        cl_gg_ccl_lin[i,j] = ccl.angular_cl(cosmo_lin, cluster_lin[i], cluster_lin[j], l_gc) 
        cl_lg_ccl_nl[i,j] = ccl.angular_cl(cosmo_nl, lens[i], cluster[j], l_xc) 
        cl_lg_ccl_lin[i,j] = ccl.angular_cl(cosmo_lin, lens_lin[i], cluster_lin[j], l_xc) 


# add noise 
for i in range(nbin):
    cl_ll_ccl_nl[i,i] += MGL.Survey.noise['LL']
    cl_ll_ccl_lin[i,i] += MGL.Survey.noise['LL']
    cl_gg_ccl_nl[i,i] += MGL.Survey.noise['GG']
    cl_gg_ccl_lin[i,i] += MGL.Survey.noise['GG']

# ---- # 
# plot
# ---- #
print('C_LL plotting...')
#plot_cl_mgl_vs_ccl(l_wl, l_wl_max, cl_ll_mgl, cl_ll_mgl_lin, cl_ll_ccl_nl, cl_ll_ccl_lin, err_cl_ll, title='$C^{\\rm LL}_\ell$ Shear-Shear', show=True, filename='mgl_vs_cll_shear')
#plot_cl_mgl_vs_ccl(l_gc, l_gc_max, cl_gg_mgl, cl_gg_mgl_lin, cl_gg_ccl_nl, cl_gg_ccl_lin, err_cl_gg, title='$C^{\\rm GG}_\ell$ Galaxy-Galaxy', show=False, filename='mgl_vs_cll_galclust_abs')
plot_cl_mgl_vs_ccl(l_xc, l_gc_max, cl_lg_mgl, cl_lg_mgl_lin, cl_lg_ccl_nl, cl_lg_ccl_lin, err_cl_lg, title='$C^{\\rm LG}_\ell$ Shear-Galaxy', show=False, filename='mgl_vs_cll_crosscorr_abs')

