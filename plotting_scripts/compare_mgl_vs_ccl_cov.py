import pyccl as ccl
import itertools
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

# non-linear matter power spectrum from ccl
bemu_nl = ccl.BaccoemuNonlinear()
cosmo_nl = ccl.Cosmology(Omega_c=params['Omega_c'], Omega_b=params['Omega_b'], h=params['h'], n_s=params['ns'], A_s=params['As'],
                      m_nu=params['Mnu'], transfer_function='boltzmann_camb',
                      matter_power_spectrum=bemu_nl)
bemu_lin =ccl.BaccoemuLinear()
cosmo_lin = ccl.Cosmology(Omega_c=params['Omega_c'], Omega_b=params['Omega_b'], h=params['h'], n_s=params['ns'], A_s=params['As'],
                      m_nu=params['Mnu'], transfer_function='boltzmann_camb',
                      matter_power_spectrum=bemu_lin)

# tracers
ccl_tracers = {}
tracer_noise = {}
bias_ia = params['a1_IA']*(1.+zz)**params['eta1_IA']

for i in range(nbin):
    ccl_tracers['src'+str(i)] = ccl.WeakLensingTracer(cosmo_nl, dndz=(zz, MGL.Survey.eta_z_s[:,i]),ia_bias=(zz,bias_ia)) #CCL automatically normalizes dNdz
    tracer_noise['src'+str(i)] = MGL.Survey.noise['LL']
    bias_gal = bias1_arr[i]*np.ones(len(zz))
    ccl_tracers['lens'+str(i)] = ccl.NumberCountsTracer(cosmo_nl, has_rsd=False, dndz=(zz, MGL.Survey.eta_z_l[:,i]), bias=(zz, bias_gal))
    tracer_noise['lens'+str(i)] = MGL.Survey.noise['GG']


def get_covariance_block(
        tracer_comb1, #e.g. shear-shear is (0,0)
        tracer_comb2,
    ):
        """Compute a single covariance matrix for a given pair of C_ell.

        Args:
            tracer_comb1 (list): List of the pair of tracer names of C_ell^1
            tracer_comb2 (list): List of the pair of tracer names of C_ell^2
        
            lmax (int, optional): Maximum ell up to which to compute the
            covariance

        Returns:
            array: The covariance block
        """
        cosmo = cosmo_nl 
        ell_bins, ell_edges = MGL.Survey.l_wl_bin_centers, MGL.Survey.ell_bin_edges
        ell = np.arange(MGL.Survey.lmin, MGL.Survey.lmax+1).astype(int)
        #ccl_tracers, tracer_Noise = [lens, cluster], [MGL.Survey.noise['LL'], MGL.Survey.noise['GG']]


        cl = {}
        cl[13] = ccl.angular_cl(
            cosmo,
            ccl_tracers[tracer_comb1[0]],
            ccl_tracers[tracer_comb2[0]],
            ell,
        )
        cl[24] = ccl.angular_cl(
            cosmo,
            ccl_tracers[tracer_comb1[1]],
            ccl_tracers[tracer_comb2[1]],
            ell,
        )
        cl[14] = ccl.angular_cl(
            cosmo,
            ccl_tracers[tracer_comb1[0]],
            ccl_tracers[tracer_comb2[1]],
            ell,
        )
        cl[23] = ccl.angular_cl(
            cosmo,
            ccl_tracers[tracer_comb1[1]],
            ccl_tracers[tracer_comb2[0]],
            ell,
        )

        SN = {}
        SN[13] = (
            tracer_noise[tracer_comb1[0]] if tracer_comb1[0] == tracer_comb2[0] else 0
        )
        SN[24] = (
            tracer_noise[tracer_comb1[1]] if tracer_comb1[1] == tracer_comb2[1] else 0
        )
        SN[14] = (
            tracer_noise[tracer_comb1[0]] if tracer_comb1[0] == tracer_comb2[1] else 0
        )
        SN[23] = (
            tracer_noise[tracer_comb1[1]] if tracer_comb1[1] == tracer_comb2[0] else 0
        )

        cov = np.diag(
            (cl[13] + SN[13]) * (cl[24] + SN[24])
            + (cl[14] + SN[14]) * (cl[23] + SN[23])
        )


        norm = (2 * ell + 1) * np.gradient(ell) * MGL.Survey.fsky
        cov /= norm

        lb, cov = bin_cov(r=ell, r_bins=ell_edges, cov=cov)

        return cov

def bin_cov(r, cov, r_bins):  # works for cov and skewness
    """Function to apply the binning operator.

    This function works on both one dimensional vectors and two dimensional
    covariance covrices.

    Args:
        r: theta or ell values at which the un-binned vector is computed.
        cov: Unbinned covariance. It also works for a vector of C_ell or xi
        r_bins: theta or ell bins to which the values should be binned.

    Returns:
        array_like: Binned covariance or vector of C_ell or xi
    """
    bin_center = 0.5 * (r_bins[1:] + r_bins[:-1])
    n_bins = len(bin_center)
    ndim = len(cov.shape)
    cov_int = np.zeros([n_bins] * ndim, dtype="float64")
    norm_int = np.zeros([n_bins] * ndim, dtype="float64")
    bin_idx = np.digitize(r, r_bins) - 1
    r2 = np.sort(
        np.unique(np.append(r, r_bins))
    )  # this takes care of problems around bin edges
    dr = np.gradient(r2)
    r2_idx = [i for i in np.arange(len(r2)) if r2[i] in r]
    dr = dr[r2_idx]
    r_dr = r * dr

    ls = ["i", "j", "k", "ell"]
    s1 = ls[0]
    s2 = ls[0]
    r_dr_m = r_dr
    for i in np.arange(ndim - 1):
        s1 = s2 + "," + ls[i + 1]
        s2 += ls[i + 1]
        # works ok for 2-d case
        r_dr_m = np.einsum(s1 + "->" + s2, r_dr_m, r_dr)

    cov_r_dr = cov * r_dr_m
    for indxs in itertools.product(
        np.arange(min(bin_idx), n_bins), repeat=ndim
    ):
        norm_ijk = 1
        cov_t = []
        for nd in np.arange(ndim):
            slc = [slice(None)] * (ndim)
            slc[nd] = bin_idx == indxs[nd]
            if nd == 0:
                cov_t = cov_r_dr[slc[0]][:, slc[1]]
            else:
                cov_t = cov_t[slc[0]][:, slc[1]]
            norm_ijk *= np.sum(r_dr[slc[nd]])
        if norm_ijk == 0:
            continue
        cov_int[indxs] = np.sum(cov_t) / norm_ijk
        norm_int[indxs] = norm_ijk
    return bin_center, cov_int

# TXPipe: https://github.com/LSSTDESC/TXPipe/blob/master/txpipe/covariance.py
# TJPCov: https://github.com/LSSTDESC/TJPCov/blob/master/tjpcov/covariance_gaussian_fsky.py

tracer_combs = [('src0', 'src0'), ('src0', 'src1'), ('src0', 'src2'), ('src0', 'src3'), ('src0', 'src4'),
                                  ('src1', 'src1'), ('src1', 'src2'), ('src1', 'src3'), ('src1', 'src4'),
                                                    ('src2', 'src2'), ('src2', 'src3'), ('src2', 'src4'),
                                                                      ('src3', 'src3'), ('src3', 'src4'),
                                                                                        ('src4', 'src4'),
                ('src0', 'lens0'), ('src0', 'lens1'), ('src0', 'lens2'), ('src0', 'lens3'), ('src0', 'lens4'),
                ('src1', 'lens0'), ('src1', 'lens1'), ('src1', 'lens2'), ('src1', 'lens3'), ('src1', 'lens4'),
                ('src2', 'lens0'), ('src2', 'lens1'), ('src2', 'lens2'), ('src2', 'lens3'), ('src2', 'lens4'),
                ('src3', 'lens0'), ('src3', 'lens1'), ('src3', 'lens2'), ('src3', 'lens3'), ('src3', 'lens4'),
                ('src4', 'lens0'), ('src4', 'lens1'), ('src4', 'lens2'), ('src4', 'lens3'), ('src4', 'lens4'),
                ('lens0', 'lens0'), ('lens0', 'lens1'), ('lens0', 'lens2'), ('lens0', 'lens3'), ('lens0', 'lens4'),
                                    ('lens1', 'lens1'), ('lens1', 'lens2'), ('lens1', 'lens3'), ('lens1', 'lens4'),
                                                        ('lens2', 'lens2'), ('lens2', 'lens3'), ('lens2', 'lens4'),
                                                                            ('lens3', 'lens3'), ('lens3', 'lens4'),
                                                                                                ('lens4', 'lens4')
                
                ]
N2pt = len(tracer_combs)
cov_full = np.zeros((MGL.Survey.lbin * N2pt, MGL.Survey.lbin * N2pt))
for i in range(N2pt):
    tracer_comb1 = tracer_combs[i]
    for j in range(i, N2pt):
        tracer_comb2 = tracer_combs[j]
        print(f"Computing {tracer_comb1} x {tracer_comb2}: chunk ({i},{j}) of ({N2pt},{N2pt})")
        cov_ij = get_covariance_block(
                        tracer_comb1=tracer_comb1,
                        tracer_comb2=tracer_comb2
                    )
        # Find the right location in the matrix
        start_i = i * MGL.Survey.lbin
        start_j = j * MGL.Survey.lbin
        end_i = start_i + MGL.Survey.lbin
        end_j = start_j + MGL.Survey.lbin
        # and fill it in, and the transpose component
        cov_full[start_i:end_i, start_j:end_j] = cov_ij
        cov_full[start_j:end_j, start_i:end_i] = cov_ij.T


ccl_cov = cov_full
# ---- # 
# plot
# ---- #
cmap = 'seismic'
#cov = MGL.data_covariance
cov = ccl_cov
ndata = cov.shape[0]
pp_norm = np.zeros((ndata,ndata))
for i in range(ndata):
    for j in range(ndata):
        pp_norm[i][j] = cov[i][j]/np.sqrt(cov[i][i]*cov[j][j])      
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im3 = ax.imshow(pp_norm, cmap=cmap, vmin=-1, vmax=1)
fig.colorbar(im3, orientation='vertical')
plt.show()
