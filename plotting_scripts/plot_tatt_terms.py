import numpy as np
import fastpt as fpt
from scipy import integrate, interpolate
import baccoemu 
import matplotlib.pyplot as plt
H0_h_c = 1./2997.92458 
# =100/c in Mpc/h Hubble constant conversion factor
C_IA = 0.0134 
# =dimensionsless, C x rho_crit 
PIVOT_REDSHIFT = 0.
    
def ia_tatt_terms(k_lin,  plin,  C_window=.75):
        """
        Computes the terms of the IA TATT model, at 1-loop order.
        For reference on the equations, see arxiv:1708.09247.

        Parameters
        ----------
        wavenumber: float or numpy.ndarray
            wavemode(s) at which to evaluate the intrinsic alignment: klin
        C_window: float
            It removes high frequency modes to avoid ringing effects.
            Default value set to C_window = 0.75 (tested with fast-pt).

        Returns
        -------
        a00e, c00e, a0e0e, a0b0b, ae2e2, ab2b2, a0e2, b0e2, d0ee2, d0bb2:
        numpy.ndarray
            Value(s) of the intrinsic alignment 1-loop order terms as a
            function of the wavemode(s) at a fixed redshift z = 0.
        """

        P_window = None
        # This parameter sets the FAST-PT quantities needed in initialization
        pad_factor = 1
        n_pad = int(pad_factor * len(k_lin))
        f_pt = fpt.FASTPT(k_lin, to_do=['IA'],
                                low_extrap=-5,
                                high_extrap=3,
                                n_pad=n_pad)

        a00e, c00e, a0e0e, a0b0b = f_pt.IA_ta(plin,
                                                   P_window=P_window,
                                                   C_window=C_window)
        ae2e2, ab2b2 = f_pt.IA_tt(plin, P_window=P_window, C_window=C_window)

        a0e2, b0e2, d0ee2, d0bb2 = f_pt.IA_mix(plin,
                                                    P_window=P_window,
                                                    C_window=C_window)

        a00e = interpolate.interp1d(k_lin, a00e, kind='linear',
                                    fill_value='extrapolate')

        c00e = interpolate.interp1d(k_lin, c00e, kind='linear',
                                    fill_value='extrapolate')

        a0e0e = interpolate.interp1d(k_lin, a0e0e, kind='linear',
                                     fill_value='extrapolate')

        a0b0b = interpolate.interp1d(k_lin, a0b0b, kind='linear',
                                     fill_value='extrapolate')

        ae2e2 = interpolate.interp1d(k_lin, ae2e2, kind='linear',
                                     fill_value='extrapolate')

        ab2b2 = interpolate.interp1d(k_lin, ab2b2, kind='linear',
                                     fill_value='extrapolate')

        a0e2 = interpolate.interp1d(k_lin, a0e2, kind='linear',
                                    fill_value='extrapolate')

        b0e2 = interpolate.interp1d(k_lin, b0e2, kind='linear',
                                    fill_value='extrapolate')

        d0ee2 = interpolate.interp1d(k_lin, d0ee2, kind='linear',
                                     fill_value='extrapolate')

        d0bb2 = interpolate.interp1d(k_lin, d0bb2, kind='linear',
                                     fill_value='extrapolate')

        return a00e, c00e, a0e0e, a0b0b, ae2e2, ab2b2, a0e2, b0e2, d0ee2, d0bb2




def normalize_tatt_parameters(params, dz, redshift):
        omegam = params['Omm']
        growth = dz[:, None]
        a1_ia = params['a1_ia']
        a2_ia = params['a2_ia']
        b1_ia = params['b1_ia']
        eta1_ia = params['eta1_ia']
        eta2_ia = params['eta2_ia']
        c1 = -a1_ia * C_IA * omegam * \
            ((1 + redshift) / (1 + PIVOT_REDSHIFT)) ** eta1_ia / growth
        c2 = a2_ia * 5 * C_IA * omegam * \
            ((1 + redshift) / (1 + PIVOT_REDSHIFT)) ** eta2_ia / (growth**2)
        c1d = b1_ia * c1

        return c1, c1d, c2

Omega_m = 0.3
Obh2 = 0.02268
h = 0.68
As = np.exp(3.044)*1e-10
ns = 0.97
params = {
            'ns'            :  ns,
            'A_s'           :  As,
            'hubble'        :  h,
            'omega_baryon'  :  Obh2/h/h,
            'omega_cold'    :  Omega_m,
            'neutrino_mass' :  0.,
            'w0'            : -1.0, # These params are fixed
            'wa'            :  0.0,
            'expfactor'     :  1
        }

def get_plin(kmin, kmax, nbins):
    k_lin = np.logspace(np.log10(kmin), np.log10(kmax), nbins)
    # The emulator only works up to k=50, so split the k vector
    kcut = k_lin[np.where(k_lin <= 50)]
    kext = k_lin[np.where(k_lin > 50)]
    baccoemulator = baccoemu.Matter_powerspectrum()
    _, pl = baccoemulator.get_linear_pk(k=kcut, cold=False, **params)
    # Extrapolation with power law
    m = np.log(pl[-1] / pl[-2]) / np.log(kcut[-1] / kcut[-2])
    pl_ext = pl[-1] / kcut[-1]**m * kext**m
    plin =  np.hstack((pl, pl_ext))
    return k_lin, plin



term_names = ['$A_{0|0E}$', '$C_{0|0E}$', '$A_{0E|0E}$', '$A_{0B|0B}$', '$A_{E2|E2}$'
              '$A_{B2|B2}$', '$A_{0|E2}$', '$B_{0|E2}$', '$D_{0E|E2}$', '$D_{0B|B2}$']
kmin_arr = [1e-4, 1e-3, 1e-3, 1e-3]
kmax_arr = [100, 100, 50, 50]
nbins_arr = [512, 512, 512, 256]
fig, ax = plt.subplots(figsize=(8, 16), nrows = 5, ncols=2, sharex=True,  facecolor='w')
for ind in range(len(nbins_arr)):
       k_lin, plin = get_plin(kmin_arr[ind], kmax_arr[ind], nbins_arr[ind])
       terms = ia_tatt_terms(k_lin, plin)
       for i in range(5):
            for j in range(2):
                if i==j==0:
                    label=f'kmin={kmin_arr[ind]}, kmax={kmax_arr[ind]}, nbins={nbins_arr[ind]}'
                else:
                    label=''    
                ax[i, j].loglog(k_lin, terms[i+2*j], label=label)
                ax[i, j].legend(loc='upper right', title_fontsize=10, title=term_names[i+2*j])    
ax[-1, 0].set_xlabel('$k$ [Mpc/$h$]')            
ax[-1, 1].set_xlabel('$k$ [Mpc/$h$]')            
#plt.show()       
plt.savefig('./figs/tatt_terms.png')