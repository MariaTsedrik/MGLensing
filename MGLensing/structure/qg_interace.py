from scipy.interpolate import RectBivariateSpline
from scipy import interpolate as itp
import numpy as np
from cosmopower import cosmopower_NN
import MGrowth as mg
import os
from math import log10, log
from scipy import interpolate as itp
from scipy.integrate import odeint, quad
from scipy.integrate import cumulative_trapezoid
dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 

C_KM_S_MPC = 2997.92458

emu_ranges_all = {
        'Omega_c':      {'p1': 0.1,         'p2': 0.8},    
        'Omega_b':      {'p1':0.01,         'p2': 0.1},  
        'h':            {'p1': 0.4,         'p2': 1.},      
        'As':           {'p1': 0.495e-9,    'p2': 5.459e-9},
        'ns':           {'p1': 0.6,         'p2': 1.2}, 
        'Mnu':          {'p1': 0.0,         'p2': 0.5},
        'w0':           {'p1': -3.,         'p2': -0.3},    
        'wa':           {'p1': -3.,         'p2': 3.},  
        'log10Tagn':    {'p1': 7.6,         'p2': 8.3}
}

def powerlaw_highk_extrap(pk_or_boost, log_k, k_last, kh_high, zz_num):
    last_entry, lastlast_entry = pk_or_boost[:, -1], pk_or_boost[:, -2]
    m = np.array([log(np.abs(last_entry[i] / lastlast_entry[i])) / log_k for i in range(zz_num)])
    highk_extrap = last_entry[:, np.newaxis] * (kh_high[np.newaxis, :]/k_last)**m[:, np.newaxis]
    return highk_extrap 

def fill_in_ell_z_array(interp, k, lbin, zz_integr, zmax=10.):
    array = np.zeros((lbin, len(zz_integr)), 'float64')
    #old implementation:
    # index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
    # for index_l, index_z in index_pknn:
    #         array[index_l, index_z] = interp(min(zz_integr[index_z], zmax), k[index_l,index_z])  
    mask = (k > k_min_h_by_mpc) & (k < k_max_h_by_mpc)
    index_l, index_z = np.where(mask)
    z_pts = zz_integr[index_z]
    k_pts = k[index_l, index_z]
    array[index_l, index_z] = np.ravel(interp(z_pts, k_pts, grid=False))
    return array

#--- Quantum gravity functions ---
# Equation of state functions
def ratio_R(z, z_q):
    z = np.asarray(z)
    return (1.0 + z_q) / (1.0 + z)

def mu_from_m(m):
    val = 1.0 + 0.75 * m
    if val < 0.0:
        raise ValueError("mu is only real for 1 + 3m/4 >= 0.")
    return np.sqrt(val)


def beta_from_m(m):
    val = -1.0 - 0.75 * m
    if val <= 0.0:
        raise ValueError("beta is only real for 1 + 3m/4 < 0.")
    return 2.0 * np.sqrt(val)

# Equation C10 
def y1_of_m(m):
    mu = mu_from_m(m)
    return -2.0 * (3.0 / 8.0 - ((1.0 - mu) ** 2) / (m**2)) / (1.0 + 2.0 * mu)

# Approximate w0 for log and power law models (see following Eq.23b)
def delta_w0_from_Bm(m, B):
    mu = mu_from_m(m)
    return 2.0 * (m**2) * (B**2) * (1.0 - mu) * y1_of_m(m)


def oscillatory_coeffs(m, A, B):
    # Valid for 1 + 3m/4 < 0
    beta = beta_from_m(m)

    cA = A - 0.5 * beta * B
    cB = B + 0.5 * beta * A

    kappa_A = 1.5 * (m**2) * (A**2) - 4.0 * (cA**2)
    kappa_B = 1.5 * (m**2) * (B**2) - 4.0 * (cB**2)
    kappa_AB = 3.0 * (m**2) * A * B - 8.0 * cA * cB

    delta_w0 = -(kappa_A + kappa_B)

    denom = 2.0 * (1.0 + beta**2)
    kappa_tilde_1 = (
        ((2.0 - beta**2) * (kappa_A - kappa_B) - 3.0 * beta * kappa_AB) / denom
    )
    kappa_tilde_2 = (
        (3.0 * beta * (kappa_A - kappa_B) + (2.0 - beta**2) * kappa_AB) / denom
    )

    delta_w1 = -kappa_tilde_1
    delta_w2 = -kappa_tilde_2
    return delta_w0, delta_w1, delta_w2, beta


def delta_w_oscillatory(z, m, z_q, A, B):
    if 1.0 + 0.75 * m >= 0.0:
        raise ValueError("Oscillatory branch requires 1 + 3m/4 < 0.")
    delta_w0, delta_w1, delta_w2, beta = oscillatory_coeffs(m, A, B)
    R = ratio_R(z, z_q)
    Phi = 1.5 * beta * np.log(R)
    return R**(-6.0) * (
        delta_w0 + delta_w1 * np.cos(2.0 * Phi) + delta_w2 * np.sin(2.0 * Phi)
    )


def delta_w_powerlaw(z, m, z_q, B):
    mu = mu_from_m(m)
    if not (0.0 < mu <= 1.0):
        raise ValueError("Power-law branch requires 0 < mu <= 1.")
    if m > 0.0:
        raise ValueError("Power-law branch requires m <= 0.")
    R = ratio_R(z, z_q)
    delta_w0 = delta_w0_from_Bm(m, B)
    return delta_w0 * R ** (6.0 * (mu - 1.0))


def delta_w_logarithmic(z, z_q, Bqg):
    R = ratio_R(z, z_q)
    delta_w0 = delta_w0_from_Bm(-4/3, Bqg)
    return 9.0 * delta_w0 * R**(-6.0) * np.log(R) ** 2


def w_of_z_osc(z, mqg, zeqqg, Aqg, Bqg):
    """
    Oscillatory quantum-gravity dark-energy equation of state.
    """
    z = np.asarray(z)
    if 1.0 + 0.75 * mqg >= 0.0:
        raise ValueError("w_model='osc' requires m_qg < -4/3.")
    return -1.0 + delta_w_oscillatory(z, m=mqg, z_q=zeqqg, A=Aqg, B=Bqg)


def _log_de_turnoff(z, z_cut, width=None):
    """Cosine taper: 1 for z <= z_cut, fading to 0 over (z_cut, z_cut + width]."""
    z_arr = np.asarray(z, dtype=float)
    if width is None:
        width = max(0.25, 0.2 * z_cut)
    z_hi = z_cut + width
    window = np.ones_like(z_arr)
    window[z_arr > z_hi] = 0.0
    in_transition = (z_arr > z_cut) & (z_arr <= z_hi)
    t = (z_arr[in_transition] - z_cut) / width
    window[in_transition] = 0.5 * (1.0 + np.cos(np.pi * t))
    return window


def w_of_z_log(z, zeqqg, Bqg, z_de_active=None):
    """
    Logarithmic quantum-gravity dark-energy equation of state.

    For z <= z_de_active the profile is unchanged.  Above z_de_active it fades
    smoothly to zero (w -> -1), avoiding blow-up of R^{-6}[log R]^2 at z >> z_q
    when integrating to recombination for the theta -> H0 inversion.
    """
    z_arr = np.asarray(z)
    delta_w = delta_w_logarithmic(z_arr, z_q=zeqqg, Bqg=Bqg)
    if z_de_active is not None:
        delta_w *= _log_de_turnoff(z_arr, z_de_active)
    return -1.0 + delta_w


def w_of_z_pl(z, mqg, zeqqg, Bqg):
    """
    Power-law quantum-gravity dark-energy equation of state.
    """
    z = np.asarray(z)
    mu = mu_from_m(mqg)
    if not (0.0 < mu <= 1.0):
        raise ValueError("w_model='pl' requires 0 < mu(m_qg) <= 1.")
    if mqg > 0.0:
        raise ValueError("w_model='pl' requires m_qg <= 0.")
    return -1.0 + delta_w_powerlaw(z, m=mqg, z_q=zeqqg, B=Bqg)



def w_of_z_cpl(z, w0, wa):
    """
    CPL dark-energy equation of state.
    """
    z = np.asarray(z)
    return w0 + wa * z / (1.0 + z)


def signed_log_parameter(x):
    return np.sign(x) * (10.0**abs(x) - 1.0)


def sampled_m_to_physical_m(mqg, model):
    if model == "osc":
        return -10.0**mqg
    return mqg


def check_AB_prior(mqg, zeqqg, Aqg, Bqg, model, zin=2.0, safety=0.1):
    """
    Hard AB prior:
        |A| << rho_in^2
        |B| << model-dependent bound

    with rho_in = [(1+zq)/(1+zin)]^(3/2).

    `safety` converts << into < safety * bound.
    """

    if model == "cpl":
        return

    rho_in = ((1.0 + zeqqg) / (1.0 + zin))**1.5

    Amax = safety * rho_in**2

    if model == "pl":
        mu = mu_from_m(mqg)
        if not (0.0 < mu <= 1.0):
            raise ValueError(f"Power-law model requires 0 < mu <= 1; got mu={mu}")

        Bmax = safety * rho_in**(2.0 * (1.0 - mu))

    elif model == "log":
        log_rho = np.log(rho_in)
        if np.isclose(log_rho, 0.0):
            raise ValueError("AB prior undefined for log model when rho_in approx 1")

        Bmax = safety * rho_in**2 / abs(log_rho)

    elif model == "osc":
        Bmax = safety * rho_in**2

    else:
        raise ValueError(f"Unknown w_model '{model}'")

    if abs(Aqg) > Amax:
        raise ValueError(
            f"AB prior failed: |A|={abs(Aqg):.4g} > {Amax:.4g} "
            f"for rho_in={rho_in:.4g}"
        )

    if abs(Bqg) > Bmax:
        raise ValueError(
            f"AB prior failed: |B|={abs(Bqg):.4g} > {Bmax:.4g} "
            f"for rho_in={rho_in:.4g}, model={model}"
        )


def w_of_z(z, mqg, zeqqg, Aqg, Bqg, model, zin, safety, ABprior, amp, alpha, beta, z_de_active=None):
    #transformation is applied before calling this function
    if ABprior:
        check_AB_prior(
            mqg=mqg,
            zeqqg=zeqqg,
            Aqg=Aqg,
            Bqg=Bqg,
            model=model,
            zin=zin,
            safety=safety,
        )
    if model == "osc":
        return w_of_z_osc(z, mqg, zeqqg, Aqg, Bqg)
    if model == "log":
        return w_of_z_log(z, zeqqg, Bqg, z_de_active=z_de_active)
    if model == "pl":
        return w_of_z_pl(z, mqg, zeqqg, Bqg)
    if model == "osc_param":
        return w_of_z_osc_param(z, amp, alpha, beta, z_de_active=z_de_active)
    raise ValueError(f"Unknown w_model '{model}'. Expected osc!")

# --- log-model helpers ---
Z_PIV = 0.6
M_LOG = -4.0 / 3.0

def B_from_delta_w0(m, delta_w0, sign=1.0):
    """Recover B from delta_w0 = 2 m^2 B^2 (1-mu) y1(m)."""
    mu = mu_from_m(m)
    denom = 2.0 * (m ** 2) * (1.0 - mu) * y1_of_m(m)
    if denom == 0.0:
        return np.nan
    B2 = delta_w0 / denom
    if B2 < 0.0:
        return np.nan
    return sign * np.sqrt(B2)

def log_pivot_profile(zq, z_piv=Z_PIV):
    """P(z_q) = 9 R_piv^{-6} [log R_piv]^2 in delta_w_log(z_piv) = delta_w0 * P(z_q)."""
    R_piv = (1.0 + zq) / (1.0 + z_piv)
    if R_piv <= 0.0:
        return np.nan
    logRp = np.log(R_piv)
    return 9.0 * R_piv ** (-6.0) * logRp ** 2

def wpiv_basis_to_physical_log(log10_zq, w_piv, z_piv=Z_PIV, B_sign=1.0):
    """Invert w_piv = -1 + delta_w0 P(z_q) for delta_w0, then B."""
    zq = 10.0 ** log10_zq
    dw_piv = w_piv + 1.0
    if dw_piv <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    P = log_pivot_profile(zq, z_piv)
    if not np.isfinite(P) or P <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    delta_w0 = dw_piv / P
    B = B_from_delta_w0(M_LOG, delta_w0, sign=B_sign)
    if not np.isfinite(B):
        return np.nan, np.nan, np.nan, np.nan
    return M_LOG, zq, 0.0, B

def beta_to_m(beta):
    """Physical m from beta (always m < -4/3)."""
    return -(4.0 + beta**2) / 3.0

def C_factor(m_phys):
    """C(m) = delta_w_0 / r^2 in the analytic factorisation."""
    return -1.5 * m_phys * (m_phys + 2.0)


def M_factor(m_phys):
    """Complex coefficient |M| e^{i phi_M} of the oscillatory part.

    Returns (M_magnitude, M_phase) such that the oscillatory contribution to
        delta_w_1 cos(2 Phi) + delta_w_2 sin(2 Phi)
    equals  - r^2 * |M| * cos(2 Phi - 2 theta + phi_M)
    where theta = arctan2(B, A).
    """
    if 1.0 + 0.75 * m_phys >= 0.0:
        return 0.0, 0.0
    beta = 2.0 * np.sqrt(-1.0 - 0.75 * m_phys)
    alpha = 1.5 * m_phys**2 - 3.0 * m_phys - 8.0
    gamma = 4.0 * beta
    denom = 2.0 * (1.0 + beta**2)
    coeff = -((2.0 - beta**2) + 1j * 3.0 * beta) / denom * (alpha - 1j * gamma)
    return abs(coeff), np.angle(coeff)

def w_of_z_osc_param(z, amp, alpha, beta, z_de_active=None):
    """
    Oscillatory quantum-gravity dark-energy equation of state.
    """
    zz = np.asarray(z)
    m  = beta_to_m(beta)
    Cm = C_factor(m)
    Mm, _ = M_factor(m)
    delta_w = amp*(1.+zz)**6 * (Cm/Mm + np.cos(3.0 * beta * np.log(1.+zz) + alpha))
    if z_de_active is not None:
        delta_w *= _log_de_turnoff(zz, z_de_active)
    return -1.0 + delta_w

def w_piv_to_amp_osc(w_piv, alpha, beta, z_piv):
    """w_piv to amp for oscillatory model."""
    m  = beta_to_m(beta)
    Cm = C_factor(m)
    Mm, _ = M_factor(m)
    A1 = Cm/Mm
    amp = (w_piv+1.)/(1.+z_piv)**6 /(A1 + np.cos(3.0 * beta * np.log(1.+z_piv) + alpha))
    return amp


class QuantumGravity():
    def __init__(self, option=None):
        self.zz_pk = np.linspace(0., 3., 256, endpoint=True)
        self.aa_pk = np.array(1./(1.+self.zz_pk[::-1])) # should be increasing
        self.nz_pk = len(self.zz_pk)
        self.zz_max = self.zz_pk[-1]

        self.cp_nl_hmcode_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../../emulators/hmcode2020/log10_total_matter_nonlinear_emu',
                      )
        self.kh_nl = self.cp_nl_hmcode_model.modes # 0.01..50. h/Mpc    
        self.cp_lin_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../../emulators/hmcode2020/log10_total_matter_linear_emu',
                      )
        self.kh_lin = self.cp_lin_model.modes # 3.7e-4..50. h/Mpc IMPORTANT LATER USED IN TATT
        self.kh_lin_left = self.kh_lin[self.kh_lin<self.kh_nl[0]]
        self.kh_tot = np.concatenate((self.kh_lin_left, self.kh_nl))
        self.a_start = 1.0e-4
        self.a_fine = np.hstack((np.logspace(np.log10(self.a_start), np.log10(0.5), 256), np.linspace(0.501, 1.0, 256)))
        self.z_growth = 1./self.a_fine[::-1] - 1.0
        self.config = {
            "w_model": "osc_param",
            "zin": 3.0,
            "safety": 0.1,
            "ABprior": False
        }

        print('initialising quantum gravity')
        self.emu_name = 'quantum gravity'


        if option=='linear':
            self.get_pk_nl =  self.get_pk_lin 
        elif option=='pseudo':
            self.get_pk_nl = self.get_pk_pseudo
        else:
            self.get_pk_nl = self.get_pk_pseudo


    def check_pars(self, params):
        emu_ranges = emu_ranges_all.copy()
        if 'log10Tagn' not in params:
            del emu_ranges['log10Tagn'] 
        eva_pars = emu_ranges.keys()     
        if not all(emu_ranges[par_i]['p1'] <= params[par_i] <= emu_ranges[par_i]['p2'] for par_i in eva_pars):
            return False
        if params['w0']+params['wa']>=0:
            return False 
        

        # w_piv = params["w_piv"]
        # z_piv = Z_PIV
        # alpha = params["alpha"]
        # beta = params["beta"]
        # amp = w_piv_to_amp_osc(w_piv, alpha, beta, z_piv)
        amp = params["amp"]
        if amp<0:
            return False
        return True
    
    def check_pars_ini(self, params):
        emu_ranges = emu_ranges_all.copy()
        if 'log10Tagn' not in params:
            del emu_ranges['log10Tagn'] 
        eva_pars = emu_ranges.keys()     
        # parameters currently available
        avail_pars = [coo for coo in params.keys()]    
        # parameters needed for a computation
        comp_pars = list(set(eva_pars)-set(avail_pars))
        miss_pars = list(set(comp_pars))
        # check missing parameters
        if miss_pars:
            print(f"HMcode2020 emulator:")
            print(f"  Please add the parameter(s) {miss_pars}"
                  f" to your parameters!")
            raise KeyError(f"HMcode2020 emulator: coordinates need the"
                           f" following parameter(s): ", miss_pars)
        pp = [params[p] for p in eva_pars]    
        for i, par in enumerate(eva_pars):
                val = pp[i]
                message = "Parameter {}={} out of bounds [{}, {}]".format(
                par, val, emu_ranges[par]['p1'],
                emu_ranges[par]['p2'])
                assert (np.all(val >= emu_ranges[par]['p1'])
                    & np.all(val <= emu_ranges[par]['p2'])
                    ), message
        # check the w0-wa condition        
        if params['w0']+params['wa']>=0:
             raise KeyError("Stability condition: w0+wa must be negative!")        
        return True

    def get_pk_hmcode_lin_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        params_hmcode = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'hubble'        :  np.full(self.nz_pk, h),
                'omega_baryon'  :  np.full(self.nz_pk, omega_b),
                'omega_cdm'     :  np.full(self.nz_pk, omega_c),
                'neutrino_mass' :  np.zeros(self.nz_pk),
                'w0'            :  np.full(self.nz_pk, -1.),
                'wa'            :  np.zeros(self.nz_pk),
                'z'             :  self.zz_pk
            }
        plin_cp = self.cp_lin_model.ten_to_predictions_np(params_hmcode)
        self.pklin_z0_lcdm = plin_cp[0] # zz_pk[0] must be 0.!
        plin_interp = RectBivariateSpline(self.zz_pk,
                            self.kh_lin,
                            plin_cp,
                            kx=1, ky=1)     
        return  plin_interp
    

    def get_pk_hmcode_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        params_hmcode = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  a_s if isinstance(a_s, np.ndarray) and len(a_s) == self.nz_pk else np.full(self.nz_pk, a_s),
                'hubble'        :  np.full(self.nz_pk, h),
                'omega_baryon'  :  np.full(self.nz_pk, omega_b),
                'omega_cdm'     :  np.full(self.nz_pk, omega_c),
                'neutrino_mass' :  np.zeros(self.nz_pk),
                'w0'            :  np.full(self.nz_pk, -1.),
                'wa'            :  np.zeros(self.nz_pk),
                'z'             :  self.zz_pk
            }
        pnl_cp  = self.cp_nl_hmcode_model.ten_to_predictions_np(params_hmcode)
        plin_cp = self.cp_lin_model.ten_to_predictions_np(params_hmcode)
        self.pklin_z0_lcdm = plin_cp[0] # zz_pk[0] must be 0.!
        plin_left = plin_cp[:, self.kh_lin<self.kh_nl[0]]
        pnl  = np.concatenate((plin_left, pnl_cp),axis=1)
        pnl_interp = RectBivariateSpline(self.zz_pk,
                            self.kh_tot,
                            pnl,
                            kx=1, ky=1)     
        return  pnl_interp
    
    def get_pk_lin(self, params_dic, k, lbin, zz_integr):
        pk_l_interp = self.get_pk_hmcode_lin_interp(params_dic)
        # TO-DO implemenet this propery later
        #self.pklin_z0 = self.pklin_z0_lcdm # later used in tatt 
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        
        dz_qg, dz0_qg = self.get_growth(params_dic, self.zz_pk)
        dz_qg_notnorm = dz_qg*dz0_qg
        _, dz0_lcdm = self.get_growth_lcdm(params_dic, self.zz_pk)
        dz_norm = (dz_qg_notnorm/dz0_lcdm)**2

        d2_mg_lcdm_z_interp  = itp.interp1d(self.zz_pk,
                            dz_norm, bounds_error=False, kind='cubic') 
        #D_MG(z, k)^2/D_GR(z=0)^2 P_L,GR(z=0, k)  
        #index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        #for index_l, index_z in index_pknn:
        #    pk_m_l[index_l, index_z] = pk_l_interp(0., k[index_l,index_z])*d2_mg_lcdm_z_interp(k[index_l,index_z], zz_integr[index_z])

        array = np.zeros((lbin, len(zz_integr)), 'float64')
        mask = (k > k_min_h_by_mpc) & (k < k_max_h_by_mpc)
        index_l, index_z = np.where(mask)
        z_pts = zz_integr[index_z]
        k_pts = k[index_l, index_z]
        array[index_l, index_z] = np.ravel(pk_l_interp(0., k_pts, grid=False)*d2_mg_lcdm_z_interp(z_pts))
        pk_m_l = array
        return pk_m_l  

    def get_pk_pseudo(self, params_dic, k, lbin, zz_integr):
        d_mg_z, d_mg_z0 = self.get_growth(params_dic, self.zz_pk)
        d_lcdm_z, d_lcdm_z0 = self.get_growth_lcdm(params_dic, self.zz_pk)
        d2_mg_lcdm_z = (d_mg_z*d_mg_z0/d_lcdm_z/d_lcdm_z0)**2

        As_orig = params_dic['As']
        params_new = params_dic.copy()
        params_new['As'] = d2_mg_lcdm_z * As_orig

        pk_l_interp = self.get_pk_hmcode_interp(params_new)
        pk_m_l = fill_in_ell_z_array(pk_l_interp, k, lbin, zz_integr)
        return pk_m_l 
    
    def get_interpolators(self, params_dic):
        # if self.config["w_model"] == "log":
        #     if "w_piv" in params_dic:
        #         log10_zq = params_dic[ "log10_zq"]
        #         w_piv = params_dic["w_piv"]
        #         mqg, zeqqg, Aqg, Bqg = wpiv_basis_to_physical_log(log10_zq, w_piv, z_piv=Z_PIV, B_sign=1.0)
        #     else:
        #         mqg = -3./4.
        #         zeqqg = 10.0**params_dic["log10_zq"]
        #         Aqg = 0.
        #         Bqg = signed_log_parameter(params_dic["Bqg_log"])
        #w_piv = params_dic['w_piv']
        #z_piv = Z_PIV
        alpha = params_dic['alpha']
        beta = params_dic['beta']
        #amp = w_piv_to_amp_osc(w_piv, alpha, beta, z_piv)
        amp = params_dic['amp']
        w_grid = np.asarray(
                w_of_z(
                    self.z_growth,
                    0., 0., 0., 0.,
                    self.config["w_model"],
                    self.config["zin"],
                    self.config["safety"],
                    self.config["ABprior"],
                    z_de_active=self.zz_max,
                    amp=amp, alpha=alpha, beta=beta
                ),
                dtype=float,
            )
        self.w_interp = itp.interp1d(self.a_fine, w_grid[::-1], kind='cubic', fill_value="extrapolate")
        de_integrand = (1.0 + w_grid) / (1.0 + self.z_growth)
        de_int = cumulative_trapezoid(de_integrand, self.z_growth, initial=0.0)
        de_density_factor = np.exp(3.0 * de_int)
        self.fDE_interp = itp.interp1d(self.a_fine, de_density_factor[::-1], kind='cubic', fill_value="extrapolate")
        omega0 = params_dic['Omega_m']
        omega_de = 1.0 - omega0
        E_grid = np.sqrt(
            omega0 * (1.0 + self.z_growth) ** 3
            + omega_de * de_density_factor
        )
        r_dimless = cumulative_trapezoid(1.0 / E_grid, self.z_growth, initial=0.0)
        self.E_interp = itp.interp1d(self.z_growth, E_grid, kind='cubic', fill_value="extrapolate")
        self.rcom_interp = itp.interp1d(self.z_growth, r_dimless, kind='cubic', fill_value="extrapolate")
    
    def get_growth_lcdm(self, params_dic, zz_integr):
        aa_integr =  np.array(1./(1.+zz_integr[::-1]))
        background ={
            'Omega_m': params_dic['Omega_m'],
            'h' : params_dic['h'],
            'w0': -1.,
            'wa': 0.,
            'a_arr': np.hstack((aa_integr, 1.))
            }
        cosmo = mg.LCDM(background)
        da, _ = cosmo.growth_parameters()  
        dz = da[::-1] 
        # growth factor should be normalised to z=0
        # return array of z
        dz0 = dz[0]
        dz = dz[1:]/dz0
        return dz, dz0
    
    def get_growth(self, params_dic, zz_integr):
        aa_integr =  np.array(1./(1.+zz_integr[::-1]))
        aa = np.hstack((aa_integr, 1.))
        aa_int = np.hstack(([self.a_start], aa))   # match MGrowth
        def Omega_m_DE_fast(a, omega0, fDE_interp):
            """Fast version using pre-computed cumulative integral."""
            f_DE = fDE_interp(a)
            omegaL = (1.0 - omega0) * f_DE
            E2 = omega0 / a**3 + omegaL
            return omega0 / a**3 / E2

        def dlnH_dlna_DE_fast(a, omega0, fDE_interp, w_interp):
            f_DE = fDE_interp(a)
            omegaL = (1.0 - omega0) * f_DE
            E2 = omega0 / a**3 + omegaL
            w_a = w_interp(a)
            return -1.5 * (omega0 / a**3 + (1 + w_a) * omegaL) / E2

        def DE_D_derivatives_fast(D, a, fDE_interp, w_interp, omega0):
            """Fast version using pre-computed cumulative integral."""
            D1, D2 = D
            dlnH = dlnH_dlna_DE_fast(a, omega0, fDE_interp, w_interp)
            Om = Omega_m_DE_fast(a, omega0, fDE_interp)
            dD1 = D2
            dD2 = -D2 / a * (3.0 + dlnH) + 1.5 * D1 / a**2 * Om
            return np.array([dD1, dD2])
        # Use fast version with pre-computed integrals
        D_sol, ode_info = odeint(
            DE_D_derivatives_fast,
            [self.a_start, 1.0],
            aa_int,
            args=(
                self.fDE_interp,
                self.w_interp,
                params_dic['Omega_m'],
            ),
            full_output=True,
            mxstep=50000,
        )
        # LSODA can stall near a=1 without raising; check message and solution
        if 'successful' not in ode_info.get('message', '').lower() or not np.all(np.isfinite(D_sol)):
            raise RuntimeError("Growth ODE integration failed")
        Da = D_sol[1:, 0]
        dz = Da[::-1]
        # growth factor should be normalised to z=0
        dz0 = dz[0]
        if not np.isfinite(dz0) or dz0 <= 0.:
            raise RuntimeError("Invalid growth factor normalization")
        dz = dz[1:]/dz0
        if not np.all(np.isfinite(dz)) or np.any(dz <= 0.):
            raise RuntimeError("Non-physical growth factor")
        return dz, dz0
    
    
