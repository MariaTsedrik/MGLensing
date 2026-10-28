from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.special import hyp2f1
import numpy as np
from cosmopower import cosmopower_NN
import os
from math import log10, log

dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 

emu_ranges_all = {
        'Omega_m':      {'p1':0.29,            'p2':0.34}, 
        'Omega_b':      {'p1':0.04044,         'p2':0.05686}, 
        'h':            {'p1':0.629,           'p2':0.731},
        'ns':           {'p1':0.9432,          'p2':0.9862},
        'As':           {'p1':1.5e-09,         'p2':2.7e-09},
        'Omega_nu':     {'p1':0.,              'p2':0.00317},
        'gamma0':       {'p1':0.,              'p2':1.},
        'gamma1':       {'p1':-0.7,            'p2':0.7},
        'q1_gamma':     {'p1':-2.,             'p2':2.}
}


def _ell_z_points(k, lbin, zz_integr):
    array = np.zeros((lbin, len(zz_integr)), 'float64')
    index_l, index_z = np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))
    return array, index_l, index_z, zz_integr[index_z], k[index_l, index_z]


def _lcdm_growth_ratio_squared(z, omega_m):
    """[D(z)/D(0)]^2 for flat ΛCDM.

    D(a) ∝ E(a) ∫_0^a da' / [a'^3 E(a')^3]. The prefactor cancels in the
    ratio. This is the linear-theory growth of the HMcode spectrum, not the
    modified-gravity D(z) used for intrinsic alignments.
    """
    z = np.asarray(z, dtype=np.float64)
    a_grid = np.logspace(-4., 0., 4096)
    e_a = np.sqrt(omega_m * a_grid**(-3) + (1. - omega_m))
    integrand = 1. / (a_grid**3 * e_a**3)
    integral = cumulative_trapezoid(integrand, a_grid, initial=0.)
    a0 = a_grid[0]
    integral = integral + (2. / 5.) * a0**2.5 / omega_m**1.5
    growth = e_a * integral
    growth /= growth[-1]
    return np.interp(1. / (1. + z), a_grid, growth)**2


def powerlaw_highk_extrap(pk_or_boost, log_k, k_last, kh_high, zz_num):
    last_entry, lastlast_entry = pk_or_boost[:, -1], pk_or_boost[:, -2]
    m = np.array([log(np.abs(last_entry[i] / lastlast_entry[i])) / log_k for i in range(zz_num)])
    highk_extrap = last_entry[:, np.newaxis] * (kh_high[np.newaxis, :]/k_last)**m[:, np.newaxis]
    return highk_extrap 

class GammaReACT():
    def __init__(self, option=None):
        #self.zz_pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5, 3.0]) # these numers are hand-picked
        self.zz_pk = np.linspace(0., 3., 128, endpoint=True)
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

        print('initialising Growth Index Parametrisation')
        self.cp_nl_gammaz_model = cosmopower_NN(restore=True, 
                        restore_filename=dirname+'/../../emulators/gamma/react_boost_gamma_z_q1', 
                        )
        self.kh_nl_boost = self.cp_nl_gammaz_model.modes # 0.01..5. h/Mpc 
        self.zz_boost = np.minimum(self.zz_pk, 2.5)
        self.kh_lin_left_boost = self.kh_lin[self.kh_lin<self.kh_nl_boost[0]]
        self.kh_nl_right_boost = self.kh_nl[self.kh_nl>self.kh_nl_boost[-1]]
        self.kh_nl_boost_tot = np.concatenate((self.kh_lin_left_boost, self.kh_nl_boost, self.kh_nl_right_boost))
        self.k_nl_boost_last = self.kh_nl_boost[-1]
        self.k_nl_boost_lastlast = self.kh_nl_boost[-2]
        self.log_k = log(self.k_nl_boost_last / self.k_nl_boost_lastlast)
        self.emu_name = 'Gamma_z_ReACT'

        if option=='linear':
            self.get_pk_nl =  self.get_pk_lin 
        elif option=='pseudo':
            self.get_pk_nl = self.get_pk_pseudo
        else:
            self.get_pk_nl = self.get_pk_nl_

    def check_pars(self, params):
        emu_ranges = emu_ranges_all.copy()
        eva_pars = emu_ranges.keys()     
        if not all(emu_ranges[par_i]['p1'] <= params[par_i] <= emu_ranges[par_i]['p2'] for par_i in eva_pars):
            return False
        return True
    
    
    def check_pars_ini(self, params):
        emu_ranges = emu_ranges_all.copy()
        eva_pars = emu_ranges.keys()     
        # parameters currently available
        avail_pars = [coo for coo in params.keys()]    
        # parameters needed for a computation
        comp_pars = list(set(eva_pars)-set(avail_pars))
        miss_pars = list(set(comp_pars))
        # check missing parameters
        if miss_pars:
            print(f"ReACT growth index emulator:")
            print(f"  Please add the parameter(s) {miss_pars}"
                  f" to your parameters!")
            raise KeyError(f"ReACT gamma-z emulator: coordinates need the"
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
        if params['w0']!=-1. or params['wa']!=0.:
             raise KeyError("Applicable only for Lambda as dark energy!")        
        return True    

    def get_mg_boost_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        omega_b = params_dic['Omega_b']
        omega_m = params_dic['Omega_m']
        omega_nu  = params_dic['Omega_nu']
        gamma0    = params_dic['gamma0']
        gamma1    = params_dic['gamma1']
        q1    = params_dic['q1_gamma']
        params_react = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'H0'            :  np.full(self.nz_pk, h*100),
                'Omega_b'       :  np.full(self.nz_pk, omega_b),
                'Omega_m'       :  np.full(self.nz_pk, omega_m),
                'Omega_nu'      :  np.full(self.nz_pk, omega_nu),
                'gamma0'        :  np.full(self.nz_pk, gamma0),
                'gamma1'        :  np.full(self.nz_pk, gamma1),
                'q1'            :  np.full(self.nz_pk, q1),
                'z'             :  self.zz_boost
            }
        mg_boost = self.cp_nl_gammaz_model.predictions_np(params_react) 
        self.d2_mg_lcdm = mg_boost[0, 0] # zz_boost[0] must be 0.!
        self.d2_mg_lcdm_z = mg_boost[:, 0]
        # constant extrapolation for k<0.01 h/Mpc
        mg_boost_left = np.full((self.nz_pk, len(self.kh_lin_left_boost)), mg_boost[:, [0]])
        # power law extrapolation for k>5 h/Mpc
        mg_boost_right = powerlaw_highk_extrap(mg_boost, self.log_k, self.k_nl_boost_last, self.kh_nl_right_boost, self.nz_pk)
        # combine mg_boost at all scales
        mg_boost_k = np.concatenate((mg_boost_left, mg_boost, mg_boost_right), axis=1)
        # interpolate
        mg_boost_interp = RectBivariateSpline(self.zz_pk,
                    self.kh_nl_boost_tot,
                    mg_boost_k,
                    kx=1, ky=1)
        return  mg_boost_interp
    
    def _lcdm_linear_power(self, params_dic):
        """ΛCDM linear P(k, z) on the emulator redshift grid.

        The network is evaluated once at z=0. With massless neutrinos the
        growth is scale-independent, so P_lin(k, z) = P_lin(k, 0) [D(z)/D(0)]^2.
        A redshift-dependent As, used by the pseudo spectrum, scales the same way.
        """
        ns = params_dic['ns']
        a_s = params_dic['As']
        h = params_dic['h']
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        as_per_z = isinstance(a_s, np.ndarray) and np.size(a_s) == self.nz_pk
        as_z0 = a_s[0] if as_per_z else a_s
        params_z0 = {
                'ns'            :  np.array([ns]),
                'As'            :  np.array([as_z0]),
                'hubble'        :  np.array([h]),
                'omega_baryon'  :  np.array([omega_b]),
                'omega_cdm'     :  np.array([omega_c]),
                'neutrino_mass' :  np.zeros(1),
                'w0'            :  np.array([-1.]),
                'wa'            :  np.zeros(1),
                'z'             :  np.zeros(1),
            }
        plin_z0 = self.cp_lin_model.ten_to_predictions_np(params_z0)[0]
        self.pklin_z0_lcdm = plin_z0
        growth2 = _lcdm_growth_ratio_squared(self.zz_pk, params_dic['Omega_m'])
        if as_per_z:
            growth2 = growth2 * (a_s / as_z0)
        return plin_z0[None, :] * growth2[:, None]

    def get_pk_hmcode_lin_interp(self, params_dic):
        plin_cp = self._lcdm_linear_power(params_dic)
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
        plin_cp = self._lcdm_linear_power(params_dic)
        plin_left = plin_cp[:, self.kh_lin<self.kh_nl[0]]
        pnl  = np.concatenate((plin_left, pnl_cp),axis=1)
        pnl_interp = RectBivariateSpline(self.zz_pk,
                            self.kh_tot,
                            pnl,
                            kx=1, ky=1)     
        return  pnl_interp
    
    def get_pk_react(self, params_dic, k, lbin, zz_integr, mg_boost_l_interp):
        pk_l_interp = self.get_pk_hmcode_interp(params_dic)
        # d2_mg_lcdm is computed in self.get_mg_boost_interp
        self.pklin_z0 = self.d2_mg_lcdm * self.pklin_z0_lcdm # later used in tatt
        pk_m_l, index_l, index_z, z_pts, k_pts = _ell_z_points(k, lbin, zz_integr)
        pk_m_l[index_l, index_z] = (
            np.ravel(pk_l_interp(z_pts, k_pts, grid=False))
            * np.ravel(mg_boost_l_interp(np.minimum(z_pts, 2.5), k_pts, grid=False))
        )
        return pk_m_l  


    def get_pk_nl_(self, params_dic, k, lbin, zz_integr):
        mg_boost_l_interp = self.get_mg_boost_interp(params_dic)
        pk_m_l = self.get_pk_react(params_dic, k, lbin, zz_integr, mg_boost_l_interp)
        return pk_m_l
    
    def get_pk_pseudo(self, params_dic, k, lbin, zz_integr):
        # call to compute self.d2_mg_lcdm and self.d2_mg_lcdm_z
        _ = self.get_mg_boost_interp(params_dic)
        # call linear lcdm with original As  
        _ = self.get_pk_hmcode_lin_interp(params_dic)
        self.pklin_z0 = self.d2_mg_lcdm * self.pklin_z0_lcdm # later used in tatt 
        As_orig = params_dic['As']
        params_new = params_dic.copy()
        params_new['As'] = self.d2_mg_lcdm_z * As_orig
        pk_l_interp = self.get_pk_hmcode_interp(params_new)
        pk_m_l, index_l, index_z, z_pts, k_pts = _ell_z_points(k, lbin, zz_integr)
        pk_m_l[index_l, index_z] = np.ravel(pk_l_interp(z_pts, k_pts, grid=False))
        return pk_m_l 
    
    def get_pk_lin(self, params_dic, k, lbin, zz_integr):
        # call to compute self.d2_mg_lcdm and self.d2_mg_lcdm_z
        _ = self.get_mg_boost_interp(params_dic)
        d2_mg_lcdm_z_interp = interp1d(self.zz_pk, self.d2_mg_lcdm_z)
        # call linear lcdm with original As  
        pk_l_interp = self.get_pk_hmcode_lin_interp(params_dic)
        self.pklin_z0 = self.d2_mg_lcdm * self.pklin_z0_lcdm # later used in tatt
        pk_m_l, index_l, index_z, z_pts, k_pts = _ell_z_points(k, lbin, zz_integr)
        pk_m_l[index_l, index_z] = np.ravel(pk_l_interp(z_pts, k_pts, grid=False)) * d2_mg_lcdm_z_interp(z_pts)
        return pk_m_l

    def get_growth(self, params_dic, zz_integr):
        # Linder gamma(a) = gamma0 + gamma1 (1-a)^2/a, integrated once in ln a.
        # Same D(a=1e-4) normalisation as MGrowth.Linder_gamma_a; the returned
        # array is D(z)/D(a=1) on zz_integr. Flat Lambda only (w0=-1, wa=0).
        omega0 = params_dic['Omega_m']
        gamma0 = params_dic['gamma0']
        gamma1 = params_dic['gamma1']
        a_start = 1.e-4
        a_out = np.hstack((1./(1. + zz_integr[::-1]), 1.))
        lna = np.linspace(np.log(a_start), 0., 8192)
        a = np.exp(lna)
        omega_m_a = omega0 / (omega0 + (1. - omega0) * a**3)
        gamma_a = gamma0 + gamma1 * (1. - a)**2 / a
        integral = cumulative_trapezoid(np.exp(gamma_a * np.log(omega_m_a)), lna, initial=0.)
        d_ini = a_start * hyp2f1(1./3., 1., 11./6., (omega0 - 1.) / omega0 * a_start**3)
        growth_a = np.interp(a_out, a, d_ini * np.exp(integral))
        dz = growth_a[::-1]
        dz0 = dz[0]
        dz = dz[1:] / dz0
        return dz, dz0