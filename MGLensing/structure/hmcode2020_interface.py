from scipy.interpolate import RectBivariateSpline
import numpy as np
from cosmopower import cosmopower_NN
import os
import MGrowth as mg

dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 

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

def fill_in_ell_z_array(interp, k, lbin, zz_integr):
    array = np.zeros((lbin, len(zz_integr)), 'float64')
    index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
    for index_l, index_z in index_pknn:
            array[index_l, index_z] = interp(zz_integr[index_z], k[index_l,index_z])    
    return array


class HMcode2020():
    """
    HMcode2020 class for handling power spectra and baryonic boost factor calculations using cosmopower emulators.

    Attributes
    ----------
    cp_nl_model : cosmopower_NN
        An instance of the cosmopower_NN class for non-linear power spectrum emulation.
    kh_nl : numpy.ndarray
        Array of non-linear wavenumbers in units of h/Mpc.
    cp_lin_model : cosmopower_NN
        An instance of the cosmopower_NN class for linear power spectrum emulation.
    kh_lin : numpy.ndarray
        Array of linear wavenumbers in units of h/Mpc.
    kh_lin_left : numpy.ndarray
        Array of linear wavenumbers less than the minimum non-linear wavenumber.
    kh_tot : numpy.ndarray
        Concatenated array of linear and non-linear wavenumbers.
    cp_barboost_model : cosmopower_NN
        An instance of the cosmopower_NN class for baryonic boost factor emulation.
    kh_bb : numpy.ndarray
        Array of wavenumbers for baryonic boost factor emulation in units of h/Mpc.
    boost_left : numpy.ndarray
        Array of ones for linear wavenumbers less than the minimum baryonic boost wavenumber.
    kh_barboost : numpy.ndarray
        Concatenated array of linear and baryonic boost wavenumbers.
    cp_sigma8_model : cosmopower_NN
        An instance of the cosmopower_NN class for sigma8 emulation.
    zz_pk : numpy.ndarray
        Array of redshifts for power spectrum emulation.
    aa_pk : numpy.ndarray
        Array of scale factors corresponding to zz_pk.
    nz_pk : int
        Number of redshift bins for power spectrum emulation.
    zz_max : float
        Maximum redshift for power spectrum emulation.
    emu_name : str
        Name of the emulator.

    Methods
    -------
    __init__():
        Initializes the HMcode2020 class, setting up the cosmopower emulated models and relevant parameters.
    check_pars(params):
        Checks if the provided parameters are within the emulator's valid range.
    check_pars_ini(params):
        Checks if the provided initial parameters are within the emulator's valid range and raises errors for missing or out-of-bound parameters.
    get_pk_interp(params_dic):
        Interpolates the non-linear power spectrum based on the provided cosmological parameters.
    get_pk_lin_interp(params_dic):
        Interpolates the linear power spectrum based on the provided cosmological parameters.
    get_barboost_interp(params_dic):
        Interpolates the baryonic boost factor based on the provided cosmological parameters.
    get_growth(params_dic, zz_integr):
        Computes the growth factor for given redshifts.
    get_barboost(params_dic, k, lbin, zz_integr):
        Computes the baryonic boost factor for given wavenumbers and redshifts.
    get_pk_nl(params_dic, k, lbin, zz_integr):
        Computes the non-linear power spectrum for given wavenumbers and redshifts.
    get_pk_lin(params_dic, k, lbin, zz_integr):
        Computes the linear power spectrum for given wavenumbers and redshifts.
    get_sigma8(params_dic):
        Computes the sigma8 parameter based on the provided cosmological parameters.
    """
    def __init__(self):
        print('initialising hmcode')
        # these numers are hand-picked
        self.zz_pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5, 3.0]) 
        self.aa_pk = np.array(1./(1.+self.zz_pk[::-1])) # should be increasing
        self.nz_pk = len(self.zz_pk)
        self.zz_max = self.zz_pk[-1]

        self.cp_nl_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../../emulators/log10_total_matter_nonlinear_emu',
                      )
        self.kh_nl = self.cp_nl_model.modes # 0.01..50. h/Mpc    
        self.cp_lin_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../../emulators/log10_total_matter_linear_emu',
                      )
        self.kh_lin = self.cp_lin_model.modes # 3.7e-4..50. h/Mpc IMPORTANT LATER USED IN TATT
        self.kh_lin_left = self.kh_lin[self.kh_lin<self.kh_nl[0]]
        self.kh_tot = np.concatenate((self.kh_lin_left, self.kh_nl))

        self.cp_barboost_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../../emulators/total_matter_bar_boost_emu',
                      )
        self.kh_bb = self.cp_barboost_model.modes # 0.01..50. h/Mpc 
        kbin_left_bb = len(self.kh_lin[self.kh_lin<self.kh_bb[0]])
        self.kh_barboost = np.concatenate((self.kh_lin[self.kh_lin<self.kh_bb[0]], self.kh_bb))
        self.boost_left = np.ones((self.nz_pk, kbin_left_bb))
        self.cp_sigma8_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../../emulators/sigma8_emu',
                      )
        self.emu_name = 'HMcode2020'


    def check_pars(self, params):
        emu_ranges = emu_ranges_all.copy()
        if 'log10Tagn' not in params:
            del emu_ranges['log10Tagn'] 
        eva_pars = emu_ranges.keys()     
        if not all(emu_ranges[par_i]['p1'] <= params[par_i] <= emu_ranges[par_i]['p2'] for par_i in eva_pars):
            return False
        if params['w0']+params['wa']>=0:
            return False 
        return True
    
    
    def check_pars_ini(self, params):
        emu_ranges = emu_ranges_all.copy()
        if 'log10Tagn' not in params:
            print('here')
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


    def get_pk_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        m_nu  = params_dic['Mnu']
        params_hmcode = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'hubble'        :  np.full(self.nz_pk, h),
                'omega_baryon'  :  np.full(self.nz_pk, omega_b),
                'omega_cdm'     :  np.full(self.nz_pk, omega_c),
                'neutrino_mass' :  np.full(self.nz_pk, m_nu),
                'w0'            :  np.full(self.nz_pk, w0),
                'wa'            :  np.full(self.nz_pk, wa),
                'z'             :  self.zz_pk
            }
        pnl_cp  = self.cp_nl_model.ten_to_predictions_np(params_hmcode)
        plin_cp = self.cp_lin_model.ten_to_predictions_np(params_hmcode)
        self.pklin_z0 = plin_cp[0] # zz_pk[0] must be 0.!
        plin_left = plin_cp[:, self.kh_lin<self.kh_nl[0]]
        pnl  = np.concatenate((plin_left, pnl_cp),axis=1)
        pnl_interp = RectBivariateSpline(self.zz_pk,
                                    self.kh_tot,
                                    pnl,
                                    kx=1, ky=1)
        return  pnl_interp    
    
    def get_pk_lin_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        m_nu  = params_dic['Mnu']
        params_hmcode = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'hubble'        :  np.full(self.nz_pk, h),
                'omega_baryon'  :  np.full(self.nz_pk, omega_b),
                'omega_cdm'     :  np.full(self.nz_pk, omega_c),
                'neutrino_mass' :  np.full(self.nz_pk, m_nu),
                'w0'            :  np.full(self.nz_pk, w0),
                'wa'            :  np.full(self.nz_pk, wa),
                'z'             :  self.zz_pk
            }
        plin_cp = self.cp_lin_model.ten_to_predictions_np(params_hmcode)
        plin_interp = RectBivariateSpline(self.zz_pk,
                                    self.kh_lin,
                                    plin_cp,
                                    kx=1, ky=1)
        return  plin_interp    
    
    def get_barboost_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        m_nu  = params_dic['Mnu']
        logt_agn = params_dic['log10T_AGN']
        params_hmcode_bar = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'hubble'        :  np.full(self.nz_pk, h),
                'omega_baryon'  :  np.full(self.nz_pk, omega_b),
                'omega_cdm'     :  np.full(self.nz_pk, omega_c),
                'neutrino_mass' :  np.full(self.nz_pk, m_nu),
                'w0'            :  np.full(self.nz_pk, w0),
                'wa'            :  np.full(self.nz_pk, wa),
                'log10TAGN'     :  np.full(self.nz_pk, logt_agn),
                'z'             :  self.zz_pk
            }
        boost = self.cp_barboost_model.predictions_np(params_hmcode_bar) 
        boost_k  = np.concatenate((self.boost_left, boost),axis=1)
        bhm_interp = RectBivariateSpline(self.zz_pk,
                                    self.kh_barboost,
                                    boost_k,
                                    kx=1, ky=1)
        return  bhm_interp 
    
    def get_growth(self, params_dic, zz_integr):
        aa_integr =  np.array(1./(1.+zz_integr[::-1]))
        background ={
            'Omega_m': params_dic['Omega_m'],
            'h' : params_dic['h'],
            'w0': params_dic['w0'],
            'wa': params_dic['wa'],
            'a_arr': np.hstack((aa_integr, 1.))
            }
        cosmo = mg.w0waCDM(background)   
        da, _ = cosmo.growth_parameters() 
        dz = da[::-1] 
        # growth factor should be normalised to z=0
        dz0 = dz[0]
        dz = dz[1:]/dz0
        return dz, dz0

    def get_barboost(self, params_dic, k, lbin, zz_integr):
        boost_bar_interp = self.get_barboost_interp(params_dic)
        boost_bar = fill_in_ell_z_array(boost_bar_interp, k, lbin, zz_integr)
        return boost_bar

    def get_pk_nl(self, params_dic, k, lbin, zz_integr):
        pk_l_interp = self.get_pk_interp(params_dic)
        pk_nl_l = fill_in_ell_z_array(pk_l_interp, k, lbin, zz_integr)
        return pk_nl_l
    
    def get_pk_lin(self, params_dic, k, lbin, zz_integr):
        pk_l_interp = self.get_pk_lin_interp(params_dic)
        pk_lin_l = fill_in_ell_z_array(pk_l_interp, k, lbin, zz_integr)
        return pk_lin_l
    
         
    
    def get_sigma8(self, params_dic, flag_gr=False):
        ns   = params_dic['ns']
        len_chain = len(ns)
        a_s   = params_dic['As']
        h    = params_dic['h']
        w0    = params_dic['w0'] if not flag_gr else np.full(len_chain ,-1.)
        wa    = params_dic['wa'] if not flag_gr else np.zeros(len_chain)
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        m_nu  = params_dic['Mnu']
        params_hmcode = {
                'ns'            :  ns,
                'As'            :  a_s,
                'hubble'        :  h,
                'omega_baryon'  :  omega_b,
                'omega_cdm'     :  omega_c,
                'neutrino_mass' :  m_nu,
                'w0'            :  w0,
                'wa'            :  wa,
                'z'             :  np.zeros(len_chain)
            }
        sigma8_emu = self.cp_sigma8_model.predictions_np(params_hmcode)
        sigma8 = sigma8_emu[:, 0]
        return sigma8