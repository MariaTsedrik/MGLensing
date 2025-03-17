from scipy.interpolate import RectBivariateSpline
import numpy as np
import baccoemu 
import os
import MGrowth as mg
from math import log10, log
from .hmcode2020_interface import HMcode2020

dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 

emu_ranges_all = {
        'Omega_cb':     {'p1': 0.23,         'p2': 0.4},    
        'Omega_b':      {'p1':0.04,          'p2': 0.06},  
        'h':            {'p1': 0.6,          'p2': 0.8},      
        'sigma8_cb':    {'p1': 0.73,         'p2': 0.9},
        'ns':           {'p1': 0.92,         'p2': 1.01}, 
        'Mnu':          {'p1': 0.0,          'p2': 0.4},
        'w0':           {'p1': -1.15,        'p2': -0.85},    
        'wa':           {'p1': -0.3,         'p2': 0.3}, 
        'log10Mc_bc':   {'p1': 9.,           'p2': 15.}, 
        'eta_bc':       {'p1': -0.69,        'p2': 0.69},
        'beta_bc':      {'p1': -1.,          'p2': 0.69},
        'log10Mz0_bc':  {'p1': 9.,           'p2': 13.},
        'thetaout_bc':  {'p1': 0.,           'p2': 0.47},
        'thetainn_bc':  {'p1': -2.,          'p2': -0.523},
       'log10Minn_bc': {'p1': 9.,           'p2': 13.5}
        }
baryonic_params = [ 'log10Mc_bc', 'eta_bc', 'beta_bc', 'log10Mz0_bc', 'thetaout_bc', 'thetainn_bc', 'log10Minn_bc']

emu_ranges_lin = {
        'Omega_cb':     {'p1': 0.06,         'p2': 0.7},    
        'Omega_b':      {'p1':0.03,          'p2': 0.07},  
        'h':            {'p1': 0.5,          'p2': 0.9},      
        'ns':           {'p1': 0.6,         'p2': 1.2}, 
        'Mnu':          {'p1': 0.0,          'p2': 1.},
        'w0':           {'p1': -2.,        'p2': -0.5},    
        'wa':           {'p1': -0.5,         'p2': 0.5}, 
        }  


def fill_in_ell_z_array(interp, k, lbin, zz_integr, zmax=10.):
    array = np.zeros((lbin, len(zz_integr)), 'float64')
    index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
    for index_l, index_z in index_pknn:
            array[index_l, index_z] = interp(min(zz_integr[index_z], zmax), k[index_l,index_z])    
    return array

def powerlaw_highk_extrap(pk_or_boost, log_k, k_last, kh_high, zz_num):
    last_entry, lastlast_entry = pk_or_boost[:, -1], pk_or_boost[:, -2]
    m = np.array([log(last_entry[i] / lastlast_entry[i]) / log_k for i in range(zz_num)])
    highk_extrap = last_entry[:, np.newaxis] * (kh_high[np.newaxis, :]/k_last)**m[:, np.newaxis]
    return highk_extrap 

class BaccoEmu:
    """
    BaccoEmu class for computing power spectra and baryonic boost factors using the BACCO emulator.
    
    Attributes
    ----------
    baccoemulator : baccoemu.Matter_powerspectrum
        An instance of the Matter_powerspectrum class for power spectrum emulation.
    heftemulator : baccoemu.Lbias_expansion
        An instance of the Lbias_expansion class for halo expansion emulation.
    kh_nl : numpy.ndarray
        Array of non-linear wavenumbers in units of h/Mpc.
    kh_bb : numpy.ndarray
        Array of wavenumbers for baryonic boost emulation in units of h/Mpc.
    kh_heft : numpy.ndarray
        Array of wavenumbers for HEFT emulation in units of h/Mpc.
    kh_lin : numpy.ndarray
        Array of linear wavenumbers in units of h/Mpc.
    z_nl_bacco : numpy.ndarray
        Array of redshifts for non-linear BACCO emulation.
    z_lin_bacco : numpy.ndarray
        Array of redshifts for linear BACCO emulation.
    kh_lin_left : numpy.ndarray
        Array of linear wavenumbers less than the minimum non-linear wavenumber.
    kh_lin_right : numpy.ndarray
        Array of linear wavenumbers greater than the maximum non-linear wavenumber.
    kh_tot : numpy.ndarray
        Concatenated array of linear and non-linear wavenumbers.
    kh_barboost_tot : numpy.ndarray
        Concatenated array of wavenumbers for baryonic boost emulation.
    z_lin_high : numpy.ndarray
        Array of redshifts higher than the maximum non-linear redshift.
    aa_lin_high : numpy.ndarray
        Array of scale factors corresponding to z_lin_high.
    aa_nl : numpy.ndarray
        Array of scale factors corresponding to z_nl_bacco.
    aa_all : numpy.ndarray
        Concatenated array of all scale factors.
    zz_all_bacco : numpy.ndarray
        Concatenated array of all redshifts for BACCO emulation.
    zz_max : float
        Maximum redshift for power spectrum emulation.
    emu_name : str
        Name of the emulator.

    Methods
    -------
    __init__():
        Initializes the BaccoEmu class with default parameters and sets up the BACCO emulators.
    check_pars(params):
        Checks if the given parameters are within the allowed ranges for the full emulator.
    check_pars_lin(params):
        Checks if the given parameters are within the allowed ranges for the linear emulator.
    check_pars_ini(params):
        Checks if the given parameters are within the allowed ranges and raises errors for missing or out-of-bound parameters.
    get_pk_interp(params_dic):
        Interpolates the non-linear power spectrum and extrapolates using the linear BACCO emulator for z>1.5.
    get_pk_lin_interp(params_dic):
        Interpolates the linear power spectrum using the BACCO emulator.
    get_barboost_interp(params_dic):
        Interpolates the baryonic boost factor using the BACCO emulator.
    get_heft_interp(params_dic):
        Interpolates the non-linear power spectrum in the Hybrid Effective Field Theory (HEFT) approach using the BACCO emulator.
    get_pk_nl(params_dic, k, lbin, zz_integr):
        Computes the non-linear power spectrum for given parameters, wavenumbers, and redshifts.
    get_pk_lin(params_dic, k, lbin, zz_integr):
        Computes the linear power spectrum for given parameters, wavenumbers, and redshifts.
    get_barboost(params_dic, k, lbin, zz_integr):
        Computes the baryonic boost factor for given parameters, wavenumbers, and redshifts.
    get_heft(params_dic, k, lbin, zz_integr):
        Computes the non-linear power spectrum for halo expansion for given parameters, wavenumbers, and redshifts.
    get_growth(params_dic, zz_integr):
        Computes the growth factor for given parameters and redshifts.
    get_a_s(params):
        Computes the scalar amplitude A_s for given parameters.
    get_sigma8_cb(params):
        Computes the sigma8 for cold dark matter for given parameters.
    """
    def __init__(self, option=None):
        print('initialising baccoemu')
        self.baccoemulator = baccoemu.Matter_powerspectrum()
        self.heftemulator = baccoemu.Lbias_expansion()
        k_min_lin = 1.e-4
        k_max_lin = 50.
        z_min = 0.
        z_max_lin = 3. #add a comment about global z_max
        k_min_nl = 0.01001
        k_max_nl = 4.903235148249275
        k_max_boost = 4.692772528625323
        z_max_nl = 1.5
        k_min_heft = 0.01001
        k_max_heft = 0.71


        self.kh_nl = np.logspace(log10(k_min_nl), log10(k_max_nl), num=512)
        self.kh_bb = np.logspace(log10(k_min_nl), log10(k_max_boost), num=256)
        self.kh_heft = np.logspace(log10(k_min_heft), log10(k_max_heft), num=128)
        kh_lin_ = np.logspace(log10(k_min_lin), log10(k_max_lin), num=512)
        self.z_nl_bacco = np.linspace(z_min, z_max_nl, 32)
        self.z_lin_bacco = np.linspace(z_min, z_max_lin, 32)

        self.kh_lin_left = kh_lin_[kh_lin_<k_min_nl]
        self.kh_lin_right = kh_lin_[kh_lin_>k_max_nl]
        self.kh_nl_last = self.kh_nl[-1]
        self.log_kh_nl = log(self.kh_nl_last / self.kh_nl[-2])
        self.kh_tot = np.concatenate((self.kh_lin_left, self.kh_nl, self.kh_lin_right))

        self.boost_left = np.ones((len(self.z_nl_bacco), len(self.kh_lin_left)))
        self.kh_lin_bb_right = kh_lin_[kh_lin_>k_max_boost]
        self.kh_bb_last = self.kh_bb[-1]
        self.log_kh_bb = log(self.kh_bb_last / self.kh_bb[-2])
        self.kh_barboost_tot = np.concatenate((self.kh_lin_left, self.kh_bb, self.kh_lin_bb_right))

        self.kh_lin = self.kh_tot # IMPORTANT LATER USED IN TATT
        #self.kh_lin_last = self.kh_lin[-1]
        #self.log_kh_lin = log(self.kh_lin_last / self.kh_lin[-2])
        #self.kh_lin_extrap = np.logspace(log10(k_max_lin+1), 3, num=512)
        #self.kh_lin_tot = np.concatenate((self.kh_lin, self.kh_lin_extrap))


        self.z_lin_high = self.z_lin_bacco[self.z_lin_bacco>self.z_nl_bacco[-1]]
        self.aa_lin_high = 1./(1.+self.z_lin_high)[::-1]
        self.aa_nl = 1./(1.+self.z_nl_bacco)[::-1]
        self.aa_all = np.concatenate((self.aa_lin_high, self.aa_nl))
        self.zz_all_bacco = np.concatenate((self.z_nl_bacco, self.z_lin_high))
        self.zz_max = z_max_lin
        self.emu_name = 'BACCO'

        if option=='linear':
            self.get_pk_nl =  self.get_pk_lin 
        elif option=='z_extrap_linear' or option==None:
            self.get_pk_nl = self.get_pk_nl_zextr_lin
        elif option=='z_extrap_hmcode':
            self.HMcodeEmu = HMcode2020()
            self.get_pk_nl = self.get_pk_nl_zextr_hmcode
        else:
            raise KeyError("Invalid bacco z>1.5 extrapolation.")

    def check_pars(self, params):
        emu_ranges = emu_ranges_all.copy()
        for bar_par in baryonic_params:
            if bar_par not in params:
                del emu_ranges[bar_par] 
        eva_pars = emu_ranges.keys()     
        if not all(emu_ranges[par_i]['p1'] <= params[par_i] <= emu_ranges[par_i]['p2'] for par_i in eva_pars):
            return False
        if params['w0']+params['wa']>=0:
            return False 
        return True
    
    def check_pars_lin(self, params):
        emu_ranges = emu_ranges_lin.copy()
        eva_pars = emu_ranges.keys()     
        if not all(emu_ranges[par_i]['p1'] <= params[par_i] <= emu_ranges[par_i]['p2'] for par_i in eva_pars):
            return False
        if params['w0']+params['wa']>=0:
            return False 
        return True
    
    
    def check_pars_ini(self, params):
        emu_ranges = emu_ranges_all.copy()
        for bar_par in baryonic_params:
            if bar_par not in params:
                del emu_ranges[bar_par] 
        eva_pars = emu_ranges.keys()     
        # parameters currently available
        avail_pars = [coo for coo in params.keys()]    
        # parameters needed for a computation
        comp_pars = list(set(eva_pars)-set(avail_pars))
        miss_pars = list(set(comp_pars))
        # check missing parameters
        if miss_pars:
            print(f"BACCO emulator:")
            print(f"  Please add the parameter(s) {miss_pars}"
                  f" to your parameters!")
            raise KeyError(f"BACCO emulator: coordinates need the"
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
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_cb = params_dic['Omega_cb']
        omega_b = params_dic['Omega_b']
        m_nu  = params_dic['Mnu']
        sigma8_cb = params_dic['sigma8_cb']
        params_bacco = {
                'ns'            :  ns,
                'hubble'        :  h,
                'sigma8_cold'   :  sigma8_cb,
                'omega_baryon'  :  omega_b,
                'omega_cold'    :  omega_cb,
                'neutrino_mass' :  m_nu,
                'w0'            :  w0,
                'wa'            :  wa,
                'expfactor'     :  self.aa_nl
            }
        _, pnl_bacco  = self.baccoemulator.get_nonlinear_pk(k=self.kh_nl, cold=False, **params_bacco)
        # power law extrapolation for k>5 h/Mpc
        pnl_right = powerlaw_highk_extrap(pnl_bacco, self.log_kh_nl, self.kh_nl_last, self.kh_lin_right, len(self.aa_nl))
        # extrapolation with linear power spectrum for k<0.01 h/Mpc and z>1.5
        params_bacco_lin = params_bacco
        params_bacco_lin['expfactor'] = self.aa_all
        _, plin_bacco = self.baccoemulator.get_linear_pk(k=self.kh_tot, cold=False, **params_bacco_lin)
        self.pklin_z0 = plin_bacco[-1] # at redshift 0, later used in fast-pt
        plin_left = plin_bacco[:, self.kh_tot<self.kh_nl[0]]
        plin_left = plin_left[self.aa_all>=self.aa_nl[0], :]
        pnl  = np.concatenate((plin_left, pnl_bacco, pnl_right),axis=1)
        pl_highz = plin_bacco[self.aa_all<self.aa_nl[0], :]
        pnl  = np.concatenate((pl_highz, pnl),axis=0)
        # interpolate
        pnl_interp = RectBivariateSpline(self.zz_all_bacco, 
                                    self.kh_tot,
                                    pnl[::-1, :],
                                    kx=1, ky=1)
        return  pnl_interp 
    
    def get_pk_lin_interp(self, params_dic):
        ns   = params_dic['ns']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_cb = params_dic['Omega_cb']
        omega_b = params_dic['Omega_b']
        m_nu  = params_dic['Mnu']
        sigma8_cb = params_dic['sigma8_cb']
        params_bacco = {
                'ns'            :  ns,
                'hubble'        :  h,
                'sigma8_cold'   :  sigma8_cb,
                'omega_baryon'  :  omega_b,
                'omega_cold'    :  omega_cb,
                'neutrino_mass' :  m_nu,
                'w0'            :  w0,
                'wa'            :  wa,
                'expfactor'     :  self.aa_all
            }
        _, plin_bacco = self.baccoemulator.get_linear_pk(k=self.kh_tot, cold=False, **params_bacco)
        plin_interp = RectBivariateSpline(self.zz_all_bacco,
                                    self.kh_tot,
                                    plin_bacco[::-1, :],
                                    kx=1, ky=1)
        #plin_right = powerlaw_highk_extrap(plin_bacco, self.log_kh_lin, self.kh_lin_last, self.kh_lin_extrap, len(self.aa_all))
        #plin_tot  = np.concatenate((plin_bacco, plin_right),axis=1)
        
        #plin_interp = RectBivariateSpline(self.zz_all_bacco,
        #                            self.kh_lin_tot,
        #                            plin_tot[::-1, :],
        #                            kx=1, ky=1)
        return  plin_interp    
    
    def get_barboost_interp(self, params_dic):      
        ns   = params_dic['ns']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_cb = params_dic['Omega_cb']
        omega_b = params_dic['Omega_b']
        sigma8_cb = params_dic['sigma8_cb']
        m_nu  = params_dic['Mnu']
        log10mc_bc = params_dic['log10Mc_bc']
        eta_bc = params_dic['eta_bc']
        beta_bc = params_dic['beta_bc']
        log10mcen_bc = params_dic['log10Mz0_bc']
        theout_bc = params_dic['thetaout_bc']
        thetinn_bc = params_dic['thetainn_bc']
        log10minn_bc = params_dic['log10Minn_bc']
        params_bacco = {
                'ns'            :  ns,
                'hubble'        :  h,
                'sigma8_cold'   :  sigma8_cb,
                'omega_baryon'  :  omega_b,
                'omega_cold'    :  omega_cb,
                'neutrino_mass' :  m_nu,
                'w0'            :  w0,
                'wa'            :  wa,
                'expfactor'     :  self.aa_nl,

                'M_c'           : log10mc_bc,
                'eta'           : eta_bc,
                'beta'          : beta_bc,
                'M1_z0_cen'     : log10mcen_bc,
                'theta_out'     : theout_bc,
                'theta_inn'     : thetinn_bc,
                'M_inn'         : log10minn_bc
            } 

        _, boost = self.baccoemulator.get_baryonic_boost(k=self.kh_bb, cold=False, **params_bacco)
        boost_right = powerlaw_highk_extrap(boost, self.log_kh_bb, self.kh_bb_last, self.kh_lin_bb_right, len(self.aa_nl))
        boost_k  = np.concatenate((self.boost_left, boost, boost_right),axis=1)
        bfc_interpolator = RectBivariateSpline(self.z_nl_bacco, 
                                               self.kh_barboost_tot,
                                               boost_k[::-1, :],
                                               kx=1, ky=1)
        return  bfc_interpolator   
    
    def get_heft_interp(self, params_dic):
        ns   = params_dic['ns']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        sigma8_cb = params_dic['sigma8_cb']
        omega_cb = params_dic['Omega_cb']
        omega_b = params_dic['Omega_b']
        m_nu  = params_dic['Mnu']
        params_bacco = {
                'ns'            :  ns,
                'hubble'        :  h,
                'sigma8_cold'   :  sigma8_cb,
                'omega_baryon'  :  omega_b,
                'omega_cold'    :  omega_cb,
                'neutrino_mass' :  m_nu,
                'w0'            :  w0,
                'wa'            :  wa,
                'expfactor'     :  self.aa_nl
            }
        _, pnn = self.heftemulator.get_nonlinear_pnn(k=self.kh_heft, **params_bacco)
        params_bacco_l = params_bacco
        params_bacco_l['expfactor'] = self.aa_all
        _, plin_bacco = self.baccoemulator.get_linear_pk(k=self.kh_tot, cold=False, **params_bacco_l)
        plin_interp = RectBivariateSpline(self.zz_all_bacco,
                                    self.kh_tot,
                                    plin_bacco[::-1, :],
                                    kx=1, ky=1)
        pnn_interp = [RectBivariateSpline(self.z_nl_bacco, 
                                    self.kh_heft,
                                    pnn_i[::-1, :],
                                    kx=1, ky=1) for pnn_i in pnn]
        return pnn_interp, plin_interp 
    
    
    def get_barboost(self, params_dic, k, lbin, zz_integr):
        boost_bar_interp = self.get_barboost_interp(params_dic)
        boost_bar = fill_in_ell_z_array(boost_bar_interp, k, lbin, zz_integr, zmax=1.5)
        return boost_bar

    def get_pk_nl_zextr_lin(self, params_dic, k, lbin, zz_integr):
        pk_l_interp = self.get_pk_interp(params_dic)
        pk_nl_l = fill_in_ell_z_array(pk_l_interp, k, lbin, zz_integr)
        return pk_nl_l
    
    def get_pk_nl_zextr_hmcode(self, params_dic, k, lbin, zz_integr):
        pk_l_interp = self.get_pk_interp(params_dic)
        params_dic['As'] = self.get_a_s(params_dic)
        pl_l_interp_hmcode = self.HMcodeEmu.get_pk_interp(params_dic)
        pk_nl_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        for index_l, index_z in index_pknn:
            if zz_integr[index_z]<=1.5:
                pk_nl_l[index_l, index_z] = pk_l_interp(zz_integr[index_z], k[index_l,index_z])    
            else:
                pk_nl_l[index_l, index_z] = pl_l_interp_hmcode(zz_integr[index_z], k[index_l,index_z])        
        return pk_nl_l
    
    def get_pk_lin(self, params_dic, k, lbin, zz_integr):
        pk_l_interp = self.get_pk_lin_interp(params_dic)
        pk_lin_l = fill_in_ell_z_array(pk_l_interp, k, lbin, zz_integr)
        #pk_lin_l = np.zeros((lbin, len(zz_integr)), 'float64')
        #index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < 1000.))).transpose()
        #for index_l, index_z in index_pknn:
        #    pk_lin_l[index_l, index_z] = pk_l_interp(zz_integr[index_z], k[index_l,index_z])    
        return pk_lin_l
    
    
    def get_barboost(self, params_dic, k, lbin, zz_integr):
        boost_bar = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        boost_bar_interp = self.get_barboost_interp(params_dic)
        for index_l, index_z in index_pknn:
            boost_bar[index_l, index_z] = boost_bar_interp(min(zz_integr[index_z], 1.5), k[index_l,index_z])
        return boost_bar
    

    def get_heft(self, params_dic, k, lbin, zz_integr):
        pk_nn_l  = np.zeros((15, lbin, len(zz_integr)), 'float64')
        pk_lin_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k<k_max_h_by_mpc))).transpose()
        pnn_l_interp, plin_l_interp = self.get_heft_interp(params_dic)
        for index_l, index_z in index_pknn:    
            for index_i in np.arange(15): 
                if zz_integr[index_z]<=1.5 and k[index_l,index_z]>=self.kh_nl[0]:
                    pk_nn_l[index_i, index_l, index_z] = pnn_l_interp[index_i](zz_integr[index_z], k[index_l,index_z])   
            if zz_integr[index_z]>1.5 or k[index_l,index_z]<self.kh_nl[0]:
                pk_lin_l[index_l, index_z] = plin_l_interp(zz_integr[index_z], k[index_l,index_z])    
        return pk_nn_l, pk_lin_l
    
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
    
    def get_a_s(self, params):
        # to-do: add from_sigma8_to_sigma8_cb
        # make distinction between sigma8_cold and sigma8
        #status = self.check_pars_lin(params)
        params_bacco = {
        'ns'            :  params['ns'],
        'sigma8_cold'   :  params['sigma8_cb'], 
        'hubble'        :  params['h'],
        'omega_baryon'  :  params['Omega_b'],
        'omega_cold'    :  params['Omega_cb'], 
        'neutrino_mass' :  params['Mnu'],
        'w0'            :  params['w0'],
        'wa'            :  params['wa'],
        'expfactor'     :  1    
        }
        a_s = self.baccoemulator.get_A_s(**params_bacco)
        return a_s
            
    def get_sigma8_cb(self, params):
        params_bacco = {
        'ns'            :  params['ns'],
        'A_s'           :  params['As'], 
        'hubble'        :  params['h'],
        'omega_baryon'  :  params['Omega_b'],
        'omega_cold'    :  params['Omega_cb'], 
        'neutrino_mass' :  params['Mnu'],
        'w0'            :  params['w0'],
        'wa'            :  params['wa'],
        'expfactor'     :  1    
        }
        sigma8_cb = self.baccoemulator.get_sigma8(cold=True, **params_bacco)
        return sigma8_cb        