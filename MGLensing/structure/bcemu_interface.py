from scipy.interpolate import RectBivariateSpline
import numpy as np
import BCemu
import os
from math import log10, log

dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 

emu_ranges_all = {
        'log10Mc_bcemu':    {'p1': 11.4,         'p2': 14.6}, 
        'thej_bcemu':       {'p1': 2.6,          'p2': 7.4},
        'mu_bcemu':         {'p1': 0.2,          'p2': 1.8},
        'gamma_bcemu':      {'p1': 1.3,          'p2': 3.7},
        'delta_bcemu':      {'p1': 3.8,          'p2': 10.2},
        'eta_bcemu':        {'p1': 0.085,        'p2': 0.365},
        'deta_bcemu':       {'p1': 0.085,        'p2': 0.365}
        }

def powerlaw_highk_extrap(pk_or_boost, log_k, k_last, kh_high, zz_num):
    last_entry, lastlast_entry = pk_or_boost[:, -1], pk_or_boost[:, -2]
    m = np.array([log(last_entry[i] / lastlast_entry[i]) / log_k for i in range(zz_num)])
    highk_extrap = last_entry[:, np.newaxis] * (kh_high[np.newaxis, :]/k_last)**m[:, np.newaxis]
    return highk_extrap 

class BCemulator():
    """
    A class to emulate baryonic feedback effects on power spectra.

    Attributes
    ----------
    bfcemu : BCemu.BCM_7param
        An instance of the BCM_7param class for baryonic feedback emulation.
    kh_l : numpy.ndarray
        Array of linear wavenumbers in units of h/Mpc.
    k_bfc : numpy.ndarray
        Array of wavenumbers for baryonic feedback emulation.
    z_bfc : numpy.ndarray
        Array of redshifts for baryonic feedback emulation.
    zz_max : float
        Maximum redshift for baryonic feedback emulation.
    kh_barboost : numpy.ndarray
        Concatenated array of linear and baryonic feedback wavenumbers.
    boost_left : numpy.ndarray
        Array of ones for linear wavenumber bins less than the minimum baryonic feedback wavenumber.
    kmax_bcemu : float
        Maximum wavenumber for baryonic feedback emulation.

    Methods
    -------
    __init__():
        Initializes the BCemulator class with default parameters and sets up the BCemu emulator.
    get_barboost_interp(params_dic):
        Interpolates the baryonic boost factor using the BCemu emulator.
    get_barboost(params_dic, k, lbin, zz_integr):
        Computes the baryonic boost factor for given parameters, wavenumbers, and redshifts.
        Extrapolation for k<0.01 h/Mpc and k>12.51 h/Mpc is a constant first and last values of the baryonic boost, respectively.
        Extrapolation for z>2 is the baryonic boost at z=2.
    """
    def __init__(self):
        print('initialising bcemu')
        self.bfcemu = BCemu.BCM_7param(verbose=False)
        bcemu_k_bins = 200
        bcemu_z_bins = 20
        k_bin = 512
        self.kh_lin = np.logspace(-4, log10(k_max_h_by_mpc), k_bin)
        kmin_bcemu = 0.0342
        kmax_bcemu = 12.51
        zz_max = 3.
        self.k_bfc = np.logspace(log10(kmin_bcemu), log10(kmax_bcemu), bcemu_k_bins)
        self.z_bfc = np.linspace(0., min(2, zz_max), bcemu_z_bins)
        self.zz_max = zz_max
        kbin_left_bb = len(self.kh_lin[self.kh_lin<self.k_bfc[0]])
        self.kh_lin_bb_right = self.kh_lin[self.kh_lin>self.k_bfc[-1]]
        self.kh_bb_last = self.k_bfc[-1]
        self.log_kh_bb = log(self.kh_bb_last / self.k_bfc[-2])
        self.kh_barboost_tot = np.concatenate((self.kh_lin[self.kh_lin<self.k_bfc[0]], self.k_bfc, self.kh_lin_bb_right))
        self.boost_left = np.ones((len(self.z_bfc), kbin_left_bb))
        self.kmax_bcemu = kmax_bcemu
        self.emu_name = 'BCEmu'
        

    def get_barboost_interp(self, params_dic):
        log10mc_bc = params_dic['log10Mc_bcemu']
        thej_bc = params_dic['the_bcemuj']
        mu_bc = params_dic['mu_bcemu']
        gamma_bc = params_dic['gamma_bcemu']
        delta_bc = params_dic['delta_bcemu']
        eta_bc = params_dic['eta_bcemu']
        deta_bc = params_dic['deta_bcemu']
        bcemu_dict ={
            'log10Mc' : log10mc_bc,
            'mu'     : mu_bc,
            'thej'   : thej_bc,  
            'gamma'  : gamma_bc,
            'delta'  : delta_bc, 
            'eta'    : eta_bc, 
            'deta'   : deta_bc, 
                }
        fb = params_dic['fb']
        boost = [self.bfcemu.get_boost(z_i,bcemu_dict, self.k_bfc,fb) for z_i in self.z_bfc]
        boost_right = powerlaw_highk_extrap(boost, self.log_kh_bb, self.kh_bb_last, self.kh_lin_bb_right, len(self.z_bfc))
        boost_k  = np.concatenate((self.boost_left, boost, boost_right),axis=1)
        bfc_interpolator = RectBivariateSpline(self.z_bfc, 
                                               self.kh_barboost_tot,
                                               boost_k,
                                               kx=1, ky=1)
        return  bfc_interpolator   


    def get_barboost(self, params_dic, k, lbin, zz_integr):
        boost_bar = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k<k_max_h_by_mpc))).transpose()
        boost_bar_interp = self.get_barboost_interp(params_dic)
        for index_l, index_z in index_pknn:
                boost_bar[index_l, index_z] = boost_bar_interp(min(zz_integr[index_z], 2.), k[index_l,index_z])
        return boost_bar
    
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
            print(f"BCemu emulator:")
            print(f"  Please add the parameter(s) {miss_pars}"
                  f" to your parameters!")
            raise KeyError(f"BCemu emulator: coordinates need the"
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
        return True
    
