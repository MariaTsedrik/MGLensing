from scipy.interpolate import RectBivariateSpline, interp1d
import numpy as np
from cosmopower import cosmopower_NN
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
        'Omega_c':      {'p1': 0.1,         'p2': 0.8},    
        'Omega_b':      {'p1':0.01,         'p2': 0.1},  
        'h':            {'p1': 0.4,         'p2': 1.},      
        'As':           {'p1': 0.495e-9,    'p2': 5.459e-9},
        'ns':           {'p1': 0.6,         'p2': 1.2}, 
        'Mnu':          {'p1': 0.0,         'p2': 0.5},
        'w0':           {'p1': -3.,         'p2': -0.3},    
        'wa':           {'p1': -3.,         'p2': 3.},  
}

def powerlaw_highk_extrap(pk_or_boost, log_k, k_last, kh_high, zz_num):
    last_entry, lastlast_entry = pk_or_boost[:, -1], pk_or_boost[:, -2]
    m = np.array([log(np.abs(last_entry[i] / lastlast_entry[i])) / log_k for i in range(zz_num)])
    highk_extrap = last_entry[:, np.newaxis] * (kh_high[np.newaxis, :]/k_last)**m[:, np.newaxis]
    return highk_extrap 

class MGrowthMuKZ():
    def __init__(self, option=None):
        self.zz_pk = np.linspace(0., 3., 64, endpoint=True)
        self.aa_pk = np.array(1./(1.+self.zz_pk[::-1])) # should be increasing
        self.nz_pk = len(self.zz_pk)
        self.zz_max = self.zz_pk[-1]

        self.cp_nl_hmcode_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../../emulators/log10_total_matter_nonlinear_emu',
                      )
        self.kh_nl = self.cp_nl_hmcode_model.modes # 0.01..50. h/Mpc    
        self.cp_lin_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../../emulators/log10_total_matter_linear_emu',
                      )
        self.kh_lin = self.cp_lin_model.modes # 3.7e-4..50. h/Mpc IMPORTANT LATER USED IN TATT
        self.kh_lin_left = self.kh_lin[self.kh_lin<self.kh_nl[0]]
        self.kh_tot = np.concatenate((self.kh_lin_left, self.kh_nl))

        #self.kh_lin_less_bins = np.logspace(np.log10(3.7e-4), np.log10(50.), 64, endpoint=True)
        self.kh_lin_less_bins = np.logspace(np.log10(3.7e-4), np.log10(50.), 50, endpoint=True)
        print('initialising MGrowth scale-dep mu')
        self.HMcodeEmu = HMcode2020()
        self.emu_name = 'MGrowth_HMcode'
        self.get_pk_nl =  self.get_pk_lin 

        self.aa_for_ode = np.linspace(1e-3, 1, 256)


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
            print(f"HMcode with MGrowth emulator:")
            print(f"  Please add the parameter(s) {miss_pars}"
                  f" to your parameters!")
            raise KeyError(f"HMcode with MGrowth emulator: coordinates need the"
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
    
    

    
    def get_pk_lin(self, params_dic, k, lbin, zz_integr):
        pk_l_interp = self.HMcodeEmu.get_pk_lin_interp(params_dic)
        # TO-DO implemenet this propery later
        #self.pklin_z0 = self.pklin_z0_lcdm # later used in tatt 
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        
        dz_fr0, dz0_fr0 = self.get_growth(params_dic, self.zz_pk)
        dz_fr0_notnorm = dz_fr0*dz0_fr0
        _, dz0_lcdm = self.get_growth_lcdm(params_dic, self.zz_pk)
        dz_norm = (dz_fr0_notnorm/dz0_lcdm)**2
        d2_mg_lcdm_z_interp  = RectBivariateSpline(
                            self.kh_lin_less_bins,
                            self.zz_pk,
                            dz_norm,
                            kx=1, ky=1)  
          
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(0., k[index_l,index_z])*d2_mg_lcdm_z_interp(k[index_l,index_z], zz_integr[index_z])
        return pk_m_l  
    
    def get_growth_binned(self, params_dic, k, lbin,  zz_integr):
        dz_mu0k, dz0_mu0k = self.get_growth(params_dic, self.zz_pk)
        dz_mu0k_interp  = RectBivariateSpline(
                            self.kh_lin_less_bins,
                            self.zz_pk,
                            dz_mu0k,
                            kx=1, ky=1)  
        dz0_mu0k_interp  = interp1d(
                            self.kh_lin_less_bins,
                            dz0_mu0k[:, 0])  
        # ones instead of zeros, because dz is in the denominator 
        # of the intrinsic alignment signal
        dz  = np.ones((lbin, len(zz_integr)), 'float64') 
        dz0  = np.ones((lbin, len(zz_integr)), 'float64') 
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose() 
        for index_l, index_z in index_pknn:
            dz[index_l, index_z] = dz_mu0k_interp(k[index_l,index_z], zz_integr[index_z])
            dz0[index_l, index_z] = dz0_mu0k_interp(k[index_l,index_z])
        return dz, dz0
    
    def mu_DE(self, a, k, omega0, mu0, c1, lamb, w0=-1, wa=0.):
        omegaL = (1.-omega0) * a**(-3.*(1.+w0+wa)) * np.exp(3.*(-1.+a)*wa)
        omegaL0 = (1.-omega0) 
        E = np.sqrt(omega0/a**3 + omegaL)
        mu_DE = 1. + mu0*(omegaL/E**2)/omegaL0*((1+c1*(lamb*E/k)**2)/(1.+(lamb*E/k)**2))
        return mu_DE
        

    def get_growth(self, params_dic, zz_integr):
        aa_integr =  np.array(1./(1.+zz_integr[::-1]))
        background ={
            'Omega_m': params_dic['Omega_m'],
            'h' : params_dic['h'],
            'w0': -1.,
            'wa': 0.,
            'a_arr': np.hstack((aa_integr, 1.))
            }
        cosmo = mg.mu_a(background) 
        mu_de_k = np.array([self.mu_DE(aa_integr,  k_i, params_dic['Omega_m'], params_dic['mu0'], params_dic['c1'], params_dic['lambda']) for k_i in self.kh_lin_less_bins])
        mu_interpolator_k = np.array([interp1d(aa_integr, mu_de_k_i, bounds_error=False,
                    kind='cubic',
                    fill_value=(mu_de_k_i[0], mu_de_k_i[-1])) for mu_de_k_i in mu_de_k]) 
        d_f_i =  [cosmo.growth_parameters(mu_interp=mu_interpolator_k_i) for mu_interpolator_k_i in mu_interpolator_k]
        da = np.array([d_i for d_i, _ in d_f_i])
        dz = da[:, ::-1] 
        # growth factor should be normalised to z=0
        # return array of k and z
        dz0 = dz[:, 0]
        dz = dz[:, 1:]/dz0[:, None]
        return dz, dz0[:, None]
    
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