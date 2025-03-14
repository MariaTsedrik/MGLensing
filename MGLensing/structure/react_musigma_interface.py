from scipy.interpolate import RectBivariateSpline
from scipy import interpolate as itp
import numpy as np
from cosmopower import cosmopower_NN
import os
import MGrowth as mg
from math import log10, log

dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 

a_arr_for_mu = np.logspace(-5., 1., 512)

emu_ranges_all = {
        'Omega_m':      {'p1':0.2,            'p2':0.6}, 
        'Omega_b':      {'p1':0.03,           'p2':0.07}, 
        'h':            {'p1':0.58,           'p2':0.8},
        'ns':           {'p1':0.93,           'p2':1.0},
        'As':           {'p1':0.5e-09,        'p2':5.e-09},
        'Mnu':          {'p1':0.,             'p2':0.6},
        'mu0':          {'p1':-0.9999,        'p2':2.9999},
        'sigma0':       {'p1':-1.,            'p2':3.},
        'q1':           {'p1':-2.,            'p2':2.}
}


def powerlaw_highk_extrap(pk_or_boost, log_k, k_last, kh_high, zz_num):
    last_entry, lastlast_entry = pk_or_boost[:, -1], pk_or_boost[:, -2]
    m = np.array([log(last_entry[i] / lastlast_entry[i]) / log_k for i in range(zz_num)])
    highk_extrap = last_entry[:, np.newaxis] * (kh_high[np.newaxis, :]/k_last)**m[:, np.newaxis]
    return highk_extrap 

class MuSigmaReACT():
    def __init__(self):
        self.zz_pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5, 3.0]) # these numers are hand-picked
        self.aa_pk = np.array(1./(1.+self.zz_pk[::-1])) # should be increasing
        self.nz_pk = len(self.zz_pk)
        self.zz_max = self.zz_pk[-1]
        print('initialising mu-Sigma')
        self.cp_nl_musigma_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/MuSigma_nonlinearboost_extAs', 
                      )
        self.kh_nl_boost = self.cp_nl_musigma_model.modes # 0.01..10. h/Mpc
        self.cp_lin_musigma_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/MuSigma_linear_log10ps_extAs', 
                      )
        self.kh_lin = self.cp_lin_musigma_model.modes # 1.e-4..50. h/Mpc IMPORTANT LATER USED IN TATT

        self.zz_boost = np.minimum(self.zz_pk, 2.5)
        self.kh_lin_left_boost = self.kh_lin[self.kh_lin<self.kh_nl_boost[0]]
        self.kh_nl_right_boost = self.kh_lin[self.kh_lin>self.kh_nl_boost[-1]]
        self.kh_nl_boost_tot = np.concatenate((self.kh_lin_left_boost, self.kh_nl_boost, self.kh_nl_right_boost))
        self.k_nl_boost_last = self.kh_nl_boost[-1]
        self.k_nl_boost_lastlast = self.kh_nl_boost[-2]
        self.log_k = log(self.k_nl_boost_last / self.k_nl_boost_lastlast)
        self.emu_name = 'mu_Sigma_ReACT'

    def check_pars(self, params):
        emu_ranges = emu_ranges_all.copy()
        eva_pars = emu_ranges.keys()     
        if not all(emu_ranges[par_i]['p1'] <= params[par_i] <= emu_ranges[par_i]['p2'] for par_i in eva_pars):
            return False
        if params['w0']!=-1. or params['wa']!=0.:
            return False 
        # check the mu-Sigma condition
        if params['mu0'] > (2.*params['sigma0']+1.):
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
            print(f"ReACT mu-Sigma emulator:")
            print(f"  Please add the parameter(s) {miss_pars}"
                  f" to your parameters!")
            raise KeyError(f"ReACT mu-Sigma emulator: coordinates need the"
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
        # check the mu-Sigma condition
        if params['mu0'] > (2.*params['sigma0']+1.):
             raise KeyError("Stability condition: mu0<=2 Sigma0+1!")        
        return True    

    def get_pk_nl_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        omega_b = params_dic['Omega_b']
        omega_m = params_dic['Omega_m']
        omega_nu  = params_dic['Omega_nu']
        mu0    = params_dic['mu0']
        sigma0    = params_dic['sigma0']
        q1    = params_dic['q1']
        params_react = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'H0'            :  np.full(self.nz_pk, h*100),
                'Omega_b'       :  np.full(self.nz_pk, omega_b),
                'Omega_m'       :  np.full(self.nz_pk, omega_m),
                'Omega_nu'      :  np.full(self.nz_pk, omega_nu),
                'mu0'           :  np.full(self.nz_pk, mu0),
                'sigma0'        :  np.full(self.nz_pk, sigma0),
                'q1'            :  np.full(self.nz_pk, q1),
                'z'             :  self.zz_boost
            }
        mg_boost = self.cp_nl_musigma_model.predictions_np(params_react) 
        mg_pk_lin = self.cp_lin_musigma_model.ten_to_predictions_np(params_react)
        self.pklin_z0 = mg_pk_lin[0] # zz_pk[0] must be 0.! later used in tatt
        # constant extrapolation for k<0.01 h/Mpc
        mg_boost_left = np.ones(len(self.kh_lin_left_boost))
        # power law extrapolation for k>10 h/Mpc
        mg_boost_right = powerlaw_highk_extrap(mg_boost, self.log_k, self.k_nl_boost_last, self.kh_nl_right_boost, self.zz_boost)
        # combine mg_boost at all scales
        mg_boost_k = np.concatenate((mg_boost_left, mg_boost, mg_boost_right))
        # interpolate
        mg_boost_interp = RectBivariateSpline(self.zz_boost,
                    self.kh_nl_boost_tot,
                    mg_boost_k,
                    kx=1, ky=1)
        mg_plin_interp = RectBivariateSpline(self.zz_boost,
                    self.kh_lin,
                    mg_pk_lin,
                    kx=1, ky=1)
        return  mg_boost_interp, mg_plin_interp


    def get_pk_nl(self, params_dic, k, lbin, zz_integr):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        mg_boost_l_interp, pk_l_interp = self.get_pk_nl_interp(params_dic)
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(min(zz_integr[index_z], 2.5), k[index_l,index_z])*mg_boost_l_interp(min(zz_integr[index_z], 2.5), k[index_l,index_z])
        return pk_m_l  
    

    def get_growth(self, params_dic, zz_integr):
        aa_integr =  np.array(1./(1.+zz_integr[::-1]))
        background ={
            'Omega_m': params_dic['Omega_m'],
            'h' : params_dic['h'],
            'w0': -1.,
            'wa': 0.,
            'a_arr': np.hstack((aa_integr, 1.))
            }
        omega_m = params_dic['Omega_m']
        w0 = params_dic['w0']
        wa = params_dic['wa']
        omega_lambda = (1.-omega_m)* pow(a_arr_for_mu, -3.*(1.+w0+wa)) * np.exp(-3.*wa*(1.-a_arr_for_mu))
        e2 = omega_m/a_arr_for_mu**3+omega_lambda
        cosmo = mg.mu_a(background)
        mu0 = params_dic['mu0'] 
        mu0_arr = 1.+mu0/e2
        mu_interpolator = itp.interp1d(a_arr_for_mu, mu0_arr, bounds_error=False,
            kind='cubic') 
        da, _ = cosmo.growth_parameters(mu_interp=mu_interpolator) 
        dz = da[::-1] 
        # growth factor should be normalised to z=0
        dz0 = dz[0]
        dz = dz[1:]/dz0
        return dz, dz0