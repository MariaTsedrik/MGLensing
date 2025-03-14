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


emu_ranges_all = {
        'Ombh2':        {'p1':0.01875,            'p2':0.02625}, 
        'Omch2':        {'p1':0.05,           'p2':0.255}, 
        'h':            {'p1':0.64,           'p2':0.82},
        'ns':           {'p1':0.84,           'p2':1.1},
        'S8':           {'p1':0.6,        'p2':0.9},
        'Mnu':          {'p1':0.,             'p2':0.2},
        'w0':           {'p1':-1.3,            'p2':-0.7},
        'Ads':           {'p1':-30.,            'p2':30.}
}

def powerlaw_highk_extrap(pk_or_boost, log_k, k_last, kh_high, zz_num):
    last_entry, lastlast_entry = pk_or_boost[:, -1], pk_or_boost[:, -2]
    m = np.array([log(last_entry[i] / lastlast_entry[i]) / log_k for i in range(zz_num)])
    highk_extrap = last_entry[:, np.newaxis] * (kh_high[np.newaxis, :]/k_last)**m[:, np.newaxis]
    return highk_extrap 


class DarkScatteringReACT():
    def __init__(self):
        self.zz_pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5, 3.0]) # these numers are hand-picked
        self.aa_pk = np.array(1./(1.+self.zz_pk[::-1])) # should be increasing
        self.nz_pk = len(self.zz_pk)
        self.zz_max = self.zz_pk[-1]
        print('initialising Dark Scattering')
        self.cp_nl_ds_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/DS_nonlinear_cp_NN_S8', 
                      )
        self.kh_nl_boost = self.cp_nl_ds_model.modes # 1e-3..10. h/Mpc

        self.cp_lin_ds_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/DS_linear_cp_NN_S8', 
                      )
        self.kh_lin_boost = self.cp_lin_ds_model.modes # 1e-3..10. h/Mpc later used in tatt


        kh_lin_ = np.logspace(log10(self.kh_nl_boost[0]), log10(k_max_h_by_mpc), num=256)
        self.zz_boost = np.minimum(self.zz_pk, 2.5)
        self.kh_nl_right_boost = kh_lin_[kh_lin_>self.kh_nl_boost[-1]]
        self.kh_nl_boost_tot = np.concatenate((self.kh_nl_boost, self.kh_nl_right_boost))
        self.k_nl_boost_last = self.kh_nl_boost[-1]
        self.k_nl_boost_lastlast = self.kh_nl_boost[-2]
        self.log_k = log(self.k_nl_boost_last / self.k_nl_boost_lastlast)
        self.emu_name = 'Dark_Scattering_ReACT'

    def check_pars(self, params):
        emu_ranges = emu_ranges_all.copy()
        eva_pars = emu_ranges.keys()     
        if not all(emu_ranges[par_i]['p1'] <= params[par_i] <= emu_ranges[par_i]['p2'] for par_i in eva_pars):
            return False
        # check the DS condition
        if params['w0']!=-1.:
            if params['Ads']/(1. + params['w0']) < 0:
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
            print(f"ReACT Dark Scattering emulator:")
            print(f"  Please add the parameter(s) {miss_pars}"
                  f" to your parameters!")
            raise KeyError(f"ReACT DS emulator: coordinates need the"
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
        # check the DS condition
        if params['w0']!=-1.:
            if params['Ads']/(1. + params['w0']) < 0:
                raise KeyError("Positive scattering amplitude requires Ads/(1+w) to be non-negative!")       
        if params['wa']!=0.0:
            raise KeyError("Only constant equation of state for dark energy is allowed!")      
        return True       
        

    def get_pk_nl_interp(self, params_dic):
        ns   = params_dic['ns']
        s8   = params_dic['S8']
        h    = params_dic['h']
        ombh2 = params_dic['Ombh2']
        omch2 = params_dic['Omch2']
        m_nu  = params_dic['Mnu']
        w0    = params_dic['w0']
        ads    = params_dic['Ads']
        params_react = {
                'n_s'           :  np.full(self.nz_pk, ns),
                'S_8'           :  np.full(self.nz_pk, s8),
                'h'             :  np.full(self.nz_pk, h),
                'omega_b'       :  np.full(self.nz_pk, ombh2),
                'omega_cdm'     :  np.full(self.nz_pk, omch2),
                'm_nu'          :  np.full(self.nz_pk, m_nu),
                'w'             :  np.full(self.nz_pk, w0),
                'A'             :  np.full(self.nz_pk, ads),
                'z'             :  self.zz_boost
            }
        mg_pk_nl = self.cp_nl_ds_model.ten_to_predictions_np(params_react)
        mg_pk_lin = self.cp_lin_ds_model.ten_to_predictions_np(params_react)
        self.pklin_z0 = mg_pk_lin[0] # zz_pk[0] must be 0.! later used in tatt
        # power law extrapolation for k>10 h/Mpc
        mg_pk_nl_right = powerlaw_highk_extrap(mg_pk_nl, self.log_k, self.k_nl_boost_last, self.kh_nl_right_boost, self.zz_boost)
        # combine mg_boost at all scales
        mg_pnl_k = np.concatenate((mg_pk_nl, mg_pk_nl_right))
        # interpolate
        mg_pnl_interp = RectBivariateSpline(self.zz_boost,
                    self.kh_nl_boost_tot,
                    mg_pnl_k,
                    kx=1, ky=1)
        return  mg_pnl_interp
    
    def get_pk_nl(self, params_dic, k, lbin, zz_integr):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        pk_l_interp = self.get_pk_nl_interp(params_dic)
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(min(zz_integr[index_z], 2.5), k[index_l,index_z])
        return pk_m_l  
    
    def get_growth(self, params_dic, zz_integr):
        aa_integr =  np.array(1./(1.+zz_integr[::-1]))
        background ={
            'Omega_m': params_dic['Omega_m'],
            'h' : params_dic['h'],
            'w0': params_dic['w0'],
            'wa': 0.,
            'a_arr': np.hstack((aa_integr, 1.))
            }
        cosmo = mg.IDE(background)
        xi = params_dic['Ads']/(1.+params_dic['w0'])
        da, _ = cosmo.growth_parameters(xi=xi)  
        dz = da[::-1] 
        # growth factor should be normalised to z=0
        dz0 = dz[0]
        dz = dz[1:]/dz0
        return dz, dz0