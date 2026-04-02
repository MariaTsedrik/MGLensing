from scipy.interpolate import RectBivariateSpline
from scipy import interpolate as itp
import numpy as np
from cosmopower import cosmopower_NN
import os
import MGrowth as mg
from math import log10, log
from scipy.interpolate import CubicSpline
dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 

a_arr_for_mu = np.logspace(-5., 1., 512)

# Constants
nbin = 3
zstart, ztanh = 3.0, 4.0
astart, atanh, aend = 1.0 / (1.0 + zstart), 1.0 / (1.0 + ztanh), 1.0
# Arrays
a_arr = np.zeros(2 * nbin, dtype=np.float64)
mu_arr = np.zeros(2 * nbin, dtype=np.float64)
for i in range(1, nbin + 1):
        a_arr[i - 1] = atanh * float(i - 1) / float(nbin - 1)
        a_arr[nbin + i - 1] = astart + (1.0 - astart) * float(i - 1) / float(nbin - 1)
a_arr[0]=1e-3

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

class MuSigmaSpline():
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

        
        print('initialising scale-indep mu')
        self.emu_name = 'mu-spline interpolation'


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
        
        dz_mu0, dz0_mu0 = self.get_growth(params_dic, self.zz_pk)
        dz_mu0_notnorm = dz_mu0*dz0_mu0
        _, dz0_lcdm = self.get_growth_lcdm(params_dic, self.zz_pk)
        dz_norm = (dz_mu0_notnorm/dz0_lcdm)**2

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

    def mu_recon(self, mu_bins):
        for i in range(nbin+1):
            mu_arr[nbin-1+i]=mu_bins[i]
        for j in range(nbin-1):
            mu_arr[j] = (mu_arr[nbin-1] - 1.0) / 2.0 * (1.0 + np.tanh((a_arr[j] - atanh / 2.0) / 0.04)) + 1.0
        # Spline interpolation
        spline_mu = CubicSpline(a_arr, mu_arr)
        return spline_mu
    

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
        # mu0 is for a=1/z=0, ...
        # for 3 bins we will have 4 mu-values
        mu_bins_in_z = np.array([params_dic['mu'+str(i)] for i in range(0, nbin+1)])
        mu_bins_in_a = mu_bins_in_z[::-1]
        mu_interpolator = self.mu_recon(mu_bins_in_a)
        da, _ = cosmo.growth_parameters(mu_interp=mu_interpolator) 
        dz = da[::-1] 
        # growth factor should be normalised to z=0
        dz0 = dz[0]
        dz = dz[1:]/dz0
        return dz, dz0

    
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