from tkinter import W
from scipy.interpolate import RectBivariateSpline, interp1d
import numpy as np
from cosmopower import cosmopower_NN
import os
import MGrowth as mg
from math import log10, log
try: import pyhmcode  
except: print('PyHMCode not installed!')
from .hmcode2020_interface import HMcode2020
dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 

emu_ranges_all = {
        'Omega_m':      {'p1':0.24,            'p2':0.4}, 
        'Omega_b':      {'p1':0.049,         'p2':0.049}, 
        'h':            {'p1':0.6,           'p2':0.84},
        'ns':           {'p1':0.9649,          'p2':0.9649},
        'As':           {'p1':1.7e-09,         'p2':2.5e-09},
        'Omega_nu':     {'p1':0.,              'p2':0.},
        'w0':           {'p1':-1.5,              'p2':-0.5},
        'wa':          {'p1':-0.5,            'p2':0.5},

        'mu0':           {'p1':-0.999,             'p2':3.},
        'c1':           {'p1':-0.3333,             'p2':1.},
        'lam':           {'p1':0.,             'p2':2.},

        'q1':           {'p1':-2.,             'p2':2.},
        'q2':           {'p1':-2.,             'p2':2.},
        'q3':           {'p1':-2.,             'p2':2.}
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

class MuKZReACT():
    def __init__(self, option=None):
        #self.zz_pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5, 3.0]) # these numers are hand-picked
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

        print('initialising Scale-dependent mu-Sigma Parametrisation')
        self.cp_nl_mukz_model = cosmopower_NN(restore=True, 
                        restore_filename=dirname+'/../../emulators/mu_sigma/mu_z_k/react_boost_mu_z_k_q123', 
                        )
        self.kh_nl_boost = self.cp_nl_mukz_model.modes # 0.01..5. h/Mpc 
        self.zz_boost = np.minimum(self.zz_pk, 2.5)
        self.kh_lin_left_boost = self.kh_lin[self.kh_lin<self.kh_nl_boost[0]]
        self.kh_nl_right_boost = self.kh_nl[self.kh_nl>self.kh_nl_boost[-1]]
        self.kh_nl_boost_tot = np.concatenate((self.kh_lin_left_boost, self.kh_nl_boost, self.kh_nl_right_boost))
        self.k_nl_boost_last = self.kh_nl_boost[-1]
        self.k_nl_boost_lastlast = self.kh_nl_boost[-2]
        self.log_k = log(self.k_nl_boost_last / self.k_nl_boost_lastlast)
        self.kh_lin_less_bins = np.logspace(np.log10(3.7e-4), np.log10(50.), 128, endpoint=True)
        print('initialising ReACT scale-dep mu')
        self.HMcodeEmu = HMcode2020()
        self.emu_name = 'ReACT mu-Sigma-k-z'

        # load pyhmcode objects
        self.hmc = pyhmcode.Cosmology()
        # Set the halo model in HMcode
        # Options: HMcode2015, HMcode2016, HMcode2020
        self.hmod = pyhmcode.Halomodel(pyhmcode.HMcode2020, verbose=False)

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
            print(f"ReACT mu(k,z) emulator:")
            print(f"  Please add the parameter(s) {miss_pars}"
                  f" to your parameters!")
            raise KeyError(f"ReACT mu-k-z emulator: coordinates need the"
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
             raise KeyError("Applicable only for Lambda as dark energy for now!")      
        # check the mu-Sigma condition
        if params['mu0'] > (2.*params['sigma0']+1.):
             raise KeyError("Stability condition: mu0<=2 Sigma0+1!")   
        return True    

    def get_mg_boost_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        omega_b = params_dic['Omega_b']
        omega_m = params_dic['Omega_m']
        omega_nu  = params_dic['Omega_nu']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        mu0    = params_dic['mu0']
        c1    = params_dic['c1']
        lam    = params_dic['lam']
        q1    = params_dic['q1']
        q2    = params_dic['q2']
        q3    = params_dic['q3']
        params_react = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'H0'            :  np.full(self.nz_pk, h*100),
                'Omega_b'       :  np.full(self.nz_pk, omega_b),
                'Omega_m'       :  np.full(self.nz_pk, omega_m),
                'Omega_nu'      :  np.full(self.nz_pk, omega_nu),
                'w0'        :  np.full(self.nz_pk, w0),
                'wa'        :  np.full(self.nz_pk, wa),
                'mu0'        :  np.full(self.nz_pk, mu0),
                'c1'        :  np.full(self.nz_pk, c1),
                'lam'        :  np.full(self.nz_pk, lam),
                'q1'            :  np.full(self.nz_pk, q1),
                'q2'            :  np.full(self.nz_pk, q2),
                'q3'            :  np.full(self.nz_pk, q3),
                'z'             :  self.zz_boost
            }
        mg_boost = self.cp_nl_mukz_model.predictions_np(params_react) 
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
    
    def get_pk_react(self, params_dic, k, lbin, zz_integr, mg_boost_l_interp):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        pk_l_interp = self.get_pk_hmcode_interp(params_dic)

        # index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        # for index_l, index_z in index_pknn:
        #     pk_m_l[index_l, index_z] = pk_l_interp(zz_integr[index_z], k[index_l,index_z])*mg_boost_l_interp(min(zz_integr[index_z], 2.5), k[index_l,index_z])
        # print(pk_m_l.shape)
        array = np.zeros((lbin, len(zz_integr)), 'float64')
        mask = (k > k_min_h_by_mpc) & (k < k_max_h_by_mpc)
        index_l, index_z = np.where(mask)
        z_pts = zz_integr[index_z]
        k_pts = k[index_l, index_z]
        array[index_l, index_z] = np.ravel(
            pk_l_interp(z_pts, k_pts, grid=False)
            * mg_boost_l_interp(np.minimum(z_pts, 2.5), k_pts, grid=False)
        )
        pk_m_l = array
        return pk_m_l  


    def get_pk_nl_(self, params_dic, k, lbin, zz_integr):
        mg_boost_l_interp = self.get_mg_boost_interp(params_dic)
        pk_m_l = self.get_pk_react(params_dic, k, lbin, zz_integr, mg_boost_l_interp)
        return pk_m_l

    def sigma_lensing(self, params_dic, k, lbin, zz_integr):
        omega_m = params_dic['Omega_m']
        w0 = params_dic['w0']
        wa = params_dic['wa']
        omega_lambda = (1.-omega_m)* pow(1.+zz_integr, 3.*(1.+w0+wa)) * np.exp(-3.*wa*zz_integr/(1.+zz_integr))
        e2 = omega_m*(1.+zz_integr)**3+omega_lambda
        sigma0 = params_dic['sigma0'] 
        c2 = params_dic['c2']
        lamb = params_dic['lam']
        sigma = 1.+sigma0/e2[None, :]*omega_lambda[None, :]/(1.-omega_m)*(1+c2*(lamb*np.sqrt(e2[None, :])/k[:, None])**2)/(1.+(lamb*np.sqrt(e2[None, :])/k[:, None])**2)
        #dimensions of (ell, zz_integr, n_bin)
        return sigma[:, :, None]
    
    def get_pk_pseudo(self, params_dic, k, lbin, zz_integr):
        # call linear lcdm with original As  
        # interpolator of (zz, k)
        pk_l_interp = self.get_pk_hmcode_lin_interp(params_dic)
        # TO-DO implemenet this propery later
        #self.pklin_z0 = self.pklin_z0_lcdm # later used in tatt 
        dz_mu0, dz0_mu0 = self.get_growth(params_dic, self.zz_pk)
        dz_mu0_notnorm = dz_mu0*dz0_mu0
        dz_lcdm, dz0_lcdm = self.get_growth_lcdm(params_dic, self.zz_pk)
        dz_lcdm_notnorm = dz_lcdm*dz0_lcdm
        dz_rescale = (dz_mu0_notnorm/dz_lcdm_notnorm[None, :])**2
        
        zz_pk_short = np.linspace(0., 3., 128, endpoint=True)
        
        d2_mg_lcdm_z_interp  = RectBivariateSpline(
                            self.kh_lin_less_bins,
                            self.zz_pk,
                            dz_rescale, #dz_norm,
                            kx=1, ky=1)  
        # zz_pk, kh_lin
        pk_lin = pk_l_interp(0., self.kh_lin)*d2_mg_lcdm_z_interp(self.kh_lin, zz_pk_short).T  
        sigma8 = self.HMcodeEmu.get_sigma8_lcdm(params_dic)[0]

        # Set HMcode internal cosmological parameters
        self.hmc.om_m = params_dic['Omega_m']
        self.hmc.om_b = params_dic['Omega_b']
        self.hmc.om_v = 1.- params_dic['Omega_m']
        self.hmc.h = params_dic['h']
        self.hmc.ns = params_dic['ns']
        self.hmc.sig8 = sigma8
        self.hmc.m_nu = params_dic['Mnu'] if 'Mnu' in params_dic else 0.
        # silly re-scalling required py pyhmcode
        pnl_hmcode = []
        pk_lin_hmc_4 =  pk_lin[:4, :]
        self.hmc.set_linear_power_spectrum(self.kh_lin, zz_pk_short[:4], pk_lin_hmc_4)
        pnl_hmcode.append(pyhmcode.calculate_nonlinear_power_spectrum(self.hmc, self.hmod, verbose=False)[0])
        for i in range(len(zz_pk_short)-1):
            zz_pk_short_4 = np.linspace(0., zz_pk_short[i+1], endpoint=True, num=4)
            pk_lin_hmc_4[0] = pk_lin[i+1]
            self.hmc.set_linear_power_spectrum(self.kh_lin, zz_pk_short_4, pk_lin_hmc_4)
            pnl_hmcode.append(pyhmcode.calculate_nonlinear_power_spectrum(self.hmc, self.hmod, verbose=False)[-1])

        pnl_hmcode = np.array(pnl_hmcode)
        pnl_hmcode_interp = RectBivariateSpline(
                                    zz_pk_short,
                                    self.kh_lin,
                                    pnl_hmcode,
                                    kx=1, ky=1)
        #index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        #for index_l, index_z in index_pknn:
        #    pk_m_l[index_l, index_z] = pnl_hmcode_interp(zz_integr[index_z], k[index_l,index_z])
        pk_m_l = fill_in_ell_z_array(pnl_hmcode_interp, k, lbin, zz_integr)
        return pk_m_l 
    
    def get_pk_lin(self, params_dic, k, lbin, zz_integr):
        pk_l_interp = self.HMcodeEmu.get_pk_lin_interp(params_dic)
        # TO-DO implemenet this propery later
        #self.pklin_z0 = self.pklin_z0_lcdm # later used in tatt 
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        
        dz_mu0, dz0_mu0 = self.get_growth(params_dic, self.zz_pk)
        dz_mu0_notnorm = dz_mu0*dz0_mu0
        _, dz0_lcdm = self.get_growth_lcdm(params_dic, self.zz_pk)
        dz_norm = (dz_mu0_notnorm/dz0_lcdm)**2
        d2_mg_lcdm_z_interp  = RectBivariateSpline(
                            self.kh_lin_less_bins,
                            self.zz_pk,
                            dz_norm,
                            kx=1, ky=1)  
        #D_MG(z, k)^2/D_GR(z=0)^2 P_L,GR(z=0, k)  
        #index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        #for index_l, index_z in index_pknn:
        #    pk_m_l[index_l, index_z] = pk_l_interp(0., k[index_l,index_z])*d2_mg_lcdm_z_interp(k[index_l,index_z], zz_integr[index_z])

        array = np.zeros((lbin, len(zz_integr)), 'float64')
        mask = (k > k_min_h_by_mpc) & (k < k_max_h_by_mpc)
        index_l, index_z = np.where(mask)
        z_pts = zz_integr[index_z]
        k_pts = k[index_l, index_z]
        array[index_l, index_z] = np.ravel(pk_l_interp(0., k_pts, grid=False)*d2_mg_lcdm_z_interp(k_pts, z_pts, grid=False))
        pk_m_l = array
        return pk_m_l  

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
        aa_interp = np.linspace(1e-3, 1, 128)
        mu_de_k = np.array([self.mu_DE(aa_interp,  k_i, params_dic['Omega_m'], params_dic['mu0'], params_dic['c1'], params_dic['lam']) for k_i in self.kh_lin_less_bins])
        mu_interpolator_k = np.array([interp1d(aa_interp, mu_de_k_i, bounds_error=False,
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