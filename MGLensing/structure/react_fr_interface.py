from scipy.interpolate import RectBivariateSpline, interp1d
import numpy as np
from cosmopower import cosmopower_NN
import os
import MGrowth as mg
from math import log10, log
from .hmcode2020_interface import HMcode2020
try: import pyhmcode
except: print('PyHMCode not installed!')

dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 

emu_ranges_all = {
        'Omega_m':      {'p1':0.24,           'p2':0.35}, 
        'Omega_b':      {'p1':0.040,        'p2':0.06}, 
        'h':            {'p1':0.63,           'p2':0.75},
        'ns':           {'p1':0.9,           'p2':1.01},
        'As':           {'p1':1.7e-09,        'p2':2.5e-09},
        'Omega_nu':          {'p1':0.,             'p2':0.00317},
        'log10f_R0':{'p1':-10.,            'p2':-4.},
}

def powerlaw_highk_extrap(pk_or_boost, log_k, k_last, kh_high, zz_num):
    last_entry, lastlast_entry = pk_or_boost[:, -1], pk_or_boost[:, -2]
    m = np.array([log(np.abs(last_entry[i] / lastlast_entry[i])) / log_k for i in range(zz_num)])
    highk_extrap = last_entry[:, np.newaxis] * (kh_high[np.newaxis, :]/k_last)**m[:, np.newaxis]
    return highk_extrap 

class FofRReACT():
    def __init__(self, option=None):
        self.zz_pk = np.linspace(0., 3., 64, endpoint=True)
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

        self.kh_lin_short = np.logspace(np.log10(3.7e-4), np.log10(50.), 50, endpoint=True)

        print('initialising f(R) ReACT')
        self.cp_nl_fofr_model = cosmopower_NN(restore=True, 
                        restore_filename=dirname+'/../../emulators/fofr/react_boost_fofr',
                        )
        self.kh_nl_boost = self.cp_nl_fofr_model.modes # 0.01..3. h/Mpc 
        self.zz_boost = np.minimum(self.zz_pk, 2.)
        self.kh_lin_left_boost = self.kh_lin[self.kh_lin<self.kh_nl_boost[0]]
        self.kh_nl_right_boost = self.kh_nl[self.kh_nl>self.kh_nl_boost[-1]]
        self.kh_nl_boost_tot = np.concatenate((self.kh_lin_left_boost, self.kh_nl_boost, self.kh_nl_right_boost))
        self.k_nl_boost_last = self.kh_nl_boost[-1]
        self.k_nl_boost_lastlast = self.kh_nl_boost[-2]
        self.log_k = log(self.k_nl_boost_last / self.k_nl_boost_lastlast)
        self.emu_name = 'fR_ReACT'

        # load for sigma8 computation 
        self.HMcodeEmu = HMcode2020()

        if option=='linear':
            self.get_pk_nl =  self.get_pk_lin 
        elif option=='pseudo':
            self.get_pk_nl = self.get_pk_pseudo
            # load pyhmcode objects
            self.hmc = pyhmcode.Cosmology()
            # Set the halo model in HMcode
            # Options: HMcode2015, HMcode2016, HMcode2020
            self.hmod = pyhmcode.Halomodel(pyhmcode.HMcode2020, verbose=False)
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
            print(f"ReACT f(R) emulator:")
            print(f"  Please add the parameter(s) {miss_pars}"
                  f" to your parameters!")
            raise KeyError(f"ReACT f(R) emulator: coordinates need the"
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
        fr0    = 10**params_dic['log10f_R0']
        params_react = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'H0'            :  np.full(self.nz_pk, h*100),
                'Omega_b'       :  np.full(self.nz_pk, omega_b),
                'Omega_m'       :  np.full(self.nz_pk, omega_m),
                'Omega_nu'      :  np.full(self.nz_pk, omega_nu),
                'fR0'           :  np.full(self.nz_pk, fr0),
                'z'             :  self.zz_boost
            }
        mg_boost = self.cp_nl_fofr_model.predictions_np(params_react) #(zz_boost, kh_nl_boost)
        #self.d2_mg_lcdm = mg_boost[0, 0] # zz_boost[0] must be 0.!
        #self.d2_mg_lcdm_z = mg_boost[:, 0]
        # constant extrapolation for k<0.01 h/Mpc
        mg_boost_left = np.full((self.nz_pk, len(self.kh_lin_left_boost)), mg_boost[:, [0]])
        #print('mg_boost: ', mg_boost.shape)
        #print('mg_boost_left: ', mg_boost_left.shape)
        # power law extrapolation for k>5 h/Mpc
        mg_boost_right = powerlaw_highk_extrap(mg_boost, self.log_k, self.k_nl_boost_last, self.kh_nl_right_boost, self.nz_pk)
        #print('mg_boost_right: ', mg_boost_right.shape)
        # combine mg_boost at all scales
        mg_boost_k = np.concatenate((
            mg_boost_left,
            mg_boost,
            mg_boost_right
        ), axis=1)
        #print('mg_boost_k: ', mg_boost_k.shape)
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
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        pk_l_interp = self.get_pk_hmcode_interp(params_dic)
        # TO-DO implemenet this propery later
        # d2_mg_lcdm is computed in self.get_mg_boost_interp
        #self.pklin_z0 = self.d2_mg_lcdm * self.pklin_z0_lcdm # later used in tatt 
        self.pklin_z0 = self.pklin_z0_lcdm # later used in tatt 
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(zz_integr[index_z], k[index_l,index_z])*mg_boost_l_interp(min(zz_integr[index_z], 2.), k[index_l,index_z])
        return pk_m_l  


    def get_pk_nl_(self, params_dic, k, lbin, zz_integr):
        mg_boost_l_interp = self.get_mg_boost_interp(params_dic)
        pk_m_l = self.get_pk_react(params_dic, k, lbin, zz_integr, mg_boost_l_interp)
        return pk_m_l
    
    def get_pk_pseudo(self, params_dic, k, lbin, zz_integr):
        # call linear lcdm with original As  
        # interpolator of (zz, k)
        pk_l_interp = self.get_pk_hmcode_lin_interp(params_dic)
        # TO-DO implemenet this propery later
        #self.pklin_z0 = self.pklin_z0_lcdm # later used in tatt 
        dz_fr0, dz0_fr0 = self.get_growth(params_dic, self.zz_pk)
        dz_fr0_notnorm = dz_fr0*dz0_fr0
        dz_lcdm, dz0_lcdm = self.get_growth_lcdm(params_dic, self.zz_pk)
        dz_lcdm_notnorm = dz_lcdm*dz0_lcdm
        #dz_norm = (dz_fr0_notnorm/dz0_lcdm)**2
        dz_rescale = (dz_fr0_notnorm/dz_lcdm_notnorm[None, :])**2
        
        zz_pk_short = np.linspace(0., 3., 32, endpoint=True)
        
        d2_mg_lcdm_z_interp  = RectBivariateSpline(
                            self.kh_lin_short,
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

        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pnl_hmcode_interp(zz_integr[index_z], k[index_l,index_z])
        return pk_m_l  


    def get_pk_lin(self, params_dic, k, lbin, zz_integr):
        # call linear lcdm with original As  
        pk_l_interp = self.get_pk_hmcode_lin_interp(params_dic)
        # TO-DO implemenet this propery later
        #self.pklin_z0 = self.pklin_z0_lcdm # later used in tatt 
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        dz_fr0, dz0_fr0 = self.get_growth(params_dic, self.zz_pk)
        dz_fr0_notnorm = dz_fr0*dz0_fr0
        _, dz0_lcdm = self.get_growth_lcdm(params_dic, self.zz_pk)
        dz_norm = (dz_fr0_notnorm/dz0_lcdm)**2
        d2_mg_lcdm_z_interp  = RectBivariateSpline(
                            self.kh_lin_short,
                            self.zz_pk,
                            dz_norm,
                            kx=1, ky=1)    
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(0., k[index_l,index_z])*d2_mg_lcdm_z_interp(k[index_l,index_z], zz_integr[index_z])
        return pk_m_l  
    
    def get_growth_binned(self, params_dic, k, lbin,  zz_integr):
        dz_fr0, dz0_fr0 = self.get_growth(params_dic, self.zz_pk)
        dz_fr0_interp  = RectBivariateSpline(
                            self.kh_lin_short,
                            self.zz_pk,
                            dz_fr0,
                            kx=1, ky=1)  
        dz0_fr0_interp  = interp1d(
                            self.kh_lin_short,
                            dz0_fr0[:, 0])  
        # ones instead of zeros, because dz is in the denominator 
        # of the intrinsic alignment signal
        dz  = np.ones((lbin, len(zz_integr)), 'float64') 
        dz0  = np.ones((lbin, len(zz_integr)), 'float64') 
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose() 
        for index_l, index_z in index_pknn:
            dz[index_l, index_z] = dz_fr0_interp(k[index_l,index_z], zz_integr[index_z])
            dz0[index_l, index_z] = dz0_fr0_interp(k[index_l,index_z])
        return dz, dz0
    

    def get_growth(self, params_dic, zz_integr):
        aa_integr =  np.array(1./(1.+zz_integr[::-1]))
        background ={
            'Omega_m': params_dic['Omega_m'],
            'h' : params_dic['h'],
            'w0': -1.,
            'wa': 0.,
            'a_arr': np.hstack((aa_integr, 1.))
            }
        cosmo = mg.fR_HS(background)
        fr0    = 10**params_dic['log10f_R0']
        da, _ = cosmo.growth_parameters(self.kh_lin_short, fr0)  
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