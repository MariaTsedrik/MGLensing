import BCemu
import math
from scipy import interpolate as itp
from scipy.interpolate import interp1d, RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.special import erf
import numpy as np
import baccoemu as bacco
import cosmopower as cp
from cosmopower import cosmopower_NN
import math
import baccoemu 


class PowerSpectra:
    def __init__(self):
        self.zz_Pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5])
        #self.zz_Pk = np.linspace(0., zmax, num=32)
        self.aa_Pk = np.array(1./(1.+self.zz_Pk[::-1])) ##should be increasing
        self.k_min_h_by_Mpc = 0.001
        self.k_max_h_by_Mpc = 50.0 #limit of bacco's linear emulator
        self.nz_Pk = len(self.zz_Pk)
        self.kLbin = 512
        self.kL = np.logspace(-4, np.log10(self.k_max_h_by_Mpc), self.kLbin)
        #self.zz_max = self.zz_integr[-1]
        #print(self.zz_max)

    #def get_pk_interp(self, params_dic):
    #    raise NotImplementedError("Subclasses must implement this method")
        
    #def get_barboost_interp(self, params_dic):
    #    raise NotImplementedError("Subclasses must implement this method")

    #def get_heft_interp(self, params_dic):
    #    raise NotImplementedError("Subclasses must implement this method")


class HMcode2020(PowerSpectra):
    def __init__(self):
        super().__init__() 
        print('initialising hmcode')
        self.cp_nl_model = cosmopower_NN(restore=True, 
                      restore_filename='emulators/log10_total_matter_nonlinear_emu',
                      )
        self.kh_nl = self.cp_nl_model.modes #0.01..50.
        k_min_l = 3.7e-4
        k_max_l = 50.
        k_min_nl = 0.01
        k_max_nl = 50.
        
        #self.cp_l_model = cosmopower_NN(restore=True, 
        #              restore_filename='emulators/log10_total_matter_linear_emu',
        #              )
        #self.kh_l = self.cp_l_model.modes #3.7e-4..50.
        
        #self.cp_barboost_model = cosmopower_NN(restore=True, 
        #              restore_filename='emulators/total_matter_bar_boost_emu',
        #              )
        #self.kh_bb = self.cp_barboost_model.modes #0.01..50.


    def get_pk_interp(self, params_dic):
        ns   = params_dic['ns']
        As   = params_dic['As'] if 'As' in params_dic else np.exp(params_dic['log10As'])*1e-10
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        Omm = params_dic['Omega_m']
        Omb = params_dic['Omega_b']
        Omnu = params_dic['Omega_nu']
        Mnu  = params_dic['Mnu']
        Omc = Omm-Omb-Omnu
        params_hmcode = {
                'ns'            :  ns*np.ones(self.nz_Pk),
                'As'            :  As*np.ones(self.nz_Pk),
                'hubble'        :  h*np.ones(self.nz_Pk),
                'omega_baryon'  :  Omb*np.ones(self.nz_Pk),
                'omega_cdm'     :  Omc*np.ones(self.nz_Pk),
                'neutrino_mass' :  Mnu*np.ones(self.nz_Pk),
                'w0'            :  w0*np.ones(self.nz_Pk),
                'wa'            :  wa*np.ones(self.nz_Pk),
                'z'             :  self.zz_Pk
            }
        PNL  = self.cp_nl_model.ten_to_predictions_np(params_hmcode)
        PNL_interp = RectBivariateSpline(self.zz_Pk,
                                    self.kh_nl,
                                    PNL,
                                    kx=1, ky=1)
        return  PNL_interp    
    
    def get_barboost_interp(self, params_dic):
        raise NotImplementedError("Subclasses must implement this method")

    def get_barboost(self, params_dic, k, lbin, zz_integr):
        raise NotImplementedError("Subclasses must implement this method")


    def get_pk(self, params_dic, k, lbin, zz_integr):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > self.k_min_h_by_Mpc) & (k<self.k_max_h_by_Mpc))).transpose()
        Pk_l_interp = self.get_pk_interp(params_dic)
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = Pk_l_interp(zz_integr[index_z], k[index_l,index_z])
        return pk_m_l

class BaccoEmu(PowerSpectra):
    def __init__(self):
        super().__init__() 
        print('initialising baccoemu')
        self.baccoemulator = baccoemu.Matter_powerspectrum()
        self.heftemulator = baccoemu.Lbias_expansion()
        k_min_l = 1.e-4
        k_max_l = 50.
        z_min = 0.
        z_max_l = 3.
        k_min_nl = 0.01001
        k_max_nl = 4.903235148249275
        z_max_nl = 1.5
        k_min_heft = 0.01001
        k_max_heft = 0.71
        self.kh_nl = np.logspace(np.log10(k_min_nl), np.log10(k_max_nl), num=256)
        self.kh_heft = np.logspace(np.log10(k_min_heft), np.log10(k_max_heft), num=128)
        self.kh_l = np.logspace(np.log10(k_min_l), np.log10(k_max_l), num=256)
        self.z_nl_bacco = np.linspace(z_min, z_max_nl, 28)
        self.z_l_bacco = np.linspace(z_min, z_max_l, 28)

    def get_pk_interp(self, params_dic):
        ns   = params_dic['ns']
        As   = params_dic['As'] if 'As' in params_dic else np.exp(params_dic['log10As'])*1e-10
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        Omm = params_dic['Omega_m']
        Omb = params_dic['Omega_b']
        Omnu = params_dic['Omega_nu']
        Mnu  = params_dic['Mnu']
        Omc = Omm-Omb-Omnu
        aa = 1./(1.+self.z_nl_bacco)[::-1]
        params_bacco = {
                'ns'            :  ns,
                'A_s'           :  As,
                'hubble'        :  h,
                'omega_baryon'  :  Omb,
                'omega_cold'    :  Omc+Omb,
                'neutrino_mass' :  Mnu,
                'w0'            :  w0,
                'wa'            :  wa,
                'expfactor'     :  aa
            }
        _, PNL  = self.baccoemulator.get_nonlinear_pk(k=self.kh_nl, cold=False, **params_bacco)
        PNL_interp = RectBivariateSpline(self.z_nl_bacco, 
                                    self.kh_nl,
                                    PNL[::-1, :],
                                    kx=1, ky=1)
        return  PNL_interp 
    

    def get_pk(self, params_dic, k, lbin, zz_integr):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > self.k_min_h_by_Mpc) & (k<self.k_max_h_by_Mpc))).transpose()
        Pk_l_interp = self.get_pk_interp(params_dic)
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = Pk_l_interp(min(zz_integr[index_z], 1.5), k[index_l,index_z])
        return pk_m_l
    
    def get_barboost_interp(self, params_dic):      
        ns   = params_dic['ns']
        As   = params_dic['As'] if 'As' in params_dic else np.exp(params_dic['log10As'])*1e-10
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        Omm = params_dic['Omega_m']
        Omb = params_dic['Omega_b']
        Omnu = params_dic['Omega_nu']
        Mnu  = params_dic['Mnu']
        Omc = Omm-Omb-Omnu
        aa = 1./(1.+self.z_nl_bacco)[::-1]

        Mc = params_dic['log10Mc_bc']
        eta = params_dic['eta_bc']
        beta = params_dic['beta_bc']
        Mcen = params_dic['log10Mz0_bc']
        theout = params_dic['thetaout_bc']
        thetinn = params_dic['thetainn_bc']
        Minn = params_dic['log10Minn_bc']
        params_bacco = {
                'ns'            :  ns,
                'A_s'           :  As,
                'hubble'        :  h,
                'omega_baryon'  :  Omb,
                'omega_cold'    :  Omc+Omb,
                'neutrino_mass' :  Mnu,
                'w0'            :  w0,
                'wa'            :  wa,
                'expfactor'     :  aa,

                'M_c'           : Mc,
                'eta'           : eta,
                'beta'          : beta,
                'M1_z0_cen'     : Mcen,
                'theta_out'     : theout,
                'theta_inn'     : thetinn,
                'M_inn'         : Minn
            } 
        _, Boost = self.baccoemulator.get_baryonic_boost(k=self.kh_nl, cold=False, **params_bacco)
        Boost = Boost[::-1, :]
        Boost_itp = [interp1d(self.kh_nl, Boost[i], bounds_error=False,
                    kind='cubic',
                    fill_value=(Boost[i][0], Boost[i][-1])) for i in range(len(self.z_nl_bacco))]
        Boost_k = np.array([Boost_itp[z](self.kL) for z in range(len(self.z_nl_bacco))], dtype=np.float64)
        BFC_interp = RectBivariateSpline(self.z_nl_bacco, self.kL, Boost_k)
        return  BFC_interp   
    
    def get_barboost(self, params_dic, k, lbin, zz_integr):
        boost_bar = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > self.k_min_h_by_Mpc) & (k<self.k_max_h_by_Mpc))).transpose()
        boost_bar_interp = self.get_barboost_interp(params_dic)
        for index_l, index_z in index_pknn:
                boost_bar[index_l, index_z] = boost_bar_interp(min(zz_integr[index_z], 1.5), k[index_l,index_z])
            
        return boost_bar
    
    def get_heft_interp(self, params_dic):
        ns   = params_dic['ns']
        As   = params_dic['As'] if 'As' in params_dic else np.exp(params_dic['log10As'])*1e-10
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        Omm = params_dic['Omega_m']
        Omb = params_dic['Omega_b']
        Omnu = params_dic['Omega_nu']
        Mnu  = params_dic['Mnu']
        Omc = Omm-Omb-Omnu
        aa = 1./(1.+self.z_nl_bacco)[::-1]
        params_bacco = {
                'ns'            :  ns,
                'A_s'           :  As,
                'hubble'        :  h,
                'omega_baryon'  :  Omb,
                'omega_cold'    :  Omc+Omb,
                'neutrino_mass' :  Mnu,
                'w0'            :  w0,
                'wa'            :  wa,
                'expfactor'     :  aa
            }
        
        _, pnn = self.heftemulator.get_nonlinear_pnn(k=self.kh_heft, **params_bacco)
        pnn_interp = [RectBivariateSpline(self.z_nl_bacco, 
                                    self.kh_heft,
                                    pnn_i[::-1, :],
                                    kx=1, ky=1) for pnn_i in pnn]
        return pnn_interp 
    
    def get_heft(self, params_dic, k, lbin, zz_integr):
        pk_nn_l  = np.zeros((15, lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > self.k_min_h_by_Mpc) & (k<self.k_max_h_by_Mpc))).transpose()
        pnn_l_interp = self.get_heft_interp(params_dic)
        for index_i in np.arange(15): 
            for index_l, index_z in index_pknn:
                pk_nn_l[index_i, index_l, index_z] = pnn_l_interp[index_i](min(zz_integr[index_z], 1.5), k[index_l,index_z])
        return pk_nn_l


class BCemulator(PowerSpectra):
    def __init__(self, zz_max):
        super().__init__() 
        print('initialising bcemu')
        self.bfcemu = BCemu.BCM_7param(verbose=False)
        ###for baryonic feedback###
        BCemu_k_bins = 200
        BCemu_z_bins = 20


        kmin_bcemu = 0.0342
        kmax_bcemu = 12.51
        self.k_bfc = np.logspace(np.log10(kmin_bcemu), np.log10(kmax_bcemu), BCemu_k_bins)
        self.z_bfc = np.linspace(0., min(2, zz_max), BCemu_z_bins)



    def get_barboost_interp(self, params_dic):
        log10Mc = params_dic['log10Mc_bcemu']
        thej = params_dic['the_bcemuj']
        mu = params_dic['mu_bcemu']
        gamma_bc = params_dic['gamma_bcemu']
        delta = params_dic['delta_bcemu']
        eta = params_dic['eta_bcemu']
        deta = params_dic['deta_bcemu']
        bcemu_dict ={
            'log10Mc' : log10Mc,
            'mu'     : mu,
            'thej'   : thej,  
            'gamma'  : gamma_bc,
            'delta'  : delta, 
            'eta'    : eta, 
            'deta'   : deta, 
                }

        fb = params_dic['fb']
        ###last element extrapolation###
        Boost = [self.bfcemu.get_boost(z_i,bcemu_dict, self.k_bfc,fb) for z_i in self.z_bfc]
        Boost_itp = [interp1d(self.k_bfc, Boost[i], bounds_error=False,
                    kind='cubic',
                    fill_value=(Boost[i][0], Boost[i][-1])) for i in range(self.BCemu_z_bins)]
        Boost_k = np.array([Boost_itp[z](self.kL) for z in range(self.BCemu_z_bins)], dtype=np.float64)
        BFC_interpolator = RectBivariateSpline(self.z_bfc, self.kL, Boost_k)
        return  BFC_interpolator   


    def get_barboost(self, params_dic, k, lbin, zz_integr):
        boost_bar = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > self.k_min_h_by_Mpc) & (k<self.k_max_h_by_Mpc))).transpose()
        boost_bar_interp = self.get_barboost_interp(params_dic)
        for index_l, index_z in index_pknn:
                boost_bar[index_l, index_z] = boost_bar_interp(min(zz_integr[index_z], 2.), k[index_l,index_z])
        return boost_bar
