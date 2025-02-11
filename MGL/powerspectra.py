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

    def get_pk_interp(self, params_dic, NL_model):
        raise NotImplementedError("Subclasses must implement this method")
        

    def get_barboost_interp(self, params_dic, baryon_model):
        raise NotImplementedError("Subclasses must implement this method")




class HMcode2020(PowerSpectra):
    def __init__(self):
        super().__init__() 
        print('initialising hmcode')
        self.cp_nl_model = cosmopower_NN(restore=True, 
                      restore_filename='emulators/log10_total_matter_nonlinear_emu',
                      )
        self.kh_nl = self.cp_nl_model.modes #0.01..50.

        
        #self.cp_l_model = cosmopower_NN(restore=True, 
        #              restore_filename='emulators/log10_total_matter_linear_emu',
        #              )
        #self.kh_l = self.cp_l_model.modes #3.7e-4..50.
        

    #def get_linear_pk(self, params_dic):
    #    return 'ahoi'

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
        return 'ahoi'
    
class BCemulator(PowerSpectra):
    def __init__(self, zz_max):
        #super().__init__() 
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



