import BCemu
from scipy.interpolate import RectBivariateSpline
import numpy as np
from cosmopower import cosmopower_NN
import baccoemu 
import os
from math import log10, log

dirname = os.path.split(__file__)[0]

# extrapolation ranges
# limits of bacco's linear emulator
k_min_h_by_mpc = 0.001
k_max_h_by_mpc = 50.0 
                

class HMcode2020():
    """
    HMcode2020 class for handling power spectra and baryonic boost factor calculations using cosmopower emulators.

    Attributes
    ----------
    cp_nl_model : cosmopower_NN
        An instance of the cosmopower_NN class for non-linear power spectrum emulation.
    kh_nl : numpy.ndarray
        Array of non-linear wavenumbers in units of h/Mpc.
    cp_l_model : cosmopower_NN
        An instance of the cosmopower_NN class for linear power spectrum emulation.
    kh_l : numpy.ndarray
        Array of linear wavenumbers in units of h/Mpc.
    kh_l_left : numpy.ndarray
        Array of linear wavenumbers less than the minimum non-linear wavenumber.
    kh : numpy.ndarray
        Concatenated array of linear and non-linear wavenumbers.
    cp_barboost_model : cosmopower_NN
        An instance of the cosmopower_NN class for baryonic boost factor emulation.
    kh_bb : numpy.ndarray
        Array of wavenumbers for baryonic boost factor emulation in units of h/Mpc.
    kbin_left_bb : int
        Number of linear wavenumber bins less than the minimum baryonic boost wavenumber.
    zz_pk : numpy.ndarray
        Array of redshifts for power spectrum emulation.
    aa_pk : numpy.ndarray
        Array of scale factors corresponding to zz_pk.
    nz_pk : int
        Number of redshift bins for power spectrum emulation.
    zz_pk_max : float
        Maximum redshift for power spectrum emulation.

    Methods
    -------
    __init__():
        Initializes the HMcode2020 class, setting up the cosmopower emulated models and relevant parameters.
    get_pk_interp(params_dic):
        Interpolates the non-linear power spectrum based on the provided cosmological parameters.
    get_barboost_interp(params_dic):
        Interpolates the baryonic boost factor based on the provided cosmological parameters.
    get_barboost(params_dic, k, lbin, zz_integr):
        Computes the baryonic boost factor for given wavenumbers and redshifts.
    get_pk(params_dic, k, lbin, zz_integr):
        Computes the power spectrum for given wavenumbers and redshifts.
    """
    def __init__(self):
        print('initialising hmcode')
        self.zz_pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5, 3.0]) # these numers are hand-picked
        self.aa_pk = np.array(1./(1.+self.zz_pk[::-1])) # should be increasing
        self.nz_pk = len(self.zz_pk)
        self.zz_max = self.zz_pk[-1]
        self.cp_nl_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/log10_total_matter_nonlinear_emu',
                      )
        self.kh_nl = self.cp_nl_model.modes # 0.01..50. h/Mpc    
        self.cp_l_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/log10_total_matter_linear_emu',
                      )
        self.kh_l = self.cp_l_model.modes # 3.7e-4..50. h/Mpc 
        self.kh_l_left = self.kh_l[self.kh_l<self.kh_nl[0]]
        self.kh = np.concatenate((self.kh_l_left, self.kh_nl))
        self.cp_barboost_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/total_matter_bar_boost_emu',
                      )
        self.kh_bb = self.cp_barboost_model.modes # 0.01..50. h/Mpc 
        kbin_left_bb = len(self.kh_l[self.kh_l<self.kh_bb[0]])
        self.kh_barboost = np.concatenate((self.kh_l[self.kh_l<self.kh_bb[0]], self.kh_bb))
        self.boost_left = np.ones((self.nz_pk, kbin_left_bb))
        self.cp_sigma8_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/sigma8_emu',
                      )
        self.emu_name = 'HMcode2020'



    def get_pk_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        m_nu  = params_dic['Mnu']
        params_hmcode = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'hubble'        :  np.full(self.nz_pk, h),
                'omega_baryon'  :  np.full(self.nz_pk, omega_b),
                'omega_cdm'     :  np.full(self.nz_pk, omega_c),
                'neutrino_mass' :  np.full(self.nz_pk, m_nu),
                'w0'            :  np.full(self.nz_pk, w0),
                'wa'            :  np.full(self.nz_pk, wa),
                'z'             :  self.zz_pk
            }
        pnl_cp  = self.cp_nl_model.ten_to_predictions_np(params_hmcode)
        plin_cp = self.cp_l_model.ten_to_predictions_np(params_hmcode)
        plin_left = plin_cp[:, self.kh_l<self.kh_nl[0]]
        pnl  = np.concatenate((plin_left, pnl_cp),axis=1)
        pnl_interp = RectBivariateSpline(self.zz_pk,
                                    self.kh,
                                    pnl,
                                    kx=1, ky=1)
        return  pnl_interp    
    
    def get_barboost_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        m_nu  = params_dic['Mnu']
        logt_agn = params_dic['log10T_AGN']
        params_hmcode_bar = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'hubble'        :  np.full(self.nz_pk, h),
                'omega_baryon'  :  np.full(self.nz_pk, omega_b),
                'omega_cdm'     :  np.full(self.nz_pk, omega_c),
                'neutrino_mass' :  np.full(self.nz_pk, m_nu),
                'w0'            :  np.full(self.nz_pk, w0),
                'wa'            :  np.full(self.nz_pk, wa),
                'log10TAGN'     :  np.full(self.nz_pk, logt_agn),
                'z'             :  self.zz_pk
            }
        boost = self.cp_barboost_model.predictions_np(params_hmcode_bar) 
        boost_k  = np.concatenate((self.boost_left, boost),axis=1)
        bhm_interp = RectBivariateSpline(self.zz_pk,
                                    self.kh_barboost,
                                    boost_k,
                                    kx=1, ky=1)
        return  bhm_interp 


    def get_barboost(self, params_dic, k, lbin, zz_integr):
        boost_bar = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        boost_bar_interp = self.get_barboost_interp(params_dic)
        for index_l, index_z in index_pknn:
                boost_bar[index_l, index_z] = boost_bar_interp(zz_integr[index_z], k[index_l,index_z])
            
        return boost_bar


    def get_pk(self, params_dic, k, lbin, zz_integr):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k<k_max_h_by_mpc))).transpose()
        pk_l_interp = self.get_pk_interp(params_dic)
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(zz_integr[index_z], k[index_l,index_z])
        return pk_m_l
    
    def get_sigma8(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_b = params_dic['Omega_b']
        omega_c = params_dic['Omega_c']
        m_nu  = params_dic['Mnu']
        len_chain = len(ns)
        params_hmcode = {
                'ns'            :  ns,
                'As'            :  a_s,
                'hubble'        :  h,
                'omega_baryon'  :  omega_b,
                'omega_cdm'     :  omega_c,
                'neutrino_mass' :  m_nu,
                'w0'            :  w0,
                'wa'            :  wa,
                'z'             :  np.zeros(len_chain)
            }
        sigma8_emu = self.cp_sigma8_model.predictions_np(params_hmcode)
        sigma8 = sigma8_emu[:, 0]
        return sigma8


class BaccoEmu():
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
    kh_heft : numpy.ndarray
        Array of wavenumbers for HEFT emulation in units of h/Mpc.
    kh_l : numpy.ndarray
        Array of linear wavenumbers in units of h/Mpc.
    z_nl_bacco : numpy.ndarray
        Array of redshifts for non-linear BACCO emulation.
    z_l_bacco : numpy.ndarray
        Array of redshifts for linear BACCO emulation.
    kh_l_left : numpy.ndarray
        Array of linear wavenumbers less than the minimum non-linear wavenumber.
    kh : numpy.ndarray
        Concatenated array of linear and non-linear wavenumbers.
    z_l_high : numpy.ndarray
        Array of redshifts higher than the maximum non-linear redshift.
    aa_l_high : numpy.ndarray
        Array of scale factors corresponding to z_l_high.
    aa_nl : numpy.ndarray
        Array of scale factors corresponding to z_nl_bacco.
    aa_all : numpy.ndarray
        Concatenated array of all scale factors.
    zz_all_bacco : numpy.ndarray
        Concatenated array of all redshifts for BACCO emulation.
    zz_pk_max : float
        Maximum redshift for power spectrum emulation.
    
    Methods
    -------
    __init__():
        Initializes the BaccoEmu class with default parameters and sets up the BACCO emulators.
    get_pk_interp(params_dic):
        Interpolates the non-linear power spectrum and extrapolates using the linear BACCO emulator for z>1.5.
    get_pk(params_dic, k, lbin, zz_integr):
        Computes the power spectrum for given parameters, wavenumbers, and redshifts.
    get_barboost_interp(params_dic):
        Interpolates the baryonic boost factor using the BACCO emulator.
    get_barboost(params_dic, k, lbin, zz_integr):
        Computes the baryonic boost factor for given parameters, wavenumbers, and redshifts.
        Extrapolation for k<0.01 h/Mpc and k>5 h/Mpc is a constrant first and last values of the baryonic boost, respectively.
        Extrapolation for z>1.5 is the baryonic boost at z=1.5.
    get_heft_interp(params_dic):
        Interpolates the non-linear power spectrum in the Hybrid Effective Field Theory (HEFT) 
        approach using the BACCO emulator. Extrapolates using the linear BACCO emulator for z>1.5 and k<0.01 h/Mpc.
    get_heft(params_dic, k, lbin, zz_integr):
        Computes the non-linear power spectrum for halo expansion for given parameters, wavenumbers, and redshifts.
    """
    def __init__(self):
        print('initialising baccoemu')
        self.baccoemulator = baccoemu.Matter_powerspectrum()
        self.heftemulator = baccoemu.Lbias_expansion()
        k_min_l = 1.e-4
        k_max_l = 50.
        z_min = 0.
        z_max_l = 3. #add a comment about global z_max
        k_min_nl = 0.01001
        k_max_nl = 4.903235148249275
        k_max_boost = 4.692772528625323
        z_max_nl = 1.5
        k_min_heft = 0.01001
        k_max_heft = 0.71
        self.kh_nl = np.logspace(log10(k_min_nl), log10(k_max_nl), num=256)
        self.kh_boost = np.logspace(log10(k_min_nl), log10(k_max_boost), num=256)
        self.kh_heft = np.logspace(log10(k_min_heft), log10(k_max_heft), num=128)
        self.kh_l = np.logspace(log10(k_min_l), log10(k_max_l), num=256)
        self.z_nl_bacco = np.linspace(z_min, z_max_nl, 12)
        self.z_l_bacco = np.linspace(z_min, z_max_l, 16)
        self.kh_l_left = self.kh_l[self.kh_l<self.kh_nl[0]]
        self.boost_left = np.ones((len(self.z_nl_bacco), len(self.kh_l_left)))
        self.kh = np.concatenate((self.kh_l_left, self.kh_nl))
        self.kh_boost_tot = np.concatenate((self.kh_l_left, self.kh_boost))
        self.z_l_high = self.z_l_bacco[self.z_l_bacco>self.z_nl_bacco[-1]]
        self.aa_l_high = 1./(1.+self.z_l_high)[::-1]
        self.aa_nl = 1./(1.+self.z_nl_bacco)[::-1]
        self.aa_all = np.concatenate((self.aa_l_high, self.aa_nl))
        self.zz_all_bacco = np.concatenate((self.z_nl_bacco, self.z_l_high))
        self.zz_max = z_max_l
        self.emu_name = 'BACCO'

    def get_pk_interp(self, params_dic):
        ns   = params_dic['ns']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_cb = params_dic['Omega_cb']
        omega_b = params_dic['Omega_b']
        m_nu  = params_dic['Mnu']
        params_bacco = {
                'ns'            :  ns,
                'hubble'        :  h,
                'omega_baryon'  :  omega_b,
                'omega_cold'    :  omega_cb,
                'neutrino_mass' :  m_nu,
                'w0'            :  w0,
                'wa'            :  wa,
                'expfactor'     :  self.aa_nl
            }
        if 'As' in params_dic:
            params_bacco['A_s']   = params_dic['As'] 
        elif 'sigma8' in params_dic:
            params_bacco['sigma8_cold']   = params_dic['sigma8']
        _, pnl_bacco  = self.baccoemulator.get_nonlinear_pk(k=self.kh_nl, cold=False, **params_bacco)
        # to-do: add power-law extrapolation for 5<k[h/Mpc]<50
        params_bacco_lin = params_bacco
        params_bacco_lin['expfactor'] = self.aa_all
        _, plin_bacco = self.baccoemulator.get_linear_pk(k=self.kh, cold=False, **params_bacco_lin)
        plin_left = plin_bacco[:, self.kh<self.kh_nl[0]]
        plin_left = plin_left[self.aa_all>=self.aa_nl[0], :]
        pnl  = np.concatenate((plin_left, pnl_bacco),axis=1)
        pl_high = plin_bacco[self.aa_all<self.aa_nl[0], :]
        pnl  = np.concatenate((pl_high, pnl),axis=0)
        pnl_interp = RectBivariateSpline(self.zz_all_bacco, 
                                    self.kh,
                                    pnl[::-1, :],
                                    kx=1, ky=1)
        return  pnl_interp 

    def get_pk(self, params_dic, k, lbin, zz_integr):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        pk_l_interp = self.get_pk_interp(params_dic)
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(zz_integr[index_z], k[index_l,index_z])
        return pk_m_l
    
    def get_barboost_interp(self, params_dic):      
        ns   = params_dic['ns']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_cb = params_dic['Omega_cb']
        omega_b = params_dic['Omega_b']
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
        if 'As' in params_dic:
            params_bacco['A_s']   = params_dic['As'] 
        elif 'sigma8' in params_dic:
            params_bacco['sigma8_cold']   = params_dic['sigma8']
        _, boost = self.baccoemulator.get_baryonic_boost(k=self.kh_boost, cold=False, **params_bacco)
        # to-do: add power-law extrapolation for 4.7<k[h/Mpc]<50
        boost = boost[::-1, :]
        boost_k  = np.concatenate((self.boost_left, boost),axis=1)
        bfc_interpolator = RectBivariateSpline(self.z_nl_bacco, 
                                               self.kh_boost_tot,
                                               boost_k,
                                               kx=1, ky=1)
        return  bfc_interpolator   
    
    def get_barboost(self, params_dic, k, lbin, zz_integr):
        boost_bar = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        boost_bar_interp = self.get_barboost_interp(params_dic)
        for index_l, index_z in index_pknn:
                boost_bar[index_l, index_z] = boost_bar_interp(min(zz_integr[index_z], 1.5), k[index_l,index_z])
            
        return boost_bar
    
    def get_heft_interp(self, params_dic):
        ns   = params_dic['ns']
        h    = params_dic['h']
        w0    = params_dic['w0']
        wa    = params_dic['wa']
        omega_cb = params_dic['Omega_cb']
        omega_b = params_dic['Omega_b']
        m_nu  = params_dic['Mnu']
        params_bacco = {
                'ns'            :  ns,
                'hubble'        :  h,
                'omega_baryon'  :  omega_b,
                'omega_cold'    :  omega_cb,
                'neutrino_mass' :  m_nu,
                'w0'            :  w0,
                'wa'            :  wa,
                'expfactor'     :  self.aa_nl
            }
        if 'As' in params_dic:
            params_bacco['A_s']   = params_dic['As'] #add option that we sample in S8 bzw sigma8
        elif 'sigma8' in params_dic:
            params_bacco['sigma8_cold']   = params_dic['sigma8']
        _, pnn = self.heftemulator.get_nonlinear_pnn(k=self.kh_heft, **params_bacco)
        params_bacco_l = params_bacco
        params_bacco_l['expfactor'] = self.aa_all
        _, plin_bacco = self.baccoemulator.get_linear_pk(k=self.kh, cold=False, **params_bacco_l)
        plin_interp = RectBivariateSpline(self.zz_all_bacco,
                                    self.kh,
                                    plin_bacco[::-1, :],
                                    kx=1, ky=1)
        pnn_interp = [RectBivariateSpline(self.z_nl_bacco, 
                                    self.kh_heft,
                                    pnn_i[::-1, :],
                                    kx=1, ky=1) for pnn_i in pnn]
        return pnn_interp, plin_interp 

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
        self.kh_l = np.logspace(-4, log10(k_max_h_by_mpc), k_bin)
        kmin_bcemu = 0.0342
        kmax_bcemu = 12.51
        zz_max = 3.
        self.k_bfc = np.logspace(log10(kmin_bcemu), log10(kmax_bcemu), bcemu_k_bins)
        self.z_bfc = np.linspace(0., min(2, zz_max), bcemu_z_bins)
        self.zz_max = zz_max
        kbin_left_bb = len(self.kh_l[self.kh_l<self.k_bfc[0]])
        self.kh_barboost = np.concatenate((self.kh_l[self.kh_l<self.k_bfc[0]], self.k_bfc))
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
        # to-do: add power-law extrapolation for 12.51<k[h/Mpc]<50
        boost_k  = np.concatenate((self.boost_left, boost),axis=1)
        bfc_interpolator = RectBivariateSpline(self.z_bfc, 
                                               self.kh_barboost,
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

class BoostReACT():
    def __init__(self):
        self.zz_pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5, 3.0]) # these numers are hand-picked
        self.aa_pk = np.array(1./(1.+self.zz_pk[::-1])) # should be increasing
        self.nz_pk = len(self.zz_pk)
        self.zz_max = self.zz_pk[-1]
        self.cp_nl_hmcode_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/log10_total_matter_nonlinear_emu',
                      )
        self.kh_nl = self.cp_nl_model.modes # 0.01..50. h/Mpc    
        self.cp_l_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/log10_total_matter_linear_emu',
                      )
        self.kh_l = self.cp_l_model.modes # 3.7e-4..50. h/Mpc 
        self.kh_l_left = self.kh_l[self.kh_l<self.kh_nl[0]]
        self.kh = np.concatenate((self.kh_l_left, self.kh_nl))
        
    def get_pk_react(self, params_dic, k, lbin, zz_integr, mg_boost_l_interp):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        pk_l_interp = self.get_pk_hmcode_interp(params_dic)
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(zz_integr[index_z], k[index_l,index_z])*mg_boost_l_interp(min(zz_integr[index_z], 2.5), k[index_l,index_z])
        return pk_m_l  

    def get_pk_hmcode_interp(self, params_dic):
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
                'wa'            :  np.zeros(self.nz_pk, 0.),
                'z'             :  self.zz_pk
            }
        pnl_cp  = self.cp_nl_model.ten_to_predictions_np(params_hmcode)
        plin_cp = self.cp_l_model.ten_to_predictions_np(params_hmcode)
        plin_left = plin_cp[:, self.kh_l<self.kh_nl[0]]
        pnl  = np.concatenate((plin_left, pnl_cp),axis=1)
        pnl_interp = RectBivariateSpline(self.zz_pk,
                            self.kh,
                            pnl,
                            kx=1, ky=1)     
        return  pnl_interp

def boost_right_extrap(mg_boost, log_k, k_last, kh_nl_right_boost, zz_boost):
    boost_last_entry, boost_lastlast_entry = mg_boost[:, -1], mg_boost[:, -2]
    m = np.array([log(boost_last_entry[i] / boost_lastlast_entry[i]) / log_k for i in range(zz_boost)])
    mg_boost_right = boost_last_entry[:, np.newaxis] *(kh_nl_right_boost[np.newaxis, :]/k_last)**m[:, np.newaxis]
    return mg_boost_right    
class DGPReACT(BoostReACT):
    def __init__(self):
        super().__init__()
        print('initialising nDGP')
        self.cp_nl_ngdp_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/react_boost_nDGP_emu_v2',
                      )
        self.kh_nl_boost = self.cp_nl_ngdp_model.modes # 0.01..5. h/Mpc 
        self.zz_boost = np.minimum(self.zz_pk, 2.5)
        self.kh_l_left_boost = self.kh_l[self.kh_l<self.kh_nl_boost[0]]
        self.kh_nl_right_boost = self.kh_nl[self.kh_nl>self.kh_nl_boost[-1]]
        self.kh_nl_boost_tot = np.concatenate((self.kh_l_left_boost, self.kh_nl_boost, self.kh_nl_right_boost))
        self.k_nl_boost_last = self.kh_nl_boost[-1]
        self.k_nl_boost_lastlast = self.kh_nl_boost[-2]
        self.log_k = log(self.k_nl_boost_last / self.k_nl_boost_lastlast)
        self.emu_name = 'nDGP_ReACT'
        

    def get_mg_boost_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        omega_b = params_dic['Omega_b']
        omega_m = params_dic['Omega_m']
        omega_nu  = params_dic['Omega_nu']
        omegarc    = 10**params_dic['log10Omega_rc']
        params_react = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'H0'            :  np.full(self.nz_pk, h*100),
                'Omega_b'       :  np.full(self.nz_pk, omega_b),
                'Omega_m'       :  np.full(self.nz_pk, omega_m),
                'Omega_nu'      :  np.full(self.nz_pk, omega_nu),
                'omegarc'       :  np.full(self.nz_pk, omegarc),
                'z'             :  self.zz_boost
            }
        mg_boost = self.cp_nl_ngdp_model.predictions_np(params_react) 
        # constant extrapolation for k<0.01 h/Mpc
        mg_boost_left = np.full(len(self.kh_l_left_boost), mg_boost[0])
        # power law extrapolation for k>5 h/Mpc
        mg_boost_right = boost_right_extrap(mg_boost, self.log_k, self.k_nl_boost_last, self.kh_nl_right_boost, self.zz_boost)
        # combine mg_boost at all scales
        mg_boost_k = np.concatenate((mg_boost_left, mg_boost, mg_boost_right))
        # interpolate
        mg_boost_interp = RectBivariateSpline(self.zz_boost,
                    self.kh_nl_boost_tot,
                    mg_boost_k,
                    kx=1, ky=1)
        return  mg_boost_interp
    
    def get_pk(self, params_dic, k, lbin, zz_integr):
        mg_boost_l_interp = self.get_mg_boost_interp(params_dic)
        pk_m_l = self.get_pk_react(params_dic, k, lbin, zz_integr, mg_boost_l_interp)
        return pk_m_l
    

class GammaReACT(BoostReACT):
    def __init__(self):
        super().__init__()
        print('initialising Growth Index Parametrisation')
        # to-do: update with new version
        self.cp_nl_ngdp_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/react_boost_gLEMURSzzz', 
                      )
        self.kh_nl_boost = self.cp_nl_ngdp_model.modes # 0.01..5. h/Mpc 
        self.zz_boost = np.minimum(self.zz_pk, 2.5)
        self.kh_l_left_boost = self.kh_l[self.kh_l<self.kh_nl_boost[0]]
        self.kh_nl_right_boost = self.kh_nl[self.kh_nl>self.kh_nl_boost[-1]]
        self.kh_nl_boost_tot = np.concatenate((self.kh_l_left_boost, self.kh_nl_boost, self.kh_nl_right_boost))
        self.k_nl_boost_last = self.kh_nl_boost[-1]
        self.k_nl_boost_lastlast = self.kh_nl_boost[-2]
        self.log_k = log(self.k_nl_boost_last / self.k_nl_boost_lastlast)
        self.emu_name = 'Linder_z_ReACT'
        

    def get_mg_boost_interp(self, params_dic):
        ns   = params_dic['ns']
        a_s   = params_dic['As']
        h    = params_dic['h']
        omega_b = params_dic['Omega_b']
        omega_m = params_dic['Omega_m']
        omega_nu  = params_dic['Omega_nu']
        gamma0    = params_dic['gamma0']
        gamma1    = params_dic['gamma1']
        q1    = params_dic['q1']
        params_react = {
                'ns'            :  np.full(self.nz_pk, ns),
                'As'            :  np.full(self.nz_pk, a_s),
                'H0'            :  np.full(self.nz_pk, h*100),
                'Omega_b'       :  np.full(self.nz_pk, omega_b),
                'Omega_m'       :  np.full(self.nz_pk, omega_m),
                'Omega_nu'      :  np.full(self.nz_pk, omega_nu),
                'gamma0'        :  np.full(self.nz_pk, gamma0),
                'gamma1'        :  np.full(self.nz_pk, gamma1),
                'q1'            :  np.full(self.nz_pk, q1),
                'z'             :  self.zz_boost
            }
        mg_boost = self.cp_nl_ngdp_model.predictions_np(params_react) 
        # constant extrapolation for k<0.01 h/Mpc
        mg_boost_left = np.full(len(self.kh_l_left_boost), mg_boost[0])
        # power law extrapolation for k>5 h/Mpc
        mg_boost_right = boost_right_extrap(mg_boost, self.log_k, self.k_nl_boost_last, self.kh_nl_right_boost, self.zz_boost)
        # combine mg_boost at all scales
        mg_boost_k = np.concatenate((mg_boost_left, mg_boost, mg_boost_right))
        # interpolate
        mg_boost_interp = RectBivariateSpline(self.zz_boost,
                    self.kh_nl_boost_tot,
                    mg_boost_k,
                    kx=1, ky=1)
        return  mg_boost_interp
    
    def get_pk(self, params_dic, k, lbin, zz_integr):
        mg_boost_l_interp = self.get_mg_boost_interp(params_dic)
        pk_m_l = self.get_pk_react(params_dic, k, lbin, zz_integr, mg_boost_l_interp)
        return pk_m_l
    

class MuSigmaReACT():
    def __init__(self):
        print('initialising mu-Sigma')
        self.zz_pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5, 3.0]) # these numers are hand-picked
        self.aa_pk = np.array(1./(1.+self.zz_pk[::-1])) # should be increasing
        self.nz_pk = len(self.zz_pk)
        self.zz_max = self.zz_pk[-1]
        # to-do: update with new version
        self.cp_nl_musigma_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/MuSigma_nonlinearboost_extAs', 
                      )
        self.kh_nl_boost = self.cp_nl_musigma_model.modes # 0.01..10. h/Mpc
        self.cp_lin_musigma_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/MuSigma_linear_log10ps_extAs', 
                      )
        self.kh_l = self.cp_lin_musigma_model.modes # 1.e-4..50. h/Mpc
        self.zz_boost = np.minimum(self.zz_pk, 2.5)
        self.kh_l_left_boost = self.kh_l[self.kh_l<self.kh_nl_boost[0]]
        self.kh_nl_right_boost = self.kh_l[self.kh_l>self.kh_nl_boost[-1]]
        self.kh_nl_boost_tot = np.concatenate((self.kh_l_left_boost, self.kh_nl_boost, self.kh_nl_right_boost))
        self.k_nl_boost_last = self.kh_nl_boost[-1]
        self.k_nl_boost_lastlast = self.kh_nl_boost[-2]
        self.log_k = log(self.k_nl_boost_last / self.k_nl_boost_lastlast)
        self.emu_name = 'mu_Sigma_ReACT'
        

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
        # constant extrapolation for k<0.01 h/Mpc
        mg_boost_left = np.ones(len(self.kh_l_left_boost))
        # power law extrapolation for k>10 h/Mpc
        mg_boost_right = boost_right_extrap(mg_boost, self.log_k, self.k_nl_boost_last, self.kh_nl_right_boost, self.zz_boost)
        # combine mg_boost at all scales
        mg_boost_k = np.concatenate((mg_boost_left, mg_boost, mg_boost_right))
        # interpolate
        mg_boost_interp = RectBivariateSpline(self.zz_boost,
                    self.kh_nl_boost_tot,
                    mg_boost_k,
                    kx=1, ky=1)
        mg_plin_interp = RectBivariateSpline(self.zz_boost,
                    self.kh_l,
                    mg_pk_lin,
                    kx=1, ky=1)
        return  mg_boost_interp, mg_plin_interp
    
    def get_pk(self, params_dic, k, lbin, zz_integr):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        mg_boost_l_interp, pk_l_interp = self.get_pk_nl_interp(params_dic)
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(min(zz_integr[index_z], 2.5), k[index_l,index_z])*mg_boost_l_interp(min(zz_integr[index_z], 2.5), k[index_l,index_z])
        return pk_m_l  
    

class DarkScatteringReACT(BoostReACT):
    def __init__(self):
        super().__init__()
        print('initialising Dark Scattering')
        self.cp_nl_ds_model = cosmopower_NN(restore=True, 
                      restore_filename=dirname+'/../emulators/DS_nonlinear_cp_NN_S8', 
                      )
        self.kh_nl_boost = self.cp_nl_ds_model.modes # 1e-3..10. h/Mpc
        self.zz_boost = np.minimum(self.zz_pk, 2.5)
        self.kh_nl_right_boost = self.kh_l[self.kh_l>self.kh_nl_boost[-1]]
        self.kh_nl_boost_tot = np.concatenate((self.kh_nl_boost, self.kh_nl_right_boost))
        self.k_nl_boost_last = self.kh_nl_boost[-1]
        self.k_nl_boost_lastlast = self.kh_nl_boost[-2]
        self.log_k = log(self.k_nl_boost_last / self.k_nl_boost_lastlast)
        self.emu_name = 'Dark_Scattering_ReACT'
        

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
                'n_s'            :  np.full(self.nz_pk, ns),
                'S_8'            :  np.full(self.nz_pk, s8),
                'h'            :  np.full(self.nz_pk, h),
                'omega_b'       :  np.full(self.nz_pk, ombh2),
                'omega_cdm'     :  np.full(self.nz_pk, omch2),
                'm_nu'          :  np.full(self.nz_pk, m_nu),
                'w'             :  np.full(self.nz_pk, w0),
                'A'             :  np.full(self.nz_pk, ads),
                'z'             :  self.zz_boost
            }
        mg_pk_nl = self.cp_nl_ds_model.ten_to_predictions_np(params_react)
        # power law extrapolation for k>10 h/Mpc
        mg_pk_nl_right = boost_right_extrap(mg_pk_nl, self.log_k, self.k_nl_boost_last, self.kh_nl_right_boost, self.zz_boost)
        # combine mg_boost at all scales
        mg_pnl_k = np.concatenate((mg_pk_nl, mg_pk_nl_right))
        # interpolate
        mg_pnl_interp = RectBivariateSpline(self.zz_boost,
                    self.kh_nl_boost_tot,
                    mg_pnl_k,
                    kx=1, ky=1)
        return  mg_pnl_interp
    
    def get_pk(self, params_dic, k, lbin, zz_integr):
        pk_m_l  = np.zeros((lbin, len(zz_integr)), 'float64')
        index_pknn = np.array(np.where((k > k_min_h_by_mpc) & (k < k_max_h_by_mpc))).transpose()
        pk_l_interp = self.get_pk_nl_interp(params_dic)
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = pk_l_interp(min(zz_integr[index_z], 2.5), k[index_l,index_z])
        return pk_m_l  