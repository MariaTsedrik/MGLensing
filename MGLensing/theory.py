import numpy as np
import MGrowth as mg
from scipy.integrate import trapezoid, quad
from scipy import interpolate as itp
from .powerspectra import HMcode2020, BCemulator, BaccoEmu, DGPReACT, GammaReACT, MuSigmaReACT, DarkScatteringReACT
from math import sqrt, log, exp, pow, log10
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
# Suppress TensorFlow warnings


H0_h_c = 1./2997.92458 
# =100/c in Mpc/h Hubble constant conversion factor
C_IA = 0.0134 
# =dimensionsless, C x rho_crit 
a_arr_for_mu = np.logspace(-5., 1., 512)

NL_MODEL_HMCODE = 0
NL_MODEL_BACCO = 1
NL_MODEL_NDGP = 2
NL_MODEL_GAMMAZ = 3
NL_MODEL_MUSIGMA = 4
NL_MODEL_DS = 5


NO_BARYONS = 0
BARYONS_HMCODE = 1
BARYONS_BCEMU = 2
BARYONS_BACCO = 3

IA_NLA = 0
IA_TATT = 1

BIAS_LIN = 0
BIAS_B1B2 = 1
BIAS_HEFT = 2


EmulatorRanges = {
    'HMcode':
        {
        'Omega_c':      {'p1': 0.1,         'p2': 0.8},    
        'Omega_b':      {'p1':0.01,         'p2': 0.1},  
        'h':            {'p1': 0.4,         'p2': 1.},      
        'As':           {'p1': 0.495e-9,    'p2': 5.459e-9},
        'ns':           {'p1': 0.6,         'p2': 1.2}, 
        'Mnu':          {'p1': 0.0,         'p2': 0.5},
        'w0':           {'p1': -3.,         'p2': -0.3},    
        'wa':           {'p1': -3.,         'p2': 3.},  
        },
    'bacco':
        {
        'Omega_cb':     {'p1': 0.23,         'p2': 0.4},    
        'Omega_b':      {'p1':0.04,          'p2': 0.06},  
        'h':            {'p1': 0.6,          'p2': 0.8},      
        'sigma8_cb':    {'p1': 0.73,         'p2': 0.9},
        'ns':           {'p1': 0.92,         'p2': 1.01}, 
        'Mnu':          {'p1': 0.0,          'p2': 0.4},
        'w0':           {'p1': -1.15,        'p2': -0.85},    
        'wa':           {'p1': -0.3,         'p2': 0.3}, 
        },    
    'bacco_lin':
        {
        'Omega_cb':     {'p1': 0.06,         'p2': 0.7},    
        'Omega_b':      {'p1':0.03,          'p2': 0.07},  
        'h':            {'p1': 0.5,          'p2': 0.9},      
        'ns':           {'p1': 0.6,         'p2': 1.2}, 
        'Mnu':          {'p1': 0.0,          'p2': 1.},
        'w0':           {'p1': -2.,        'p2': -0.5},    
        'wa':           {'p1': -0.5,         'p2': 0.5}, 
        },        
    'bcemu':
        {
        'log10Mc_bcemu':    {'p1': 11.4,         'p2': 14.6}, 
        'thej_bcemu':       {'p1': 2.6,          'p2': 7.4},
        'mu_bcemu':         {'p1': 0.2,          'p2': 1.8},
        'gamma_bcemu':      {'p1': 1.3,          'p2': 3.7},
        'delta_bcemu':      {'p1': 3.8,          'p2': 10.2},
        'eta_bcemu':        {'p1': 0.085,        'p2': 0.365},
        'deta_bcemu':       {'p1': 0.085,        'p2': 0.365},
        },
    'hm_bar':
        {    
        'log10Tagn':    {'p1': 7.6,         'p2': 8.3}
        },
    'bacco_bfc':
        {    
        'log10Mc_bc':   {'p1': 9.,           'p2': 15.}, 
        'eta_bc':       {'p1': -0.69,        'p2': 0.69},
        'beta_bc':      {'p1': -1.,          'p2': 0.69},
        'log10Mz0_bc':  {'p1': 9.,           'p2': 13.},
        'thetaout_bc':  {'p1': 0.,           'p2': 0.47},
        'thetainn_bc':  {'p1': -2.,          'p2': -0.523},
        'log10Minn_bc': {'p1': 9.,           'p2': 13.5}
        }

}


cosmo_names = ["Omega_m", "Omega_b", "Omega_c", "Omega_cb", "Omega_nu", "Ombh2", "Omnuh2", "fb", "h", "Mnu", "ns", "w0", "wa"]
def check_zmax(zmax, emu_obj):
    emu_z_max = emu_obj.zz_max
    if zmax > emu_z_max:
        raise ValueError(f'Survey z_max must not exceed zz_max in the power spectrum computation for {emu_obj.emu_name}!')
        

class Theory:
    def __init__(self, SurveyClass):
        self.Survey = SurveyClass
        # load classes with emulators
        # to-do: load only the ones actually used in the code
        self.HMcode2020Emulator = HMcode2020()
        check_zmax(self.Survey.zmax, self.HMcode2020Emulator)
        self.BCemulator = BCemulator()
        check_zmax(self.Survey.zmax, self.BCemulator)
        self.BaccoEmulator = BaccoEmu()
        check_zmax(self.Survey.zmax, self.BaccoEmulator)
        self.DGPEmulator = DGPReACT()
        check_zmax(self.Survey.zmax, self.DGPEmulator)
        self.GammazEmulator = GammaReACT()
        check_zmax(self.Survey.zmax, self.GammazEmulator)
        self.MuSigmaEmulator = MuSigmaReACT()
        check_zmax(self.Survey.zmax, self.MuSigmaEmulator)
        self.DSEmulator = DarkScatteringReACT()
        check_zmax(self.Survey.zmax, self.DSEmulator)

    def check_consistency(self, params): 
        if 'h' not in params.keys():
            raise KeyError("h not found in dictionary")
        conditions = [
        'Omega_m',
        ('Omega_c', 'Omch2'),
        ('Omega_b', 'Ombh2'),
        ('Omega_nu', 'Mnu', 'Omnuh2')]
        all_present = all(any(key in params for key in condition) if isinstance(condition, tuple) else condition in params for condition in conditions)
        if all_present:
            raise ValueError("Invalid parameter set: too many densities are present in the dictionary.")
        conditions_amp = [
        'log10As',
        'As',
        'sigma8',
        'S8']
        amplitude_params = [param for param in conditions_amp if param in params]
        if len(amplitude_params) > 1:
            raise ValueError(f"Invalid parameter set: multiple amplitude parameters present: {amplitude_params}")
        elif len(amplitude_params) == 0:  
            raise ValueError(f"Invalid parameter set: no amplitude parameters present")
        params_new = self.apply_relations(params)
        still_unspecified = [param for param in cosmo_names if param not in params_new]
        if len(still_unspecified) > 1:
            raise ValueError(f"Invalid parameter set: missing parameters: {still_unspecified}")
        return params_new
        
    
    def apply_relations(self, params): 
        params['TCMB'] = params.get('TCMB', 2.7255)
        params['nnu'] = params.get( 'nnu', 3.044)
        if 'Omega_b' in params.keys():
            params['Ombh2'] = params['Omega_b']*params['h']*params['h']
        elif 'Ombh2' in params.keys():
            params['Omega_b'] = params['Ombh2']/params['h']/params['h']
        if 'Omch2' in params.keys():
            params['Omega_c'] = params['Omch2']/params['h']/params['h']
        elif 'Omega_c' in params.keys():
            params['Omch2'] = params['Omega_c']*params['h']*params['h']
        if 'Omega_nu' in params.keys():
            params['Omnuh2'] = params['Omega_nu']*params['h']*params['h']   
            params['Mnu'] = params['Omnuh2'] / ((params['nnu'] / 3.0) ** 0.75 / 94.06410581217612 * (params['TCMB']/2.7255)**3)    
        elif 'Omnuh2' in params.keys():
            params['Omega_nu'] = params['Omnuh2'] / params['h']/params['h'] 
            params['Mnu'] = params['Omnuh2'] / ((params['nnu'] / 3.0) ** 0.75 / 94.06410581217612 * (params['TCMB']/2.7255)**3)
        elif 'Mnu' in params.keys():
            params['Omnuh2'] = params['Mnu'] * ((params['nnu'] / 3.0) ** 0.75 / 94.06410581217612 * (params['TCMB']/2.7255)**3)
            params['Omega_nu'] = params['Omnuh2'] / params['h']/params['h']
        if 'Omega_m' not in params.keys():
            params['Omega_m'] = params['Omega_b'] + params['Omega_c'] + params['Omega_nu']    
        params['Omega_cb'] = params['Omega_m'] - params['Omega_nu']
        params['Omega_c'] = params['Omega_cb'] - params['Omega_b']
        params['fb'] = params['Omega_b']/params['Omega_m']
        if 'As' in params.keys():
            params['log10As'] = log(1e10*params['As'])
        elif 'log10As' in params.keys():
            params['As'] = exp(params['log10As'])*1e-10
        if 'S8' in params.keys():
            params['sigma8'] = params['S8']/sqrt(params['Omega_m']/0.3)   
        return params

    def get_emu_status(self, params, which_emu, flag_once=False):
        """
        Checks the relevant boundaries of emulators.

        Parameters:
        ----------
        params (dict): a set of coordinates in parameter space
        which_emu (str) : kind of emulator: options are 'HMcode', 'bacco', 'bacco_lin',
                                            'baryons'
        flag_once (boolean): set to false within a chain, set to true for a single data computation

        Returns:
        -------
        boolean: status of the ranges check

        Raises:
        ------
        KeyError: If the parameter is outside its allowed ranges: what is this parameter and which are the correct ranges.
        """
        # parameters strictly needed to evaluate the emulator
        eva_pars = EmulatorRanges[which_emu].keys()
        pp = [params[p] for p in eva_pars]
        if flag_once:
            for i, par in enumerate(eva_pars):
                val = pp[i]
                message = 'Parameter {}={} out of bounds [{}, {}]'.format(
                par, val, EmulatorRanges[which_emu][par]['p1'],
                EmulatorRanges[which_emu][par]['p2'])
                assert (np.all(val >= EmulatorRanges[which_emu][par]['p1'])
                    & np.all(val <= EmulatorRanges[which_emu][par]['p2'])
                    ), message
        else:    
            if not all(EmulatorRanges[which_emu][par_i]['p1'] <= params[par_i] <= EmulatorRanges[which_emu][par_i]['p2'] for par_i in eva_pars):
                return False
        return True
        
        
    def get_a_s(self, params, flag_once=False):
        if 'As' in params:
            return True, params['As']
        else:
            # to-do: add from_sigma8_to_sigma8_cb
            # make distinction between sigma8_cold and sigma8
            params['sigma8_cb'] = params['sigma8'] 
            status = self.get_emu_status(params, 'bacco_lin', flag_once)
            if not status:
                return False, np.nan
            else:
                params_bacco = {
                'ns'            :  params['ns'],
                'sigma8_cold'   :  params['sigma8_cb'], 
                'hubble'        :  params['h'],
                'omega_baryon'  :  params['Omega_b'],
                'omega_cold'    :  params['Omega_cb'], 
                'neutrino_mass' :  params['Mnu'],
                'w0'            :  params['w0'],
                'wa'            :  params['wa'],
                'expfactor'     :  1    
                }
                a_s = self.BaccoEmulator.baccoemulator.get_A_s(**params_bacco)
                return True, a_s
    
    def get_sigma8_cb(self, params, flag_once=False):
        if 'sigma8' in params:
            # to-do: add from_sigma8_to_sigma8_cb
            # make distinction between sigma8_cold and sigma8
            params['sigma8_cb'] = params['sigma8']     
            return True, params['sigma8_cb'] 
        else: 
            status = self.get_emu_status(params, 'bacco_lin', flag_once)
            if not status:
                return False, np.nan
            else:
                params_bacco = {
                'ns'            :  params['ns'],
                'A_s'           :  params['As'], 
                'hubble'        :  params['h'],
                'omega_baryon'  :  params['Omega_b'],
                'omega_cold'    :  params['Omega_cb'], 
                'neutrino_mass' :  params['Mnu'],
                'w0'            :  params['w0'],
                'wa'            :  params['wa'],
                'expfactor'     :  1    
                }
                sigma8_cb = self.BaccoEmulator.baccoemulator.get_sigma8(cold=True, **params_bacco)
                return True, sigma8_cb
    
    
    def check_ranges(self, params, model, flag_once=False):
        """
        Check the parameter ranges in emulators for the model.

        Parameters:
        -----------
        params (dict): Dictionary containing the parameters to be checked.                  
        model (dict): Dictionary containing the model-specific parameters.
        flag_once (boolean): set to false within a chain, set to true for a single data computation 

        Returns:
        -------
        boolean: status of the parameter range check.
        """
        if  model['nl_model']==NL_MODEL_HMCODE:
            status_, params['As'] = self.get_a_s(params, flag_once)
            if not status_:
                return status_
            status_nl = self.get_emu_status(params, 'HMcode', flag_once)
        elif  model['nl_model']==NL_MODEL_BACCO:       
            status_, params['sigma8_cb'] = self.get_sigma8_cb(params)
            if not status_:
                return status_
            status_nl = self.get_emu_status(params, 'bacco', flag_once)
        if  model['baryon_model']==BARYONS_BCEMU:
            status_b = self.get_emu_status(params, 'bcemu', flag_once)
        elif  model['baryon_model']==BARYONS_HMCODE:
            status_b = self.get_emu_status(params, 'hm_bar', flag_once)
        elif  model['baryon_model']==BARYONS_BACCO:
            status_b = self.get_emu_status(params, 'bacco_bfc', flag_once)
        status = status_nl and status_b
        # check the w0-wa condition
        if params['w0'] + params['wa'] >= 0:
            status = False
        # check the mu-Sigma condition
        if params['mu0'] > (2.*params['sigma0']+1.):
            status = False    
        # check the DS condition
        if params['w0']!=-1.:
            if params['Ads']/(1. + params['w0']) < 0:
                status = False       
        return  status 


    def check_pars_ini(self, param_dic, model_dic, flag_once=True):
        """
        Initial check and validate the parameters for the model and data.

        Parameters:
        ----------
        param_dic (dict): Dictionary containing the parameters to be checked.
        model_dic (dict): Dictionary containing the model-specific parameters.

        Returns:
        --------
        tuple: A tuple containing the updated parameter dictionary and the status
               of the parameter range check.

        Raises:
        -------        
        KeyError: If the required parameter 'h' is not found in the parameter dictionary.
        KeyError: If some cosmological parameters are missing from the parameter dictionary.
        """
        param_dic_all = self.check_consistency(param_dic)
        status = self.check_ranges(param_dic_all, model_dic, flag_once)
        return param_dic_all, status
    
    def check_pars(self, param_dic, model_dic):
        """
        Check and validate the parameters for the model within a chain.

        Parameters:
        ----------
        param_dic (dict): Dictionary containing the parameters to be checked.
        model_dic (dict): Dictionary containing the model-specific parameters.

        Returns:
        --------
        tuple: A tuple containing the updated parameter dictionary and the status
               of the parameter range check.
        """
        param_dic_all = self.apply_relations(param_dic)
        status = self.check_ranges(param_dic_all, model_dic, False)
        return param_dic_all, status

    def get_ez_rz_k(self, params_dic, zz):
        """
        Calculate the E(z), r_com(z), and k(z)=(ell+1/2)/r_com(z) grids based on cosmological parameters.

        Parameters:
        -----------
        params_dic : dict
            Dictionary containing cosmological parameters:
            - 'Omega_m': Matter density parameter.
            - 'w0': Equation of state parameter w0.
            - 'wa': Equation of state parameter wa.
        zz : array-like
            Array of redshift values.

        Returns:
        --------
        e_z_grid : numpy.ndarray
            Array of E(z) values corresponding to the input redshift values.
        r_z_grid : numpy.ndarray
            Array of r_com(z) values corresponding to the input redshift values, in units of Mpc/h.
        k_grid : numpy.ndarray
            Array of k(z) values corresponding to the input redshift values, in units of h/Mpc.
        """
        omega_m = params_dic['Omega_m']
        w0 = params_dic['w0']
        wa = params_dic['wa']
        omega_lambda_func = lambda z: (1.-omega_m) * pow(1.+z, 3.*(1.+w0+wa)) * np.exp(-3.*wa*z/(1.+z))
        e_z_func = lambda z: np.sqrt(omega_m*pow(1.+z, 3) + omega_lambda_func(z))
        r_z_int = lambda z: 1./e_z_func(z)
        r_z_func = lambda z_in: quad(r_z_int, 0, z_in)[0]
        r_z_grid = np.array([r_z_func(zz_i) for zz_i in zz])/H0_h_c 
        e_z_grid = np.array([e_z_func(zz_i) for zz_i in zz])
        k_grid =(self.Survey.ell[:,None]+0.5)/r_z_grid
        return e_z_grid, r_z_grid, k_grid
    

    
    def get_growth(self, params_dic, zz_integr,  nl_model=0):
        """
        Calculate the growth factor for different cosmological models.

        Parameters:
        -----------
        params_dic : dict
            Dictionary containing cosmological parameters. Expected keys are:
            'Omega_m', 'h', 'w0', 'wa', and depending on the model, 'log10Omega_rc', 'gamma0', 'gamma1'.
        zz_integr : array-like
            Array of redshift values for integration.
        nl_model : int, optional
            Integer specifying the modified gravity model to use. Default is 0.
            - 0 or 1: w0waCDM model
            - 2: nDGP model
            - 3: Linder_gamma_a model

        Returns:
        --------
        dz : numpy.ndarray
            Normalized growth factor array corresponding to the input redshift values.
        dz0 : float
            Growth factor at redshift zero.
            
        Raises:
        -------
        ValueError
            If an invalid mg_model option is provided.
        """
        aa_integr =  np.array(1./(1.+zz_integr[::-1]))
        background ={
            'Omega_m': params_dic['Omega_m'],
            'h' : params_dic['h'],
            'w0': params_dic['w0'],
            'wa': params_dic['wa'],
            'a_arr': np.hstack((aa_integr, 1.))
            }
        if nl_model==NL_MODEL_HMCODE or nl_model==NL_MODEL_BACCO:
            cosmo = mg.w0waCDM(background)   
            da, _ = cosmo.growth_parameters() 
        elif nl_model==NL_MODEL_NDGP:
            cosmo = mg.nDGP(background)
            log10omega_rc = params_dic['log10Omega_rc'] 
            da, _ = cosmo.growth_parameters(10**log10omega_rc)  
        elif nl_model==NL_MODEL_GAMMAZ:
            cosmo = mg.Linder_gamma_a(background)
            gamma0 = params_dic['gamma0'] 
            gamma1 = params_dic['gamma1'] 
            da, _ = cosmo.growth_parameters(gamma=gamma0, gamma1=gamma1)
        elif nl_model==NL_MODEL_MUSIGMA:
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
        elif nl_model==NL_MODEL_DS:
            cosmo = mg.IDE(background)
            xi = params_dic['Ads']/(1.+params_dic['w0'])
            da, _ = cosmo.growth_parameters(xi=xi)             
        else:
            raise ValueError("Invalid mg_model option.")    
        dz = da[::-1] 
        # growth factor should be normalised to z=0
        dz0 = dz[0]
        dz = dz[1:]/dz0
        return dz, dz0
    
    def get_ia_kernel(self, omega_m, params_dic, ez, dz, eta_z_s, zz_integr, ia_model=0):
        """
        Calculate the intrinsic alignment (IA) kernel in units of h/Mpc. The amplitude is given by 

        .. math::
            A(z) = - a_{\rm IA} C_{\rm IA} \frac{\Omega_{\rm m}}{D(z)} (1+z)^{\eta_{\rm IA}} L(z)^{\beta_{\rm IA}}
        with :math:`C_{\rm IA} = \bar{C}_{\rm IA} \rho_{\rm crit}` and 
        :math:`\bar{C}_{\rm IA}=5 \times 10^{-14} M^{-1}_\odot h^{-2} \rm{Mpc}^3`, and luminosity funsiton L.
        The kernel is given by

        .. math::
            W_i^{\rm IA}(z) = A(z) \frac{n_i(z)}{c/H(z)}
        with :math:`n_i(z)` being the source distribution in bin-i.

        Parameters:
        -----------
        omega_m : float
            Matter density parameter.
        params_dic : dict
            Dictionary containing IA model parameters:
            - 'aIA': Amplitude of the intrinsic alignment.
            - 'etaIA': Redshift evolution parameter for IA.
            - 'betaIA': Luminosity-function dependence parameter for IA.
        ez : array_like
            Redshift-dependent Hubble parameter values.
        dz : array_like
            Growth factor values.
        eta_z_s : array_like
            Redshift-dependent source galaxy distribution.
        zz_integr : array_like
            Redshift integration grid.
        ia_model : int, optional
            Intrinsic alignment model to use (default is 0).
            - 0: Standard IA model.
            - 1: TATT.

        Returns:
        --------
        w_ia : array_like
            Intrinsic alignment kernel.
        """
        if ia_model==IA_NLA:
            w_ia_p = eta_z_s * ez[:,None] * H0_h_c
            a_ia = params_dic['aIA'] 
            eta_ia = params_dic['etaIA'] 
            beta_ia = params_dic['betaIA']
            f_ia = (1.+zz_integr)**eta_ia * (self.Survey.lum_func(zz_integr))**beta_ia
            dz = dz[None,:] 
            # growth factor can be scale-dependent for f(R)
            w_ia = - a_ia*C_IA*omega_m*f_ia[None,:,None]/dz[:,:,None] * w_ia_p[None,:,:]
        else:
            raise NotImplementedError('TATT not implememnted yet')
        return w_ia
    
    def get_wl_kernel(self, omega_m, params_dic, ez, rz, dz, ia_model=0):
        """
        Calculate the weak lensing kernel, in units of units of h/Mpc: 
        .. math::
            W_i^{L} = W_i^{\gamma} + W_i^{\rm IA}
        where
        .. math::
            W_i^{\gamma} = \frac{3}{2} \left( \frac{H_0}{c} \right)^2 \Omega_{\rm m} (1+z) r_{\rm com}(z) \bar{W}_i(z)
        with
        .. math:: 
            \bar{W}_i(z) = \int \mathrm{d}z' n_i(z')\left[ 1-\frac{r_{\rm com}(z)}{r_{\rm com}(z')} \right]   

        Parameters:
        -----------
        omega_m : float
            Matter density parameter.
        params_dic : dict
            Dictionary of parameters.
        ez : array-like
            E(z) function values.
        rz : array-like
            Comoving distance values.
        dz : array-like
            Growth factor values.
        ia_model : int, optional
            Intrinsic alignment model (default is 0).

        Returns:
        --------
        w_l : array-like
            Weak lensing kernel.
        """
        # to-do: change to a function with varying photo-z errors
        eta_z_s =  self.Survey.eta_z_s 
        # in the integrand dimensions are (bin_i, zz_integr, zz_integr)
        integrand = 3./2.*H0_h_c**2. * omega_m * rz[None,:,None]*(1.+self.Survey.zz_integr[None,:,None])*eta_z_s.T[:,None,:]*(1.-rz[None,:,None]/rz[None,None,:])
        # integrate along the third dimension in zz_integr
        w_gamma  = trapezoid(np.triu(integrand), self.Survey.zz_integr,axis=-1).T
        # add an extra dimension to w_gamma as we might have ell-dependence in the IA-kernel due to the scale-dependent linear growth
        # sum the lensing and intrinsic alignment kernels together
        w_l = w_gamma[None,:,:] + self.get_ia_kernel(omega_m, params_dic, ez, dz, eta_z_s, self.Survey.zz_integr, ia_model)
        return w_l
    
    def get_cell_shear(self, params_dic, ez, rz, dz, pk, ia_model=0):
        """
        Calculate the weak lensing power spectrum (C_ell):
        .. math::
            C^{\rm LL}_{ij}(\ell) = \frac{c}{H_0} \int \mathrm{d}z \frac{W^{\rm L}_i(z)W^{\rm L}_j(z)}{E(z) r^2_{\rm com}(z)} P_{\rm mm}(k(\ell, z), z)
        and weak lensing kernel.
            
        Parameters:
        -----------
        params_dic : dict
            Dictionary containing cosmological parameters, including 'Omega_m'.
        ez : array_like
            Array of E(z) values, where E(z) is the dimensionless Hubble parameter.
        rz : array_like
            Array of comoving radial distances.
        dz : array_like
            Array of growth factors.
        pk : array_like
            Array of matter power spectrum values.
        ia_model : int, optional
            Intrinsic alignment model (default is 0).

        Returns:
        --------
        cl_ll : array_like
            Weak lensing power spectrum (C_ell) for different redshift bins.
        w_l : array_like
            Weak lensing kernel.
        """
        omega_m = params_dic['Omega_m']
        # compute weak lensing kernel
        w_l = self.get_wl_kernel(omega_m, params_dic, ez, rz, dz, ia_model)
        # compute the integrand with dimensions (ell, z_integr, bin_i, bin_j)
        cl_ll_int = w_l[:,:,:,None] * w_l[:,:,None,:] * pk[:,:,None,None] / ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] / H0_h_c
        # integrate along the z_integr-direction
        cl_ll = trapezoid(cl_ll_int, self.Survey.zz_integr, axis=1)[:self.Survey.nell_wl, :, :]
        # add noise to the auto-correlated bins
        for i in range(self.Survey.nbin):
            cl_ll[:, i, i] += self.Survey.noise['LL']
        return cl_ll, w_l
    
    def get_bpgg(self, params_dic, k, pgg, pgg_extr, nbin, bias_model):
        """
        Calculate the galaxy-galaxy power spectrum given the parameters and bias model,
        in units of (Mpc/h)^3.

        Parameters:
        -----------
        params_dic : dict
            Dictionary containing the bias parameters.
        k : numpy.ndarray
            Array of wavenumbers.
        pgm : numpy.ndarray
            Array of power spectrum values.
        pgm_extr : numpy.ndarray
            Array of extrapolated power spectrum values.
        nbin : int
            Number of bins.
        bias_model : str
            The bias model to use. Options are 'BIAS_LIN', 'BIAS_B1B2', 'BIAS_HEFT'.

        Returns:
        --------
        bpgm : numpy.ndarray
            The matter-galaxy power spectrum.
        
        Raises:
        -------
        ValueError
            If an invalid bias_model option is provided.
        """
        if bias_model == BIAS_LIN:
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(nbin)])
            bpgg = bias1[None, None, :, None] * bias1[None, None, None, :] * pgg[:,:,None, None]
        elif bias_model == BIAS_B1B2:  
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(nbin)])
            bias2 = np.array([params_dic['b2_'+str(i+1)] for i in range(nbin)])  
            bpgg = ( bias1[None, None,:, None] + bias2[None, None, :, None] * k[:, :, None, None]**2 )*( bias1[None, None, None, :] + bias2[None, None, None, :] * k[:, :, None, None]**2 )* pgg[:,:,None,None]
        elif bias_model == BIAS_HEFT:
            bL1 = np.array([params_dic['b1L_'+str(i+1)] for i in range(nbin)])
            bL2 = np.array([params_dic['b2L_'+str(i+1)] for i in range(nbin)])
            bs2 = np.array([params_dic['bs2L_'+str(i+1)] for i in range(nbin)])
            blapl = np.array([params_dic['blaplL_'+str(i+1)] for i in range(nbin)])
            p_dmdm = pgg[0, :, :]        
            p_dmd1 = pgg[1, :, :]    
            p_dmd2 = pgg[2, :, :]        
            p_dms2 = pgg[3, :, :]        
            p_dmk2 = pgg[4, :, :]      
            p_d1d1 = pgg[5, :, :]     
            p_d1d2 = pgg[6, :, :]        
            p_d1s2 = pgg[7, :, :]         
            p_d1k2 = pgg[8, :, :]         
            p_d2d2 = pgg[9, :, :]        
            p_d2s2 = pgg[10, :, :]         
            p_d2k2 = pgg[11, :, :]
            p_s2s2 = pgg[12, :, :] 
            p_s2k2 = pgg[13, :, :] 
            p_k2k2 = pgg[14, :, :] 
            bpgg = (p_dmdm[:,:,None,None]  +
               (bL1[None,None,:,None]+bL1[None,None, None, :]) * p_dmd1[:,:,None,None] +
               (bL1[None,None, :,None]*bL1[None,None, None, :]) * p_d1d1[:,:,None,None] +
               (bL2[None,None, :,None] + bL2[None,None, None, :]) * p_dmd2[:,:,None,None] +
               (bs2[None,None, :,None] + bs2[None,None, None, :]) * p_dms2[:,:,None,None] +
               (bL1[None,None, :,None]*bL2[None,None, None, :] + bL1[None,None, None, :]*bL2[None,None, :,None]) * p_d1d2[:,:,None,None] +
               (bL1[None,None, :,None]*bs2[None,None, None, :] + bL1[None,None, None, :]*bs2[None,None, :,None]) * p_d1s2[:,:,None,None] +
               (bL2[None,None, :,None]*bL2[None,None, None, :]) * p_d2d2[:,:,None,None] +
               (bL2[None,None, :,None]*bs2[None,None, None, :] + bL2[None,None, None, :]*bs2[None,None, :,None]) * p_d2s2[:,:,None,None] +
               (bs2[None,None, :,None]*bs2[None,None, None, :])* p_s2s2[:,:,None,None] +
               (blapl[None,None, :,None] + blapl[None,None, None, :]) * p_dmk2[:,:,None,None] +
               (bL1[None,None, None, :] * blapl[None,None, :,None] + bL1[None,None, :,None] * blapl[None,None, None, :]) * p_d1k2[:,:,None,None] +
               (bL2[None,None, None, :] * blapl[None,None, :,None] + bL2[None,None, :,None] * blapl[None,None, None, :]) * p_d2k2[:,:,None,None] +
               (bs2[None,None, None, :] * blapl[None,None, :,None] + bs2[None,None, :,None] * blapl[None,None, None, :]) * p_s2k2[:,:,None,None] +
               (blapl[None,None, :,None] * blapl[None,None, None, :]) * p_k2k2[:,:,None,None])
            bpgg_extr = (1.+bL1[None,None, :,None])*(1.+bL1[None,None, None, :]) * pgg_extr[:,:,None,None]
            bpgg += bpgg_extr 
        else:
            raise ValueError("Invalid bias_model option.")
        return bpgg

    def get_gg_kernel(self, ez):
        """
        Calculate the galaxy-galaxy lensing kernel, , in units of units of h/Mpc: 
        .. math::
            W_i^{G} = n_i(z)\frac{H(z)}{c}\, ,
        where n_i(z) is the lense distribution.    
        

        Parameters:
        ez : numpy.ndarray
            Array of E(z) values, where E(z) is the dimensionless Hubble parameter.

        Returns:
        numpy.ndarray
            The galaxy-galaxy lensing kernel with shape (1, zbin_integr, nbin).
        """
        # to-do: change to a function with varying photo-z errors
        eta_z_l = self.Survey.eta_z_l  
        w_g = np.zeros((self.Survey.zbin_integr, self.Survey.nbin), 'float64')
        w_g = ez[:, None] * H0_h_c * eta_z_l
        # add an extra dimension, now (ell, z_integr, bin_i)
        w_g = w_g[None, :, :]
        return w_g
    
    def get_cell_galclust(self, params_dic, ez, rz, k, pgg, pgg_extr, bias_model=0):
        """
        Compute the galaxy clustering angular power spectrum:
        .. math::
            C^{\rm GG}_{ij}(\ell) = c \int \mathrm{d}z \frac{W^{\rm G}_i(z) W^{\rm G}_j(z)}{H(z) r^2_{\rm com}(z)} P_{\rm gg}(k(\ell, z), z)

        Parameters:
        -----------
        params_dic : dict
            Dictionary containing the parameters for the computation.
        ez : array_like
            Array of E(z) values.
        rz : array_like
            Array of comoving radial distances.
        k : array_like
            Array of wavenumbers.
        pgg : array_like
            Galaxy-galaxy power spectrum.
        pgg_extr : array_like
            Extrapolated galaxy-galaxy power spectrum.
        bias_model : int, optional
            Bias model to use (default is 0).

        Returns:
        --------
        cl_gg : array_like
            Galaxy clustering angular power spectrum.
        w_g : array_like
            Photo galaxy clustering kernel.
        """
        # compute photo galaxy clustering kernel
        w_g = self.get_gg_kernel(ez)
        # compute galaxy-galaxy power spectrum
        bpgg = self.get_bpgg(params_dic, k, pgg, pgg_extr, self.Survey.nbin, bias_model)
        # compute integrand with the dimensions of (ell, z_integr, bin_i, bin_j)
        cl_gg_int = w_g[:,:,:,None] * w_g[:,: , None, :] * bpgg / ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] / H0_h_c    
        # integrate along the z_integr direction
        cl_gg = trapezoid(cl_gg_int, self.Survey.zz_integr, axis=1)[:self.Survey.nell_gc, :, :]
        # add noise
        for i in range(self.Survey.nbin):
            cl_gg[:, i, i] += self.Survey.noise['GG']
        return cl_gg, w_g    
    
    def get_bpgm(self,params_dic, k, pgm, pgm_extr, nbin, bias_model):
        """
        Calculate the matter-galaxy power spectrum given the parameters and bias model,
        in units of (Mpc/h)^3.

        Parameters:
        -----------
        params_dic : dict
            Dictionary containing the bias parameters.
        k : numpy.ndarray
            Array of wavenumbers.
        pgm : numpy.ndarray
            Array of power spectrum values.
        pgm_extr : numpy.ndarray
            Array of extrapolated power spectrum values.
        nbin : int
            Number of bins.
        bias_model : str
            The bias model to use. Options are 'BIAS_LIN', 'BIAS_B1B2', 'BIAS_HEFT'.

        Returns:
        --------
        bpgm : numpy.ndarray
            The matter-galaxy power spectrum.
        
        Raises:
        -------
        ValueError
            If an invalid bias_model option is provided.
        """
        if bias_model == BIAS_LIN:
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(nbin)])
            bpgm = bias1[None,None, :] * pgm[:,:,None]
        elif bias_model == BIAS_B1B2:
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(nbin)])
            bias2 = np.array([params_dic['b2_'+str(i+1)] for i in range(nbin)])  
            bpgm = ( bias1[None, None,:] + bias2[None, None, :] * k[:, :, None]**2 )*pgm[:,:,None]
        elif bias_model == BIAS_HEFT:
            p_dmdm = pgm[0, :, :]        
            p_dmd1 = pgm[1, :, :]    
            p_dmd2 = pgm[2, :, :]        
            p_dms2 = pgm[3, :, :]        
            p_dmk2 = pgm[4, :, :]      
            bL1 = np.array([params_dic['b1L_'+str(i+1)] for i in range(nbin)])
            bL2 = np.array([params_dic['b2L_'+str(i+1)] for i in range(nbin)])
            bs2 = np.array([params_dic['bs2L_'+str(i+1)] for i in range(nbin)])
            blapl = np.array([params_dic['blaplL_'+str(i+1)] for i in range(nbin)])
            bpgm = (p_dmdm[:,:,None]  +
                bL1[None,None,:] * p_dmd1[:,:,None] +
                bL2[None,None,:] * p_dmd2[:,:,None] +
                bs2[None,None,:] * p_dms2[:,:,None] +
                blapl[None,None,:] * p_dmk2[:,:,None])   
            bpgm_extr = (1.+bL1[None,None,:]) * pgm_extr[:,:,None]
            bpgm += bpgm_extr
        else:
            raise ValueError("Invalid bias_model option.")         
        return bpgm

    def get_cell_cross(self, params_dic, ez, rz, k, pgm, pgm_extr, w_l, w_g, bias_model=0):
        """
        Compute the galaxy-galaxy lensing or cross-correlation angular power spectrum:
        .. math::
            C^{\rm LG}_{ij}(\ell) = c \int \mathrm{d}z \frac{W^{\rm L}_i(z) W^{\rm G}_j(z)}{H(z) r^2_{\rm com}(z)} P_{\rm gm}(k(\ell, z), z)
        and 
        .. math::
            C^{\rm GL}_{ij}(\ell) = c \int \mathrm{d}z \frac{W^{\rm G}_i(z) W^{\rm L}_j(z)}{H(z) r^2_{\rm com}(z)} P_{\rm gm}(k(\ell, z), z)
        

        Parameters:
        -----------
        params_dic : dict
            Dictionary containing the parameters for the computation.
        ez : array-like
            Array of E(z) values.
        rz : array-like
            Array of comoving distance values.
        k : array-like
            Array of wavenumber values.
        pgm : array-like
            Array of power spectrum values.
        pgm_extr : array-like
            Array of extrapolated power spectrum values.
        w_l : array-like
            Array of lensing window functions.
        w_g : array-like
            Array of galaxy clustering window functions.
        bias_model : int, optional
            Bias model to be used (default is 0).

        Returns:
        --------
        cl_lg : array-like
            Computed lensing-clustering cross power spectrum.
        cl_gl : array-like
            Transposed clustering-lensing cross power spectrum.
        """
        # compute power galaxy-galaxy spectrum 
        bpgm = self.get_bpgm(params_dic, k, pgm, pgm_extr, self.Survey.nbin, bias_model)
        # compute integrand with dimensions (ell, z_integr, bin_i, bin_j)
        cl_lg_int = w_l[:,:,:,None] * w_g[:, :, None, :] * bpgm[:,:,None,:] / ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] / H0_h_c
        # integrate along the z-integr direction
        cl_lg = trapezoid(cl_lg_int, self.Survey.zz_integr, axis=1)[:self.Survey.nell_xc,:,:]
        # transpose LG to get GL
        cl_gl = np.transpose(cl_lg, (0, 2, 1))  
        return cl_lg, cl_gl
    
    def get_pmm(self, params_dic, k, lbin, zz_integr, nl_model=0, baryon_model=0):
        """
        Calculate the matter-matter power spectrum (P_mm) with optional non-linear and baryonic corrections.

        Parameters:
        -----------
        params_dic : dict
            Dictionary containing the cosmological parameters.
        k : array-like
            Wavenumber array.
        lbin : array-like
            Bin edges for the lensing kernel.
        zz_integr : array-like
            Redshift integration array.
        nl_model : int, optional
            Non-linear model to use. Options are:
            - NL_MODEL_HMCODE: Use HMcode2020 Emulator.
            - NL_MODEL_BACCO: Use Bacco Emulator.
            Default is 0 (no non-linear model).
        baryon_model : int, optional
            Baryonic model to use. Options are:
            - NO_BARYONS: No baryonic corrections.
            - BARYONS_HMCODE: Use HMcode2020 Emulator for baryonic corrections.
            - BARYONS_BCEMU: Use BCemulator for baryonic corrections.
            - BARYONS_BACCO: Use Bacco Emulator for baryonic corrections.
            Default is 0 (no baryonic model).

        Returns:
        --------
        pk : array-like
            The matter power spectrum with the specified non-linear and baryonic corrections applied.

        Raises:
        -------
        ValueError
            If an invalid nl_model or baryon_model option is provided.
        """

        if nl_model == NL_MODEL_HMCODE:
            pk = self.HMcode2020Emulator.get_pk(params_dic, k, lbin, zz_integr)
        elif nl_model == NL_MODEL_BACCO:
            pk = self.BaccoEmulator.get_pk(params_dic, k, lbin, zz_integr)
        elif nl_model == NL_MODEL_NDGP:
            pk = self.DGPEmulator.get_pk(params_dic, k, lbin, zz_integr)
        elif nl_model == NL_MODEL_GAMMAZ:
            pk = self.GammazEmulator.get_pk(params_dic, k, lbin, zz_integr)
        elif nl_model == NL_MODEL_MUSIGMA:
            pk = self.MuSigmaEmulator.get_pk(params_dic, k, lbin, zz_integr)
        elif nl_model == NL_MODEL_DS:
            pk = self.DSEmulator.get_pk(params_dic, k, lbin, zz_integr)      
        else:
            raise ValueError("Invalid nl_model option.")
        # add baryonic boost
        if baryon_model != NO_BARYONS:
            if baryon_model == BARYONS_HMCODE:
                boost_bar = self.HMcode2020Emulator.get_barboost(params_dic, k, lbin, zz_integr)
            elif baryon_model == BARYONS_BCEMU:
                boost_bar = self.BCemulator.get_barboost(params_dic, k, lbin, zz_integr)
            elif baryon_model == BARYONS_BACCO:
                boost_bar = self.BaccoEmulator.get_barboost(params_dic, k, lbin, zz_integr)
            else:
                raise ValueError("Invalid baryon_model option.")
            pk *= boost_bar
        return pk
    
    def get_pk_nl(self, params_dic, k, lbin, zz_integr, nl_model=0):
        if nl_model == NL_MODEL_HMCODE:
            pk = self.HMcode2020Emulator.get_pk(params_dic, k, lbin, zz_integr)
        elif nl_model == NL_MODEL_BACCO:
            pk = self.BaccoEmulator.get_pk(params_dic, k, lbin, zz_integr)
        else:
            raise ValueError("Invalid nl_model option.")
        return pk

    def get_bar_boost(self, params_dic, k, lbin, zz_integr, baryon_model=0):
        if baryon_model == NO_BARYONS:
            boost_bar = np.ones((lbin, len(zz_integr)), 'float64')
        elif baryon_model == BARYONS_HMCODE:
            boost_bar = self.HMcode2020Emulator.get_barboost(params_dic, k, lbin, zz_integr)
        elif baryon_model == BARYONS_BCEMU:
            boost_bar = self.BCemulator.get_barboost(params_dic, k, lbin, zz_integr)
        elif baryon_model == BARYONS_BACCO:
            boost_bar = self.BaccoEmulator.get_barboost(params_dic, k, lbin, zz_integr)
        else:
            raise ValueError("Invalid baryon_model option.")
        return boost_bar

    def compute_covariance_wl(self, params_dic, model_dic):
        """
        This function computes the (covariance) matrix for weak lensing angular power spectra by first calculating 
        the necessary cosmological functions (e.g., growth factor, power spectrum) and then 
        interpolating the shear power spectrum to the desired multipole values from ell_min to ell_max_wl.

        Parameters:
        -----------
        params_dic : dict
            Dictionary containing cosmological parameters.
        model_dic : dict
            Dictionary containing model parameters such as 'nl_model', 'baryon_model', and 'ia_model'.

        Returns:
        --------
        cov_theory : numpy.ndarray
            Theoretical covariance matrix for weak lensing, with shape 
            (len(self.Survey.ells_wl), self.Survey.nbin, self.Survey.nbin).

        """
        # compute background
        ez, rz, k = self.get_ez_rz_k(params_dic, self.Survey.zz_integr)
        # compute growth factor
        dz, _ = self.get_growth(params_dic, self.Survey.zz_integr, model_dic['nl_model'])
        # compute matter-matter power spectrum
        pmm = self.get_pmm(params_dic, k, self.Survey.lbin, self.Survey.zz_integr, model_dic['nl_model'], model_dic['baryon_model'])
        # compute weak lensing angular power spectra
        cl_wl, _ = self.get_cell_shear(params_dic, ez, rz, dz, pmm, model_dic['ia_model'])
        # create an interpolator at the binned ells
        spline_ll = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list, 3))
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_ll[bin1, bin2] = list(itp.splrep(self.Survey.l_wl[:], cl_wl[:, bin1, bin2]))
        cov_theory = np.zeros((len(self.Survey.ells_wl), self.Survey.nbin, self.Survey.nbin), 'float64')
        # interpolate at all integer values of ell
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                cov_theory[:, bin1, bin2] = itp.splev(self.Survey.ells_wl[:], spline_ll[bin1, bin2])
        return cov_theory

    def compute_covariance_gc(self, params_dic, model_dic):
        """
        This function computes the (covariance) matrix for galaxy clustering angular power spectra by first calculating 
        the necessary cosmological functions (e.g., growth factor, power spectrum) and then 
        interpolating the photo-GC power spectrum to the desired multipole values from ell_min to ell_max_gc.

        Parameters:
        -----------
        params_dic : dict
            Dictionary containing cosmological parameters.
        model_dic : dict
            Dictionary containing model parameters such as 'nl_model', 'baryon_model', and 'bias_model'.

        Returns:
        --------
        cov_theory : numpy.ndarray
            Theoretical covariance matrix for photo-GC, with shape 
            (len(self.Survey.ells_gc), self.Survey.nbin, self.Survey.nbin).

        """
        # compute background
        ez, rz, k = self.get_ez_rz_k(params_dic, self.Survey.zz_integr)
        # compute growth factor
        pk = self.get_pmm(params_dic, k, self.Survey.lbin, self.Survey.zz_integr, model_dic['nl_model'], model_dic['baryon_model'])
        pgg = pk
        pgg_extr = None
        if model_dic['bias_model'] == BIAS_HEFT:
            if model_dic['nl_model']!=0 or model_dic['nl_model']!=1:
                # re-scale for modified gravity
                # pgm dimensions are (15, ell, z_integr)
                dz, _ = self.get_growth(params_dic, self.Survey.zz_integr, model_dic['nl_model'])
                dz_norm, _ = self.get_growth(params_dic, self.Survey.zz_integr,  nl_model=0)
                dz_rescale = dz/dz_norm
                pgg, pgg_extr = self.BaccoEmulator.get_heft(params_dic, k, self.Survey.lbin, self.Survey.zz_integr)*dz_rescale[np.newaxis, np.newaxis, :]*dz_rescale[np.newaxis, np.newaxis, :]
            else:    
                pgg, pgg_extr = self.BaccoEmulator.get_heft(params_dic, k, self.Survey.lbin, self.Survey.zz_integr)               
        cl_gg, _ = self.get_cell_galclust(params_dic, ez, rz, k, pgg, pgg_extr, model_dic['bias_model'])
        # create an interpolator at the binned ells
        spline_gg = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list, 3))
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_gg[bin1, bin2] = list(itp.splrep(self.Survey.l_gc[:], cl_gg[:, bin1, bin2]))
        # interpolate at all integer values of ell
        cov_theory = np.zeros((len(self.Survey.ells_gc), self.Survey.nbin, self.Survey.nbin), 'float64')
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                cov_theory[:, bin1, bin2] = itp.splev(self.Survey.ells_gc[:], spline_gg[bin1, bin2])
        return cov_theory

    def compute_covariance_3x2pt(self, params_dic, model_dic):
        # compute background
        ez, rz, k = self.get_ez_rz_k(params_dic, self.Survey.zz_integr)
        # compute growth factor
        dz, _ = self.get_growth(params_dic, self.Survey.zz_integr, model_dic['nl_model'])
        if model_dic['bias_model'] == BIAS_HEFT:
            # compute matter-matter power spectrum with gravity only
            pk = self.get_pk_nl(params_dic, k, self.Survey.lbin, self.Survey.zz_integr, model_dic['nl_model'])
            bar_boost = self.get_bar_boost(params_dic, k, self.Survey.lbin, self.Survey.zz_integr, model_dic['baryon_model'])
            pmm = pk*bar_boost
            if model_dic['nl_model']!=0 or model_dic['nl_model']!=1:
                # re-scale for modified gravity
                # pgm dimensions are (15, ell, z_integr)
                dz, _ = self.get_growth(params_dic, self.Survey.zz_integr, model_dic['nl_model'])
                dz_norm, _ = self.get_growth(params_dic, self.Survey.zz_integr,  nl_model=0)
                dz_rescale = dz/dz_norm
                pgg, pgg_extr = pgm, pgm_extr = self.BaccoEmulator.get_heft(params_dic, k, self.Survey.lbin, self.Survey.zz_integr) * dz_rescale[np.newaxis, np.newaxis, :]*dz_rescale[np.newaxis, np.newaxis, :]
            else:
                # find matter-galaxy and galaxy-galaxy power spectra without galaxy-bias
                pgg, pgg_extr = pgm, pgm_extr = self.BaccoEmulator.get_heft(params_dic, k, self.Survey.lbin, self.Survey.zz_integr) 
            pgm, pgm_extr = pgm*np.sqrt(bar_boost), pgm_extr*np.sqrt(bar_boost) 
            
        else: 
            # compute matter-matter power spectrum with baryonic feedback
            pmm = self.get_pmm(params_dic, k, self.Survey.lbin, self.Survey.zz_integr, model_dic['nl_model'], model_dic['baryon_model'])
            # find matter-galaxy and galaxy-galaxy power spectra without galaxy-bias
            pgg, pgg_extr = pmm, None
            pgm, pgm_extr = pmm, None    
        # compute weak lensing angular power spectra cl_ll(l, bin_i, bin_j)
        # window function w_l(l,z,bin) in units of h/Mpc
        cl_ll, w_l = self.get_cell_shear(params_dic, ez, rz, dz, pmm, model_dic['ia_model'])
        spline_ll = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list, 3))
        # create an interpolator at the binned ells
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_ll[bin1, bin2] = list(itp.splrep(
                    self.Survey.l_wl[:], cl_ll[:, bin1, bin2]))    
        # compute photometric galaxy clustring angular power spectra cl_gg(l, bin_i, bin_j)
        # window function w_g(l,z,bin) in units of h/Mpc
        cl_gg, w_g = self.get_cell_galclust(params_dic, ez, rz, k, pgg, pgg_extr, model_dic['bias_model'])    
        spline_gg = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list, 3))
        # create an interpolator at the binned ells
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_gg[bin1, bin2] = list(itp.splrep(
                    self.Survey.l_gc[:], cl_gg[:, bin1, bin2]))
        # compute cross-correlated or galaxy-galaxy lensing angular power spectra cl_lg(l, bin_i, bin_j) and cl_gl(l, bin_i, bin_j)
        cl_lg, cl_gl = self.get_cell_cross(params_dic, ez, rz, k, pgm, pgm_extr, w_l, w_g, model_dic['bias_model'])   
        spline_lg = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list, 3))
        spline_gl = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list, 3))
        # create an interpolator at the binned ells
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_lg[bin1, bin2] = list(itp.splrep(
                    self.Survey.l_xc[:], cl_lg[:, bin1, bin2]))
                spline_gl[bin1, bin2] = list(itp.splrep(
                    self.Survey.l_xc[:], cl_gl[:, bin1, bin2]))
        # compose a matrix
        # C_LL | C_LG
        # C_GL | C_GG   
        # and a "high"-matrix for ell>ell_max_gc with weak lensing anggular power spectra, as we assume that ell_max_wl>ell_max_gc
        cov_theory = np.zeros((len(self.Survey.ells_gc), 2 * self.Survey.nbin, 2 * self.Survey.nbin), 'float64')
        cov_theory_high = np.zeros(((len(self.Survey.ells_wl) - self.Survey.ell_jump), self.Survey.nbin, self.Survey.nbin), 'float64')    
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                cov_theory[:, bin1, bin2] = itp.splev(
                    self.Survey.ells_gc[:], spline_ll[bin1, bin2])
                cov_theory[:, self.Survey.nbin + bin1, bin2] = itp.splev(
                    self.Survey.ells_gc[:], spline_gl[bin1, bin2])
                cov_theory[:, bin1, self.Survey.nbin + bin2] = itp.splev(
                    self.Survey.ells_gc[:], spline_lg[bin1, bin2])
                cov_theory[:, self.Survey.nbin + bin1, self.Survey.nbin + bin2] = itp.splev(
                    self.Survey.ells_gc[:], spline_gg[bin1, bin2])
                cov_theory_high[:, bin1, bin2] = itp.splev(
                    self.Survey.ells_wl[self.Survey.ell_jump:], spline_ll[bin1, bin2])
        return cov_theory, cov_theory_high
    

 