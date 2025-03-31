import numpy as np
import MGrowth as mg
from scipy.integrate import trapezoid, quad
from scipy import interpolate as itp
from .structure.hmcode2020_interface import HMcode2020
from .structure.bacco_interface import BaccoEmu
from .structure.bcemu_interface import BCemulator
from .structure.react_ndgp_interface import DGPReACT
from .structure.react_gamma_interface import GammaReACT
from .structure.react_musigma_interface import MuSigmaReACT
from .structure.react_ide_interface import DarkScatteringReACT
from math import sqrt, log, exp, pow, log10
import os
import fastpt as fpt
from scipy.interpolate import interp1d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
# Suppress TensorFlow warnings


H0_h_c = 1./2997.92458 
# =100/c in Mpc/h Hubble constant conversion factor
C_IA = 0.0134 
# =dimensionsless, C x rho_crit 
PIVOT_REDSHIFT = 0.

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


PHOTOZ_NONE = 0
PHOTOZ_ADD = 1
PHOTOZ_MULT = 2

def check_zmax(zmax, emu_obj):
    emu_z_max = emu_obj.zz_max
    if zmax > emu_z_max:
        raise ValueError(f'Survey z_max must not exceed zz_max in the power spectrum computation for {emu_obj.emu_name}!')

def compute_spline(c_ell, binned_ell, nbin):
    spline_c_ell = np.empty((nbin, nbin), dtype=(list, 3))
    # create an interpolator at the binned ells
    for bin1 in range(nbin):
        for bin2 in range(nbin):
            spline_c_ell[bin1, bin2] = list(itp.splrep(
                binned_ell[:], c_ell[:, bin1, bin2]))    
    return spline_c_ell        

def build_data_matrix(c_ell, binned_ell, all_int_ell, nbin):
    # create an interpolator at the binned ells
    spline_c_ell = compute_spline(c_ell, binned_ell, nbin)
    cov_theory = np.zeros((len(all_int_ell), nbin, nbin), 'float64')
    # interpolate at all integer values of ell
    for bin1 in range(nbin):
        for bin2 in range(nbin):
            cov_theory[:, bin1, bin2] = itp.splev(all_int_ell[:], spline_c_ell[bin1, bin2])
    return cov_theory  

def build_data_matrix_3x2pt(cl_ll, cl_gg, cl_lg, cl_gl,  binned_ell_wl, binned_ell_gc, binned_ell_xc, 
                            all_int_ell_wl, all_int_ell_gc, ell_jump, nbin):
    # create an interpolator at the binned ells
    spline_ll = compute_spline(cl_ll, binned_ell_wl, nbin)
    spline_gg = compute_spline(cl_gg, binned_ell_gc, nbin)
    spline_lg = compute_spline(cl_lg, binned_ell_xc, nbin)
    spline_gl = compute_spline(cl_gl, binned_ell_xc, nbin)
    cov_theory = np.zeros((len(all_int_ell_gc), 2 * nbin, 2 * nbin), 'float64')
    cov_theory_high = np.zeros(((len(all_int_ell_wl) - ell_jump), nbin, nbin), 'float64')  
    for bin1 in range(nbin):
        for bin2 in range(nbin):
            cov_theory[:, bin1, bin2] = itp.splev(
                all_int_ell_gc[:], spline_ll[bin1, bin2])
            cov_theory[:, nbin + bin1, bin2] = itp.splev(
                all_int_ell_gc[:], spline_gl[bin1, bin2])
            cov_theory[:, bin1, nbin + bin2] = itp.splev(
                all_int_ell_gc[:], spline_lg[bin1, bin2])
            cov_theory[:, nbin + bin1, nbin + bin2] = itp.splev(
                all_int_ell_gc[:], spline_gg[bin1, bin2])
            cov_theory_high[:, bin1, bin2] = itp.splev(
                all_int_ell_wl[ell_jump:], spline_ll[bin1, bin2])           
    return cov_theory, cov_theory_high


class Theory:
    def __init__(self, SurveyClass, model: dict):
        """
        Initialize the theory class with the given survey and model parameters.

        Parameters:
        ----------
        SurveyClass : object
            An instance of the survey class containing survey-specific information.
        model : dict
            A dictionary containing model parameters with the following keys:

            - 'nl_model': Nonlinear model option. Valid options are:
                - NL_MODEL_HMCODE
                - NL_MODEL_BACCO
                - NL_MODEL_NDGP
                - NL_MODEL_GAMMAZ
                - NL_MODEL_MUSIGMA
                - NL_MODEL_DS
            - 'baryon_model': Baryonic model option. Valid options are:
                - NO_BARYONS
                - BARYONS_HMCODE
                - BARYONS_BCEMU
                - BARYONS_BACCO
            - 'ia_model': Intrinsic alignment model option. Valid options are:
                - IA_NLA
                - IA_TATT
            - 'bias_model': Galaxy bias model option. Valid options are:
                - BIAS_LIN
                - BIAS_B1B2
                - BIAS_HEFT
            - 'photoz_err_model': Redshift uncertainty model option. Valid options are:
                - PHOTOZ_NONE 
                - PHOTOZ_ADD 
                - PHOTOZ_MULT

        Raises:
        ------
        ValueError:
            If an invalid option is provided for 'nl_model', 'baryon_model', 'ia_model', or 'bias_model'.
            If 'ia_model' is set to IA_TATT and 'bias_model' is not BIAS_LIN.
        """
        self.Survey = SurveyClass
        # load emulators
        # assign nonlinear prescription
        nl_models = {
            NL_MODEL_HMCODE: HMcode2020,
            NL_MODEL_BACCO: BaccoEmu,
            NL_MODEL_NDGP: DGPReACT,
            NL_MODEL_GAMMAZ: GammaReACT,
            NL_MODEL_MUSIGMA: MuSigmaReACT,
            NL_MODEL_DS: DarkScatteringReACT
        }
        try:
            if model['nl_model'] == NL_MODEL_BACCO:
                option = model['bacco_option'] if 'bacco_option' in model else 'z_extrap_linear'
                self.StructureEmu = nl_models[model['nl_model']](option)
            else:
                self.StructureEmu = nl_models[model['nl_model']]()
        except KeyError:
            raise ValueError("Invalid nl_model option.")
        check_zmax(self.Survey.zmax, self.StructureEmu)

        #if 'add_noise' in model and not model['add_noise']:
        #    self.Survey.noise['LL'] = 0.
        #    self.Survey.noise['GG'] = 0.


        # assign baryonic prescription
        baryon_models = {
            BARYONS_HMCODE: HMcode2020,
            BARYONS_BCEMU: BCemulator,
            BARYONS_BACCO: BaccoEmu
        }
        self.BaryonsEmu = baryon_models.get(model['baryon_model'], None)
        if self.BaryonsEmu:
            self.BaryonsEmu = self.BaryonsEmu()
            check_zmax(self.Survey.zmax, self.BaryonsEmu)
    
        # assign intrinsic alignment     
        ia_models = {
            IA_NLA: (self.get_pk_nla, self.get_pk_cross_nla),
            IA_TATT: (self.get_pk_tatt, self.get_pk_cross_tatt)
        }
        try:
            self.get_pk_ia, self.get_pk_cross_ia = ia_models[model['ia_model']]
            if model['ia_model'] == IA_TATT:
                pad_factor = 1
                self.klin = self.StructureEmu.kh_lin
                n_pad = int(pad_factor * len(self.klin))
                self.fpt_obj = fpt.FASTPT(self.klin, to_do=['IA'],
                            low_extrap=-5,
                            high_extrap=3,
                            n_pad=n_pad)
        except KeyError:
            raise ValueError("Invalid ia_model option.")
            
        # assign galaxy bias
        self.flag_heft = (model['bias_model'] == BIAS_HEFT)
        bias_models = {
            BIAS_LIN: (self.get_pgm_lin_bias, self.get_pgg_lin_bias),
            BIAS_B1B2: (self.get_pgm_quadr_bias, self.get_pgg_quadr_bias),
            BIAS_HEFT: (self.get_pgm_heft_bias, self.get_pgg_heft_bias)
        }
        try:
            self.get_pgm, self.get_pgg = bias_models[model['bias_model']]
            if self.flag_heft:
                self.BaccoEmuClass = BaccoEmu()
        except KeyError:
            raise ValueError("Invalid bias_model option.")
        if model['bias_model'] != BIAS_LIN and model['ia_model'] == IA_TATT:
            raise ValueError("TATT is only available with linear bias!")

        # assign photo-z model
        photoz_err_models = {
            PHOTOZ_NONE: lambda nz, deltas: nz,
            PHOTOZ_ADD: self.get_add_photoz_error,
            PHOTOZ_MULT: self.get_mult_photoz_error
        }
        self.get_n_of_z = photoz_err_models.get(model['photoz_err_model'])
        if self.get_n_of_z is None:
            raise ValueError("Invalid photoz_err_model option.")

    def get_fpt_terms(self, plin_z0,  C_window=.75):
        """
        Computes the terms of the Intrinsic Alignment (IA) TATT model at 1-loop order.
        For reference on the equations, see https://arxiv.org/pdf/1708.09247.

        Parameters:
        ----------
        plin_z0 : numpy.ndarray
            Linear power spectrum at redshift z = 0.
        C_window : float, optional
            Window function parameter to remove high frequency modes and avoid ringing effects.
            Default value is 0.75.

        Returns:
        -------
        dict
            Dictionary containing the 1-loop order terms of the intrinsic alignment model:
            - 'a00e': scipy.interpolate.interp1d
            - 'c00e': scipy.interpolate.interp1d
            - 'a0e0e': scipy.interpolate.interp1d
            - 'a0b0b': scipy.interpolate.interp1d
            - 'ae2e2': scipy.interpolate.interp1d
            - 'ab2b2': scipy.interpolate.interp1d
            - 'a0e2': scipy.interpolate.interp1d
            - 'b0e2': scipy.interpolate.interp1d
            - 'd0ee2': scipy.interpolate.interp1d
            - 'd0bb2': scipy.interpolate.interp1d
            Each entry in the dictionary is an interpolation function of the corresponding term as a function of the wavenumber.
        """

        P_window = None

        a00e, c00e, a0e0e, a0b0b = self.fpt_obj.IA_ta(plin_z0,
                                                   P_window=P_window,
                                                   C_window=C_window)
        ae2e2, ab2b2 = self.fpt_obj.IA_tt(plin_z0, P_window=P_window, C_window=C_window)

        a0e2, b0e2, d0ee2, d0bb2 = self.fpt_obj.IA_mix(plin_z0,
                                                    P_window=P_window,
                                                    C_window=C_window)

        a00e_int = itp.interp1d(self.klin, a00e, kind='linear',
                                    fill_value='extrapolate')

        c00e_int = itp.interp1d(self.klin, c00e, kind='linear',
                                    fill_value='extrapolate')

        a0e0e_int = itp.interp1d(self.klin, a0e0e, kind='linear',
                                     fill_value='extrapolate')

        a0b0b_int = itp.interp1d(self.klin, a0b0b, kind='linear',
                                     fill_value='extrapolate')

        ae2e2_int = itp.interp1d(self.klin, ae2e2, kind='linear',
                                     fill_value='extrapolate')

        ab2b2_int = itp.interp1d(self.klin, ab2b2, kind='linear',
                                     fill_value='extrapolate')

        a0e2_int = itp.interp1d(self.klin, a0e2, kind='linear',
                                    fill_value='extrapolate')

        b0e2_int = itp.interp1d(self.klin, b0e2, kind='linear',
                                    fill_value='extrapolate')

        d0ee2_int = itp.interp1d(self.klin, d0ee2, kind='linear',
                                     fill_value='extrapolate')

        d0bb2_int = itp.interp1d(self.klin, d0bb2, kind='linear',
                                     fill_value='extrapolate')

        return {'a00e': a00e_int, 'c00e': c00e_int, 'a0e0e': a0e0e_int, 'a0b0b': a0b0b_int, 
                'ae2e2': ae2e2_int, 'ab2b2': ab2b2_int, 'a0e2': a0e2_int, 'b0e2': b0e2_int, 'd0ee2': d0ee2_int, 'd0bb2': d0bb2_int}



    def check_consistency(self, params): 
        """
        Check the consistency of the provided cosmological parameters.

        Parameters:
        ----------
        params (dict): Dictionary containing cosmological parameters.

        Raises:
        ------
        KeyError: If 'h' is not found in the dictionary.

        ValueError: If too many density parameters are present, 
                    if multiple amplitude parameters are present, 
                    or if no amplitude parameters are present.

        Returns:
        -------
        dict: Updated parameters after applying relations.
        """
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
        'sigma8_cb',
        'S8']
        amplitude_params = [param for param in conditions_amp if param in params]
        if len(amplitude_params) > 1:
            raise ValueError(f"Invalid parameter set: multiple amplitude parameters present: {amplitude_params}")
        elif len(amplitude_params) == 0:  
            raise ValueError(f"Invalid parameter set: no amplitude parameters present")
        params_new = self.apply_relations(params)
        return params_new
        
    def apply_relations(self, params): 
        """Apply cosmological parameter relations and compute derived parameters.

        Parameters:
        ----------
        params (dict): Dictionary containing cosmological parameters.

        Returns:
        -------
        dict: Updated dictionary with derived cosmological parameters.
        """
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
        if 'Mnu' in params.keys():
            params['Omnuh2'] = params['Mnu'] * ((params['nnu'] / 3.0) ** 0.75 / 94.06410581217612 * (params['TCMB']/2.7255)**3)
            params['Omega_nu'] = params['Omnuh2'] / params['h']/params['h']    
        elif 'Omega_nu' in params.keys():
            params['Omnuh2'] = params['Omega_nu']*params['h']*params['h']   
            params['Mnu'] = params['Omnuh2'] / ((params['nnu'] / 3.0) ** 0.75 / 94.06410581217612 * (params['TCMB']/2.7255)**3)    
        elif 'Omnuh2' in params.keys():
            params['Omega_nu'] = params['Omnuh2'] / params['h']/params['h'] 
            params['Mnu'] = params['Omnuh2'] / ((params['nnu'] / 3.0) ** 0.75 / 94.06410581217612 * (params['TCMB']/2.7255)**3)
        if 'Omega_m' not in params.keys():
            params['Omega_m'] = params['Omega_b'] + params['Omega_c'] + params['Omega_nu']    
        params['Omega_cb'] = params['Omega_m'] - params['Omega_nu']
        params['Omega_c'] = params['Omega_cb'] - params['Omega_b']
        params['fb'] = params['Omega_b']/params['Omega_m']
        if 'As' in params.keys():
            params['log10As'] = log(1e10*params['As'])
        elif 'log10As' in params.keys():
            params['As'] = exp(params['log10As'])*1e-10  
        return params        
    
    def check_ranges(self, params):
        """
        Check if the parameters are within the valid ranges for the emulators
        at each step of MCMC.

        Parameters:
        ----------
        params (dict): Dictionary containing the parameters to be checked.

        Returns:
        -------
        boolean: True if all parameters are within the valid ranges, False otherwise.
        """
        status_nl = self.StructureEmu.check_pars(params) #including mg-conditions
        if self.BaryonsEmu!=None:
            status_b  = self.BaryonsEmu.check_pars(params) 
        else:
            status_b = True    
        status = status_nl and status_b
        return  status 
    
    def check_ranges_ini(self, params):
        """
        Check if the parameters are within the valid ranges for the emulators
        once at initialization of data and model with explicit error messages.

        Parameters:
        ----------
        params (dict): Dictionary containing the parameters to be checked.

        Returns:
        -------
        boolean: True if all parameters are within the valid ranges, False otherwise.
        """
        status_nl = self.StructureEmu.check_pars_ini(params) #including mg-conditions
        if self.BaryonsEmu!=None:
            status_b  = self.BaryonsEmu.check_pars_ini(params) 
        else:
            status_b = True    
        status = status_nl and status_b
        return  status 

    def check_pars_ini(self, param_dic):
        """Initial check and validate the parameters for the model and data.

        Parameters:
        ----------
        param_dic (dict): Dictionary containing the parameters to be checked.

        Returns:
        -------
        tuple: A tuple containing the updated parameter dictionary and the status
               of the parameter range check.

        Raises:
        ------       
        KeyError: If the required parameter 'h' is not found in the parameter dictionary.
        
        KeyError: If some cosmological parameters are missing from the parameter dictionary.
        """
        param_dic_all = self.check_consistency(param_dic)
        status = self.check_ranges_ini(param_dic_all)
        return param_dic_all, status
    
    def check_pars(self, param_dic):
        """
        Check and validate the parameters for the model within a chain.

        Parameters:
        ----------
        param_dic (dict): Dictionary containing the parameters to be checked.

        Returns:
        -------
        tuple: A tuple containing the updated parameter dictionary and the status
               of the parameter range check.
        """
        param_dic_all = self.apply_relations(param_dic)
        status = self.check_ranges(param_dic_all)
        return param_dic_all, status

    def get_ez_rz_k(self, params_dic):
        r"""
        Calculate the :math:`E(z)`, :math:`r_{\rm com}(z)`, and :math:`k(z)=(\ell+1/2)/r_{\rm com}(z)` grids based on cosmological parameters.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing cosmological parameters:

            - 'Omega_m': Matter density parameter.
            - 'w0': Equation of state parameter w0.
            - 'wa': Equation of state parameter wa.

        Returns:
        -------
        e_z_grid : numpy.ndarray
            Array of E(z) values corresponding to the survey's redshift integration grid.
        r_z_grid : numpy.ndarray
            Array of r_com(z) values corresponding to the survey's redshift integration grid, in units of Mpc/h.
        k_grid : numpy.ndarray
            Array of k(z) values corresponding to the survey's redshift integration grid, in units of h/Mpc.
        """
        omega_m = params_dic['Omega_m']
        w0 = params_dic['w0']
        wa = params_dic['wa']
        omega_lambda_func = lambda z: (1.-omega_m) * pow(1.+z, 3.*(1.+w0+wa)) * np.exp(-3.*wa*z/(1.+z))
        e_z_func = lambda z: np.sqrt(omega_m*pow(1.+z, 3) + omega_lambda_func(z))
        r_z_int = lambda z: 1./e_z_func(z)
        r_z_func = lambda z_in: quad(r_z_int, 0, z_in)[0]
        r_z_grid = np.array([r_z_func(zz_i) for zz_i in self.Survey.zz_integr])/H0_h_c 
        e_z_grid = np.array([e_z_func(zz_i) for zz_i in self.Survey.zz_integr])
        k_grid =(self.Survey.ell[:,None]+0.5)/r_z_grid
        return e_z_grid, r_z_grid, k_grid
    
    def get_add_photoz_error(self, nz, deltaz):
        """
        Additative photo-z error to the redshift distribution of galaxies:
        :math:`n(z')=n(z+\delta_z)`.

        Parameters:
        ----------
        deltaz : numpy.ndarray
            Array containing photo-z error parameters for sources or lenses.
        nz : numpy.ndarray
            Array of redshift distribution of galaxies, sources or lenses.

        Returns:
        -------
        nz_biased : numpy.ndarray
            The biased redshift distribution of galaxies after applying the additive photo-z error.
        """
        nz_biased = np.zeros((len(self.Survey.zz_integr), self.Survey.nbin))
        for i in range(self.Survey.nbin):
            f = itp.interp1d(self.Survey.zz_integr, nz[:,i], fill_value=0.0, bounds_error=False)
            # additive mode
            nz_biased[:,i] = f(self.Survey.zz_integr - deltaz[i])
            # normalize
            nz_biased[:,i] /= np.trapz(nz_biased[:,i], self.Survey.zz_integr)
        return nz_biased
    
    def get_mult_photoz_error(self, nz, deltaz):
        """
        Multiplicative photo-z error to the redshift distribution of galaxies:
        :math:`n(z')=n(z(1+\delta_z))`.

        Parameters:
        ----------
        deltaz : numpy.ndarray
            Array containing photo-z error parameters for sources or lenses.
        nz : numpy.ndarray
            Array of redshift distribution of galaxies, sources or lenses.

        Returns:
        -------
        nz_biased : numpy.ndarray
            The biased redshift distribution of galaxies after applying the multiplicative photo-z error.
        """
        nz_biased = np.zeros((len(self.Survey.zz_integr), self.Survey.nbin))
        for i in range(self.Survey.nbin):
            f = itp.interp1d(self.Survey.zz_integr, nz[:,i], fill_value=0.0, bounds_error=False)
            # multiplicative mode
            nz_biased[:,i] = f(self.Survey.zz_integr * (1 - deltaz[i]))
            # normalize
            nz_biased[:,i] /= np.trapz(nz_biased[:,i], self.Survey.zz_integr)
        return nz_biased

    
    def get_tatt_parameters(self, params):
        """Calculate the normalised tidal alignment and tidal torque (TATT) model parameters.
        See definitions of C1, C1d and C2 in https://arxiv.org/pdf/1708.09247.


        Parameters:
        ----------
        params : dict
            Dictionary containing the following keys:

            - 'Omega_m': float, matter density parameter.
            - 'a1_IA': float, amplitude of the tidal alignment model.
            - 'a2_IA': float, amplitude of the torque alignment model.
            - 'b1_IA': float, amplitude of the density-weighted linear alignment model.
            - 'eta1_IA': float, redshift evolution parameter for the tidal alignment model.
            - 'eta2_IA': float, redshift evolution parameter for the torque alignment model.

        Returns:
        -------
        tuple
            A tuple containing three numpy arrays:
            
            - c1
            - c1d
            - c2
        """
        omega_m = params['Omega_m']
        # dz is normalise at z=0
        dz = self.dz[None, :]
        a1_ia = params['a1_IA']
        a2_ia = params['a2_IA']
        b1_ia = params['b1_IA']
        eta1_ia = params['eta1_IA']
        eta2_ia = params['eta2_IA']
        c1 = -a1_ia * C_IA * omega_m * \
            ((1 + self.Survey.zz_integr) / (1 + PIVOT_REDSHIFT)) ** eta1_ia / dz
        # factor 5 comes from setting similar signal in II smoothed withing filter for a1=a2
        c2 = a2_ia * 5 * C_IA * omega_m * \
            ((1 + self.Survey.zz_integr) / (1 + PIVOT_REDSHIFT)) ** eta2_ia / (dz**2)
        c1d = b1_ia * c1
        # dimensions (lbin, z_integr)
        return c1, c1d, c2


    def get_pk_tatt(self, params_dic):
        """Calculate the power spectra for tidal alignment and tidal torquing (TATT) model.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters needed to compute the TATT power spectra.

        Returns:
        -------
        pk_delta_ia : numpy.ndarray
            Power spectrum for the intrinsic alignment density.
        pk_iaia : numpy.ndarray
            Power spectrum for the intrinsic alignment-intrinsic alignment.
        """
        c1, c1d, c2 = self.get_tatt_parameters(params_dic)
        dz = self.dz[None, :]
        # growth factor can be scale-dependent for f(R) in the future potentially
        # fpt terms are computed at redshift 0
        fpt_terms = self.get_fpt_terms(self.StructureEmu.pklin_z0)

        pk_delta_ia =   c1 * self.pmm + \
                        c1d * dz**4 * (fpt_terms['a00e'](self.k) + fpt_terms['c00e'](self.k)) + \
                        c2 * dz**4 *  (fpt_terms['a0e2'](self.k) + fpt_terms['b0e2'](self.k))


        pk_iaia =   c1**2.0 * self.pmm+ \
                    2.0 * c1 * c1d * dz**4 * (fpt_terms['a00e'](self.k) + fpt_terms['c00e'](self.k)) +\
                    c1d**2.0 * dz**4 * fpt_terms['a0e0e'](self.k) + \
                    c2**2.0 * dz**4 * fpt_terms['ae2e2'](self.k) +  \
                    2.0 * c1 * c2 * dz**4 * (fpt_terms['a0e2'](self.k)+ fpt_terms['b0e2'](self.k)) +\
                    2.0 * c1d * c2 * dz**4 * fpt_terms['d0ee2'](self.k)
        # dimensions of is power spectra: (ell, zz_integr)
        return pk_delta_ia, pk_iaia
    
    def get_pk_nla(self, params_dic):
        r"""Calculate the extended non-linear alignment (e-zNLA) power spectra:

        .. math::
            P_{\delta \mathrm{IA}}(z, k) = f_{\rm IA}(z) P_{\rm mm}(z, k) \\
            P_{\rm IAIA}(z, k) = [f_{\rm IA}(z)]^2 P_{\rm mm}(z, k) 

        with the factor

        .. math::
            f_{\rm IA} = - A_1 C_\rm {IA} \frac{\Omega_{\rm m}}{D(z)} \left(\frac{1+z}{1+z_{\rm piv}}\right)^{\eta_1} [L(z)]^\beta

        where:

        - :math:`P_{\rm mm}` is the matter-matter power spectrum.
        - :math:`D(z)` is the normalised at z=0 growth factor.
        - :math:`L(z)` is the luminosity function.
        - :math:`C_{\rm IA} = \bar{C}_{\rm IA} \rho_{\rm crit}` and  :math:`\bar{C}_{\rm IA}=5 \times 10^{-14} M^{-1}_\odot h^{-2} \rm{Mpc}^3`.
        - :math:`z_{\rm piv}` is the pivot redshift that we set to 0, often in the literature it equals 0.62.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the following keys:

            - 'Omega_m': float, matter density parameter.
            - 'a1_IA': float, intrinsic alignment amplitude.
            - 'eta1_IA': float, redshift evolution parameter for intrinsic alignment.
            - 'beta_IA': float, luminosity evolution parameter for intrinsic alignment.

        Returns:
        -------
        tuple
            A tuple containing:

            - pk_delta_ia : ndarray
                The cross power spectrum between the density field and the intrinsic alignment field.
            - pk_iaia : ndarray
                The auto power spectrum of the intrinsic alignment field.
        """
        omega_m = params_dic['Omega_m']
        a_ia = params_dic['a1_IA'] 
        eta_ia = params_dic['eta1_IA'] 
        beta_ia = params_dic['beta_IA']
        f_ia = ((1. + self.Survey.zz_integr) / (1. + PIVOT_REDSHIFT))**eta_ia * (self.Survey.lum_func(self.Survey.zz_integr))**beta_ia
        # dz (ell, zz_inegr) must be normalise to unity at z=0 to cancel the linear growth of the density field 
        # and yield a constant amplitude in the primordial alignment scenario
        dz = self.dz[None,:] 
        # growth factor can be scale-dependent for f(R) in the future potentially
        # dimensions of is power spectra: (ell, zz_integr)
        self.factor_nla = - a_ia*C_IA*omega_m*f_ia[None,:]/dz
        pk_delta_ia = self.factor_nla * self.pmm
        pk_iaia = (self.factor_nla)**2 * self.pmm
        return pk_delta_ia, pk_iaia
    
    def get_wl_kernel(self, params_dic):
        r"""Calculate the weak lensing kernel:

        .. math::
            W^{\gamma}_{i} = \frac{3}{2} \left( \frac{H_0}{c} \right)^2 \Omega_{\rm m} (1+z) r_{\rm com}(z) \bar{W}_i(z)
        with

        .. math:: 
            \bar{W}_i(z) = \int \mathrm{d}z' n_i(z')\left[ 1-\frac{r_{\rm com}(z)}{r_{\rm com}(z')} \right]   

        where:

        - :math:`n_i(z)` is the redshift distribution of the sourses.
        - :math:`E(z)` is the expansion function.
        - :math:`r_{\rm com}(z)` is the comoving distance.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing cosmological parameters. Must include:

            - 'Omega_m': float, the matter density parameter.

        Returns:
        -------
        w_gamma : numpy.ndarray
            The weak lensing kernel. The shape of the array is (1, num_bins, num_bins),
            where num_bins is the number of redshift bins.
        """
        omega_m = params_dic['Omega_m']
        # can be later changed 
        #deltas = np.array([params_dic['deltaz_'+str(i+1)] for i in range(self.Survey.nbin)])
        eta_z_s =  self.get_n_of_z(self.Survey.eta_z_s, self.deltaz_s)

        # in the integrand dimensions are (bin_i, zz_integr, zz_integr)
        integrand = 3./2.*H0_h_c**2. * omega_m * self.rz[None,:,None]*(1.+self.Survey.zz_integr[None,:,None])*eta_z_s.T[:,None,:]*(1.-self.rz[None,:,None]/self.rz[None,None,:])
        # integrate along the third dimension in zz_integr
        w_gamma  = trapezoid(np.triu(integrand), self.Survey.zz_integr,axis=-1).T
        # add an extra dimension to w_gamma as we might have ell-dependence in the IA-kernel due to the scale-dependent linear growth
        w_gamma = w_gamma[None,:,:]
        return w_gamma
    
    def get_ia_kernel(self, params_dic):
        r"""This function computes the intrinsic alignment (IA) kernel, which is a function of redshift and 
        cosmological parameters:

        .. math::
            W^{\rm IA}_{i}(z) = n_i(z) H(z)
        
        where:

        - :math:`n_i(z)` is the redshift distribution of the sourses.
        - :math:`H(z)` is the expansion function.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the necessary parameters for the calculation. 


        Returns:
        -------
        w_ia : numpy.ndarray
            A 3D array with dimensions (ell, zz_integr, bin_i) representing the 
            intrinsic alignment kernel.
        """
        # dimension (ell, zz_integr, bin_i)
        eta_z_s =  self.get_n_of_z(self.Survey.eta_z_s, self.deltaz_s)
        w_ia = eta_z_s[None, :, :] * self.ez[None, :,None] * H0_h_c 
        return w_ia
    
    def get_cell_shear(self):
        r"""Calculate the weak lensing power spectrum (C_ell):

        .. math::
            C^{\rm LL}_{ij}(\ell) = \frac{c}{H_0} \int \mathrm{d}z \frac{\left[ W^{\gamma}_i(z)W^{\gamma}_j(z)P_{\rm mm}(k(\ell, z), z) + 
            (W^{\gamma}_i(z)W^{\rm IA}_j(z)+W^{\rm IA}_i(z)W^{\gamma}_j(z))P_{\rm IA m}(k(\ell, z), z)) + 
            W^{\rm IA}_i(z)W^{\rm IA}_j(z)P_{\rm IA IA}(k(\ell, z), z)\right]}{E(z) r^2_{\rm com}(z)}\, ,

        where:

        - :math:`W^{\gamma}`, :math:`W^{\rm IA}`  are the lensing and intrinsic alignment kernels.
        - :math:`E(z)` is the expansion function.
        - :math:`r_{\rm com}(z)` is the comoving distance.
        - :math:`P_{\rm mm}(z, k)`, :math:`P_{\rm m IA}(z, k)`, :math:`P_{\rm IA IA}(z, k)` are the matter-matter, matter-IA and IA-IA power spectra.     
            
        Returns:
        -------
        cl_ll : numpy.ndarray
            Weak lensing power spectrum (C_ell) for different redshift bins.
        """
        kernel_delta_ia = (self.w_gamma[:, :, :, None]*self.w_ia[:, :, None, :] + self.w_gamma[:, :, None, :]*self.w_ia[:, :, :, None]) * self.pk_delta_ia[:, :, None, None]
        kernel_iaia = self.w_ia[:, :, :, None]*self.w_ia[:, :, None, :]  * self.pk_iaia[:, :, None, None]
        kernel_wl = self.w_gamma[:, :, :, None]*self.w_gamma[:, :, None, :]  * self.pmm[:, :, None, None]
        # compute the integrand with dimensions (ell, z_integr, bin_i, bin_j)
        cl_ll_int = (kernel_wl + kernel_delta_ia + kernel_iaia) / self.ez[None,:,None,None] / self.rz[None,:,None,None] / self.rz[None,:,None,None] / H0_h_c
        # integrate along the z_integr-direction
        cl_ll = trapezoid(cl_ll_int, self.Survey.zz_integr, axis=1)[:self.Survey.nell_wl, :, :]
        # add noise to the auto-correlated bins
        for i in range(self.Survey.nbin):
            cl_ll[:, i, i] += self.Survey.noise['LL']
        return cl_ll
    
     
    def get_pmm(self, params_dic):
        """Calculate the matter-matter power spectrum (P_mm) with optional baryonic corrections.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the cosmological parameters.

        Returns:
        -------
        pk : numpy.ndarray
            The matter power spectrum with the specified baryonic corrections applied.
        boost_bar : numpy.ndarray
            The baryonic boost factor applied to the matter power spectrum.
        """
        # return matter-matter power spectrum of dimension (lbin, z_integr)
        pk = self.StructureEmu.get_pk_nl(params_dic, self.k, self.Survey.lbin, self.Survey.zz_integr)
        if self.BaryonsEmu!=None:
            # add baryonic boost
            boost_bar = self.BaryonsEmu.get_barboost(params_dic, self.k, self.Survey.lbin, self.Survey.zz_integr)
            pk *= boost_bar
            return pk, boost_bar
        return pk, np.ones((self.Survey.lbin, len(self.Survey.zz_integr)), 'float64')
    
    def get_pgg_lin_bias(self, params_dic):
        """Calculate the galaxy-galaxy power spectrum with linear bias (constant in a given redshift bin),
        in units of (Mpc/h)^3.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the bias parameters.
  
        Returns:
        -------
        pgg : numpy.ndarray
            The galaxy-galaxy power spectrum.
        """
        bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.Survey.nbin)])
        # return dimension (lbin, z_integr, bin_i, bin_j)
        pgg = bias1[None, None, :, None] * bias1[None, None, None, :] * self.pmm[:,:,None, None]
        return pgg
    
    def get_pgg_quadr_bias(self, params_dic):
        """Calculate the galaxy-galaxy power spectrum with quadratic bias model,
        in units of (Mpc/h)^3.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the bias parameters.
  
        Returns:
        -------
        pgm : numpy.ndarray
            The galaxy-galaxy power spectrum.
        """
        bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.Survey.nbin)])
        bias2 = np.array([params_dic['b2_'+str(i+1)] for i in range(self.Survey.nbin)])  
        # return dimension (lbin, z_integr, bin_i, bin_j)
        pgg = ( bias1[None, None,:, None] + bias2[None, None, :, None] * self.k[:, :, None, None]**2 ) * \
              ( bias1[None, None, None, :] + bias2[None, None, None, :] * self.k[:, :, None, None]**2 ) * self.pmm[:,:,None,None]
        return pgg
    
    def get_pgm_lin_bias(self, params_dic):
        """Calculate the galaxy-matter power spectrum with linear bias (constant in a given redshift bin),
        in units of (Mpc/h)^3.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the bias parameters.
  
        Returns:
        -------
        pgm : numpy.ndarray
            The matter-galaxy power spectrum.
        """
        bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.Survey.nbin)])
        # return dimension (lbin, z_integr, bin_i)
        pgm = bias1[None, None, :] * self.pmm[:,:,None]
        return pgm
    
    def get_pgm_quadr_bias(self, params_dic):
        """Calculate the galaxy-matter power spectrum with quadratic bias,
        in units of (Mpc/h)^3.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the bias parameters.
  
        Returns:
        -------
        pgm : numpy.ndarray
            The matter-galaxy power spectrum.
        """
        bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.Survey.nbin)])
        bias2 = np.array([params_dic['b2_'+str(i+1)] for i in range(self.Survey.nbin)])  
        # return dimension (lbin, z_integr, bin_i)
        pgm = ( bias1[None, None,:] + bias2[None, None, :] * self.k[:, :, None]**2 ) * self.pmm[:,:,None]
        return pgm
    
    def get_heft_pk_exp(self, params_dic):
        """Calculate the HEFT power spectrum with possible rescaling for modified gravity.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters for the calculation.

        Returns:
        -------
        pk_exp : ndarray
            The power spectrum perturbative expansion terms.
        pk_exp_extr : ndarray
            The extrapolated power spectrum.

        Notes:
        -----
        If the emulator name in `StructureEmu` is not 'BACCO', the power spectrum
        is rescaled for modified gravity using the growth factors from `StructureEmu` and `BaccoEmuClass`.
        """
        if self.StructureEmu.emu_name!='BACCO':
            # re-scale for modified gravity
            # heft pk dimensions are (15, ell, z_integr)
            dz, _ = self.StructureEmu.get_growth(params_dic, self.Survey.zz_integr)
            params_dic['sigma8_cb'] = self.BaccoEmuClass.get_sigma8_cb(params_dic)
            dz_norm, _ = self.BaccoEmuClass.get_growth(params_dic, self.Survey.zz_integr)
            dz_rescale = dz/dz_norm
            pk_exp, pk_exp_extr = self.BaccoEmuClass.get_heft(params_dic, self.k, self.Survey.lbin, self.Survey.zz_integr)
            pk_exp = pk_exp*dz_rescale[np.newaxis, np.newaxis, :]*dz_rescale[np.newaxis, np.newaxis, :]
            pk_exp_extr = pk_exp_extr*dz_rescale[np.newaxis, :]*dz_rescale[np.newaxis, :]
        else:    
            pk_exp, pk_exp_extr = self.BaccoEmuClass.get_heft(params_dic, self.k, self.Survey.lbin, self.Survey.zz_integr)  
        return pk_exp, pk_exp_extr
    
    def get_pgg_heft_bias(self, params_dic):
        """Calculate the galaxy-galaxy power spectrum with HEFT Lagrangian bias expansion,
        in units of (Mpc/h)^3.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the bias parameters.
  
        Returns:
        -------
        pgm : numpy.ndarray
            The galaxy-galaxy power spectrum.
        """
        pk_exp, pk_exp_extr = self.pk_exp, self.pk_exp_extr
        bL1 = np.array([params_dic['b1L_'+str(i+1)] for i in range(self.Survey.nbin)])
        bL2 = np.array([params_dic['b2L_'+str(i+1)] for i in range(self.Survey.nbin)])
        bs2 = np.array([params_dic['bs2L_'+str(i+1)] for i in range(self.Survey.nbin)])
        blapl = np.array([params_dic['blaplL_'+str(i+1)] for i in range(self.Survey.nbin)])    
        p_dmdm = pk_exp[0, :, :]        
        p_dmd1 = pk_exp[1, :, :]    
        p_dmd2 = pk_exp[2, :, :]        
        p_dms2 = pk_exp[3, :, :]        
        p_dmk2 = pk_exp[4, :, :]      
        p_d1d1 = pk_exp[5, :, :]     
        p_d1d2 = pk_exp[6, :, :]        
        p_d1s2 = pk_exp[7, :, :]         
        p_d1k2 = pk_exp[8, :, :]         
        p_d2d2 = pk_exp[9, :, :]        
        p_d2s2 = pk_exp[10, :, :]         
        p_d2k2 = pk_exp[11, :, :]
        p_s2s2 = pk_exp[12, :, :] 
        p_s2k2 = pk_exp[13, :, :] 
        p_k2k2 = pk_exp[14, :, :] 
        pgg = (p_dmdm[:,:,None,None]  +
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
        pgg_extr = (1.+bL1[None,None, :,None])*(1.+bL1[None,None, None, :]) * pk_exp_extr[:,:,None,None]
        pgg += pgg_extr 
        # return dimension (lbin, z_integr, bin_i, bin_j)
        return pgg
    
    def get_pgm_heft_bias(self, params_dic):
        """Calculate the galaxy-matter power spectrum with HEFT Lagrangian bias expansion,
        in units of (Mpc/h)^3.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the bias parameters.
  
        Returns:
        -------
        pgm : numpy.ndarray
            The matter-galaxy power spectrum.
        """
        pk_exp, pk_exp_extr = self.pk_exp, self.pk_exp_extr
        bL1 = np.array([params_dic['b1L_'+str(i+1)] for i in range(self.Survey.nbin)])
        bL2 = np.array([params_dic['b2L_'+str(i+1)] for i in range(self.Survey.nbin)])
        bs2 = np.array([params_dic['bs2L_'+str(i+1)] for i in range(self.Survey.nbin)])
        blapl = np.array([params_dic['blaplL_'+str(i+1)] for i in range(self.Survey.nbin)])    
        p_dmdm = pk_exp[0, :, :]        
        p_dmd1 = pk_exp[1, :, :]    
        p_dmd2 = pk_exp[2, :, :]        
        p_dms2 = pk_exp[3, :, :]        
        p_dmk2 = pk_exp[4, :, :]      
        pgm = (p_dmdm[:,:,None]  +
                bL1[None,None,:] * p_dmd1[:,:,None] +
                bL2[None,None,:] * p_dmd2[:,:,None] +
                bs2[None,None,:] * p_dms2[:,:,None] +
                blapl[None,None,:] * p_dmk2[:,:,None])   
        pgm_extr = (1.+bL1[None,None,:]) * pk_exp_extr[:,:,None]
        pgm += pgm_extr
        # return dimension (lbin, z_integr, bin_i)
        return pgm
        

    def get_gg_kernel(self, params_dic):
        r"""Calculate the galaxy-galaxy lensing kernel in units of h/Mpc.

        .. math::
            W_i^{G} = n_i(z)\frac{H(z)}{c}\, ,

        where n_i(z) is the lens distribution.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the cosmological parameters.

        Returns:
        -------
        numpy.ndarray
            The galaxy-galaxy lensing kernel with shape (1, zbin_integr, nbin).
        """
        eta_z_l = self.get_n_of_z(self.Survey.eta_z_l, self.deltaz_l) 
        w_g = np.zeros((self.Survey.zbin_integr, self.Survey.nbin), 'float64')
        w_g = self.ez[:, None] * H0_h_c * eta_z_l
        # add an extra dimension, now (ell, z_integr, bin_i)
        w_g = w_g[None, :, :]
        return w_g

    def get_cell_galclust(self):
        r"""Compute the galaxy clustering angular power spectrum:

        .. math::
            C^{\rm GG}_{ij}(\ell) = c \int \mathrm{d}z \frac{W^{\rm G}_i(z) W^{\rm G}_j(z)}{H(z) r^2_{\rm com}(z)} P_{\rm gg}(k(\ell, z), z)

        This function calculates the galaxy clustering angular power spectrum given the 
        cosmological parameters and the precomputed power spectra.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters for the computation.

        Returns:
        -------
        cl_gg : numpy.ndarray
            Galaxy clustering angular power spectrum with dimensions (ell, bin_i, bin_j).
        """
        # compute integrand with the dimensions of (ell, z_integr, bin_i, bin_j)
        cl_gg_int = self.w_g[:,:,:,None] *self. w_g[:,: , None, :] * self.pgg / self.ez[None,:,None,None] / self.rz[None,:,None,None] / self.rz[None,:,None,None] / H0_h_c    
        # integrate along the z_integr direction
        cl_gg = trapezoid(cl_gg_int, self.Survey.zz_integr, axis=1)[:self.Survey.nell_gc, :, :]
        # add noise
        for i in range(self.Survey.nbin):
            cl_gg[:, i, i] += self.Survey.noise['GG']
        return cl_gg 
    
    def get_pk_cross_nla(self, params_dic):
        r"""Computes the galaxy-intrinsic alignment power spectrum for the e-zNLA model.

        .. math::
            P_{\rm g IA}(z, k) = f_{\rm IA}(z) b_g(z, k) P_{\rm mm}(z, k) 

        where:

        - :math:`b_g(z)` is the galaxy bias.
        - :math:`f_{\rm IA}(z)` is the intrinsic alignment parameters (see get_pk_nla).
        - :math:`P_{\rm mm}(z, k)` is the matter power spectrum with baryons.

        Returns:
        --------
        numpy.ndarray
            The galaxy-intrinsic alignment power spectrum for the TATT model.

        """
        # if ia dependens on the photo-z bin
        # then change to factor_nla[:, :, :, None]*pgm[:, :, None, :]
        return self.factor_nla[:, :, None, None] * self.pgm[:, :, None, :]

    
    def get_pk_cross_tatt(self, params_dic):
        r"""Computes the galaxy-intrinsic alignment power spectrum for the TATT model.

        .. math::
            P_{\rm g IA}(z, k) = b_g(z) \left[ C_{1}P_{\rm mm}(z, k) + C_{1\delta}D(z)^{4}[ A_{0|0E}(k) +  C_{0|0E}(k)] + C_{2}D(z)^{4}[ A_{0|E2}(k) + B_{0|E2}(k)] \right]

        where:

        - :math:`D(z)` is the growth factor normalized at :math:`z=0`.
        - :math:`b_g(z)` is the galaxy bias.
        - :math:`C_{1}`, :math:`C_{1\delta}`, and :math:`C_{2}` are the intrinsic alignment parameters.
        - :math:`P_{\rm mm}(z, k)` is the matter power spectrum.
        - :math:`A_{0|0E}(k)`, :math:`C_{0|0E}(k)`, :math:`A_{0|E2}(k)`, and :math:`B_{0|E2}(k)` are the FAST-PT terms.

        Returns:
        --------
        numpy.ndarray
            The galaxy-intrinsic alignment power spectrum for the TATT model.
        """
        bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.Survey.nbin)])
        # return dimension (lbin, z_integr, bin_i)
        # if ia dependens on the photo-z bin
        # then change to bias1[None, None, None, :]*pk_delta_ia[:, :, :, None]
        pk_gal_ia =   bias1[None, None, None, :] * self.pk_delta_ia[:, :, None, None]
        return pk_gal_ia
    
    def get_cell_cross(self):
        r"""Compute the galaxy-galaxy lensing or cross-correlation angular power spectrum given by:

        .. math::
            C^{\rm LG}_{ij}(\ell) = c \int \mathrm{d}z \frac{\left( W^{\gamma}_i(z)P_{\rm gm}(k(\ell, z), z)+ W^{\rm IA}_i(z)P_{\rm IAg}(k(\ell, z), z) \right) W^{\rm G}_j(z)}{H(z) r^2_{\rm com}(z)} 
        and 

        .. math::
            C^{\rm GL}_{ij}(\ell) = c \int \mathrm{d}z \frac{W^{\rm G}_i(z) \left( W^{\gamma}_j(z) P_{\rm gm} (k(\ell, z), z)+ W^{\rm IA}_j(z) P_{\rm gIA}(k(\ell, z), z)\right)}{H(z) r^2_{\rm com}(z)}
        
        where:

        - :math:`W^{\gamma}`, :math:`W^{\rm IA}`, :math:`W^{\rm G}`  are the lensing, intrinsic alignment and clustering kernels.
        - :math:`H(z)` is the expansion function.
        - :math:`r_{\rm com}(z)` is the comoving distance.
        - :math:`P_{\rm gm}(z, k)`, :math:`P_{\rm g IA}(z, k)` are the galaxy-matter, and galaxy-IA power spectra.
            
        Returns:
        -------
        cl_lg : numpy.ndarray
            The galaxy-galaxy lensing angular power spectrum with dimensions (ell, bin_i, bin_j).
        cl_gl : numpy.ndarray
            The galaxy-lensing angular power spectrum with dimensions (ell, bin_i, bin_j).
        """
        # compute components of the integrand
        kernel_ia_gal = self.w_ia[:, :, :, None] * self.pk_gal_ia
        kernel_wl_gal = self.w_gamma[:, :, :, None] * self.pgm[:,:,None,:]
        # compute integrand with dimensions (ell, z_integr, bin_i, bin_j)
        cl_lg_int = (kernel_wl_gal + kernel_ia_gal) * self.w_g[:, :, None, :] / self.ez[None,:,None,None] / self.rz[None,:,None,None] / self.rz[None,:,None,None] / H0_h_c
        # integrate along the z-integr direction
        cl_lg = trapezoid(cl_lg_int, self.Survey.zz_integr, axis=1)[:self.Survey.nell_xc,:,:]
        # transpose LG to get GL
        cl_gl = np.transpose(cl_lg, (0, 2, 1))  
        return cl_lg, cl_gl
    
    def compute_shear(self, params_dic):
        """Compute the shear power spectrum given a set of cosmological parameters.
        Use this when in likelihood computations.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the cosmological parameters.

        Returns:
        -------
        numpy.ndarray
            The shear angular power spectrum.
        """
        # compute background
        self.ez, self.rz, self.k = self.get_ez_rz_k(params_dic)
        # compute normalized growth factor, of size (z_integr)
        self.dz, _ = self.StructureEmu.get_growth(params_dic, self.Survey.zz_integr)
        # get redshift uncertainties
        self.deltaz_s, self.deltaz_l = self.get_deltaz(params_dic)
        # compute matter-matter power spectrum
        self.pmm, _ = self.get_pmm(params_dic)
        # compute weak lensing kernels 
        self.w_gamma = self.get_wl_kernel(params_dic)
        # compute ia kernels 
        self.w_ia = self.get_ia_kernel(params_dic) 
        # compute intrinsinc alignment components of the integrand
        self.pk_delta_ia, self.pk_iaia = self.get_pk_ia(params_dic)
        return self.get_cell_shear()
    

    def compute_galclust(self, params_dic):
        """
        Compute the photo-GC correlation functions. Use this for likelihood computations.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing cosmological and survey parameters.

        Returns:
        -------
        cl_gg : ndarray
            Photometric galaxy clustering angular power spectra of length Survey.nell_gc.
        """
        # compute background
        self.ez, self.rz, self.k = self.get_ez_rz_k(params_dic)
        # compute growth factor
        self.dz, _ = self.StructureEmu.get_growth(params_dic, self.Survey.zz_integr)
        # get redshift uncertainties
        self.deltaz_s, self.deltaz_l = self.get_deltaz(params_dic)
        if self.flag_heft:
            self.pk_exp, self.pk_exp_extr = self.get_heft_pk_exp(params_dic)
        else:
            self.pmm, _ = self.get_pmm(params_dic)  
        self.pgg = self.get_pgg(params_dic)
        self.w_g = self.get_gg_kernel(params_dic)
        # compute photometric galaxy clustring angular power spectra cl_gg(l, bin_i, bin_j)
        # window function w_g(l,z,bin) in units of h/Mpc
        cl_gg = self.get_cell_galclust()    
        return cl_gg

    def compute_3x2pt(self, params_dic):
        """
        Compute the 3x2pt correlation functions. Use this when all 3 correlation funcions are required,
        as it is designed to compute properties needed in different computations just once.
        Attention: order of IA-components is importnant!

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing cosmological and survey parameters.

        Returns:
        -------
        cl_ll : ndarray
            Weak lensing angular power spectra of length Survey.nell_wl.
        cl_gg : ndarray
            Photometric galaxy clustering angular power spectra of length Survey.nell_gc.
        cl_lg : ndarray
            Cross-correlated galaxy-galaxy lensing angular power spectra of length Survey.nell_xc.
        cl_gl : ndarray
            Cross-correlated galaxy-galaxy lensing angular power spectra of length Survey.nell_xc.
        """
        # compute background
        self.ez, self.rz, self.k = self.get_ez_rz_k(params_dic)
        # compute growth factor
        self.dz, _ = self.StructureEmu.get_growth(params_dic, self.Survey.zz_integr)
        # get redshift uncertainties
        self.deltaz_s, self.deltaz_l = self.get_deltaz(params_dic)
        # compute matter-matter power spectrum
        self.pmm, bar_boost = self.get_pmm(params_dic)
        # compute galaxy-matter power spectrum
        if self.flag_heft:
            self.pk_exp, self.pk_exp_extr = self.get_heft_pk_exp(params_dic)
            self.pgm = self.get_pgm(params_dic)*np.sqrt(bar_boost[:, :, None]) 
        else:
            self.pgm = self.get_pgm(params_dic)  
        # compute galaxy-galaxy power spectrum     
        self.pgg = self.get_pgg(params_dic)
        # compute properties used in various calculations just once:
        self.w_gamma = self.get_wl_kernel(params_dic)
        self.w_ia = self.get_ia_kernel(params_dic)
        self.pk_delta_ia, self.pk_iaia = self.get_pk_ia(params_dic)
        self.w_g = self.get_gg_kernel(params_dic)
        # order is important: call this only after get_pk_ia
        self.pk_gal_ia = self.get_pk_cross_ia(params_dic)
        # compute weak lensing angular power spectra cl_ll(l, bin_i, bin_j)
        # window function w_l(l,z,bin) in units of h/Mpc
        cl_ll = self.get_cell_shear()
        # compute photometric galaxy clustring angular power spectra cl_gg(l, bin_i, bin_j)
        # window function w_g(l,z,bin) in units of h/Mpc
        cl_gg = self.get_cell_galclust()    
        # compute cross-correlated or galaxy-galaxy lensing angular power spectra cl_lg(l, bin_i, bin_j) and cl_gl(l, bin_i, bin_j)
        cl_lg, cl_gl = self.get_cell_cross() 
        return cl_ll, cl_gg, cl_lg, cl_gl  
    
    def get_deltaz(self, params):
        deltaz_s = deltaz_l = None
        if 'deltaz_1' in params:     
            deltaz_s = np.array([params[f'deltaz_{i+1}'] for i in range(self.Survey.nbin)])
            deltaz_l = deltaz_s.copy()
        if 'deltaz_1_s' in params:  
            deltaz_s = np.array([params[f'deltaz_{i+1}_s'] for i in range(self.Survey.nbin)])  
        if 'deltaz_1_l' in params:    
            deltaz_l = np.array([params[f'deltaz_{i+1}_l'] for i in range(self.Survey.nbin)])  
        return deltaz_s, deltaz_l

    def compute_data_matrix_3x2pt(self, params_dic):
        r"""This function computes the 3x2pt angular power spectra and composes a matrix 
        for computing the likelihood with determinants. The resulting matrix is structured as:
        
        .. math::

            \begin{pmatrix}
            C_{LL} & C_{LG} \\
            C_{GL} & C_{GG}
            \end{pmatrix}

        for the same scale-cuts per-redshift bin, i.e. ell_max_gc. We also compute a
        "high"-matrix for ell>ell_max_gc with weak lensing angular power spectra, 
        as we assume that ell_max_wl>ell_max_gc.  
        These matrices are later used in the likelihood computation via determinants.         

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters required for computing the 3x2pt angular power spectra.

        Returns:
        -------
        cov_theory : numpy.ndarray
            The data matrix for the 3x2pt angular power spectra.
        cov_theory_high : numpy.ndarray
            The data matrix for ell > ell_max_gc with weak lensing angular power spectra.
        """
        # compute 3x2pt angular power spectra
        cl_ll, cl_gg, cl_lg, cl_gl = self.compute_3x2pt(params_dic)
        # compose a matrix for computing likelihood with determinants
        # and a "high"-matrix where
        # ell_jump is ell_max_gc
        cov_theory, cov_theory_high = build_data_matrix_3x2pt(cl_ll, cl_gg, cl_lg, cl_gl, 
                                                              self.Survey.l_wl, self.Survey.l_gc, self.Survey.l_xc,
                                                              self.Survey.ells_wl, self.Survey.ells_gc, self.Survey.ell_jump, 
                                                              self.Survey.nbin)
        # dimensions: 
        # cov_theory (len(all_int_ell_wl), 2 * nbin, 2 * nbin)
        # cov_theory_high (len(all_int_ell_wl) - ell_jump), nbin, nbin)
        return cov_theory, cov_theory_high    

    def compute_data_vector_3x2pt(self, params_dic):
        """This function calculates the 3x2pt angular power spectra and constructs a data vector
        by combining the power spectra for different bins. It then applies a mask to the data
        vector based on the survey's scale-cuts per photo-z bin.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters required for computing the 3x2pt angular power spectra.

        Returns:
        -------
        data_vector_masked : numpy.ndarray
            The masked data vector containing the 3x2pt angular power spectra for the given parameters.
        """
        # compute 3x2pt angular power spectra
        cl_ll, cl_gg, cl_lg, cl_gl = self.compute_3x2pt(params_dic)
        data_vector = np.zeros(((self.Survey.nell_wl+self.Survey.nell_gc)*self.Survey.nbin_flat+self.Survey.nell_xc*self.Survey.nbin**2), 'float64')
        # construct a vector
        idx_start = 0
        start_lg = self.Survey.nell_wl*self.Survey.nbin_flat
        start_gc = start_lg+self.Survey.nell_xc*self.Survey.nbin**2
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(bin1, self.Survey.nbin):
                data_vector[idx_start*self.Survey.nell_wl:(idx_start+1)*self.Survey.nell_wl] = cl_ll[:, bin1, bin2]
                data_vector[idx_start*self.Survey.nell_gc+start_gc:(idx_start+1)*self.Survey.nell_gc+start_gc] = cl_gg[:, bin1, bin2]
                idx_start += 1
        idx_start = 0
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                # for compute_covariance_3x2pt:
                # data_vector[idx_start*self.Survey.nell_xc+start_lg:(idx_start+1)*self.Survey.nell_xc+start_lg] = cl_lg[:, bin1, bin2] 
                # for direct comparison with cosmosis:
                data_vector[idx_start*self.Survey.nell_xc+start_lg:(idx_start+1)*self.Survey.nell_xc+start_lg] = cl_gl[:, bin1, bin2]
                idx_start += 1
        # apply mask for different scale-cuts per photo-z bin        
        data_vector_masked = data_vector[self.Survey.mask_data_vector_3x2pt]
        return data_vector_masked
    
    def compute_covariance_3x2pt(self, params_dic):
        r"""Compute the gaussian (only diagonal components with :math:`\ell=\ell'`) covariance matrix for the 3x2pt analysis.
        This function calculates the covariance matrix for the 3x2pt analysis, which includes 
        weak lensing (LL), galaxy clustering (GG), and their cross-correlation (LG and GL). 
        The covariance matrix is computed for different bins and multipoles as follows

        .. math::
            \mathrm{Cov}[C_{ij}^{AB}(\ell)C_{kl}^{CD}(\ell)]=\frac{1}{2f_{\rm sky}(2 \ell+1) \Delta \ell} 
            \left( C^{AC}_{ik}C^{BD}_{jl}+C^{AD}_{il}C^{BC}_{jk}\right)\, ,

        where :math:`A,B,C,D = \{L, G\}` and :math:`i,j` correspond to redshift-bins.
        
        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters required for the computation of the 3x2pt 
            angular power spectra.
        Returns:
        -------
        cov_flat_masked : numpy.ndarray
            The masked covariance matrix for the 3x2pt analysis.
        """

        # compute all angular power spectra
        cl_ll, cl_gg, cl_lg, cl_gl = self.compute_3x2pt(params_dic)
        # l_wl, l_xc, l_gc are centers of the lbins
        # now we compute at the same number of ell-bins for all probes
        # and then apply a mask
        denom = ((2*self.Survey.ell+1)*self.Survey.fsky*self.Survey.d_ell_bin)
        cov_flat = np.zeros((2*self.Survey.lbin*self.Survey.nbin_flat+self.Survey.lbin*self.Survey.nbin**2, 2*self.Survey.lbin*self.Survey.nbin_flat+self.Survey.lbin*self.Survey.nbin**2), 'float64')
        counter_x = 0
        counter_y = 0
        start_lg = self.Survey.lbin*self.Survey.nbin_flat 
        start_gc = start_lg+self.Survey.lbin*self.Survey.nbin**2
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(bin1, self.Survey.nbin):
                for bin3 in range(self.Survey.nbin):
                    for bin4 in range(bin3, self.Survey.nbin):
                        for ell_i in range(self.Survey.lbin):
                            #ABCD=LLLL
                            cov_flat[self.Survey.lbin*counter_y+ell_i, self.Survey.lbin*counter_x+ell_i] = (cl_ll[ell_i, bin1, bin3] * cl_ll[ell_i, bin2, bin4] +
                            cl_ll[ell_i, bin1, bin4] * cl_ll[ell_i, bin2, bin3])/denom[ell_i] 

                            #ABCD=LLGG
                            cov_flat[self.Survey.lbin*counter_y+ell_i, start_gc+self.Survey.lbin*counter_x+ell_i] = (cl_lg[ell_i, bin1, bin3] * cl_lg[ell_i, bin2, bin4] +
                            cl_lg[ell_i, bin1, bin4] * cl_lg[ell_i, bin2, bin3])/denom[ell_i]  

                            #ABCD=GGLL
                            cov_flat[start_gc+self.Survey.lbin*counter_y+ell_i, self.Survey.lbin*counter_x+ell_i] = (cl_gl[ell_i, bin1, bin3] * cl_gl[ell_i, bin2, bin4] +
                            cl_gl[ell_i, bin1, bin4] * cl_gl[ell_i, bin2, bin3])/denom[ell_i] 
                            
                            #ABCD=GGGG
                            cov_flat[start_gc+self.Survey.lbin*counter_y+ell_i, start_gc+self.Survey.lbin*counter_x+ell_i] = (cl_gg[ell_i, bin1, bin3] * cl_gg[ell_i, bin2, bin4] +
                            cl_gg[ell_i, bin1, bin4] * cl_gg[ell_i, bin2, bin3])/denom[ell_i]    

                        if counter_x<self.Survey.nbin_flat-1:
                            counter_x+=1
                        else:
                            counter_x = 0
                            counter_y+=1       
        counter_x = 0
        counter_y = 0
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(bin1, self.Survey.nbin):
                for bin3 in range(self.Survey.nbin):
                    for bin4 in range(self.Survey.nbin):
                        for ell_i in range(self.Survey.lbin):
                            #ABCD=LLLG
                            cov_flat[self.Survey.lbin*counter_y+ell_i, start_lg+self.Survey.lbin*counter_x+ell_i] = (cl_ll[ell_i, bin1, bin3] * cl_lg[ell_i, bin2, bin4] +
                            cl_lg[ell_i, bin1, bin4] * cl_ll[ell_i, bin2, bin3])/denom[ell_i]   
                            #ABCD=GGLG
                            cov_flat[start_gc+self.Survey.lbin*counter_y+ell_i, start_lg+self.Survey.lbin*counter_x+ell_i] = (cl_gl[ell_i, bin1, bin3] * cl_gg[ell_i, bin2, bin4] +
                            cl_gg[ell_i, bin1, bin4] * cl_gl[ell_i, bin2, bin3])/denom[ell_i] 
                        if counter_x<self.Survey.nbin**2-1:
                            counter_x+=1
                        else:
                            counter_x = 0
                            counter_y+=1  
        counter_x = 0
        counter_y = 0
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                for bin3 in range(self.Survey.nbin):
                    for bin4 in range(self.Survey.nbin):
                        for ell_i in range(self.Survey.lbin):
                            #ABCD=LGLG
                            cov_flat[start_lg+self.Survey.lbin*counter_y+ell_i, start_lg+self.Survey.lbin*counter_x+ell_i] = (cl_ll[ell_i, bin1, bin3] * cl_gg[ell_i, bin2, bin4] +
                            cl_lg[ell_i, bin1, bin4] * cl_gl[ell_i, bin2, bin3])/denom[ell_i] 
                        if counter_x<self.Survey.nbin**2-1:
                            counter_x+=1
                        else:
                            counter_x = 0
                            counter_y+=1      
                    
        counter_x = 0
        counter_y = 0
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                for bin3 in range(self.Survey.nbin):
                    for bin4 in range(bin3, self.Survey.nbin):
                        for ell_i in range(self.Survey.lbin):            
                            #ABCD=LGLL
                            cov_flat[start_lg+self.Survey.lbin*counter_y+ell_i, self.Survey.lbin*counter_x+ell_i] = (cl_ll[ell_i, bin1, bin3] * cl_gl[ell_i, bin2, bin4] +
                            cl_ll[ell_i, bin1, bin4] * cl_gl[ell_i, bin2, bin3])/denom[ell_i]     
                            #ABCD=LGGG
                            cov_flat[start_lg+self.Survey.lbin*counter_y+ell_i, start_gc+self.Survey.lbin*counter_x+ell_i] = (cl_lg[ell_i, bin1, bin3] * cl_gg[ell_i, bin2, bin4] +
                            cl_lg[ell_i, bin1, bin4] * cl_gg[ell_i, bin2, bin3])/denom[ell_i] 
                        if counter_x<self.Survey.nbin_flat-1:
                            counter_x+=1
                        else:
                            counter_x = 0
                            counter_y+=1         

        print('cov_flat: ', cov_flat.shape)                                      
        cov_flat_masked = cov_flat[self.Survey.mask_cov_3x2pt]
        print('cov_flat_masked: ', cov_flat_masked.shape)  
        return cov_flat_masked 
    
    def get_cov_diag_ijkl(self, theory_spectra, name1, name2, bin_ij, bin_kl): 
        c_ij_12 = theory_spectra[name1]
        c_kl_34 = theory_spectra[name2]

        type_1, type_2 = c_ij_12['types']
        type_3, type_4 = c_kl_34['types']

        i,j = bin_ij
        k,l = bin_kl

        bin_pairs = [ (i,k), (j,l), (i,l), (j,k) ]
        type_pairs = [ (type_1,type_3), (type_2,type_4), (type_1,type_4), (type_2,type_3) ]
        c_ells = []
        for bin_pair,type_pair in zip(bin_pairs, type_pairs):
            bin1, bin2 = bin_pair
            t1, t2 = type_pair
            types = (t1, t2)
            if types not in self.types:
                #If we don't have a spectra with these types, we probably 
                #have an equivalent one with the types the other way round.
                #In this case, we also need to swap bin1 and bin2, because
                #we are accessing e.g. C^{ik}_{13} via C^{ki}_{31}.
                types = (types[1], types[0])
                bin1, bin2 = bin2, bin1
            s = theory_spectra[ self.types.index( types ) ]
            c_ells.append(s['cls'][:, bin1, bin2])

        cl2_sum = c_ells[0]*c_ells[1] + c_ells[2]*c_ells[3]
        # alternative for covariance computed at the center of bins:
        # denom = ((2*self.Survey.ell+1)*self.Survey.fsky*self.Survey.d_ell_bin) 
        denom = self.Survey.fsky * (2*self.Survey.ell+1)
        return ( cl2_sum ) / denom

        

    
    def compute_covariance_cosmosis_3x2pt(self, params_dic):
        n_ell = self.Survey.lbin
        ell_lims = self.Survey.ell_bin_edges
        ell_all_int = np.arange(self.Survey.lmin, self.Survey.lmax+1).astype(int)
        self.Survey.lbin = 100
        ell_vals = np.logspace(log10(self.Survey.lmin), log10(self.Survey.lmax), num=self.Survey.lbin, endpoint=True) 
        self.Survey.ell = self.Survey.l_wl = self.Survey.l_xc = self.Survey.l_gc = ell_vals
        self.Survey.nell_wl = self.Survey.nell_xc = self.Survey.nell_gc = len(self.Survey.l_wl)

        cl_ll_, cl_gg_, cl_lg_, _ = self.compute_3x2pt(params_dic)
        cl_ll = build_data_matrix(cl_ll_, ell_vals, ell_all_int, self.Survey.nbin)
        cl_gg = build_data_matrix(cl_gg_, ell_vals, ell_all_int, self.Survey.nbin)
        cl_lg = build_data_matrix(cl_lg_, ell_vals, ell_all_int, self.Survey.nbin)
        self.Survey.ell = self.Survey.l_wl = self.Survey.l_xc = self.Survey.l_gc = ell_all_int
        self.Survey.lbin = len(ell_all_int)
        self.Survey.nell_wl = self.Survey.nell_xc = self.Survey.nell_gc = len(self.Survey.l_wl)

        cl_ll_dic = {'bin_pairs': [(i, j) for i in range(self.Survey.nbin) for j in range(i, self.Survey.nbin)], 
                    'cls': cl_ll,
                    'is_auto': True,
                    'name': 'LL',
                    'types': (0, 0)
                    }
        cl_gg_dic = {'bin_pairs': [(i, j) for i in range(self.Survey.nbin) for j in range(i, self.Survey.nbin)], 
                    'cls': cl_gg,
                    'is_auto': True,
                    'name': 'GG',
                    'types': (1, 1)
                    }
        cl_lg_dic = {'bin_pairs': [(j, i) for i in range(self.Survey.nbin) for j in range(self.Survey.nbin)], #changed from (i, j) to recreate cosmosis
                    'cls': cl_lg,
                    'is_auto': False,
                    'name': 'LG',
                    'types': (0, 1)
                    }
        theory_spectra = [cl_ll_dic, cl_lg_dic, cl_gg_dic]
        self.types = [cl_ll_dic['types'], cl_lg_dic['types'], cl_gg_dic['types']]
        # Get the starting index in the full datavector for each spectrum
        # this will be used later for adding covariance blocks to the full matrix.
        cl_lengths = [n_ell*self.Survey.nbin_flat, n_ell*self.Survey.nbin**2, n_ell*self.Survey.nbin_flat]
        cl_starts = []
        for i in range(3):
            cl_starts.append( int(sum(cl_lengths[:i])) )
        covmat = np.zeros((2*n_ell*self.Survey.nbin_flat+n_ell*self.Survey.nbin**2, 2*n_ell*self.Survey.nbin_flat+n_ell*self.Survey.nbin**2), 'float64')
        # Now loop through pairs of Cls and pairs of bin pairs filling the covariance matrix
        for i_cl in range(3):
            cl_spec_i = theory_spectra[i_cl]
            for j_cl in range(i_cl, 3):
                cl_spec_j = theory_spectra[j_cl]
                cov_blocks = {} #collect cov_blocks in this dictionary
                for i_bp, bin_pair_i in enumerate(cl_spec_i['bin_pairs']):
                     for j_bp, bin_pair_j in enumerate(cl_spec_j['bin_pairs']):
                        print(f"Computing covariance {i_cl},{j_cl} pairs <{bin_pair_i} {bin_pair_j}>")
                        # First check if we've already calculated this
                        if (i_cl == j_cl) and cl_spec_i['is_auto'] and ( j_bp < i_bp ):
                            cl_var_binned = cov_blocks[j_bp, i_bp]
                        else:    
                            #First calculate the unbinned Cl covariance
                            cl_var_unbinned = self.get_cov_diag_ijkl( theory_spectra, i_cl, 
                                    j_cl, bin_pair_i, bin_pair_j)
                            #Now bin this diaginal covariance
                            #Var(binned_cl) = \sum_l Var(w_l^2 C(l)) / (\sum_l w_l)^2
                            #where w_l = 2*l+1
                            cl_var_binned = np.zeros(n_ell)
                            for ell_bin, (ell_low, ell_high) in enumerate(zip(ell_lims[:-1], ell_lims[1:])):
                                #Get the ell values for this bin:
                                ell_vals_bin = np.arange(ell_low, ell_high).astype(int)
                                #Get the indices in cl_var_binned these correspond to:
                                ell_vals_bin_inds = ell_vals_bin - int(ell_lims[0])
                                cl_var_unbinned_bin = cl_var_unbinned[ell_vals_bin_inds]
                                cl_var_binned[ell_bin] = np.sum((2*ell_vals_bin+1)**2 * 
                                    cl_var_unbinned_bin) / np.sum(2*ell_vals_bin+1)**2
                            # alternative for covariance computed at the center of bins:    
                            # cl_var_binned = self.get_cov_diag_ijkl(theory_spectra, i_cl, 
                            #        j_cl, bin_pair_i, bin_pair_j)    
                            cov_blocks[i_bp, j_bp] = cl_var_binned

                        # Now work out where this goes in the full covariance matrix
                        # and add it there.
                        inds_i = np.arange( cl_starts[i_cl] + n_ell*i_bp, 
                            cl_starts[i_cl] + n_ell*(i_bp+1) )
                        inds_j = np.arange( cl_starts[j_cl] + n_ell*j_bp, 
                            cl_starts[j_cl] + n_ell*(j_bp+1) )
                        cov_inds = np.ix_( inds_i, inds_j )
                        covmat[ cov_inds ] = np.diag(cl_var_binned)
                        cov_inds_T = np.ix_( inds_j, inds_i )
                        covmat[ cov_inds_T ] = np.diag(cl_var_binned)        
        print('cov_flat: ', covmat.shape)                                      
        covmat_masked = covmat[self.Survey.mask_cov_3x2pt]
        print('cov_flat_masked: ', covmat_masked.shape)  
        self.Survey.ell = self.Survey.l_wl = self.Survey.l_xc = self.Survey.l_gc = self.Survey.l_wl_bin_centers
        self.Survey.lbin = n_ell
        self.Survey.nell_wl = self.Survey.nell_xc = self.Survey.nell_gc = n_ell
        return covmat_masked
    
    
    def compute_data_matrix_wl(self, params_dic):
        """This function calculates the weak lensing angular power spectra using the provided parameters
        and constructs the corresponding covariance matrix. This is later used in lieklihood with determinants.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters required for computing the shear power spectra.

        Returns:
        -------
        cov_theory : numpy.ndarray
            The computed data matrix for the weak lensing data.
        """
        # compute weak lensing angular power spectra
        cl_wl = self.compute_shear(params_dic)
        # construct matrix
        cov_theory = build_data_matrix(cl_wl, self.Survey.l_wl, self.Survey.ells_wl, self.Survey.nbin)
        return cov_theory
    
    def compute_data_vector_wl(self, params_dic):
        """This function calculates the weak lensing angular power spectra and constructs
        the data vector for weak lensing, applying a mask to the resulting data vector.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters required for the computation of the shear.

        Returns:
        -------
        data_vector_masked : numpy.ndarray
            The masked data vector for weak lensing, containing the angular power spectra
            for the specified survey configuration.
        """
        # compute weak lensing angular power spectra
        cl_wl = self.compute_shear(params_dic)
        data_vector = np.zeros((self.Survey.nell_wl*self.Survey.nbin_flat), 'float64')
        idx_start = 0
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(bin1, self.Survey.nbin):
                data_vector[idx_start*self.Survey.nell_wl:(idx_start+1)*self.Survey.nell_wl] = cl_wl[:, bin1, bin2]
                idx_start += 1
        data_vector_masked = data_vector[self.Survey.mask_data_vector_wl]
        return data_vector_masked
    
    def compute_covariance_wl(self, params_dic):
        r"""Compute the Gaussian (only diagonal components with :math:`\ell=\ell'`) covariance matrix for the weak lensing (WL) analysis.
        This function calculates the covariance matrix for the weak lensing analysis, which includes 
        the auto-correlation of shear (LL). The covariance matrix is computed for different bins and multipoles as follows:

        .. math::
            \mathrm{Cov}[C_{ij}^{LL}(\ell)C_{kl}^{LL}(\ell)]=\frac{1}{2f_{\rm sky}(2 \ell+1) \Delta \ell} 
            \left( C^{LL}_{ik}C^{LL}_{jl}+C^{LL}_{il}C^{LL}_{jk}\right)\, ,

        where :math:`L` corresponds to weak lensing and :math:`i,j` correspond to redshift bins.
        
        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters required for the computation of the weak lensing 
            angular power spectra.
        Returns:
        -------
        cov_flat_masked : numpy.ndarray
            The masked covariance matrix for the weak lensing analysis.
        """
        # compute weak lensing angular power spectra
        cl_wl = self.compute_shear(params_dic)
        # l_wl are centers of the lbins
        denom = ((2*self.Survey.l_wl+1)*self.Survey.fsky*self.Survey.d_ell_bin_cut_wl)
        cov_flat = np.zeros((self.Survey.nell_wl*self.Survey.nbin_flat, self.Survey.nell_wl*self.Survey.nbin_flat), 'float64')
        counter_x = 0
        counter_y = 0
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(bin1, self.Survey.nbin):
                for bin3 in range(self.Survey.nbin):
                    for bin4 in range(bin3, self.Survey.nbin):
                        for i in range(self.Survey.nell_wl):
                            cov_flat[self.Survey.nell_wl*counter_y+i, self.Survey.nell_wl*counter_x+i] = (cl_wl[i, bin1, bin3] * cl_wl[i, bin2, bin4] +
                            cl_wl[i, bin1, bin4]* cl_wl[i, bin2, bin3])/denom[i] 
                        if counter_x<self.Survey.nbin_flat-1:
                            counter_x+=1
                        else:
                            counter_x = 0
                            counter_y+=1                       
        print('cov_flat: ', cov_flat.shape)                                      
        cov_flat_masked = cov_flat[self.Survey.mask_cov_wl]
        print('cov_flat_masked: ', cov_flat_masked.shape)  
        return cov_flat_masked 
    

    def compute_data_matrix_gc(self, params_dic):
        """This function computes the angular power spectra for galaxy clustering
        using the provided parameters dictionary, and then constructs the 
        data matrix based on the computed power spectra and survey 
        parameters. This is later used in likelihood with determinants.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters required for computing the galaxy clustering spectra.

        Returns:
        -------
        cov_theory : numpy.ndarray
            The computed data matrix for the photo-GC data.
        """
        # compute angular power spectra
        cl_gc = self.compute_galclust(params_dic)
        # construct matrix
        cov_theory = build_data_matrix(cl_gc, self.Survey.l_gc, self.Survey.ells_gc, self.Survey.nbin)
        return cov_theory
    
    def compute_data_vector_gc(self, params_dic):
        """This method computes the angular power spectra for galaxy clustering
        using the provided parameters dictionary, constructs the data vector,
        and applies a mask to the data vector.

        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters required for computing the
            angular power spectra.

        Returns:
        -------
        data_vector_masked : numpy.ndarray
            The masked data vector for galaxy clustering.
        """
        # compute angular power spectra
        cl_gc = self.compute_galclust(params_dic)
        # construct vector
        data_vector = np.zeros((self.Survey.nell_gc*self.Survey.nbin_flat), 'float64')
        idx_start = 0
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(bin1, self.Survey.nbin):
                data_vector[idx_start*self.Survey.nell_gc:(idx_start+1)*self.Survey.nell_gc] = cl_gc[:, bin1, bin2]
                idx_start += 1
        data_vector_masked = data_vector[self.Survey.mask_data_vector_gc]
        return data_vector_masked
    
    def compute_covariance_gc(self, params_dic):
        r"""Compute the Gaussian (only diagonal components with :math:`\ell=\ell'`) covariance matrix for the galaxy clustering (GC) analysis.
        This function calculates the covariance matrix for the auto-correlation of galaxy clustering. 
        The covariance matrix is computed for different bins and multipoles as follows:

        .. math::
            \mathrm{Cov}[C_{ij}^{GG}(\ell)C_{kl}^{GG}(\ell)]=\frac{1}{2f_{\rm sky}(2 \ell+1) \Delta \ell} 
            \left( C^{GG}_{ik}C^{GG}_{jl}+C^{GG}_{il}C^{GG}_{jk}\right)\, ,

        where :math:`G` corresponds to galaxy clustering and :math:`i,j` correspond to redshift bins.
        
        Parameters:
        ----------
        params_dic : dict
            Dictionary containing the parameters required for the computation of the galaxy clustering 
            angular power spectra.
        Returns:
        -------
        cov_flat_masked : numpy.ndarray
            The masked covariance matrix for the galaxy clustering analysis.
        """
        # compute angular power spectra
        cl_gc = self.compute_galclust(params_dic)
        # l_gc are centers of the lbins
        denom = ((2*self.Survey.l_gc+1)*self.Survey.fsky*self.Survey.d_ell_bin_cut_gc)
        cov_flat = np.zeros((self.Survey.nell_gc*self.Survey.nbin_flat, self.Survey.nell_gc*self.Survey.nbin_flat), 'float64')
        counter_x = 0
        counter_y = 0
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(bin1, self.Survey.nbin):
                for bin3 in range(self.Survey.nbin):
                    for bin4 in range(bin3, self.Survey.nbin):
                        for i in range(self.Survey.nell_gc):
                            cov_flat[self.Survey.nell_gc*counter_y+i, self.Survey.nell_gc*counter_x+i] = (cl_gc[i, bin1, bin3] * cl_gc[i, bin2, bin4] +
                            cl_gc[i, bin1, bin4]* cl_gc[i, bin2, bin3])/denom[i] 
                        if counter_x<self.Survey.nbin_flat-1:
                            counter_x+=1
                        else:
                            counter_x = 0
                            counter_y+=1                       
        print('cov_flat: ', cov_flat.shape)                                      
        cov_flat_masked = cov_flat[self.Survey.mask_cov_gc]
        print('cov_flat_masked: ', cov_flat_masked.shape)  
        return cov_flat_masked 