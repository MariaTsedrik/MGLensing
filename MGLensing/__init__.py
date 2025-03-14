__author__ = ["M. Tsedrik", "O. Truttero"]
__version__ = "0.0"
__description__ = "MGL = Modified Gravity Lensing: Forecasting Pipeline"

import numpy as np
import yaml
import time
from .theory import Theory
from .specs import LSSTSetUp, EuclidSetUp
from .likelihood import MGLike
from datetime import timedelta
import MGrowth as mg
from scipy import interpolate as itp
from scipy.linalg import cholesky, solve_triangular

NL_MODEL_HMCODE = 0
NL_MODEL_BACCO = 1
NL_MODEL_NDGP = 2
NL_MODEL_GAMMAZ = 3
NL_MODEL_MUSIGMA = 4
NL_MODEL_DS = 5

COMP_DATA = 0
READ_DATA = 1

a_arr_for_mu = np.logspace(-5., 1., 512)
        
def _errorbars_for_plots(nell, n, flat_array):
    matrix_errs = np.zeros((nell, n, n))
    idx_start = 0
    for bin1 in range(n):
        for bin2 in range(bin1, n):
            matrix_errs[:, bin1, bin2] = flat_array[idx_start*nell:(idx_start+1)*nell]
            matrix_errs[:, bin2, bin1] = matrix_errs[:, bin1, bin2]
            idx_start += 1
    return matrix_errs

class DataClass():
    """
    A class to handle data operations for a given survey.

    Attributes:
    ----------

    DataModel : Theory
        An instance of the Theory class initialized with the survey and data model.
    params_data_dic : dict
        A dictionary containing the parameters for the data.
    cov_obs : np.ndarray
        The covariance matrix of the observed data.
    det_obs : float
        The determinant of the observed data covariance matrix.
    ells : list
        The ell values associated with the survey observable.
    det_obs_high : float
        The determinant of the high-resolution observed data covariance matrix.
    data_vector : np.ndarray
        The data vector for the survey observable.
    data_covariance : np.ndarray
        The covariance matrix of the data vector.
    cholesky_transform : np.ndarray
        The Cholesky decomposition of the data covariance matrix.

    Methods:
    -------
    __init__(data_model: dict, Survey):
        Initializes the DataClass with the given data model and survey.
    
    compute_data(Survey):
        Computes the data based on the survey's likelihood and observable.
    """
    def __init__(self, data_model: dict, Survey):
        if data_model['type'] == COMP_DATA:
            self.DataModel = Theory(Survey, data_model)  
            with open(data_model['params'], "r", encoding="utf-8") as file_in:
                params_data = yaml.safe_load(file_in)
            print('################################')
            print('checking parameters for data....')
            self.params_data_dic, status_data = self.DataModel.check_pars_ini(params_data)
            if status_data:
                print('all good with the data dictionary')
            print('################################')
            print('start computing mock data')
            self.compute_data(Survey)
            print('finish computing mock data')
            print('################################')    
        else:
            raise NotImplementedError('Reading data from file is not implemented yet')   


    def compute_data(self, Survey):
        if Survey.likelihood == 'determinants':
            compute_matrix = {
            'WL': self.DataModel.compute_data_matrix_wl,
            'GC': self.DataModel.compute_data_matrix_gc,
            '3x2pt': self.DataModel.compute_data_matrix_3x2pt
            }
            assign_ell = {
            'WL': Survey.ells_wl,
            'GC': Survey.ells_gc,
            '3x2pt': Survey.ells_wl
            }
            self.cov_obs = compute_matrix[Survey.observable](self.params_data_dic)
            self.det_obs = np.linalg.det(self.cov_obs) 
            self.ells = assign_ell[Survey.observable]
            if Survey.observable == '3x2pt':
                self.det_obs_high = np.linalg.det(self.cov_obs[1])
                self.cov_obs = self.cov_obs[0]
        else:
            compute_vector = {
            'WL': self.DataModel.compute_data_vector_wl,
            'GC': self.DataModel.compute_data_vector_gc,
            '3x2pt': self.DataModel.compute_data_vector_3x2pt
            }
            # gaussian covariance with diagonal components only
            compute_covariance = {
            'WL': self.DataModel.compute_covariance_wl,
            'GC': self.DataModel.compute_covariance_gc,
            '3x2pt': self.DataModel.compute_covariance_3x2pt
            }
            self.data_vector = compute_vector[Survey.observable](self.params_data_dic)
            self.data_covariance = compute_covariance[Survey.observable](self.params_data_dic)
            self.cholesky_transform = cholesky(self.data_covariance, lower=True)

            
         

class MGL():
    """
    A class to handle the initialization and operations of the MGL (Modified Gravity Lensing) pipeline.
    This class is responsible for setting up the survey, checking the probe type, initializing data and theoretical models,
    and computing various cosmological quantities such as power spectra, expansion rate, comoving distance, and error bars.
    
    Attributes:
    ----------
    config_dic : dict
        Dictionary containing the configuration parameters loaded from the provided YAML file.
    Survey : object
        An instance of the survey setup class (e.g., LSSTSetUp or EuclidSetUp) based on the survey information in the configuration.
    probe : str
        The type of observable probe (e.g., '3x2pt', 'WL', 'GC').
    path : str
        The path for output files.
    chain_name : str
        The name of the output chain.
    hdf5_name : str
        The name of the HDF5 file for MCMC sampling.
    mcmc_resume : bool
        Flag indicating whether to resume MCMC sampling.
    mcmc_verbose : bool
        Flag indicating whether to enable verbose output for MCMC sampling.
    mcmc_neff : int
        The effective number of samples for MCMC.
    mcmc_nlive : int
        The number of live points for MCMC.
    mcmc_pool : int
        The number of processes in the pool for MCMC.
    data_model_dic : dict
        Dictionary containing the data model specifications.
    theo_model_dic : dict
        Dictionary containing the theoretical model specifications.
    params_priors : dict
        Dictionary containing the prior values of parameters.
    params_model : list
        List of parameter names that are not fixed.
    params_fixed : dict
        Dictionary containing the fixed parameter values.
    params_fiducial : dict
        Dictionary containing the fiducial parameter values.
    Data : object
        An instance of the DataClass initialized with the data model and survey.
    TheoryModel : object
        An instance of the Theory class initialized with the survey and theoretical model.
    Like : object
        An instance of the MGLike class initialized with the theoretical model and data.
    
    Methods:
    -------
    get_power_spectra(params, theo_model): 
        Compute power spectra (mattre-matter, matter-galaxy, galaxy-galaxy).
    get_expansion_and_rcom(params):
        Compute expansion and comoving distance for a given cosmology.
    get_c_ells(params, theo_model):
        Compute angular power spectra for a given model.
    get_errorbars(params):
        Compute errorbars for angular power spectra from a Gaussian covariance.
    get_sigma8_from_a_s_from_chain(params, nl_model):
        Calculate the sigma8 parameter from a given set of cosmological parameters and a non-linear model from a chain.
    gen_output_header():
        Generates the output header for the configuration and parameters.
    test():
        Test-computation of a likelihood.
    """
    def __init__(self, config_file):
        """Initialization requires a config yaml-file.
        Here we save the specs, compute mock data, assign model-specs, etc.
        """
        with open(config_file, "r") as file_in:
            self.config_dic = yaml.safe_load(file_in)

        # initialise the Survey
        if 'LSST' in self.config_dic['specs']['survey_info']:
            self.Survey = LSSTSetUp(self.config_dic)
        elif 'Euclid' in self.config_dic['specs']['survey_info']:
            self.Survey = EuclidSetUp(self.config_dic)    
        else:
            raise ValueError('Invalid survey name')
        if self.Survey.zz_integr[0]==0.:
            raise ValueError('Invalid smallest redshift: z_min>0!!!')  
        
        # check the probe
        self.probe = self.config_dic['observable']
        if self.probe not in ['3x2pt', 'WL', 'GC']:
            raise ValueError(f"Invalid probe type: {self.probe}. Must be '3x2pt', 'WL', or 'GC'.")
 
        self.path = self.config_dic.get('path', './')
        self.chain_name = self.config_dic['output']['chain_name']
        sampler_dic = self.config_dic['sampler']['mcmc']
        self.hdf5_name = sampler_dic['hdf5_name']
        self.mcmc_resume = sampler_dic['resume']
        self.mcmc_verbose = sampler_dic['verbose']
        self.mcmc_neff = sampler_dic['n_eff']
        self.mcmc_nlive = sampler_dic['n_live']
        self.mcmc_pool = sampler_dic['pool']
        
        # save modelling choices
        self.data_model_dic = self.config_dic['data']
        self.theo_model_dic = self.config_dic['theory']

        
        # save values of parameters (cosmological and nuisances)
        with open(self.config_dic['theory']['params'], "r", encoding="utf-8") as file_in:
            params_dic = yaml.safe_load(file_in)

        # initialise data 
        self.Data = DataClass(self.data_model_dic, self.Survey)
        # initialise model
        self.TheoryModel = Theory(self.Survey, self.theo_model_dic)  

        # need this later for Nautilus
        self.params_priors = params_dic
        self.params_model = [par for par in params_dic if params_dic[par]['type'] != 'F']
        print('################################')
        print('checking parameters for theory....')
        self.params_fixed = {par: params_dic[par]['p0'] for par in params_dic if params_dic[par]['type'] == 'F'}
        self.params_fiducial = {par: params_dic[par]['p0'] for par in params_dic if params_dic[par]['type'] != 'F'}
        _, status_theo = self.TheoryModel.check_pars_ini(self.params_fixed|self.params_fiducial)
        if status_theo:
            print('all good with the model parameters')
            print('################################')

        # initialise the likelihood
        self.Like = MGLike(self.TheoryModel, self.Data)


    # functions bellow are not designed to be fast, but user-friendly instead
    # so that one can plot and play with different avaliable settings
    # without re-writing config-files
    def get_power_spectra(self, params, theo_model):
        """
        Calculate the power spectra for a given set of parameters and theoretical model.

        Parameters:
        ----------
        params : dict
            Dictionary containing the parameters for the theoretical model.
        theo_model : dict
            Dictionary containing the theoretical model specifications.

        Returns:
        -------
        tuple
            A tuple containing:
            - k : array-like
                The wave numbers.
            - pmm : array-like
                The matter power spectrum.
            - pgm : array-like
                The galaxy-matter cross power spectrum.
            - pgg : array-like
                The galaxy power spectrum.
        """
        NewModel = Theory(self.Survey, theo_model)
        _, _, k = NewModel.get_ez_rz_k(params)
        NewModel.k = k
        if theo_model['nl_model'] == NL_MODEL_BACCO:
            params['sigma8_cb'] = NewModel.StructureEmu.get_sigma8_cb(params)
        pmm, bar_boost = NewModel.get_pmm(params)
        NewModel.pmm = pmm
        if NewModel.flag_heft:
            NewModel.pk_exp, NewModel.pk_exp_extr = NewModel.get_heft_pk_exp(params)
            pgm = NewModel.get_pgm(params)*np.sqrt(bar_boost[:, :, None]) 
        else:
            pgm = NewModel.get_pgm(params)  
        pgg = NewModel.get_pgg(params)    
        return k, pmm, pgm, pgg
        
    
    def get_expansion_and_rcom(self, params):
        """
        Calculate the expansion rate and comoving distance.

        Parameters:
        ----------
        params (dict): A dictionary of parameters required by the TheoryModel.

        Returns:
        -------
        tuple: A tuple containing the expansion rate (ez) and comoving distance (rz).
        """
        ez, rz, _ = self.TheoryModel.get_ez_rz_k(params)
        return ez, rz
    
    
    def get_c_ells(self, params, theo_model):
        """
        Compute the angular power spectra (C_ells) for different components based on the theoretical model and parameters provided.

        Parameters:
        ----------
        params : dict
            Dictionary containing the parameters required for the theoretical model.
        theo_model : dict
            Dictionary specifying the theoretical model to be used, including non-linear model and other configurations.

        Returns:
        -------
        tuple
            A tuple containing the following angular power spectra:
            - cl_ll : array
                Angular power spectrum for shear-shear correlations.

            - cl_gg : array
                Angular power spectrum for galaxy clustering correlations.

            - cl_lg : array
                Angular power spectrum for shear-galaxy cross-correlations.

            - cl_gl : array
                Angular power spectrum for galaxy-shear cross-correlations.
        """
        NewModel = Theory(self.Survey, theo_model)  
        if theo_model['nl_model'] == NL_MODEL_BACCO and 'sigma8_cb' not in params:
            params['sigma8_cb'] = NewModel.StructureEmu.get_sigma8_cb(params) 
        NewModel.ez, NewModel.rz, NewModel.k = NewModel.get_ez_rz_k(params)
        NewModel.dz, _ = NewModel.StructureEmu.get_growth(params, self.Survey.zz_integr)
        NewModel.deltaz_s, NewModel.deltaz_l = NewModel.get_deltaz(params)
        NewModel.pmm, bar_boost = NewModel.get_pmm(params)
        if NewModel.flag_heft:
            NewModel.pk_exp, NewModel.pk_exp_extr = NewModel.get_heft_pk_exp(params)
            NewModel.pgm = NewModel.get_pgm(params)*np.sqrt(bar_boost[:, :, None]) 
        else:
            NewModel.pgm = NewModel.get_pgm(params)
        NewModel.pgg = NewModel.get_pgg(params)
        NewModel.w_gamma = NewModel.get_wl_kernel(params)
        NewModel.w_ia = NewModel.get_ia_kernel(params)
        NewModel.pk_delta_ia, NewModel.pk_iaia = NewModel.get_pk_ia(params)
        NewModel.w_g = NewModel.get_gg_kernel(params)
        NewModel.pk_gal_ia = NewModel.get_pk_cross_ia(params)
        cl_ll = NewModel.get_cell_shear()
        cl_gg = NewModel.get_cell_galclust()    
        cl_lg, cl_gl = NewModel.get_cell_cross() 
        return cl_ll, cl_gg, cl_lg, cl_gl  
    

    
    def get_errorbars(self, params):
        """
        Calculate and return the error bars for different components of the survey data.

        Parameters:
        ----------
        params : dict
            Dictionary containing the parameters required to compute the covariance matrix.

        Returns:
        -------
        tuple
            A tuple containing the error bars for the following components:

            - err_cl_ll: Error bars for the lensing-lensing component.
            - err_cl_gg: Error bars for the galaxy-galaxy component.
            - err_cl_lg: Error bars for the lensing-galaxy component.
            - err_cl_gl: Error bars for the galaxy-lensing component.
        
        Notes:
        -----
        The method computes the covariance matrix for the 3x2pt data vector and extracts the error bars
        by taking the square root of the diagonal elements of the covariance matrix. The error bars are
        then split into different components based on the survey configuration.
        """
        mask_data_vector_3x2pt = np.ones(int((self.Survey.nell_gc+2*self.Survey.nell_xc+self.Survey.nell_wl)*self.Survey.nbin_flat), dtype=bool)
        self.Data.DataModel.Survey.mask_cov_3x2pt = np.ix_(mask_data_vector_3x2pt, mask_data_vector_3x2pt)
        cov3x2pt = self.Data.DataModel.compute_covariance_3x2pt(params)
        errorbars = np.sqrt(np.diag(cov3x2pt))
        start_lg = self.Survey.nell_wl*self.Survey.nbin_flat
        start_gl = (self.Survey.nell_wl+self.Survey.nell_xc)*self.Survey.nbin_flat
        start_gc = (self.Survey.nell_wl+self.Survey.nell_xc+self.Survey.nell_xc)*self.Survey.nbin_flat
        
        errorbars_ll = errorbars[:start_lg]
        errorbars_lg = errorbars[start_lg:start_gl]
        errorbars_gl = errorbars[start_gl:start_gc]
        errorbars_gg = errorbars[start_gc:]
        #return errorbars_ll, errorbars_gg, errorbars_lg, errorbars_gl
        err_cl_ll = _errorbars_for_plots(self.Survey.nell_wl, self.Survey.nbin, errorbars_ll)
        err_cl_gg = _errorbars_for_plots(self.Survey.nell_gc, self.Survey.nbin, errorbars_gg)
        err_cl_lg = _errorbars_for_plots(self.Survey.nell_xc, self.Survey.nbin, errorbars_lg)
        err_cl_gl = _errorbars_for_plots(self.Survey.nell_xc, self.Survey.nbin, errorbars_gl)
        return  err_cl_ll, err_cl_gg, err_cl_lg, err_cl_gl    

    
    def get_sigma8_from_a_s_from_chain(self, params, nl_model):
        """
        Calculate the sigma8 parameter from a given set of cosmological parameters and a non-linear model
        from a chain.

        Parameters:
        ----------
        self : object
            Instance of the class containing this method.
        params : dict
            Dictionary containing cosmological parameters. 
        nl_model : int
            Non-linear model identifier. Expected values are NL_MODEL_HMCODE or NL_MODEL_BACCO for standard models, 
            or other values for extended cosmologies.

        Returns:
        -------
        sigma8 : numpy.ndarray
            Array of sigma8 values calculated from the input parameters and non-linear model.
        """
        # initialise a theory class with hmcode
        fid_model = {'nl_model': NL_MODEL_HMCODE, 'bias_model': 0, 'ia_model': 0, 'baryon_model': 0, 'photoz_err_model': 0.}
        FidModel = Theory(self.Survey, fid_model)  
        model =fid_model.copy()
        model['nl_model'] = nl_model
        # intialise a theory class with the target model
        NewModel = Theory(self.Survey, model) 
        _ = FidModel.StructureEmu.check_pars_ini(params)
        # if lcmd or w0wacdm, then use hmcode's sigm8-emulator 
        if nl_model==NL_MODEL_HMCODE or nl_model==NL_MODEL_BACCO:
            sigma8 = FidModel.StructureEmu.get_sigma8(params)
        # for extended cosmologies do the re-scaling using linear growth factors   
        else: 
            sigma8_gr = FidModel.StructureEmu.get_sigma8(params, flag_gr=True) 
            len_chain = len(sigma8_gr) 
            D_rescale = np.zeros(len_chain)
            for i in range(len_chain):
                print(i, '/', len_chain)
                params_growth = {
                    'Omega_m': params['Omega_m'][i],
                    'h' : params['h'][i],
                    'w0': 0.,
                    'wa': -1.,
                    'log10Omega_rc': params['log10Omega_rc'][i] if 'log10Omega_rc' in params else 1.,
                    'gamma0': params['gamma0'][i] if 'gamma0' in params else 0.55,
                    'gamma1': params['gamma1'][i] if 'gamma1' in params else 0.,
                    'mu0': params['mu0'][i] if 'mu0' in params else 0.,
                    'Ads': params['Ads'][i] if 'Ads' in params else 0.,
                    }
                _, dz0_gr = FidModel.StructureEmu.get_growth(params_growth, [1.])
                params_growth['w0'] = params['w0'][i] if 'w0' in params else -1.
                _, dz0_mg = NewModel.StructureEmu.get_growth(params_growth, [1.])
                D_rescale[i] = dz0_mg/dz0_gr
            sigma8 = sigma8_gr*D_rescale
        return sigma8
        

    def gen_output_header(self):
        """Generates output header
        """
        def get_observable_label(value):
            observable_map = {
                'WL': "shear-shear",
                'GC': "photo-clustering",
                '3x2pt': "WL+XC+GC"
            }
            return observable_map.get(value, "Unknown")
        def get_likelihood_label(value):
            name_map = {
                'determinants': "using determinants and summing over all integer ell-values",
                'binned': "using covariance and differences between data vectors in ell-bins"
            }
            return name_map.get(value, "Unknown")

        def get_model_label(value, model_type):
            model_maps = {
                "nl_model": {0: "HMcode", 1: "bacco", 2: "nDGP", 3: "gLEMURS", 4: "mu-Sigma", 5:"Dark Scattering"},
                "bias_model": {0: "b1 constant within bins", 1: "(b1, b2) constant within bins", 2: "HEFT"},
                "ia_model": {0: "zNLA", 1: "TATT"},
                "baryon_model": {0: "no baryons", 1: "Tagn HMcode", 2: "bcemu", 3: "bacco"},
                "photoz_err_model": {0: "no photo-z error", 1: "additive mode, n(z') = n(z + dz)", 2: "multiplicative mode, n(z') = n(z(1 + dz))"}
            }
            return model_maps.get(model_type, {}).get(value, "Unknown")

        config = self.config_dic
        observable = get_observable_label(config.get("observable", -1))
        like = get_likelihood_label(config.get("likelihood", -1))
        output_header = f"""
        ##############################################################
        # Cosmology Pipeline Configuration
        #------------------------------------------------------------
        # Observable: {observable}
        # Likelihood: {like}
        # Sky Fraction (fsky): {self.Survey.fsky}
        # Redshift Binning: {self.Survey.nbin} bins ({self.Survey.zmin} ≤ z ≤ {self.Survey.zmax})
        # for n(z) {self.Survey.survey_name}-like
        # Lensing & Clustering Scales:
        #   - l_min: {self.Survey.lmin}
        #   - l_max: {self.Survey.lmax}
        #   - total l_bins: {self.Survey.lbin}
        #   - cut in l_max (WL): {self.Survey.lmax_wl_vals}
        #   - cut in l_max (GC, XC): {self.Survey.lmax_gc_vals}
        #
        """
        if self.data_model_dic['type']==COMP_DATA:
            output_header += f"""    
            # Data Model:
            #   - Nonlinear Power Spectrum: {get_model_label(self.data_model_dic.get("nl_model", -1), "nl_model")} ({self.data_model_dic.get("nl_model", "N/A")})
            #   - Galaxy Bias Model: {get_model_label(self.data_model_dic.get("bias_model", -1), "bias_model")} ({self.data_model_dic.get("bias_model", "N/A")})
            #   - Intrinsic Alignments: {get_model_label(self.data_model_dic.get("ia_model", -1), "ia_model")} ({self.data_model_dic.get("ia_model", "N/A")})
            #   - Baryon Model: {get_model_label(self.data_model_dic.get("baryon_model", -1), "baryon_model")} ({self.data_model_dic.get("baryon_model", "N/A")})
            #   - Parameters: 
            """
            for par_i in self.Data.params_data_dic.keys():
                output_header += f"\n            #       - {par_i}: {self.Data.params_data_dic[par_i]}"
        else:
            output_header += f"""    
            # Data from the following file: {self.data_model_dic['data_file']}
            #
            """

        output_header += f"""
        #
        # Theory Model:
        #   - Nonlinear Power Spectrum: {get_model_label(self.theo_model_dic.get("nl_model", -1), "nl_model")} ({self.theo_model_dic.get("nl_model", "N/A")})
        #   - Galaxy Bias Model: {get_model_label(self.theo_model_dic.get("bias_model", -1), "bias_model")} ({self.theo_model_dic.get("bias_model", "N/A")})
        #   - Intrinsic Alignments: {get_model_label(self.theo_model_dic.get("ia_model", -1), "ia_model")} ({self.theo_model_dic.get("ia_model", "N/A")})
        #   - Baryon Model: {get_model_label(self.theo_model_dic.get("baryon_model", -1), "baryon_model")} ({self.theo_model_dic.get("baryon_model", "N/A")})
        #   - Photo-z Error Model: {get_model_label(self.theo_model_dic.get("photoz_err_model", -1), "photoz_err_model")} ({self.theo_model_dic.get("photoz_err_model", "N/A")})
        #   - Parameter priors:
        """
        for par_i in self.params_priors.keys():
            if self.params_priors[par_i]['type'] == 'G':
                output_header += f"\n           #       - {par_i}: N({self.params_priors[par_i]['p1']},{self.params_priors[par_i]['p2']})"
            elif self.params_priors[par_i]['type'] == 'U':
                output_header += f"\n           #       - {par_i}: [{self.params_priors[par_i]['p1']},{self.params_priors[par_i]['p2']}]"   
            elif self.params_priors[par_i]['type'] == 'F':
                output_header += f"\n           #       - {par_i}: {self.params_priors[par_i]['p0']}"
        output_header += f"""
        ##############################################################
        """
        for par_i in self.params_model:
            output_header += f"     {par_i}     "
        output_header += "   log_w   log_l"    
        return output_header


    def test(self):
        """
        Evaluates one log-likelihood of the current parameter set and records the time taken for the evaluation.

        This method constructs a dictionary of test parameters by combining fixed parameters and priors.
        It then computes the log-likelihood using these parameters and measures the time taken for this computation.
        The results, including the log-likelihood value and the time taken, are saved to a text file.
        """ 
        dic_test = {par_i: self.params_priors[par_i]['p0'] for par_i in self.params_priors.keys()}
        dic_test = self.params_fixed | dic_test
        start = time.time()
        test_like = self.Like.compute(dic_test)    
        finish = time.time()
        test_time = finish-start
        test_time_hms=timedelta(seconds=test_time)        
        test_txt = f"""
        ##############################################################
        # loglikelihood = {test_like} evaluation took  {test_time} s (--> {test_time_hms} hh:mm:ss)
        """
        np.savetxt(self.path+"chains/chain_"+self.chain_name+".txt", [], header=self.gen_output_header()+test_txt)
