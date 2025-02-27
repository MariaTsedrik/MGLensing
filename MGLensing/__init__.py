__author__ = ["M. Tsedrik", "O. Truttero"]
__version__ = "0.0"
__description__ = "MGL = Modified Gravity Lensing: Forecasting Pipeline"

import numpy as np
import yaml
import time
from .theory import Theory
from .specs import EuclidSetUp, LSSTSetUp, CustomSetUp
from .likelihood import MGLike
from datetime import timedelta
# set MGlensing as working directory
#os.chdir(os.path.dirname(os.path.realpath(__file__)))

class MGL():
    def __init__(self, config_file):
        """
        Initialization requires a config yaml-file.
        Here we save the specs, compute mock data etc.
        """
        with open(config_file, "r") as file_in:
            self.config_dic = yaml.safe_load(file_in)
        # initialise the Survey
        if self.config_dic['specs']['survey_info'] == 'Euclid_5bins' or self.config_dic['specs']['survey_info'] == 'Euclid_10bins':
            self.Survey = EuclidSetUp(self.config_dic['specs']['survey_info'], self.config_dic['specs']['scale_cuts'])  
        elif self.config_dic['specs']['survey_info'] == 'LSST_Y1' or self.config_dic['specs']['survey_info'] == 'LSST_Y10':
            self.Survey = LSSTSetUp(self.config_dic['specs']['survey_info'], self.config_dic['specs']['scale_cuts'])  
        #elif self.config_dic['specs']['survey_info'] == 'Custom':
        #    self.Survey = CustomSetUp(self.config_dic['specs']['custom_survey_path'], self.config_dic['specs']['scale_cuts'])  
        else:
            raise ValueError('Invalid survey name')   
 
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
        with open(self.config_dic['data']['params'], "r", encoding="utf-8") as file_in:
            params_data_dic = yaml.safe_load(file_in)
        with open(self.config_dic['theory']['params'], "r", encoding="utf-8") as file_in:
            params_dic = yaml.safe_load(file_in)
        
        
        self.Theo = Theory(self.Survey)
        print('################################')
        print('checking parameters for data....')
        self.params_data_dic, status_data = self.Theo.check_pars_ini(params_data_dic, self.data_model_dic)
        if status_data:
            print('all good with the data dictionary')

        self.params_priors = params_dic
        self.params_fixed = {par: params_dic[par]['p0'] for par in params_dic if params_dic[par]['type'] == 'F'}
        self.params_fiducial = {par: params_dic[par]['p0'] for par in params_dic if params_dic[par]['type'] != 'F'}
        self.params_model = [par for par in params_dic if params_dic[par]['type'] != 'F']
        print('################################')
        print('checking parameters for theory....')
        _, status_theo = self.Theo.check_pars_ini(self.params_fixed|self.params_fiducial, self.theo_model_dic)
        if status_theo:
            print('all good with the model parameters')


        self.probe = self.config_dic['observable']
        if self.probe not in ['3x2pt', 'WL', 'GC']:
            raise ValueError(f"Invalid probe type: {self.probe}. Must be '3x2pt', 'WL', or 'GC'.")
        print('################################')
        print('start computing mock data')
        self.data_dic = self.compute_data()
        print('finish computing mock data')
        print('################################')
        self.Like = MGLike(self.Theo, self.data_dic)


    def compute_data(self):
        """
        Compute the mock data vector and covariance matrix based on the specified probe.

        This method calculates the covariance matrix and its determinant for different types of probes:
        '3x2pt', 'WL', and 'GC'. Depending on the probe type, it also sets the corresponding ell values.

        Attributes:
            probe (str): The type of probe to use. Can be '3x2pt', 'WL', or 'GC'.
            params_data_dic (dict): Dictionary containing the parameters for data computation.
            data_model_dic (dict): Dictionary containing the data model information.
            survey (object): An object containing survey-specific information, including ell values.

        Sets:
            cov_observ (ndarray): The computed covariance matrix.
            cov_observ_high (ndarray, optional): The block of the covariance for l>l_jump for shear, as we assume that lmax_GC<lmax_WL.
            d_obs (float): The determinant of the covariance matrix.
            d_obs_high (float, optional): The determinant of the high-ell shear covariance matrix for '3x2pt' probe.
            ells_one_probe (ndarray, optional): The ell values for the 'WL' or 'GC' probe.
        """

        if self.probe=='3x2pt':
            cov_observ, cov_observ_high = self.Theo.compute_covariance_3x2pt(self.params_data_dic, self.data_model_dic)
            d_obs = np.linalg.det(cov_observ) 
            d_obs_high = np.linalg.det(cov_observ_high) 
            data_dic = {'cov_observ': cov_observ, 'cov_observ_high': cov_observ_high, 
                        'd_obs': d_obs, 'd_obs_high': d_obs_high,
                        'ells': self.Survey.ells_WL}
        elif self.probe=='WL':
            cov_observ = self.Theo.compute_covariance_WL(self.params_data_dic, self.data_model_dic)
            d_obs = np.linalg.det(cov_observ) 
            ells_one_probe = self.Survey.ells_WL
            data_dic = {'cov_observ': cov_observ, 
                'd_obs': d_obs,
                'ells': ells_one_probe}
        elif self.probe=='GC':
            cov_observ = self.Theo.compute_covariance_GC(self.params_data_dic, self.data_model_dic)
            d_obs = np.linalg.det(cov_observ) 
            ells_one_probe = self.Survey.ells_GC 
            data_dic = {'cov_observ': cov_observ, 
                        'd_obs': d_obs,
                        'ells': ells_one_probe}
            
        return data_dic  

    def get_Pmm(self, params, NL_model, baryon_model=0):
        _, _, k = self.Theo.get_Ez_rz_k(params, self.Survey.zz_integr)
        Pk = self.Theo.get_Pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, NL_model, baryon_model)
        return   k, Pk  
    
    def get_bPgm(self, params, NL_model, bias_model, baryon_model=0):
        _, _, k = self.Theo.get_Ez_rz_k(params, self.Survey.zz_integr)
        if bias_model==2:
            Pgm, Pgm_extr = self.Theo.baccoemulator.get_heft(params, k, self.Survey.lbin, self.Survey.zz_integr) 
        else:
            Pgm = self.Theo.get_Pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, NL_model, baryon_model)  
            Pgm_extr = None
        bPgm = self.Theo.get_bPgm(params, k, Pgm, Pgm_extr, self.Survey.nbin, bias_model)
        return k, bPgm  
    
    def get_bPgg(self, params, NL_model, bias_model, baryon_model=0):
        _, _, k = self.Theo.get_Ez_rz_k(params, self.Survey.zz_integr)
        if bias_model==2:
            Pgg, Pgg_extr = self.Theo.baccoemulator.get_heft(params, k, self.Survey.lbin, self.Survey.zz_integr) 
        else:
            Pgg = self.Theo.get_Pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, NL_model, baryon_model) 
            Pgg_extr = None
        bPgg = self.Theo.get_bPgg(params, k, Pgg, Pgg_extr, self.Survey.nbin, bias_model)
        return k, bPgg

    def get_cell_shear(self, params, NL_model, baryon_model=0, IA_model=0):
        Ez, rz, k = self.Theo.get_Ez_rz_k(params, self.Survey.zz_integr)
        Dz = self.Theo.get_growth(params, self.Survey.zz_integr, NL_model)
        Pk = self.Theo.get_Pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, NL_model, baryon_model)
        Cl_LL, _ = self.Theo.get_cell_shear(params, Ez, rz, Dz, Pk, IA_model)
        return  self.Survey.l_WL, Cl_LL
    
    def get_wl_kernel(self, params, NL_model, IA_model=0):
        Ez, rz, _ = self.Theo.get_Ez_rz_k(params, self.Survey.zz_integr)
        Dz = self.Theo.get_growth(params, self.Survey.zz_integr, NL_model)
        Omega_m = params['Omega_m']
        W_L = self.Theo.get_wl_kernel(Omega_m, params, Ez, rz, Dz, IA_model)
        return  W_L
    
    def get_ia_kernel(self, params, NL_model, IA_model=0):
        Ez, _, _ = self.Theo.get_Ez_rz_k(params, self.Survey.zz_integr)
        Dz = self.Theo.get_growth(params, self.Survey.zz_integr, NL_model)
        Omega_m = params['Omega_m']
        W_IA = self.Theo.get_ia_kernel(Omega_m, params, Ez, Dz, self.Survey.eta_z_s, self.Survey.zz_integr, IA_model)
        return  W_IA


    def get_cell_galclust(self, params, NL_model, bias_model, baryon_model=0):
        Ez, rz, k = self.Theo.get_Ez_rz_k(params, self.Survey.zz_integr)
        if bias_model==2:
            Pgg, Pgg_extr = self.Theo.baccoemulator.get_heft(params, k, self.Survey.lbin, self.Survey.zz_integr) 
        else:
            Pgg = self.Theo.get_Pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, NL_model, baryon_model)
            Pgg_extr = None
        Cl_GG, _ = self.Theo.get_cell_galclust(params, Ez, rz, k, Pgg, Pgg_extr, bias_model)    
        return  self.Survey.l_GC, Cl_GG
    
    def get_cell_galgal(self, params, NL_model, bias_model, baryon_model=0, IA_model=0):
        Ez, rz, k = self.Theo.get_Ez_rz_k(params, self.Survey.zz_integr)
        Dz = self.Theo.get_growth(params, self.Survey.zz_integr, NL_model)
        Pk = self.Theo.get_Pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, NL_model, baryon_model)
        Pmm = Pk
        Pgg = Pk
        Pgm = Pk 
        Pgm_extr = None
        Pgg_extr = None
        if bias_model==2:
            Pgg, Pgg_extr = Pgm, Pgm_extr = self.Theo.baccoemulator.get_heft(params, k, self.Survey.lbin, self.Survey.zz_integr) 
        _, W_L = self.Theo.get_cell_shear(params, Ez, rz, Dz, Pmm, IA_model)
        _, W_G = self.Theo.get_cell_galclust(params, Ez, rz, k, Pgg, Pgg_extr, bias_model)    
        Cl_LG, Cl_GL = self.Theo.get_cell_galgal(params, Ez, rz, k, Pgm, Pgm_extr, W_L, W_G, bias_model)   
        return  self.Survey.l_XC, Cl_LG, Cl_GL 
    
    def get_expansion_and_rcom(self, params):
        Ez, rz, _ = self.Theo.get_Ez_rz_k(params, self.Survey.zz_integr)
        return Ez, rz

    def gen_output_header(self):
        """
        Generates output header
        """
        def get_observable_label(value):
            observable_map = {
                'WL': "shear-shear",
                'XC': "galaxy lensing",
                'GC': "photo-clustering",
                '3x2pt': "WL+XC+GC"
            }
            return observable_map.get(value, "Unknown")
        
        def get_nofz_label(value):
            nofz_map = {
                0: "Euclid",
                1: "LSST"
            }
            return nofz_map.get(value, "Unknown")

        def get_model_label(value, model_type):
            model_maps = {
                "NL_model": {0: "HMcode", 1: "bacco", 2: "nDGP", 3: "gLEMURS"},
                "bias_model": {0: "b1 constant within bins", 1: "(b1, b2) constant within bins", 2: "bacco"},
                "IA_model": {0: "zNLA", 1: "TATT"},
                "baryon_model": {0: "no baryons", 1: "Tagn HMcode", 2: "bcemu", 3: "bacco"}
            }
            return model_maps.get(model_type, {}).get(value, "Unknown")

        config = self.config_dic
        observable = get_observable_label(config.get("observable", -1))
        specs = config.get("specs", {})
        data_models = self.data_model_dic
        theory_models = self.theo_model_dic
        params_data = self.params_data_dic
        params_priors = self.params_priors

        output_header = f"""
        ##############################################################
        # Cosmology Pipeline Configuration
        #------------------------------------------------------------
        # Observable: {observable} ({config.get("observable", "N/A")})
        # Sky Fraction (fsky): {self.Survey.fsky}
        # Redshift Binning: {self.Survey.nbin} bins ({self.Survey.zmin} ≤ z ≤ {self.Survey.zmax})
        # for n(z) {self.Survey.survey_name}-like
        # Lensing & Clustering Scales:
        #   - l_min: {self.Survey.lmin}
        #   - l_max (WL): {self.Survey.lmax_wl_vals}
        #   - l_max (GC, XC): {self.Survey.lmax_gc_vals}
        #
        # Data Model:
        #   - Nonlinear Power Spectrum: {get_model_label(data_models.get("NL_model", -1), "NL_model")} ({data_models.get("NL_model", "N/A")})
        #   - Galaxy Bias Model: {get_model_label(data_models.get("bias_model", -1), "bias_model")} ({data_models.get("bias_model", "N/A")})
        #   - Intrinsic Alignments: {get_model_label(data_models.get("IA_model", -1), "IA_model")} ({data_models.get("IA_model", "N/A")})
        #   - Baryon Model: {get_model_label(data_models.get("baryon_model", -1), "baryon_model")} ({data_models.get("baryon_model", "N/A")})
        #   - Parameters: 
        """
        for par_i in params_data.keys():
            output_header += f"\n            #       - {par_i}: {params_data[par_i]}"
        #output_header += "\n".join(f'#       - {k}: {v}' for k, v in params_data.items())    
        output_header += f"""
        #
        # Theory Model:
        #   - Nonlinear Power Spectrum: {get_model_label(theory_models.get("NL_model", -1), "NL_model")} ({theory_models.get("NL_model", "N/A")})
        #   - Galaxy Bias Model: {get_model_label(theory_models.get("bias_model", -1), "bias_model")} ({theory_models.get("bias_model", "N/A")})
        #   - Intrinsic Alignments: {get_model_label(theory_models.get("IA_model", -1), "IA_model")} ({theory_models.get("IA_model", "N/A")})
        #   - Baryon Model: {get_model_label(theory_models.get("baryon_model", -1), "baryon_model")} ({theory_models.get("baryon_model", "N/A")})
        #   - Parameter priors:
        """
        for par_i in params_priors.keys():
            if params_priors[par_i]['type'] == 'G':
                output_header += f"\n           #       - {par_i}: N({params_priors[par_i]['p1']},{params_priors[par_i]['p2']})"
            elif params_priors[par_i]['type'] == 'U':
                output_header += f"\n           #       - {par_i}: [{params_priors[par_i]['p1']},{params_priors[par_i]['p2']}]"   
            elif params_priors[par_i]['type'] == 'F':
                output_header += f"\n           #       - {par_i}: {params_priors[par_i]['p0']}"
        output_header += f"""
        ##############################################################
        """
        for par_i in self.params_model:
            output_header += f"     {par_i}     "
        output_header += "   log_w   log_l"    
        return output_header


    def test(self):
        dic_test = {par_i: self.params_priors[par_i]['p0'] for par_i in self.params_priors.keys()}
        dic_test = self.params_fixed | dic_test
        
        if self.probe=='3x2pt':
            start = time.time()
            test_like = self.Like.loglikelihood_det_3x2pt(dic_test, self.theo_model_dic)
            finish = time.time()
            test_time = finish-start
        elif self.probe=='WL':
            start = time.time()
            test_like = self.Like.loglikelihood_det_WL(dic_test, self.theo_model_dic)    
            finish = time.time()
            test_time = finish-start
        elif self.probe=='GC':
            start = time.time()
            test_like = self.Like.loglikelihood_det_GC(dic_test, self.theo_model_dic)    
            finish = time.time()
            test_time = finish-start
        test_time_hms=timedelta(seconds=test_time)        
        test_txt = f"""
        ##############################################################
        # loglikelihood = {test_like} evaluation took  {test_time} s (--> {test_time_hms} hh:mm:ss)
        """
        np.savetxt(self.path+"chains/chain_"+self.chain_name+".txt", [], header=self.gen_output_header()+test_txt)
