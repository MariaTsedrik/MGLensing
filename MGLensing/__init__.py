__author__ = ["M. Tsedrik", "O. Truttero"]
__version__ = "0.0"
__description__ = "MGL = Modified Gravity Lensing: Forecasting Pipeline"

import numpy as np
import yaml
import time
from .theory import Theory
from .specs import EuclidSetUp, LSSTSetUp
from .likelihood import MGLike
from datetime import timedelta

NL_MODEL_HMCODE = 0
NL_MODEL_BACCO = 1
NL_MODEL_NDGP = 2
NL_MODEL_GAMMAZ = 3
NL_MODEL_MUSIGMA = 4
NL_MODEL_IDE = 5

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
            cov_obs (ndarray): The computed covariance matrix.
            cov_obs_high (ndarray, optional): The block of the covariance for l>l_jump for shear, as we assume that lmax_gc<lmax_wl.
            det_obs (float): The determinant of the covariance matrix.
            det_obs_high (float, optional): The determinant of the high-ell shear covariance matrix for '3x2pt' probe.
            ells_one_probe (ndarray, optional): The ell values for the 'WL' or 'GC' probe.
        """

        if self.probe=='3x2pt':
            cov_obs, cov_obs_high = self.Theo.compute_covariance_3x2pt(self.params_data_dic, self.data_model_dic)
            det_obs = np.linalg.det(cov_obs) 
            det_obs_high = np.linalg.det(cov_obs_high) 
            data_dic = {'cov_obs': cov_obs, 'cov_obs_high': cov_obs_high, 
                        'det_obs': det_obs, 'det_obs_high': det_obs_high,
                        'ells': self.Survey.ells_wl}
        elif self.probe=='WL':
            cov_obs = self.Theo.compute_covariance_wl(self.params_data_dic, self.data_model_dic)
            det_obs = np.linalg.det(cov_obs) 
            ells_one_probe = self.Survey.ells_wl
            data_dic = {'cov_obs': cov_obs, 
                'det_obs': det_obs,
                'ells': ells_one_probe}
        elif self.probe=='GC':
            cov_obs = self.Theo.compute_covariance_gc(self.params_data_dic, self.data_model_dic)
            det_obs = np.linalg.det(cov_obs) 
            ells_one_probe = self.Survey.ells_gc 
            data_dic = {'cov_obs': cov_obs, 
                        'det_obs': det_obs,
                        'ells': ells_one_probe}
            
        return data_dic  

    def get_pmm(self, params, nl_model, baryon_model=0):
        _, _, k = self.Theo.get_ez_rz_k(params, self.Survey.zz_integr)
        pk = self.Theo.get_pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, nl_model, baryon_model)
        return k, pk
    
    def get_bpgm(self, params, nl_model, bias_model, baryon_model=0):
        _, _, k = self.Theo.get_ez_rz_k(params, self.Survey.zz_integr)
        if bias_model==2:
            pgm, pgm_extr = self.Theo.BaccoEmulator.get_heft(params, k, self.Survey.lbin, self.Survey.zz_integr) 
        else:
            pgm = self.Theo.get_pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, nl_model, baryon_model)  
            pgm_extr = None
        bpgm = self.Theo.get_bpgm(params, k, pgm, pgm_extr, self.Survey.nbin, bias_model)
        return k, bpgm  
    
    def get_bpgg(self, params, nl_model, bias_model, baryon_model=0):
        _, _, k = self.Theo.get_ez_rz_k(params, self.Survey.zz_integr)
        if bias_model==2:
            pgg, pgg_extr = self.Theo.BaccoEmulator.get_heft(params, k, self.Survey.lbin, self.Survey.zz_integr) 
        else:
            pgg = self.Theo.get_pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, nl_model, baryon_model) 
            pgg_extr = None
        bpgg = self.Theo.get_bpgg(params, k, pgg, pgg_extr, self.Survey.nbin, bias_model)
        return k, bpgg

    def get_cell_shear(self, params, nl_model, baryon_model=0, ia_model=0):
        ez, rz, k = self.Theo.get_ez_rz_k(params, self.Survey.zz_integr)
        dz, _ = self.Theo.get_growth(params, self.Survey.zz_integr, nl_model)
        pk = self.Theo.get_pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, nl_model, baryon_model)
        cl_ll, _ = self.Theo.get_cell_shear(params, ez, rz, dz, pk, ia_model)
        return  self.Survey.l_wl, cl_ll
    
    def get_wl_kernel(self, params, nl_model, ia_model=0):
        ez, rz, _ = self.Theo.get_ez_rz_k(params, self.Survey.zz_integr)
        dz, _ = self.Theo.get_growth(params, self.Survey.zz_integr, nl_model)
        omega_m = params['Omega_m']
        w_l = self.Theo.get_wl_kernel(omega_m, params, ez, rz, dz, ia_model)
        return w_l
    
    def get_ia_kernel(self, params, nl_model, ia_model=0):
        ez, _, _ = self.Theo.get_ez_rz_k(params, self.Survey.zz_integr)
        dz, _ = self.Theo.get_growth(params, self.Survey.zz_integr, nl_model)
        omega_m = params['Omega_m']
        w_ia = self.Theo.get_ia_kernel(omega_m, params, ez, dz, self.Survey.eta_z_s, self.Survey.zz_integr, ia_model)
        return w_ia


    def get_cell_galclust(self, params, nl_model, bias_model, baryon_model=0):
        ez, rz, k = self.Theo.get_ez_rz_k(params, self.Survey.zz_integr)
        if bias_model == 2:
            pgg, pgg_extr = self.Theo.BaccoEmulator.get_heft(params, k, self.Survey.lbin, self.Survey.zz_integr)
        else:
            pgg = self.Theo.get_pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, nl_model, baryon_model)
            pgg_extr = None
        cl_gg, _ = self.Theo.get_cell_galclust(params, ez, rz, k, pgg, pgg_extr, bias_model)
        return self.Survey.l_gc, cl_gg
    
    def get_cell_galgal(self, params, nl_model, bias_model, baryon_model=0, ia_model=0):
        ez, rz, k = self.Theo.get_ez_rz_k(params, self.Survey.zz_integr)
        dz, _ = self.Theo.get_growth(params, self.Survey.zz_integr, nl_model)
        pk = self.Theo.get_pmm(params, k, self.Survey.lbin, self.Survey.zz_integr, nl_model, baryon_model)
        pmm = pk
        pgg = pk
        pgm = pk 
        pgm_extr = None
        pgg_extr = None
        if bias_model == 2:
            pgg, pgg_extr = pgm, pgm_extr = self.Theo.BaccoEmulator.get_heft(params, k, self.Survey.lbin, self.Survey.zz_integr) 
        _, w_l = self.Theo.get_cell_shear(params, ez, rz, dz, pmm, ia_model)
        _, w_g = self.Theo.get_cell_galclust(params, ez, rz, k, pgg, pgg_extr, bias_model)    
        cl_lg, cl_gl = self.Theo.get_cell_cross(params, ez, rz, k, pgm, pgm_extr, w_l, w_g, bias_model)   
        return self.Survey.l_xc, cl_lg, cl_gl 
    
    def get_expansion_and_rcom(self, params):
        ez, rz, _ = self.Theo.get_ez_rz_k(params, self.Survey.zz_integr)
        return ez, rz
    
    def get_sigma8_from_a_s_from_chain(self, params, nl_model):
        _ = self.Theo.get_emu_status(params, 'HMcode', flag_once=True)
        if nl_model==NL_MODEL_HMCODE or nl_model==NL_MODEL_BACCO:
            sigma8 = self.Theo.HMcode2020Emulator.get_sigma8(params)
        else: 
            sigma8_gr = self.Theo.HMcode2020Emulator.get_sigma8(params) 
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
                _, dz0_gr = self.Theo.get_growth(params_growth, [1.],  nl_model=0)
                params_growth['w0'] = params['w0'][i] if 'w0' in params else -1.
                _, dz0_mg = self.Theo.get_growth(params_growth, [1.],  nl_model=nl_model)
                D_rescale[i] = dz0_mg/dz0_gr
            sigma8 = sigma8_gr*D_rescale
        return sigma8

        

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
                "nl_model": {0: "HMcode", 1: "bacco", 2: "nDGP", 3: "gLEMURS"},
                "bias_model": {0: "b1 constant within bins", 1: "(b1, b2) constant within bins", 2: "bacco"},
                "ia_model": {0: "zNLA", 1: "TATT"},
                "baryon_model": {0: "no baryons", 1: "Tagn HMcode", 2: "bcemu", 3: "bacco"}
            }
            return model_maps.get(model_type, {}).get(value, "Unknown")

        config = self.config_dic
        observable = get_observable_label(config.get("observable", -1))

        output_header = f"""
        ##############################################################
        # Cosmology Pipeline Configuration
        #------------------------------------------------------------
        # Observable: {observable}
        # Sky Fraction (fsky): {self.Survey.fsky}
        # Redshift Binning: {self.Survey.nbin} bins ({self.Survey.zmin} ≤ z ≤ {self.Survey.zmax})
        # for n(z) {self.Survey.survey_name}-like
        # Lensing & Clustering Scales:
        #   - l_min: {self.Survey.lmin}
        #   - l_max (WL): {self.Survey.lmax_wl_vals}
        #   - l_max (GC, XC): {self.Survey.lmax_gc_vals}
        #
        # Data Model:
        #   - Nonlinear Power Spectrum: {get_model_label(self.data_model_dic.get("nl_model", -1), "nl_model")} ({self.data_model_dic.get("nl_model", "N/A")})
        #   - Galaxy Bias Model: {get_model_label(self.data_model_dic.get("bias_model", -1), "bias_model")} ({self.data_model_dic.get("bias_model", "N/A")})
        #   - Intrinsic Alignments: {get_model_label(self.data_model_dic.get("ia_model", -1), "ia_model")} ({self.data_model_dic.get("ia_model", "N/A")})
        #   - Baryon Model: {get_model_label(self.data_model_dic.get("baryon_model", -1), "baryon_model")} ({self.data_model_dic.get("baryon_model", "N/A")})
        #   - Parameters: 
        """
        for par_i in self.params_data_dic.keys():
            output_header += f"\n            #       - {par_i}: {self.params_data_dic[par_i]}"

        output_header += f"""
        #
        # Theory Model:
        #   - Nonlinear Power Spectrum: {get_model_label(self.theo_model_dic.get("nl_model", -1), "nl_model")} ({self.theo_model_dic.get("nl_model", "N/A")})
        #   - Galaxy Bias Model: {get_model_label(self.theo_model_dic.get("bias_model", -1), "bias_model")} ({self.theo_model_dic.get("bias_model", "N/A")})
        #   - Intrinsic Alignments: {get_model_label(self.theo_model_dic.get("ia_model", -1), "ia_model")} ({self.theo_model_dic.get("ia_model", "N/A")})
        #   - Baryon Model: {get_model_label(self.theo_model_dic.get("baryon_model", -1), "baryon_model")} ({self.theo_model_dic.get("baryon_model", "N/A")})
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
