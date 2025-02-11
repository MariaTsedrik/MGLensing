import yaml
import numpy as np
from nautilus import Prior, Sampler
from scipy.stats import norm
import time
import os

os.environ["OMP_NUM_THREADS"] = '1'

EmulatorRanges = {
    'HMcode':
        {
        'Omega_c':      {'p1': 0.1,         'p2': 0.8},    
        'Omega_b':      {'p1':0.01,         'p2': 0.1},  
        'h':            {'p1': 0.4,         'p2': 1.},      
        'log10As':      {'p1': np.log(4.95),   'p2': np.log(54.59)},
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
        #'sigma8_cb':    {'p1': 0.73,         'p2': 0.9},
        'ns':           {'p1': 0.92,         'p2': 1.01}, 
        'Mnu':          {'p1': 0.0,          'p2': 0.4},
        'w0':           {'p1': -1.15,        'p2': -0.85},    
        'wa':           {'p1': -0.3,         'p2': 0.3}, 
        },    
    'baryons':
        {
        'log10Mc_bcemu':    {'p1': 11.4,         'p2': 14.6}, 
        'thej_bcemu':       {'p1': 2.6,          'p2': 7.4},
        'mu_bcemu':         {'p1': 0.2,          'p2': 1.8},
        'gamma_bcemu':      {'p1': 1.3,          'p2': 3.7},
        'delta_bcemu':      {'p1': 3.8,          'p2': 10.2},
        'eta_bcemu':        {'p1': 0.085,        'p2': 0.365},
        'deta_bcemu':       {'p1': 0.085,        'p2': 0.365},
        'log10Tagn':    {'p1': 7.6,         'p2': 8.3},
        'log10Mc_bc':   {'p1': 9.,           'p2': 15.}, 
        'eta_bc':       {'p1': -0.69,        'p2': 0.69},
        'beta_bc':      {'p1': -1.,          'p2': 0.69},
        'log10Mz0_bc':  {'p1': 9.,           'p2': 13.},
        'thetaout_bc':  {'p1': 0.,           'p2': 0.47},
        'thetainn_bc':  {'p1': -2.,          'p2': -0.52},
        'log10Minn_bc': {'p1': 9.,           'p2': 13.5}
        }
}


cosmo_pars = ['Omega_m', 'Ombh2', 'h', 'log10As', 'ns', 'w0', 'wa', 'Mnu']

class MGLike:
    def check_pars(self, param_dic, model_dic):
        param_dic_all = {par_i: param_dic[par_i] for par_i in param_dic.keys()}
        param_dic_all.update({par_i: self.ModelParsFix[par_i] for par_i in self.ModelParsFix.keys()})
        print('param_dic_all: ', set(param_dic_all))
        print('cosmo_par: ', set(cosmo_pars))
        if not set(cosmo_pars).issubset(param_dic_all.keys()):
            print("Some cosmo parameters are missing!!!")
        param_dic_all['Omega_b'] = param_dic_all['Ombh2']/param_dic_all['h']**2    
        param_dic_all['Omega_nu'] = param_dic_all['Mnu']/93.14/param_dic_all['h']**2
        param_dic_all['fb'] = param_dic_all['Omega_b']/param_dic_all['Omega_m']
        if  model_dic['NL_model']==0:
            param_dic_all['Omega_c'] = param_dic_all['Omega_m']-param_dic_all['Omega_b']-param_dic_all['Omega_nu']
            return param_dic_all, all(EmulatorRanges['HMcode'][par_i]['p1'] <= param_dic_all[par_i] <= EmulatorRanges['HMcode'][par_i]['p2'] for par_i in EmulatorRanges['HMcode'])

        elif  model_dic['NL_model']==1:       
            param_dic_all['Omega_cb'] = param_dic_all['Omega_m']-param_dic_all['Omega_nu']
            return param_dic_all, all(EmulatorRanges['bacco'][par_i]['p1'] <= param_dic_all[par_i] <= EmulatorRanges['bacco'][par_i]['p2'] for par_i in EmulatorRanges['bacco'])

        if  model_dic['baryon_model']!=0:
            if any( param_dic_all[par_i] < EmulatorRanges['baryons'][par_i]['p1'] or 
                    param_dic_all[par_i] > EmulatorRanges['baryons'][par_i]['p2']
                    for par_i in param_dic_all if par_i in EmulatorRanges['baryons']):
                return param_dic_all, False
        return param_dic_all, True
    
    def loglikelihood_det_3x2pt(self, param_dic):
        param_dic_all, status = self.check_pars(param_dic, self.theo_model_dic)
        if status:   
            chi2 = 0. 
            Cov_theory, Cov_theory_high = self.compute_covariance(param_dic_all, self.theo_model_dic)
            d_the = np.linalg.det(Cov_theory)
            d_mix = np.zeros_like(d_the)
            d_obs = self.d_obs
            d_obs_high = self.d_obs_high 
            for i in range(2*self.nbin):
                newCov = Cov_theory.copy()
                newCov[:, i] = self.Cov_observ[:, :, i]
                d_mix += np.linalg.det(newCov)

            d_the_high = np.linalg.det(Cov_theory_high)
            d_mix_high = np.zeros_like(d_the_high)
            for i in range(self.nbin):
                newCov = Cov_theory_high.copy()
                newCov[:, i] = self.Cov_observ_high[:, :, i]
                d_mix_high += np.linalg.det(newCov)
            d_the = np.concatenate([d_the,d_the_high])
            d_obs = np.concatenate([d_obs,d_obs_high])
            d_mix = np.concatenate([d_mix,d_mix_high])
            chi2 += np.sum((2*self.ells_WL+1)*self.fsky*((d_mix/d_the)+np.log(d_the/d_obs)))
            return -0.5*chi2 
        else:
            return -np.inf
        
    def loglikelihood_det_single(self, param_dic):
        param_dic_all, status = self.check_pars(param_dic, self.theo_model_dic)
        if status:  
            chi2 = 0. 
            Cov_theory = self.compute_covariance(param_dic_all, self.theo_model_dic)
            d_the = np.linalg.det(Cov_theory)
            d_mix = np.zeros_like(d_the)
            for i in range(self.nbin):
                newCov = np.copy(Cov_theory)
                d_obs = self.d_obs #np.linalg.det(Cov_observ) 
                newCov[:, i] = self.Cov_observ[:, :, i]
                d_mix += np.linalg.det(newCov)
            chi2 += np.sum((2*self.ells_one_probe+1)*self.fsky*((d_mix/d_the)+np.log(d_the/d_obs)))
            return -0.5*chi2    
        else:
            return -np.inf
    
#-------------------------------------------------------------------------------

class Sampler:
    def gen_output_header(self, mcmc=False):
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
        data_models = config.get("data", {})
        theory_models = config.get("theory", {})
        params_data = self.params_data_dic
        params_priors = self.MasterPriors
        print('params_priors', params_priors.keys())
        output_header = f"""
        ##############################################################
        # Cosmology Pipeline Configuration
        #------------------------------------------------------------
        # Observable: {observable} ({config.get("observable", "N/A")})
        # Sky Fraction (fsky): {specs.get("fsky", "N/A")}
        # Redshift Binning: {specs.get("nbin", "N/A")} bins ({specs.get("zmin", "N/A")} ≤ z ≤ {specs.get("zmax", "N/A")})
        # for n(z) {get_nofz_label(specs.get("n_of_z"))}-like
        # Lensing & Clustering Scales:
        #   - l_min: {specs.get("lmin", "N/A")}
        #   - l_bin: {specs.get("lbin", "N/A")}
        #   - l_max (WL): {specs.get("lmax_WL", "N/A")}
        #   - l_max (GC, XC): {specs.get("lmax_GC", "N/A")}
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
        if mcmc:
            for par_i in self.ModelPars:
                output_header += "   " + "   ".join(par_i)
            output_header += "   log_w   log_l"    
        return output_header

    def nautilus(self, 
                n_live_points=1000,n_pools=7, n_eff=5000, resume_option=False, verbose_option=True):
        """
        Implementation of the nautilus sampler.
        """
        prior = Prior()
        #self.ndim = len(self.ModelPars)
        for par_i in self.ModelPars:
            if self.MasterPriors[par_i]['type'] == 'G':
                prior.add_parameter(par_i, dist=norm(loc=self.MasterPriors[par_i]['p1'] , scale=self.MasterPriors[par_i]['p2']))
            elif self.MasterPriors[par_i]['type'] == 'U':
                prior.add_parameter(par_i, dist=(self.MasterPriors[par_i]['p1'] , self.MasterPriors[par_i]['p2']))
        if self.Probe=='3x2pt':
            sampler = Sampler(prior, self.loglikelihood_det_3x2pt, filepath=self.PATH+'chains/hdf5/'+self.hdf5_name+'.hdf5', resume=resume_option, n_live=n_live_points, pool=n_pools)
        else:
            sampler = Sampler(prior, self.loglikelihood_det, filepath=self.PATH+'chains/hdf5/'+self.hdf5_name+'.hdf5', resume=resume_option, n_live=n_live_points, pool=n_pools)
        start = time.time()
        sampler.run(verbose=verbose_option, discard_exploration=True, n_eff=n_eff)
        log_z = sampler.evidence()
        points, log_w, log_l = sampler.posterior()
        finish = time.time()
        self.chain_time = finish-start
        np.savetxt(self.PATH+"chains/chain"+self.chain_name+".txt", np.c_[points, log_w, log_l], header=self.gen_output_header(mcmc=True))



    def test(self, plot=False):
        dic_test = {par_i: self.MasterPriors[par_i]['p0'] for par_i in self.MasterPriors.keys()}
        dic_test.update({par_i: self.ModelParsFix[par_i] for par_i in self.ModelParsFix.keys()})
        #print(dic_test)
        if self.Probe=='3x2pt':
            start = time.time()
            test_like = self.loglikelihood_det_3x2pt(dic_test)
            finish = time.time()
            test_time = finish-start
        else:
            start = time.time()
            test_like = self.loglikelihood_det(dic_test)    
            finish = time.time()
            test_time = finish-start
        test_txt = "loglikelihood = "+ str(test_like)+ "  evaluation took  "+ str(test_time) + "  s"
        np.savetxt(self.PATH+"chains/chain_"+self.chain_name+".txt", [], header=self.gen_output_header()+test_txt)
        if plot:
            print('ahoi')
            



