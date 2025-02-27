import numpy as np
from .theory import Theory



class MGLike:
    def __init__(self, TheoClass, data):
        self.Theo = TheoClass
        self.data_dic = data

    
    def loglikelihood_det_3x2pt(self, param_dic, theo_model_dic):
        param_dic_all, status = self.Theo.check_pars(param_dic, theo_model_dic)
        if status:   
            chi2 = 0. 
            cov_theory, cov_theory_high = self.Theo.compute_covariance_3x2pt(param_dic_all, theo_model_dic)
            d_the = np.linalg.det(cov_theory)
            d_mix = np.zeros_like(d_the)
            d_obs = self.data_dic['d_obs']
            d_obs_high = self.data_dic['d_obs_high'] 
            for i in range(2*self.Theo.Survey.nbin):
                new_cov = cov_theory.copy()
                new_cov[:, i] = self.data_dic['cov_observ'][:, :, i]
                d_mix += np.linalg.det(new_cov)

            d_the_high = np.linalg.det(cov_theory_high)
            d_mix_high = np.zeros_like(d_the_high)
            for i in range(self.Theo.Survey.nbin):
                new_cov = cov_theory_high.copy()
                new_cov[:, i] = self.data_dic['cov_observ_high'][:, :, i]
                d_mix_high += np.linalg.det(new_cov)
            d_the = np.concatenate([d_the,d_the_high])
            d_obs = np.concatenate([d_obs,d_obs_high])
            d_mix = np.concatenate([d_mix,d_mix_high])
            chi2 += np.sum((2*self.data_dic['ells']+1)*self.Theo.Survey.fsky*((d_mix/d_the)+np.log(d_the/d_obs)))
            return -0.5*chi2 
        else:
            return -np.inf
        
    def loglikelihood_det_GC(self, param_dic, theo_model_dic):
        param_dic_all, status = self.Theo.check_pars(param_dic, theo_model_dic)
        if status:  
            chi2 = 0. 
            cov_theory = self.Theo.compute_covariance_GC(param_dic_all, theo_model_dic)
            d_the = np.linalg.det(cov_theory)
            d_mix = np.zeros_like(d_the)
            for i in range(self.Theo.Survey.nbin):
                new_cov = np.copy(cov_theory)
                d_obs = self.data_dic['d_obs']
                new_cov[:, i] = self.data_dic['cov_observ'][:, :, i]
                d_mix += np.linalg.det(new_cov)
            chi2 += np.sum((2*self.data_dic['ells']+1)*self.Theo.Survey.fsky*((d_mix/d_the)+np.log(d_the/d_obs)))
            return -0.5*chi2    
        else:
            return -np.inf
        
    def loglikelihood_det_WL(self, param_dic, theo_model_dic):
        param_dic_all, status = self.Theo.check_pars(param_dic, theo_model_dic)
        if status:  
            chi2 = 0. 
            cov_theory = self.Theo.compute_covariance_WL(param_dic_all, theo_model_dic)
            d_the = np.linalg.det(cov_theory)
            d_mix = np.zeros_like(d_the)
            for i in range(self.Theo.Survey.nbin):
                new_cov = np.copy(cov_theory)
                d_obs = self.data_dic['d_obs']
                new_cov[:, i] = self.data_dic['cov_observ'][:, :, i]
                d_mix += np.linalg.det(new_cov)
            chi2 += np.sum((2*self.data_dic['ells']+1)*self.Theo.Survey.fsky*((d_mix/d_the)+np.log(d_the/d_obs)))
            return -0.5*chi2    
        else:
            return -np.inf    
        
 
