import numpy as np



class MGLike:
    def __init__(self, TheoClass, data):
        self.Theo = TheoClass
        self.data_dic = data

    
    def loglikelihood_det_3x2pt(self, param_dic, theo_model_dic):
        """
        Compute the log-likelihood for a 3x2pt analysis using the determinant method from arXiv:1210.2194.
        Applicable only for the same scale-cuts per probe in all redshift bins. 

        ..math::
            \chi^2 = \sum_{\ell_{\rm min}}^{\ell_{\rm max}} (2\ell+1) f_{\rm sky} \left( \frac{d^{\rm mix}}{d^{\rm theo}} + \ln{\frac{d^{\rm theo}}{d^{\rm obs}}} \right)\, ,
        where the determinants are computed for each ell-value from NxN matrices with C_ell-values, where N is the number of photo-z bins.
        "obs" stays for the observed/mock data; "theo" -- theoreticla model prediction; "mix" -- mix from theoretical and data C_ell (see Eq. (3.2)) in 2404.11508;
        "high" corresponds to the WL C_ell's with ell>ell_max_GC.

        Parameters:
        -----------
        param_dic : dict
            Dictionary containing the parameters for the theoretical model.
        theo_model_dic : dict
            Dictionary containing the theoretical model data.

        Returns:
        --------
        float
            The log-likelihood value. Returns -0.5 * chi2 if the parameters are valid, 
            otherwise returns -np.inf.
        """
        param_dic_all, status = self.Theo.check_pars(param_dic, theo_model_dic)
        if status:   
            chi2 = 0. 
            cov_theory, cov_theory_high = self.Theo.compute_covariance_3x2pt(param_dic_all, theo_model_dic)
            det_theo = np.linalg.det(cov_theory)
            det_mix = np.zeros_like(det_theo)
            det_obs = self.data_dic['det_obs']
            det_obs_high = self.data_dic['det_obs_high'] 
            for i in range(2*self.Theo.Survey.nbin):
                new_cov = cov_theory.copy()
                new_cov[:, i] = self.data_dic['cov_observ'][:, :, i]
                det_mix += np.linalg.det(new_cov)

            det_theo_high = np.linalg.det(cov_theory_high)
            det_mix_high = np.zeros_like(det_theo_high)
            for i in range(self.Theo.Survey.nbin):
                new_cov = cov_theory_high.copy()
                new_cov[:, i] = self.data_dic['cov_observ_high'][:, :, i]
                det_mix_high += np.linalg.det(new_cov)
            det_theo = np.concatenate([det_theo,det_theo_high])
            det_obs = np.concatenate([det_obs,det_obs_high])
            det_mix = np.concatenate([det_mix,det_mix_high])
            chi2 += np.sum((2*self.data_dic['ells']+1)*self.Theo.Survey.fsky*((det_mix/det_theo)+np.log(det_theo/det_obs)))
            return -0.5*chi2 
        else:
            return -np.inf
        
    def loglikelihood_det_GC(self, param_dic, theo_model_dic):
        """
        Compute the log-likelihood for a photometric galaxy clustering (GC) analysis 
        using the determinant method. Applicable only for the same scale-cuts in all redshift bins. 

        Parameters:
        -----------
        param_dic : dict
            Dictionary containing the parameters for the model.
        theo_model_dic : dict
            Dictionary containing the theoretical model information.

        Returns:
        --------
        float
            The log-likelihood value. If the parameters are invalid, returns negative infinity.
        """
        param_dic_all, status = self.Theo.check_pars(param_dic, theo_model_dic)
        if status:  
            chi2 = 0. 
            cov_theory = self.Theo.compute_covariance_gc(param_dic_all, theo_model_dic)
            det_theo = np.linalg.det(cov_theory)
            det_mix = np.zeros_like(det_theo)
            for i in range(self.Theo.Survey.nbin):
                new_cov = np.copy(cov_theory)
                det_obs = self.data_dic['det_obs']
                new_cov[:, i] = self.data_dic['cov_observ'][:, :, i]
                det_mix += np.linalg.det(new_cov)
            chi2 += np.sum((2*self.data_dic['ells']+1)*self.Theo.Survey.fsky*((det_mix/det_theo)+np.log(det_theo/det_obs)))
            return -0.5*chi2    
        else:
            return -np.inf
        
    def loglikelihood_det_WL(self, param_dic, theo_model_dic):
        """
        Compute the log-likelihood for a shear-shear of weak lensing (WL) analysis 
        using the determinant method. Applicable only for the same scale-cuts in all redshift bins. 

        Parameters:
        -----------
        param_dic : dict
            Dictionary containing the parameters for the model.
        theo_model_dic : dict
            Dictionary containing the theoretical model information.

        Returns:
        --------
        float
            The log-likelihood value. If the parameters are invalid, returns negative infinity.
        """
        param_dic_all, status = self.Theo.check_pars(param_dic, theo_model_dic)
        if status:  
            chi2 = 0. 
            cov_theory = self.Theo.compute_covariance_wl(param_dic_all, theo_model_dic)
            det_theo = np.linalg.det(cov_theory)
            det_mix = np.zeros_like(det_theo)
            for i in range(self.Theo.Survey.nbin):
                new_cov = np.copy(cov_theory)
                det_obs = self.data_dic['det_obs']
                new_cov[:, i] = self.data_dic['cov_observ'][:, :, i]
                det_mix += np.linalg.det(new_cov)
            chi2 += np.sum((2*self.data_dic['ells']+1)*self.Theo.Survey.fsky*((det_mix/det_theo)+np.log(det_theo/det_obs)))
            return -0.5*chi2    
        else:
            return -np.inf    
        
 
