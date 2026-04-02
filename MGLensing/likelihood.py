import numpy as np
from scipy.linalg import cholesky, solve_triangular
class MGLike:
    def __init__(self, Model, Data, use_jax=False):
        self.Theo = Model
        self.Data = Data
        self._use_jax = use_jax and getattr(Model.Survey, 'observable', None) == '3x2pt' and getattr(Model.Survey, 'likelihood', None) == 'binned'
        self._log_like_jax = None
        if self._use_jax:
            try:
                from .jax_cell_like import build_log_likelihood_jax, JAX_AVAILABLE
                if JAX_AVAILABLE:
                    use_cholesky = hasattr(Data, 'cholesky_transform') and Data.cholesky_transform is not None
                    self._log_like_jax = build_log_likelihood_jax(use_cholesky=use_cholesky)
                else:
                    self._use_jax = False
            except Exception:
                self._use_jax = False
        type = Model.Survey.likelihood
        if type == 'determinants':
            if Model.Survey.observable == '3x2pt':
                self.compute = self.loglikelihood_det_3x2pt
            elif Model.Survey.observable == 'WL':
                self.compute_data_matrix = self.Theo.compute_data_matrix_wl
                self.compute = self.loglikelihood_det_wl
            elif Model.Survey.observable == 'GC':
                self.compute_data_matrix = self.Theo.compute_data_matrix_gc
                self.compute = self.loglikelihood_det_gc    
    
        else:
            if Model.Survey.observable == 'WL':
                self.compute_data_vector = self.Theo.compute_data_vector_wl
            elif Model.Survey.observable == 'GC':
                self.compute_data_vector = self.Theo.compute_data_vector_gc   
            elif Model.Survey.observable == '3x2pt':
                self.compute_data_vector = self.Theo.compute_data_vector_3x2pt    
            self.compute = self.loglikelihood_binned_wrapper      

    def loglikelihood_det_3x2pt(self, param_dic):
        r"""Compute the log-likelihood for a 3x2pt analysis using the determinant method from https://arxiv.org/pdf/1210.2194.
        Applicable only for the same scale-cuts per probe in all redshift bins. The :math:`\chi^2`-value is computed as

        .. math::
            \chi^2 = \sum_{\ell_{\rm min}}^{\ell_{\rm max}} (2\ell+1) f_{\rm sky} \left( \frac{d^{\rm mix}}{d^{\rm theo}} + \ln{\frac{d^{\rm theo}}{d^{\rm obs}}} \right)\, ,
        
        where the determinants are computed for each :math:`\ell`-value from :math:`N \times N` matrices with :math:`C_\ell`-values, :math:`N` is the number of photo-z bins.
        The superscripts meaning: "obs" stays for the observed/mock data; "theo" -- theoretical model prediction; "mix" -- mix from theoretical and data :math:`C_\ell` (see Eq. 3.2 in https://arxiv.org/pdf/2404.11508);
        "high" corresponds to the WL :math:`C_\ell`'s with :math:`\ell > \rm{max}(\ell_{\rm GC})`.

        Parameters:
        -----------
        param_dic : dict
            Dictionary containing the parameters for the theoretical model.
        theo_model_dic : dict
            Dictionary containing the theoretical model data.

        Returns:
        --------
        float
            The log-likelihood value. Returns -0.5 * \(\chi^2\) if the parameters are valid, 
            otherwise returns -np.inf.
        """
        param_dic_all, status = self.Theo.check_pars(param_dic)
        if status:   
            chi2 = 0. 
            cov_theo, cov_theo_high = self.Theo.compute_data_matrix_3x2pt(param_dic_all)
            det_theo = np.linalg.det(cov_theo)
            det_mix = np.zeros_like(det_theo)
            det_obs = self.Data.det_obs
            det_obs_high = self.Data.det_obs_high
            for i in range(self.Theo.Survey.nbin_s+self.Theo.Survey.nbin_l):
                new_cov = cov_theo.copy()
                new_cov[:, i] = self.Data.cov_obs[:, :, i]
                det_mix += np.linalg.det(new_cov)

            det_theo_high = np.linalg.det(cov_theo_high)
            det_mix_high = np.zeros_like(det_theo_high)
            for i in range(self.Theo.Survey.nbin_s):
                new_cov = cov_theo_high.copy()
                new_cov[:, i] = self.Data.cov_obs_high[:, :, i]
                det_mix_high += np.linalg.det(new_cov)
            det_theo = np.concatenate([det_theo,det_theo_high])
            det_obs = np.concatenate([det_obs,det_obs_high])
            det_mix = np.concatenate([det_mix,det_mix_high])
            chi2 += np.sum((2*self.Data.ells+1)*self.Theo.Survey.fsky*((det_mix/det_theo)+np.log(det_theo/det_obs)))
            return -0.5*chi2 
        else:
            return -np.inf


    
    def loglikelihood_det_wl(self, param_dic):
        """Compute the log-likelihood for a given set of parameters using the determinant method.

        Parameters:
        -----------
        param_dic : dict
            Dictionary containing the parameters for the model.

        Returns:
        --------
        float
            The log-likelihood value. If the parameters are invalid, returns negative infinity.
        """
        param_dic_all, status = self.Theo.check_pars(param_dic)
        if status:  
            chi2 = 0. 
            cov_theo = self.compute_data_matrix(param_dic_all)
            det_theo = np.linalg.det(cov_theo)
            det_mix = np.zeros_like(det_theo)
            for i in range(self.Theo.Survey.nbin_s):
                new_cov = np.copy(cov_theo)
                det_obs = self.Data.det_obs
                new_cov[:, i] = self.Data.cov_obs[:, :, i]
                det_mix += np.linalg.det(new_cov)
            chi2 += np.sum((2*self.Data.ells+1)*self.Theo.Survey.fsky*((det_mix/det_theo)+np.log(det_theo/det_obs)))
            return -0.5*chi2    
        else:
            return -np.inf    
        
        
    def loglikelihood_det_gc(self, param_dic):
        """Compute the log-likelihood for a given set of parameters using the determinant method.

        Parameters:
        -----------
        param_dic : dict
            Dictionary containing the parameters for the model.

        Returns:
        --------
        float
            The log-likelihood value. If the parameters are invalid, returns negative infinity.
        """
        param_dic_all, status = self.Theo.check_pars(param_dic)
        if status:  
            chi2 = 0. 
            cov_theo = self.compute_data_matrix(param_dic_all)
            det_theo = np.linalg.det(cov_theo)
            det_mix = np.zeros_like(det_theo)
            for i in range(self.Theo.Survey.nbin_l):
                new_cov = np.copy(cov_theo)
                det_obs = self.Data.det_obs
                new_cov[:, i] = self.Data.cov_obs[:, :, i]
                det_mix += np.linalg.det(new_cov)
            chi2 += np.sum((2*self.Data.ells+1)*self.Theo.Survey.fsky*((det_mix/det_theo)+np.log(det_theo/det_obs)))
            return -0.5*chi2    
        else:
            return -np.inf      
        
    def loglikelihood_binned_wrapper(self, param_dic):
        """Binned log-likelihood: use JAX path if enabled and available, else Python."""
        param_dic_all, status = self.Theo.check_pars(param_dic)
        if not status:
            return -np.inf
        if self._use_jax and self._log_like_jax is not None:
            try:
                theory_arrays = self.Theo.get_theory_arrays_3x2pt(param_dic_all)
            except Exception:
                return -np.inf
            if self.Theo.flag_nan:
                return -np.inf
            cov_info = self.Data.cholesky_transform if hasattr(self.Data, 'cholesky_transform') and self.Data.cholesky_transform is not None else self.Data.inv_data_covariance
            mask = self.Theo.Survey.mask_data_vector_3x2pt
            return self._log_like_jax(theory_arrays, self.Data.data_vector, cov_info, mask)
        return self.loglikelihood(param_dic)
 
    def loglikelihood(self, param_dic):
        """Calculate the log-likelihood for a given set of parameters using the difference between model's and data's vectors.

        Parameters:
        -----------
        param_dic : dict
            Dictionary containing the parameters for which the log-likelihood is to be calculated.

        Returns:
        --------
        float
            The log-likelihood value. If the parameters are invalid, returns negative infinity.
        """
        param_dic_all, status = self.Theo.check_pars(param_dic)
        if status:  
            chi2 = 0.
            model_data_vector = self.compute_data_vector(param_dic_all) 
            if self.Theo.flag_nan:
                #print('detected NaNs before computing likelihood')
                return -np.inf
            difference_vector = self.Data.data_vector - model_data_vector

            try:
                yt = solve_triangular(self.Data.cholesky_transform, difference_vector, lower=True)
                chi2 = yt.dot(yt)
            except:
                # If the cholesky transform fails, we can try to use the inverse covariance matrix
                #print(('solve_triangular breaks when ', param_dic))
                if hasattr(self.Data, 'inv_data_covariance') and self.Data.inv_data_covariance is not None:
                    chi2 = np.einsum('i, ij, j->', difference_vector, self.Data.inv_data_covariance, difference_vector)
                else:
                    return -np.inf  
 
            return -0.5*chi2    
        else:
            return -np.inf      
        
   