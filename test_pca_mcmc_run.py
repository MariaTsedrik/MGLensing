#VERY IMPORTANT TO KEEP THESE ENV VARIABLES SET FOR PCA RUNS!!!
#If you ignore them, then time for one model-evaluation is about 10x longer!
import os

from prompt_toolkit.layout import D
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OMP_PLACES"] = "threads"
#os.environ["OMP_PROC_BIND"] = "spread"
from nautilus import Prior, Sampler
import numpy as np
import MGLensing
import time
import yaml
import copy
import tempfile
import atexit
from scipy.stats import norm
import multiprocessing
from datetime import timedelta

HMCODE = 0
BACCO = 1
REACT_NDGP = 2
REACT_GAMMAZ = 3
REACT_MUSIGMA_Q1 = 4
REACT_DS = 5
REACT_FOFR = 6
EMANTIS_FOFR = 7
NDGP_EMU = 8
REACT_MUKZ_Q123 = 9
REACT_MUSIGMA_Q123 = 10
PSEUDO_MUSPLINE = 11

def get_model_label(value, model_type):
    nl_model_maps = {
        "nl_model": {0: "HMcode", 1: "bacco", 2: "nDGP: ReACT", 3: "gamma+q1: ReACT", 4: "mu-Sigma: ReACT", 5:"Dark Scattering: ReACT", 6:"f(R): ReACT", 7: "f(R): EMANTIS", 8: "nDGP: nDGPemu", 9: "mu-Sigma-k-z+ q123: ReACT", 10: "mu-Sigma-z+ q123: ReACT", 11: "mu-Sigma-z-spline: Pseudo"},
    }
    return nl_model_maps.get(model_type, {}).get(value, "Unknown")

# Perform PCA with numpy.linalg.svd - find rotation matrix
def findPCA(M_data, B_data, L_ch_inv):
    Delta = np.array(np.matmul(L_ch_inv, (B_data - M_data).T).T)
    Usvd, s, vh = np.linalg.svd(Delta.T, full_matrices=True)
    Usvd = Usvd.T
    return Usvd, Delta

_tmp_config_paths = []


def _cleanup_tmp_configs():
    for path in _tmp_config_paths:
        if os.path.exists(path):
            os.remove(path)


atexit.register(_cleanup_tmp_configs)


def _build_model_from_base(base_config_path, nl_model, option=None):
    with open(base_config_path, "r", encoding="utf-8") as file_in:
        config_dic = yaml.safe_load(file_in)

    config_dic = copy.deepcopy(config_dic)
    config_dic["theory"]["nl_model"] = nl_model
    if option is None:
        config_dic["theory"].pop("option", None)
    else:
        config_dic["theory"]["option"] = option

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        yaml.safe_dump(config_dic, tmp_file, sort_keys=False)
        _tmp_config_paths.append(tmp_file.name)
        return MGLensing.MGL(tmp_file.name)

# Varying model, e.g. mu-Sigma linear or pseudo
# In the config file the datavector and the corresponding model is specified
# as well as varying parameters and their priors, 
# and the fixed parameters for the models in the data reduction
_base_pca_config = "ini_files/pca/config_nDGP_lin.yaml"
MGL_model = MGLensing.MGL(_base_pca_config)

cov = MGL_model.Data.data_covariance
D_data = MGL_model.Data.data_vector

L_choleski_uncut = np.linalg.cholesky(np.matrix(cov))
L_choleski_inv_uncut = np.linalg.inv(L_choleski_uncut)
L_ch_inv = L_choleski_inv_uncut

# List of models in the data reduction and decide on the PCA option
model_list = [HMCODE, REACT_FOFR, REACT_NDGP]
pca_option = "linear"

# Build the models for the data reduction
model_1_pca_nl = _build_model_from_base(_base_pca_config, nl_model=model_list[0], option=None)
model_1_pca = _build_model_from_base(_base_pca_config, nl_model=model_list[0], option=pca_option)
model_2_pca_nl = _build_model_from_base(_base_pca_config, nl_model=model_list[1], option=None)
model_2_pca = _build_model_from_base(_base_pca_config, nl_model=model_list[1], option=pca_option)
#model_2_pca_nl = MGLensing.MGL("ini_files/pca/config_fofR_pca.yaml")
#model_2_pca = MGLensing.MGL("ini_files/pca/config_fofR_lin_pca.yaml")
#model_3_pca_nl = _build_model_from_base(_base_pca_config, nl_model=model_list[2], option=None)
#model_3_pca = _build_model_from_base(_base_pca_config, nl_model=model_list[2], option=pca_option)
model_3_pca_nl = MGLensing.MGL("ini_files/pca/config_nDGP_pca.yaml")
model_3_pca = MGLensing.MGL("ini_files/pca/config_nDGP_lin_pca.yaml")

hdf5_name = chain_name = "lsst_y1_test_nDGP_linps_pca_gr_fofr_ndgp_omegarc1"

def log_probability_function(pars):
        param_dic = pars | MGL_model.params_fixed
        param_dic_all, status = MGL_model.Like.Theo.check_pars(param_dic)

        if status:
            D_theory = MGL_model.Like.compute_data_vector(param_dic_all) 
        else:
            return -np.inf

        Diff = (D_data - D_theory)

        # Find Choleski scaled data vector
        Diff_ch = np.array(np.matmul(L_ch_inv, Diff.T))[0]

        ### COMBINE
        #different emulators have different prior-ranges, first check!
        _, status_1 = model_1_pca_nl.Like.Theo.check_pars(param_dic_all)
        _, status_2 = model_2_pca_nl.Like.Theo.check_pars(param_dic_all)
        _, status_3 = model_3_pca_nl.Like.Theo.check_pars(param_dic_all)
        if status_1 and status_2 and status_3:
            # 1: find C_ell for non-linear matter power spectrum
            B1 = model_1_pca_nl.Like.compute_data_vector(param_dic_all)  
            B2 = model_2_pca_nl.Like.compute_data_vector(param_dic_all)  
            B3 = model_3_pca_nl.Like.compute_data_vector(param_dic_all)  
            # 2: find C_ell for linear or pseudo matter power spectrum
            M1 = model_1_pca.Like.compute_data_vector(param_dic_all)  
            M2 = model_2_pca.Like.compute_data_vector(param_dic_all)  
            M3 = model_3_pca.Like.compute_data_vector(param_dic_all)  
        else:
            return -np.inf


        B_data =np.array([B1,B2,B3])
        M_data =np.array([M1,M2,M3])


        # EXTRACT PCA MATRIX
        try:
            Usvd, Delta = findPCA(M_data, B_data, L_ch_inv)
        except:
            return -np.inf
             

        # Cut data vector (choleski cov. matrix = I)
        Diff_cut = np.matmul(Usvd[len(M_data):], Diff_ch.T)
        Likelihood = -0.5*(np.matmul(Diff_cut.T,Diff_cut))
        return Likelihood



prior = Prior()
for par_i in MGL_model.params_model:
    if MGL_model.params_priors[par_i]['type'] == 'G':
        prior.add_parameter(par_i, dist=norm(loc=MGL_model.params_priors[par_i]['p1'] , scale=MGL_model.params_priors[par_i]['p2']))
    elif MGL_model.params_priors[par_i]['type'] == 'U':
        prior.add_parameter(par_i, dist=(MGL_model.params_priors[par_i]['p1'] , MGL_model.params_priors[par_i]['p2']))

 
# Ensure the directories exist
os.makedirs('chains', exist_ok=True)
os.makedirs('chains/hdf5', exist_ok=True)
pca_header = (
    "Models in the data reduction: \n"
    + get_model_label(model_list[0], "nl_model") + "(=" + str(model_list[0]) + ")\n"
    + get_model_label(model_list[1], "nl_model") + "(=" + str(model_list[1]) + ")\n"
    + get_model_label(model_list[2], "nl_model") + "(=" + str(model_list[2]) + ")\n"
    + (
        get_model_label(model_list[3], "nl_model") + "(=" + str(model_list[3]) + ")\n"
        if len(model_list) > 3 else "\n"
    )
    + "\n with PCA option: " + pca_option + "\n" + MGL_model.gen_output_header() 
)
def main():    
    sampler = Sampler(prior, log_probability_function, 
                      filepath='chains/hdf5/'+hdf5_name+'.hdf5', resume=True, #MGL_model.mcmc_resume, 
                      n_live=MGL_model.mcmc_nlive, pool=14)
    start = time.time()
    sampler.run(verbose=MGL_model.mcmc_verbose, discard_exploration=True, n_eff=MGL_model.mcmc_neff)
    log_z = sampler.evidence()
    points, log_w, log_l = sampler.posterior()
    finish = time.time()
    chain_time = finish-start

    np.savetxt("chains/chain_"+chain_name+".txt", np.c_[points, log_w, log_l], 
    header=pca_header,
    footer='log_Z = {log_z};  chain_time = {chain_time} (--> {chain_time_hms} hh:mm:ss)'.format(log_z=log_z, chain_time=chain_time, chain_time_hms=timedelta(seconds=chain_time)))


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all pools are properly closed
        multiprocessing.active_children()

