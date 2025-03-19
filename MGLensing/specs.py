import numpy as np
from scipy.special import erf
from scipy.integrate import trapezoid,quad
from scipy.interpolate import interp1d
from math import sqrt, log10, exp, pow
import sacc
import yaml
import warnings
import os
DEG2_IN_SPHERE = 4 * np.pi * (180 / np.pi)**2

dirname = os.path.split(__file__)[0]
print(dirname)
H0_h_c = 1./2997.92458 #=100/c in Mpc/h
z_bins_for_integration = 200

def get_luminosity_func(file_name):
    """
    Reads the luminosity function data from a file and returns an interpolation function.

    Parameters:
    ----------
    file_name : str
        The path to the file containing the luminosity function data.

    Returns:
    -------
    function
        An interpolation function that takes redshift values and returns the corresponding luminosity values.
    """
    lum_file = open(file_name, 'r')
    content = lum_file.readlines()
    zlum = np.zeros((len(content)))
    lum = np.zeros((len(content)))
    for index in range(len(content)):
        line = content[index]
        zlum[index] = line.split()[0]
        lum[index] = line.split()[1]
    return interp1d(zlum, lum,kind='linear')

def reduce_len_by_averaging(arr, target_len=z_bins_for_integration):
    """
    Reduces the length of an array by averaging its elements to match the target length.

    Parameters:
    ---------
    arr : list or numpy.ndarray
        The input array to be reduced.
    target_len : int, optional
        The desired length of the output array. Default is z_bins_for_integration.

    Returns:
    -------
    numpy.ndarray
        The reduced array with the specified target length.
    """
    # Number of elements to average together
    step = len(arr) // target_len  
    reduced_arr = np.array([np.mean(arr[i:i+step]) for i in range(0, len(arr), step)])
    return reduced_arr[:target_len]

def get_noise(nbins, gal_per_sqarcmn):
    """
    Calculate the noise for a given number of redshift bins and galaxy density.

    Parameters:
    ----------
    nbins : int
        The number of redshift bins.
    gal_per_sqarcmn : float
        The number of galaxies per square arcminute.

    Returns:
    -------
    float
        The noise value for the given number of redshift bins and galaxy density.
    """
    n_bar = gal_per_sqarcmn * (60.*180./np.pi)**2 #in 1/radians^2=1/sr
    n_bar /= nbins
    return n_bar

def check_length(scale_cuts_info, nbin):
    """
    Check the length of scale cuts information based on the type and number of bins.

    Parameters:
    ----------
    scale_cuts_info : dict
        Dictionary containing scale cuts information with keys 'type', 'max_WL', and 'max_GC'.
    nbin : int
        Number of bins.

    Raises:
    ------
    ValueError
        If the length of 'max_WL' or 'max_GC' does not match the expected number of bins.
    """
    labels = ['max_WL', 'max_GC']
    nbins = int(nbin*(1+nbin)/2)
    if scale_cuts_info['type'] == 'lmax':
        for label in labels:
            if len(scale_cuts_info[label])!=nbins:
                raise ValueError(label+' length must be equal to '+str(nbins)+'!')
    elif scale_cuts_info['type'] == 'kmax':
        for label in labels:
            if len(scale_cuts_info[label])!=nbin:
                raise ValueError(label+' length must be equal to '+str(nbin)+'!')


def get_rcom(params_dic, z_max):
    """
    Calculate the comoving distance to a given redshift z_max.
    
    Parameters:
    ----------
    params_dic (dict): Dictionary containing cosmological parameters. 
                        Expected keys are 'Omega_m', 'w0', and 'wa'.
                        Defaults are used if keys are not present:
                        'Omega_m' defaults to 0.31,
                        'w0' defaults to -1.0,
                        'wa' defaults to 0.0.
    z_max (float): The maximum redshift value to calculate the comoving distance for.
    
    Returns:
    -------
    float: The comoving distance to the given redshift z_max in units of Mpc/h.
    """
    omega_m = params_dic['Omega_m'] if 'Omega_m' in params_dic else 0.31
    w0 = params_dic['w0'] if 'w0' in params_dic else -1.
    wa = params_dic['wa'] if 'wa' in params_dic else 0.
    omega_lambda_func = lambda z: (1.-omega_m) * pow(1.+z, 3.*(1.+w0+wa)) * exp(-3.*wa*z/(1.+z))
    r_z_int = lambda z: 1./sqrt(omega_m*pow(1.+z, 3) + omega_lambda_func(z))
    r_z_func = lambda z_in: quad(r_z_int, 0, z_in)[0]
    r_z = r_z_func(z_max)/H0_h_c #Mpc/h
    return r_z

def from_keff_to_lmax(n, k_eff, z_mod, params_dic):
    """
    Calculate lmax for given effective wavenumbers and redshift.
    Always for a lower photo-z: e.g., bin 2-1 or bin 4-1 will take k_eff for bin 1 to compute ell_max.

    Parameters:
    ----------
    n (int): The number of bins.
    k_eff (list of float): The effective wavenumbers for each bin in h/Mpc.
    z_mod (list of float): The redshift, peak of the kernel, for each bin.
    params_dic (dict): Dictionary containing cosmological parameters.

    Returns:
    -------
    list of float: The calculated lmax values for each bin combination.
    """
    lmax = []
    for bin1 in range(n):
        for bin2 in range(bin1, n):
            z = z_mod[bin1]
            lmax_i = k_eff[bin1] * get_rcom(params_dic, z) / (1. + z)
            lmax.append( lmax_i )
            print('bin1, bin2: ', bin1, bin2, 'k_eff[bin1]: ', k_eff[bin1], 'lmax: ', lmax_i)
    return lmax

def max_ells_for_plots(n, flat_array):
    matrix_ell_cuts = np.zeros((n, n), dtype=int)
    iu1 = np.triu_indices(n)
    matrix_ell_cuts[iu1] = flat_array
    matrix_ell_cuts = matrix_ell_cuts + matrix_ell_cuts.T - np.diag(np.diag(matrix_ell_cuts))
    return matrix_ell_cuts

def setup_const_lmax(obj, likelihood):
    """
    Sets up the maximum multipole moments (lmax) for weak lensing (wl) and galaxy clustering (gc)
    based on the provided likelihood type. It also configures the ell bins, masks, and delta ells
    for covariance computation.

    Parameters:
    ----------
    obj : object
        An object containing various attributes related to ell bins and lmax values.
    likelihood : str
        The type of likelihood ('determinants' or other) to determine the setup process.

    Returns:
    -------
    None
    """
    nbin_flat = obj.nbin_flat
    if likelihood == 'determinants':
        idx_lmax_wl = int(np.argwhere(obj.ell >= int(obj.lmax_wl_vals))[0]) 
        # binned ell_wl cut at highest lmax
        obj.l_wl = obj.ell[:idx_lmax_wl+1] 
        idx_lmax_gc = int(np.argwhere(obj.ell >= int(obj.lmax_gc_vals))[0]) 
        # binned ell_gc cut at highest lmax
        obj.l_gc = obj.ell[:idx_lmax_gc+1]
        # we set ells for the cross-correlation to be equal to photo-gc
        obj.l_xc = obj.l_gc
        # number of ell-bins
        obj.nell_wl = len(obj.l_wl)
        obj.nell_gc = len(obj.l_gc)
        obj.nell_xc = len(obj.l_xc)
        # all integer ells cut at highest lmax
        obj.ells_wl = np.array(range(obj.lmin, int(obj.lmax_wl_vals)+1)) 
        obj.ells_gc = np.array(range(obj.lmin, int(obj.lmax_gc_vals)+1)) # all integer ells
        obj.ell_jump =  int(obj.lmax_gc_vals) - obj.lmin + 1
    else:
        obj.lmax_wl_vals = np.full(obj.nbin_flat, obj.lmax_wl_vals)
        obj.lmax_gc_vals = np.full(obj.nbin_flat, obj.lmax_gc_vals)
        setup_lmax(obj)

    obj.ells_wl_max = max_ells_for_plots(obj.nbin, np.full(nbin_flat, obj.lmax_wl_vals))
    obj.ells_gc_max = max_ells_for_plots(obj.nbin, np.full(nbin_flat, obj.lmax_gc_vals))    


def setup_lmax(obj):
    """
    Sets up the maximum multipole moments (lmax) for different probes in the given object.
    The function computes masks for data vectors and covariance matrices for weak lensing (wl) 
    and galaxy clustering (gc) probes.

    Parameters:
    ----------
    obj : object
        The object containing the necessary attributes for setting up lmax.

    Attributes used from obj:
    ------------------------
    - l_wl_bin_centers
    - d_ell_bin
    - ell_bin_edges
    - lmax_wl_vals
    - lmax_gc_vals
    - nbin

    Returns:
    -------
    None
    """
    # for binned ansatz:
    # we compute model/data/covariance at all available ell-bins
    # and apply mask afterwards;
    obj.l_wl = obj.l_xc = obj.l_gc = obj.l_wl_bin_centers
    obj.nell_wl = obj.nell_xc = obj.nell_gc = len(obj.l_wl)
    # delta ells for covariance computation
    obj.d_ell_bin_cut_wl = obj.d_ell_bin_cut_gc = obj.d_ell_bin_cut_xc = obj.d_ell_bin
    # compute masks
    # again we assume that lmax_gc = lmax_lg, but this can be easily changed
    mask_ells_wl = np.array([(obj.ell_bin_edges<= lmax_wl_i) for lmax_wl_i in obj.lmax_wl_vals])
    mask_ells_wl = mask_ells_wl[:, 1:]
    mask_ells_gc = np.array([(obj.ell_bin_edges<= lmax_gc_i) for lmax_gc_i in obj.lmax_gc_vals])
    mask_ells_gc = mask_ells_gc[:, 1:]
    # later add an option to define xc cuts independently
    lmax_xc_vals_2d = max_ells_for_plots(obj.nbin, obj.lmax_gc_vals)
    obj.lmax_xc_vals = np.concatenate(lmax_xc_vals_2d)
    mask_ells_xc = np.array([(obj.ell_bin_edges<= lmax_xc_i) for lmax_xc_i in obj.lmax_xc_vals])
    mask_ells_xc = mask_ells_xc[:, 1:]
    obj.mask_data_vector_wl = np.concatenate((mask_ells_wl))
    obj.mask_data_vector_gc = np.concatenate((mask_ells_gc))
    obj.mask_data_vector_xc = np.concatenate((mask_ells_xc))
    obj.mask_data_vector_3x2pt = np.concatenate((obj.mask_data_vector_wl, obj.mask_data_vector_xc, obj.mask_data_vector_gc))
    obj.mask_cov_wl = np.ix_(obj.mask_data_vector_wl, obj.mask_data_vector_wl)
    obj.mask_cov_gc = np.ix_(obj.mask_data_vector_gc, obj.mask_data_vector_gc)
    obj.mask_cov_3x2pt = np.ix_(obj.mask_data_vector_3x2pt, obj.mask_data_vector_3x2pt)
    obj.ells_wl_max = max_ells_for_plots(obj.nbin, obj.lmax_wl_vals)
    obj.ells_gc_max = max_ells_for_plots(obj.nbin, obj.lmax_gc_vals)
    


def validate_and_setup_lmax(obj, scale_cuts_info, likelihood, lmin, lmax, zz_mod_wl, zz_mod_gg):
    """
    Validates and sets up the lmax values for weak lensing (WL) and galaxy clustering (GC) 
    based on the provided scale cuts information.

    Parameters:
    ----------
    obj : object
        The object containing the necessary attributes for setting up lmax values.
    scale_cuts_info : dict
        Dictionary containing scale cuts information with keys:
        - 'type' (str): Type of scale cut ('const_lmax', 'lmax', or 'kmax').
        - 'max_WL' (list): Maximum l values for weak lensing.
        - 'max_GC' (list): Maximum l values for galaxy clustering.
        - 'cosmo' (dict, optional): Cosmological parameters required for 'kmax' type.
    likelihood : str
        The likelihood approach ('binned' or other).
    lmin : int
        Minimum l value.
    lmax : int
        Maximum l value.
    zz_mod_wl : array
        Redshift modification array for weak lensing.
    zz_mod_gg : array
        Redshift modification array for galaxy clustering.

    Raises:
    ------
    ValueError
        If any validation checks fail, such as:
        - lmax exceeding the provided maximum value.
        - Invalid scale cut type.
        - Incompatible likelihood approach for varied lmax.
        - lmax_gc being greater than lmax_wl in 'const_lmax' type.
    """
    obj.nbin_flat = int(0.5 * obj.nbin * (obj.nbin + 1))
    if any(val > lmax for val in scale_cuts_info['max_WL'] + scale_cuts_info['max_GC']):
            raise ValueError(f'lmax cannot exceed {lmax}!!!')    
    if scale_cuts_info['type'] == 'const_lmax':
        obj.lmax_wl_vals = scale_cuts_info['max_WL'][0]
        obj.lmax_gc_vals= scale_cuts_info['max_GC'][0]
        if obj.lmax_wl_vals<obj.lmax_gc_vals:
            raise ValueError('Only lmax_gc<=lmax_wl case is implemented!')
        setup_const_lmax(obj, likelihood)
    elif scale_cuts_info['type'] == 'lmax':
        if likelihood!='binned':
            raise ValueError('Varied lmax is applicable only for the binned approach!')
        check_length(scale_cuts_info, obj.nbin)
        obj.lmax_wl_vals = scale_cuts_info['max_WL']
        obj.lmax_gc_vals= scale_cuts_info['max_GC']
        setup_lmax(obj)
    elif scale_cuts_info['type'] == 'kmax':   
        if likelihood!='binned':
            raise ValueError('Varied lmax is applicable only for the binned approach!') 
        check_length(scale_cuts_info, obj.nbin)
        params_dic = scale_cuts_info['cosmo']
        lmax_wl = from_keff_to_lmax(obj.nbin, scale_cuts_info['max_WL'], zz_mod_wl, params_dic)
        lmax_wl = np.array(lmax_wl).astype(int)
        lmax_gc = from_keff_to_lmax(obj.nbin, scale_cuts_info['max_GC'], zz_mod_gg, params_dic)
        lmax_gc = np.array(lmax_gc).astype(int)
        obj.lmax_wl_vals = lmax_wl 
        obj.lmax_gc_vals = lmax_gc 
        print('lmax_wl: ', lmax_wl)
        print('lmax_gc: ', lmax_gc)
        if any(val > lmax for val in lmax_wl) or  any(val > lmax for val in lmax_gc):
            raise ValueError(f'lmax cannot exceed {lmax}!!!')  
        setup_lmax(obj)
    else:
        raise ValueError('Invalid scale-cut type') 

class LSSTSetUp:
    """
    A class to set up the LSST survey specifications and configurations.

    Attributes
    ----------
    survey_name : str
        Name of the survey, default is 'LSST'.
    observable : str
        Observable configuration from the input config.
    zmin : float
        Minimum redshift value.
    zmax : float
        Maximum redshift value.
    fsky : float
        Fraction of the sky covered by the survey.
    gal_per_sqarcmn : float
        Number of galaxies per square arcminute.
    gal_per_sqarcmn_s : float
        Number of source galaxies per square arcminute.
    gal_per_sqarcmn_l : float
        Number of lens galaxies per square arcminute.
    rms_shear : float
        Root mean square of the shear.
    nbin : int
        Number of redshift bins.
    z_bin_center_s : np.ndarray
        Centers of the source redshift bins.
    z_bin_center_l : np.ndarray
        Centers of the lens redshift bins.
    eta_z_s : np.ndarray
        Normalized source galaxy distribution as a function of redshift.
    eta_z_l : np.ndarray
        Normalized lens galaxy distribution as a function of redshift.
    zz_integr : np.ndarray
        Array of redshift values used for integration.
    zbin_integr : int
        Number of redshift bins used for integration.
    noise : dict
        Dictionary containing noise values for different combinations of lensing and galaxy-galaxy correlations.
    aa_integr : np.ndarray
        Array of scale factors corresponding to the redshift values used for integration.
    lmin : int
        Minimum multipole value.
    lbin : int
        Number of multipole bins.
    lmax : float
        Maximum multipole value.
    ell : np.ndarray
        Array of multipole values.
    ell_bin_edges : np.ndarray
        Edges of the multipole bins.
    d_ell_bin : np.ndarray
        Differences between consecutive multipole bin edges.
    l_wl_bin_centers : np.ndarray
        Centers of the multipole bins.
    lum_func : np.ndarray
        Luminosity function data.
    likelihood : str
        Likelihood configuration from the input config.
    lmax_wl_vals : np.ndarray or float
        Maximum scale-cuts for weak lensing.
    lmax_gc_vals : np.ndarray or float
        Maximum scale-cuts for galaxy clustering.
    l_wl : np.ndarray 
        Binned ell-values for weak lensing.
    l_xc : np.ndarray 
        Binned ell-values for cross-correlation.
    l_gc : np.ndarray 
        Binned ell-values for galaxy clustering.
    nell_wl : int
        Number of ell-bins for weak lensing.
    nell_xc : int
        Number of ell-bins for cross-correlation.
    nell_gc : int
        Number of ell-bins for galaxy clustering.
    ells_wl : np.ndarray 
        All integer ell-values from lmin to lmax for weak lensing.
    ells_gc : np.ndarray 
        All integer ell-values from lmin to lmax for galaxy clustering.
    ell_jump : int
        Equals to lmax_gc, as we assume that lmax_gc < lmax_wl; required for likelihood with determinants.
    mask_data_vector_wl : np.ndarray
        Boolean array to mask out data vectors for weak lensing.
    mask_data_vector_gc : np.ndarray
        Boolean array to mask out data vectors for galaxy clustering.
    mask_data_vector_3x2pt : np.ndarray
        Boolean array to mask out data vectors for 3x2pt.
    mask_cov_wl : np.ndarray 
        Boolean array to mask out covariances for weak lensing.
    mask_cov_gc : np.ndarray 
        Boolean array to mask out covariances for galaxy clustering.
    mask_cov_3x2pt : np.ndarray 
        Boolean array to mask out covariances for 3x2pt.
    d_ell_bin_cut_wl : np.ndarray 
        Delta ell or bin width required for covariance computation for weak lensing.
    d_ell_bin_cut_xc : np.ndarray 
        Delta ell or bin width required for covariance computation for cross-correlation.
    d_ell_bin_cut_gc : np.ndarray 
        Delta ell or bin width required for covariance computation for galaxy clustering.
    ells_wl_max : np.ndarray 
        Scale-cuts for weak lensing in the form appropriate for plotting scripts.
    ells_gc_max : np.ndarray 
        Scale-cuts for galaxy clustering in the form appropriate for plotting scripts.
    """
    def __init__(self, config:dict):
        survey_info = config['specs']['survey_info']
        scale_cuts_info = config['specs']['scale_cuts']
        likelihood = config['likelihood']
        self.survey_name = 'LSST'
        self.observable = config['observable']


        # redshift setup 
        self.zmin = 0.01 
        self.zmax = 3.0 #actually 4.0,  but changed because of bacco-emu 
        if survey_info == 'LSST_Y1':
            self.fsky = 0.43 #0.5
            self.gal_per_sqarcmn = 20.0
            self.gal_per_sqarcmn_s = 10.0
            self.gal_per_sqarcmn_l = 18.0
            self.rms_shear = 0.26
            self.nbin = 5 
            nz_sacc_file = dirname+'/forecast_fid.sacc'
            self.z_bin_center_s = np.array([2.253273944137912099e-01, 4.490567958326107112e-01, 6.545664037512012312e-01, 9.258332934738261466e-01, 1.590547464182611836e+00]) #from DESC github
            self.z_bin_center_l = np.array([3.083232685515536753e-01, 5.011373846158426737e-01, 6.981257392184466726e-01, 8.964730913147888058e-01, 1.095412294727920344e+00])  #from DESC github
        elif survey_info == 'LSST_Y10': 
            raise NotImplementedError('LSST_Y10 is not implemented yet')  
        # normalized galaxy distribution and noise
        self.eta_z_s, self.eta_z_l = self.get_norm_galaxy_distrib(nz_sacc_file) 
        zz_mod_wl = [self.zz_integr[np.argmax(self.eta_z_s[:, i])] for i in range (self.nbin)]
        zz_mod_gg = [self.zz_integr[np.argmax(self.eta_z_l[:, i])] for i in range (self.nbin)]
        n_bar = get_noise(self.nbin, self.gal_per_sqarcmn)
        n_bar_l = get_noise(self.nbin, self.gal_per_sqarcmn_l)
        n_bar_s = get_noise(self.nbin, self.gal_per_sqarcmn_s)
        self.noise = {
        'LL': self.rms_shear**2./n_bar_l,
        'LG': 0.,
        'GL': 0.,
        'GG': 1./n_bar_s
        }
        self.aa_integr = np.array(1./(1.+self.zz_integr[::-1])) 
        self.zbin_integr = len(self.zz_integr)        
        # \ell bins setup
        lmin = 20
        self.lmin = lmin
        self.lbin = 50 #100 
        lmax = 5e3
        self.lmax = lmax
        if likelihood == 'determinants':
            self.ell = np.logspace(log10(lmin), log10(lmax), num=self.lbin, endpoint=True) 
        elif likelihood == 'binned':
            self.ell_bin_edges = np.logspace(log10(lmin), log10(lmax), num=self.lbin+1, endpoint=True) 
            self.d_ell_bin = np.diff(self.ell_bin_edges)
            # means in log-space:
            self.l_wl_bin_centers = np.sqrt(self.ell_bin_edges[:-1] *self.ell_bin_edges[1:])
            self.ell = self.l_wl_bin_centers
            # means in lin-space:
            # self.l_wl_bin_centers = np.array([0.5*(self.ell_bin_edges[i+1]+self.ell_bin_edges[i]) for i in range(self.lbin)])
        validate_and_setup_lmax(self, scale_cuts_info, likelihood, lmin, lmax, zz_mod_wl, zz_mod_gg)
        self.lum_func = get_luminosity_func(dirname+'/scaledmeanlum_E2Sa.dat')
        self.likelihood = likelihood

    def get_norm_galaxy_distrib(self, file_name):
        """
        Reads the source and lens galaxy normalized distributions from a sacc file
        and compresses them to the required number of redshifts.

        Parameters:
        -----------
        file_name : str
            Name of the sacc file.

        Returns:
        --------
        norm_nz_s : np.ndarray
            Normalized source galaxy distribution as a function of redshift. 
            Shape: (zbin_integr, nbin).
        norm_nz_l : np.ndarray
            Normalized lens galaxy distribution as a function of redshift. 
            Shape: (zbin_integr, nbin).
        """
        s = sacc.Sacc.load_fits(file_name)
        dndz_z = []
        lens_z = []
        z_arr = s.tracers['src0'].z
        # reduce redshift range
        indx_min = np.where(z_arr==self.zmin)[0][0]
        indx_max = np.where(z_arr==self.zmax)[0][0]
        z_arr = z_arr[indx_min:indx_max]
        # read sources and lenses distributions for each bin
        for i in range(self.nbin):
            dndz_z.append(s.tracers['src'+str(i)].nz[indx_min:indx_max]) 
            lens_z.append(s.tracers['lens'+str(i)].nz[indx_min:indx_max])
            # reduce array size
            dndz_z[i] = reduce_len_by_averaging(dndz_z[i], target_len=z_bins_for_integration)
            lens_z[i] = reduce_len_by_averaging(lens_z[i], target_len=z_bins_for_integration)
        # reduce z array size
        z_arr = reduce_len_by_averaging(z_arr, target_len=z_bins_for_integration)
        self.zz_integr = z_arr
        self.zbin_integr = len(z_arr)
        
        norm_nz_s = np.array(dndz_z).T
        norm_nz_l = np.array(lens_z).T
        
        return norm_nz_s, norm_nz_l     
    

class EuclidSetUp:
    """
    A class to set up the Euclid survey specifications and configurations.

    Attributes
    ----------
    survey_name : str
        Name of the survey, default is 'Euclid'.
    observable : str
        Observable configuration from the input config.
    zmin : float
        Minimum redshift value.
    zmax : float
        Maximum redshift value.
    fsky : float
        Fraction of the sky covered by the survey.
    gal_per_sqarcmn : float
        Number of galaxies per square arcminute.
    gal_per_sqarcmn_s : float
        Number of source galaxies per square arcminute.
    gal_per_sqarcmn_l : float
        Number of lens galaxies per square arcminute.
    rms_shear : float
        Root mean square of the shear.
    nbin : int
        Number of redshift bins.
    z_bin_center_s : np.ndarray
        Centers of the source redshift bins.
    z_bin_center_l : np.ndarray
        Centers of the lens redshift bins.
    eta_z_s : np.ndarray
        Normalized source galaxy distribution as a function of redshift.
    eta_z_l : np.ndarray
        Normalized lens galaxy distribution as a function of redshift.
    zz_integr : np.ndarray
        Array of redshift values used for integration.
    zbin_integr : int
        Number of redshift bins used for integration.
    noise : dict
        Dictionary containing noise values for different combinations of lensing and galaxy-galaxy correlations.
    aa_integr : np.ndarray
        Array of scale factors corresponding to the redshift values used for integration.
    lmin : int
        Minimum multipole value.
    lbin : int
        Number of multipole bins.
    lmax : float
        Maximum multipole value.
    ell : np.ndarray
        Array of multipole values.
    ell_bin_edges : np.ndarray
        Edges of the multipole bins.
    d_ell_bin : np.ndarray
        Differences between consecutive multipole bin edges.
    l_wl_bin_centers : np.ndarray
        Centers of the multipole bins.
    lum_func : np.ndarray
        Luminosity function data.
    likelihood : str
        Likelihood configuration from the input config.
    lmax_wl_vals, lmax_gc_vals : np.ndarray or float
        Maximum scale-cuts.
    l_wl, l_xc, l_gc : np.ndarray 
        Binned ell-values for different probes, for now l_xc=l_gc.
    nell_wl, nell_xc, nell_gc : int
        Number of ell-bins per probe.
    ells_wl, ells_gc : np.ndarray 
        All integer ell-values from lmin to lmax; required for likelihood with determinants. 
    ell_jump: int
        Equals to lmax_gc, as we assume that lmax_gc<lmax_wl; required for likelihood with determinants. 
    mask_data_vector_wl, mask_data_vector_gc, mask_data_vector_3x2pt : np.ndarray
        Boolean arrays to mask out data vectors for different lmax cuts in each redshift-bin.
    mask_cov_wl, mask_cov_gc, mask_cov_3x2pt : np.ndarray 
        Boolean arrays to mask out covariances for different lmax cuts in each redshift-bin.
    d_ell_bin_cut_wl, d_ell_bin_cut_xc, d_ell_bin_cut_gc : np.ndarray 
        Delta ell or bin width required for covariance computation.
    ells_wl_max, ells_gc_max : np.ndarray 
        Scale-cuts in the form appropriate for our plotting scripts.
    """
    def __init__(self, config:dict):
        survey_info = config['specs']['survey_info']
        scale_cuts_info = config['specs']['scale_cuts']
        likelihood = config['likelihood']
        self.observable = config['observable']
        self.survey_name = 'Euclid'

        # redshift bins setup
        self.zmin = 0.001
        self.zmax  = 2.5 

        self.gal_per_sqarcmn = 30.0
        self.rms_shear = 0.30
        self.fsky = 0.375 #15k deg^2
        self.lbin = 20 #100
        self.lmin = 10
        self.lmax = 5000
        if survey_info == 'Euclid_5bins':
            self.nbin = 5 
            self.z_bin_edge = np.array([self.zmin, 0.560, 0.789, 1.019, 1.324, self.zmax])  
            self.fsky = 0.375 #15k deg^2 #sky coverage 14_700  in in deg^2, fskay=sky coverage/DEG2_IN_SPHERE

        elif survey_info == 'Euclid_Y1':
            self.nbin = 5 
            self.z_bin_edge = np.array([self.zmin, 0.560, 0.789, 1.019, 1.324, self.zmax])  
            self.fsky = 0.375/3. #5k deg^2   
            self.lbin = 30 #80
            self.lmin = 100
            self.lmax = 5000
            self.gal_per_sqarcmn = 15.0
             
        elif survey_info == 'Euclid_10bins':    
            self.nbin = 10
            self.z_bin_edge = np.array([self.zmin, 0.418, 0.560, 0.678, 0.789, 0.900, 1.019, 1.155, 1.324, 1.576, 2.5])
            self.fsky = 14_700/DEG2_IN_SPHERE #0.4 
        self.z_bin_center = np.array([(self.z_bin_edge[i]+self.z_bin_edge[i+1])/2 for i in range(self.nbin)])
        self.z_bin_center_s = self.z_bin_center_l = self.z_bin_center
        # z values for integration (!= from z bins)
        self.zbin_integr = z_bins_for_integration 
        self.zz_integr = np.linspace(self.zmin, self.zmax, num=self.zbin_integr)
        self.aa_integr = np.array(1./(1.+self.zz_integr[::-1])) 
        # normalized galaxy distribution and noise
        self.eta_z_s, self.eta_z_l = self.get_norm_galaxy_distrib()
        zz_mod_wl = [self.zz_integr[np.argmax(self.eta_z_s[:, i])] for i in range (self.nbin)]
        zz_mod_gg = [self.zz_integr[np.argmax(self.eta_z_l[:, i])] for i in range (self.nbin)]
        self.n_bar = get_noise(self.nbin, self.gal_per_sqarcmn) 
        self.noise = {
        'LL': self.rms_shear**2./self.n_bar,
        'LG': 0.,
        'GL': 0.,
        'GG': 1./self.n_bar}
        # \ell bins setup
        if likelihood == 'determinants':
            self.ell = np.logspace(log10(self.lmin), log10(self.lmax), num=self.lbin, endpoint=True) 
        elif likelihood == 'binned':
            self.ell_bin_edges = np.logspace(log10(self.lmin), log10(self.lmax), num=self.lbin+1, endpoint=True) 
            self.d_ell_bin = np.diff(self.ell_bin_edges)
            # means in log-space:
            self.l_wl_bin_centers = np.sqrt(self.ell_bin_edges[:-1] *self.ell_bin_edges[1:])
            self.ell = self.l_wl_bin_centers
            # means in lin-space:
            # self.l_wl_bin_centers = np.array([0.5*(self.ell_bin_edges[i+1]+self.ell_bin_edges[i]) for i in range(self.lbin)])
        validate_and_setup_lmax(self, scale_cuts_info, likelihood, self.lmin, self.lmax, zz_mod_wl, zz_mod_gg)
        self.lum_func = get_luminosity_func(dirname+'/scaledmeanlum_E2Sa.dat')
        self.likelihood = likelihood
        
    def galaxy_distribution(self, z):
        """
        Calculate the unnormalized galaxy distribution at a given redshift.

        Parameters:
        ----------
        z : float
            The redshift at which to calculate the galaxy distribution.

        Returns:
        -------
        float
            The unnormalized galaxy distribution at the given redshift.
        """
        zmean = 0.9
        z0 = zmean/np.sqrt(2)
        galaxy_dist = (z/z0)**2*exp(-(z/z0)**(1.5))
        return galaxy_dist
    
    def photo_z_distribution(self, z, bin):
        """
        Calculate the photo-z error distribution for a given redshift and bin.
        It follows Eqs. (112, 113, 115) in https://arxiv.org/pdf/1910.09273.

        Parameters:
        ----------
        z : float
            The redshift at which to calculate the photo-z error.
        bin : int
            The redshift bin index.

        Returns:
        -------
        float
            The photo-z error distribution at the given redshift and bin.
        """
        c0, z0, sigma_0 = 1.0, 0.1, 0.05
        cb, zb, sigma_b = 1.0, 0.0, 0.05
        f_out = 0.1
        term1 = f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin - 1]))/(sigma_0*(1+z)))/(2.*c0)
        term2 =-f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin    ]))/(sigma_0*(1+z)))/(2.*c0)
        term3 = c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin - 1]))/(sigma_b*(1+z)))/(2.*cb)
        term4 =-c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin    ]))/(sigma_b*(1+z)))/(2.*cb)
        return term1+term2+term3+term4

    def get_norm_galaxy_distrib(self):
        """
        Returns the normalized source and lens galaxy distributions for each redshift value in each redshift bin.
        This function calculates the normalized galaxy distribution `n(z)` and the photo-z error for each redshift bin.
        The galaxy distribution is normalized using the trapezoidal rule for numerical integration.

        Returns:
        ------
        tuple 
            A tuple containing two numpy arrays:
                - norm_nz (numpy.ndarray): The normalized galaxy distribution for each redshift bin. Shape: (zbin_integr, nbin).
                - norm_nz (numpy.ndarray): The same normalized galaxy distribution, returned twice as the source distribution
                  is equal to the lens distribution in Euclid. Shape: (zbin_integr, nbin).
        """
        norm_nz = np.zeros((self.zbin_integr, self.nbin), 'float64')
        photo_z = np.zeros((self.zbin_integr, self.nbin), 'float64')
        # calculate n(z) and photo-z error
        for Bin in range(self.nbin):
                for nz in range(self.zbin_integr):
                    z = self.zz_integr[nz]
                    photo_z[nz,Bin] = self.photo_z_distribution(z,Bin+1)
                    norm_nz[nz, Bin] = photo_z[nz,Bin] * self.galaxy_distribution(z)
        # normalize galaxy distribution 
        for Bin in range(self.nbin):
            norm_nz[:,Bin] /= trapezoid(norm_nz[:,Bin],self.zz_integr[:]) 
        # return twice as source distrib = lenses distribution in Euclid
        return norm_nz, norm_nz 