import numpy as np
from scipy.special import erf
from scipy.integrate import trapezoid,quad
from scipy.interpolate import interp1d
from math import sqrt, log10, exp, pow
import sacc
import yaml
import warnings
import os

dirname = os.path.split(__file__)[0]
print(dirname)
H0_h_c = 1./2997.92458 #=100/c in Mpc/h
z_bins_for_integration = 200

def get_luminosity_func(file_name=dirname+'/scaledmeanlum_E2Sa.dat'):
    """
    Reads the luminosity function data from a file and returns an interpolation function.

    Parameters:
    -----------
    file_name : str, optional
        The path to the file_name containing the luminosity function data. Default is 'scaledmeanlum_E2Sa.dat'.

    Returns:
    --------
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
    ----------
    arr (list or numpy.ndarray): The input array to be reduced.
    target_len (int): The desired length of the output array. Default is z_bins_for_integration.

    Returns:
    --------
    list or numpy.ndarray: The reduced array with the specified target length.
    """
    # Number of elements to average together
    step = len(arr) // target_len  
    reduced_arr = np.array([np.mean(arr[i:i+step]) for i in range(0, len(arr), step)])
    return reduced_arr[:target_len]

def get_noise(nbins, gal_per_sqarcmn):
    """
    Calculate the noise for a given number of redshift bins and galaxy density.

    Parameters:
    -----------
    nbins : int
        The number of redshift bins.
    gal_per_sqarcmn : float
        The number of galaxies per square arcminute.

    Returns:
    --------
    float
        The noise value for the given number of redshift bins and galaxy density.
    """
    n_bar = gal_per_sqarcmn * (60.*180./np.pi)**2 #in 1/radians^2=1/sr
    n_bar /= nbins
    return n_bar

def check_length(scale_cuts_info, nbin):
    labels = ['max_WL', 'max_GC']
    nbins = int(nbin*(1+nbin)/2)
    if scale_cuts_info['type'] == 'const_lmax' or scale_cuts_info['type'] == 'lmax':
        for label in labels:
            if len(scale_cuts_info[label])!=nbins:
                raise ValueError(label+' length must be equal to '+str(nbins)+'!')
    elif scale_cuts_info['type'] == 'kmax':
        for label in labels:
            if len(scale_cuts_info[label])!=nbin:
                raise ValueError(label+' length must be equal to '+str(nbin)+'!')

def max_ells_for_plots(n, flat_array):
    matrix_ell_cuts = np.zeros((n, n), dtype=int) 
    il1 = np.tril_indices(n)  
    matrix_ell_cuts[il1] = flat_array  
    matrix_ell_cuts = matrix_ell_cuts + matrix_ell_cuts.T - np.diag(np.diag(matrix_ell_cuts))  
    return matrix_ell_cuts


def create_mask(n, len_ell_max, flat_array):
    """
    Creates a mask for the ell values to cut up to lmax. Later used in covariance computations for single probes.

    Parameters:
    -----------
    n : int
        The number of redshift bins.
    len_ell_max : int
        Length of an array with all integer ell-values from lmin up to max(ell_max)
    flat_array : list or array
        The flattened array of n*(n+1)/2 ell_max values - lmin + 1: so that we cut-out the values in matrices
        that correpond to lmin...ell_max_in_this_bin

    Returns:
    --------
    matrix_ell_cuts_bool : np.ndarray
        A boolean array indicating the mask for the ell values.
    """
    matrix_ell_cuts_bool = np.zeros((len_ell_max, n, n), dtype=bool) 
    il1 = np.tril_indices(n)  
    matrix_ell_cuts = np.zeros((n, n), dtype=int) 
    matrix_ell_cuts[il1] = flat_array  
    matrix_ell_cuts = matrix_ell_cuts + matrix_ell_cuts.T - np.diag(np.diag(matrix_ell_cuts))  
    for bin1 in range(n):
        for bin2 in range(n):
            matrix_ell_cuts_bool[:matrix_ell_cuts[bin1, bin2], bin1, bin2] =  np.ones(matrix_ell_cuts[bin1, bin2], dtype=bool) 
    return matrix_ell_cuts_bool

def create_mask_high(n, ell_max_wl, ell_max_gg):  
    """
    Creates a boolean mask matrix based on the given ell_max_wl and ell_max_gg values.
    We assume that ell_max_wl is always larger than ell_max_gg. The high-ell values 
    are all values of ell_wl that are larger than ell_max_gg in each bin. Hence, later
    when computing cov_high we apply this mask so that if in bin_i lmax_gg = 500 while 
    min(ell_max_gg) = 400, and lmax_wl = 1000 in the same bin, we mask [False (corresponds to ell=401), 
    False (corresponds to ell=402), ..., False (corresponds to ell=500), True (corresponds to ell=501), 
    True (corresponds to ell=502), ..., True (corresponds to ell=1000), False (corresponds to ell=1001), ...
    Falls (corresponds to ell=max(ell_max_wl)) ].


    Parameters:
    -----------
    n (int): The size of the matrix (number of bins).
    ell_max_wl (list or array): The maximum ell values for weak lensing.
    ell_max_gg (list or array): The maximum ell values for galaxy-galaxy lensing.

    Returns:
    --------
    numpy.ndarray: A boolean mask matrix of shape (max(ell_max_wl)-min(ell_max_gg), n, n).
    """
    matrix_ell_cuts_bool = np.zeros((max(ell_max_wl)-min(ell_max_gg), n, n), dtype=bool) 
    il1 = np.tril_indices(n)  
    matrix_ell_cuts_gg = np.zeros((n, n), dtype=int) 
    matrix_ell_cuts_gg[il1] = ell_max_gg
    matrix_ell_cuts_gg = matrix_ell_cuts_gg + matrix_ell_cuts_gg.T - np.diag(np.diag(matrix_ell_cuts_gg))  
    il2 = np.tril_indices(n)  
    matrix_ell_cuts_wl = np.zeros((n, n), dtype=int) 
    matrix_ell_cuts_wl[il2] = ell_max_wl
    matrix_ell_cuts_wl = matrix_ell_cuts_wl + matrix_ell_cuts_wl.T - np.diag(np.diag(matrix_ell_cuts_wl))  
    for bin1 in range(n):
        for bin2 in range(n):
            matrix_ell_cuts_bool[(matrix_ell_cuts_gg[bin1, bin2]-min(ell_max_gg)):(matrix_ell_cuts_wl[bin1,bin2]-min(ell_max_gg)), bin1, bin2] =  np.ones(matrix_ell_cuts_wl[bin1, bin2]-matrix_ell_cuts_gg[bin1, bin2], dtype=bool) 
    return matrix_ell_cuts_bool

def create_mask_3x2pt(n, len_ellmax_gg, ell_max_gg):   
    """
    Creates a 3x2 point mask for given parameters. Again, we assume that in a bin ell_max_wl is always larger than ell_max_gg.
    Hence, later in the combined covariance 
    LL | LG
    GL | GG
    each block has a length that correponds to lmin...max(ell_max_gg). We then apply this matrix to cut values in a z-bin
    that correspond to ell>ell_max_gg_in_this_bin.

    Parameters:
    ----------
    n (int): Number of bins.
    len_ellmax_gg (int): Length of the ell_max_gg array, that includes lmin...max(ell_max_gg).
    ell_max_gg (array-like): Array containing the ell_max values for each bin combination in the lower triangle.

    Returns:
    -------
    numpy.ndarray: A boolean mask array of shape (len_ellmax_gg, 2*n, 2*n) indicating the ell cuts.
    """
    matrix_ell_cuts_bool = np.zeros((len_ellmax_gg, 2*n, 2*n), dtype=bool) 
    il1 = np.tril_indices(n)  
    matrix_ell_cuts = np.zeros((n, n), dtype=int) 
    matrix_ell_cuts[il1] = ell_max_gg 
    matrix_ell_cuts = matrix_ell_cuts + matrix_ell_cuts.T - np.diag(np.diag(matrix_ell_cuts))  
    for bin1 in range(n):
        for bin2 in range(n):
            matrix_ell_cuts_bool[:matrix_ell_cuts[bin1, bin2], bin1, bin2] =  np.ones(matrix_ell_cuts[bin1, bin2], dtype=bool) 
            matrix_ell_cuts_bool[:matrix_ell_cuts[bin1, bin2],n+bin1,bin2] = np.ones(matrix_ell_cuts[bin1, bin2], dtype=bool) 
            matrix_ell_cuts_bool[:matrix_ell_cuts[bin1, bin2],bin1,n+bin2] = np.ones(matrix_ell_cuts[bin1, bin2], dtype=bool) 
            matrix_ell_cuts_bool[:matrix_ell_cuts[bin1, bin2],n+bin1,n+bin2] = np.ones(matrix_ell_cuts[bin1, bin2], dtype=bool) 
    return matrix_ell_cuts_bool

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
        for bin2 in range(n):
            if bin1>=bin2:
                z = z_mod[bin2]
                lmax.append( k_eff[bin2] * get_rcom(params_dic, z) / (1. + z))
    return lmax

def setup_const_lmax(obj, lmax_wl, lmax_gc, lmin, ell, nbin):
    idx_lmax_wl = int(np.argwhere(ell >= lmax_wl)[0]) 
    obj.l_wl = ell[:idx_lmax_wl+1] 
    idx_lmax_gc = int(np.argwhere(ell >= lmax_gc)[0]) 
    obj.l_gc = obj.l_wl[:idx_lmax_gc+1]
    obj.l_xc = obj.l_gc
    obj.nell_wl = len(obj.l_wl)
    obj.nell_gc = len(obj.l_gc)
    obj.nell_xc = len(obj.l_xc)
    obj.ells_wl = np.array(range(lmin, lmax_wl+1)) # all integer ells
    obj.ells_gc = np.array(range(lmin, lmax_gc+1)) # all integer ells
    obj.ell_jump = lmax_gc - lmin + 1
    obj.mask_ells_wl = np.ones((len(obj.ells_wl), nbin, nbin), dtype=bool)
    obj.mask_ells_gc = np.ones((len(obj.ells_gc), nbin, nbin), dtype=bool)
    obj.mask_ells_high = np.ones((len(obj.ells_wl) - obj.ell_jump, nbin, nbin), dtype=bool)
    obj.mask_ells_3x2pt = np.ones((len(obj.ells_gc), 2 * nbin, 2 * nbin), dtype=bool)
    obj.ells_wl_max = max_ells_for_plots(nbin, np.full(int(0.5 * nbin * (nbin + 1)), lmax_wl))
    obj.ells_gc_max = max_ells_for_plots(nbin, np.full(int(0.5 * nbin * (nbin + 1)), lmax_gc))

def setup_lmax(obj, lmax_wl, lmax_gc, lmin, ell, nbin):
    idx_lmax_wl = int(np.argwhere(ell >= max(lmax_wl))[0]) 
    obj.l_wl = ell[:idx_lmax_wl+1] 
    idx_lmax_gc = int(np.argwhere(ell >= max(lmax_gc))[0]) 
    obj.l_gc = obj.l_wl[:idx_lmax_gc+1]
    obj.l_xc = obj.l_gc
    obj.ell_jump = min(lmax_gc) - lmin + 1
    obj.nell_wl = len(obj.l_wl)
    obj.nell_gc = len(obj.l_gc)
    obj.nell_xc = len(obj.l_xc)
    obj.ells_wl = np.array(range(lmin, max(lmax_wl)+1)) # all integer ells
    obj.ells_gc = np.array(range(lmin, max(lmax_gc)+1)) # all integer ells
    obj.mask_ells_wl = create_mask(nbin, len(obj.ells_wl), np.array(lmax_wl) - lmin + 1)
    obj.mask_ells_gc = create_mask(nbin, len(obj.ells_gc), np.array(lmax_gc) - lmin + 1)
    obj.mask_ells_high = create_mask_high(nbin, np.array(lmax_wl), np.array(lmax_gc))
    obj.mask_ells_3x2pt = create_mask_3x2pt(nbin, len(obj.ells_gc), np.array(lmax_gc) - lmin + 1)
    obj.ells_wl_max = max_ells_for_plots(nbin, lmax_wl)
    obj.ells_gc_max = max_ells_for_plots(nbin, lmax_gc)

def validate_and_setup_lmax(obj, scale_cuts_info, lmin, lmax, zz_mod_wl, zz_mod_gg):
    """
    Validates and sets up the maximum multipole moments (lmax) for weak lensing (WL) and galaxy clustering (GC).

    Parameters:
    -----------
    obj : object
        The object containing the necessary attributes such as ell, nbin, and lmax.
    scale_cuts_info : dict
        Dictionary containing scale cut information. It must include:
        - 'type': Type of scale cut ('const_lmax', 'lmax', or 'kmax').
        - 'max_WL': List of maximum multipole moments for WL.
        - 'max_GC': List of maximum multipole moments for GC.
        - 'cosmo': Dictionary of cosmological parameters (required if 'type' is 'kmax').
    lmin : int
        The minimum multipole moment.
    lmax : int
        The maximum multipole moment.
    zz_mod_wl : array-like
        Redshift peaks of kernels for weak lensing.
    zz_mod_gg : array-like
        Redshift peaks of kernels for galaxy clustering.

    Raises:
    -------
    ValueError:
        If scale_cuts_info['type'] is 'const_lmax' or 'lmax' and any value in scale_cuts_info['max_WL'] or scale_cuts_info['max_GC'] exceeds obj.lmax.
        For any scale_cuts_info['type'] if lmax_gc > lmax_wl.
        If scale_cuts_info['type'] is 'kmax' and any lmax_wl or lmax_gc exceeds obj.lmax.
        If scale_cuts_info['type'] is not one of 'const_lmax', 'lmax', or 'kmax'.
    """
    if scale_cuts_info['type'] == 'const_lmax':
        if any(val > lmax for val in scale_cuts_info['max_WL'] + scale_cuts_info['max_GC']):
            raise ValueError(f'lmax cannot exceed {lmax}!!!')
        lmax_wl = int(scale_cuts_info['max_WL'][0])
        lmax_gc = int(scale_cuts_info['max_GC'][0])
        obj.lmax_wl_vals = lmax_wl 
        obj.lmax_gc_vals= lmax_gc 
        if lmax_wl<lmax_gc:
            raise ValueError('Only lmax_gc<=lmax_wl case is implemented yet!')
        setup_const_lmax(obj, lmax_wl, lmax_gc, lmin, obj.ell, obj.nbin)
    elif scale_cuts_info['type'] == 'lmax':
        if any(val > lmax for val in scale_cuts_info['max_WL'] + scale_cuts_info['max_GC']):
            raise ValueError(f'lmax cannot exceed {lmax}!!!')
        check_length(scale_cuts_info, obj.nbin)
        lmax_wl = scale_cuts_info['max_WL']
        lmax_gc = scale_cuts_info['max_GC']
        lmax_wl = np.array(lmax_wl).astype(int)
        lmax_gc = np.array(lmax_gc).astype(int)
        obj.lmax_wl_vals = lmax_wl 
        obj.lmax_gc_vals= lmax_gc 
        if any(lmax_wl[i]<lmax_gc[i] for i in range(int(0.5*obj.nbin*(obj.nbin+1)))):
            raise ValueError('Only lmax_gc<=lmax_wl case is implemented yet!')
        setup_lmax(obj, lmax_wl, lmax_gc, lmin, obj.ell, obj.nbin)
        warnings.warn('CUTS IN DIFFERENT LMAX ARE NOT IMPLEMENTED IN THE LIKELIHOOD YET!!!')

    elif scale_cuts_info['type'] == 'kmax':
        check_length(scale_cuts_info, obj.nbin)
        params_dic = scale_cuts_info['cosmo']
        lmax_wl = from_keff_to_lmax(obj.nbin, scale_cuts_info['max_WL'], zz_mod_wl, params_dic)
        lmax_wl = np.array(lmax_wl).astype(int)
        lmax_gc = from_keff_to_lmax(obj.nbin, scale_cuts_info['max_GC'], zz_mod_gg, params_dic)
        lmax_gc = np.array(lmax_gc).astype(int)
        obj.lmax_wl_vals = lmax_wl 
        obj.lmax_gc_vals= lmax_gc 
        print('lmax_wl: ', lmax_wl)
        print('lmax_gc: ', lmax_gc)
        if any(lmax_wl[i]<lmax_gc[i] for i in range(int(0.5*obj.nbin*(obj.nbin+1)))):
            raise ValueError('Only lmax_gc<=lmax_wl case is implemented yet!')
        if any(val > lmax for val in list(lmax_gc) + list(lmax_wl)):
            raise ValueError(f'lmax cannot exceed {lmax}!!!')
        setup_lmax(obj, lmax_wl, lmax_gc, lmin, obj.ell, obj.nbin)
        warnings.warn('CUTS IN DIFFERENT KMAX ARE NOT IMPLEMENTED IN THE LIKELIHOOD YET!!!')
    else:
        raise ValueError('Invalid scale-cut type') 



class EuclidSetUp:
    """
    A class to set up and manage the Euclid survey parameters and configurations.
    Attributes:
    survey_name : str
        The name of the survey, default is 'Euclid'.
    fsky : float
        The fraction of the sky covered by the survey.
    gal_per_sqarcmn : float
        The number of galaxies per square arcminute.
    rms_shear : float
        The root mean square shear.
    zmin : float
        The minimum redshift value.
    zmax : float
        The maximum redshift value.
    nbin : int
        The number of redshift bins.
    z_bin_edge : numpy.ndarray
        The edges of the redshift bins.
    z_bin_center : numpy.ndarray
        The centers of the redshift bins.
    z_bin_center_s : numpy.ndarray
        The centers of the source redshift bins.
    z_bin_center_l : numpy.ndarray
        The centers of the lens redshift bins.
    zbin_integr : int
        The number of redshift bins for integration.
    zz_integr : numpy.ndarray
        The redshift values for integration.
    aa_integr : numpy.ndarray
        The scale factor values for integration.
    eta_z_s : numpy.ndarray
        The normalized source galaxy distribution.
    eta_z_l : numpy.ndarray
        The normalized lens galaxy distribution.
    n_bar : numpy.ndarray
        The noise values for each redshift bin.
    noise : dict
        The noise dictionary containing noise values for different combinations of lensing and galaxy-galaxy correlations.
    lbin : int
        The number of ell bins.
    lmin : int
        The minimum ell value.
    lmax : int
        The maximum ell value.
    ell : numpy.ndarray
        The ell values for the bins.
    lum_func : function
        The luminosity function.
    Methods:
    galaxy_distribution(z):
        Part of the n(z) computation.
    photo_z_distribution(z, bin):
        Part of the n(z) computation.
    get_norm_galaxy_distrib():
        Returns source and lens galaxy normalised distributions, which are the same for Euclid.
    """
    def __init__(self, survey_info, scale_cuts_info):
        self.survey_name = 'Euclid'
        self.fsky = 0.4
        self.gal_per_sqarcmn = 30.0
        self.rms_shear = 0.30
        # redshift bins setup
        self.zmin = 0.001
        self.zmax  = 2.5 
        if survey_info == 'Euclid_5bins':
            self.nbin = 5 
            self.z_bin_edge = np.array([self.zmin, 0.560, 0.789, 1.019, 1.324, self.zmax])  
        elif survey_info == 'Euclid_10bins':    
            self.nbin = 10
            self.z_bin_edge = np.array([self.zmin, 0.418, 0.560, 0.678, 0.789, 0.900, 1.019, 1.155, 1.324, 1.576, 2.5])
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
        self.lbin = 100
        lmin = 10
        self.lmin = lmin
        lmax = 5000
        self.lmax = lmax
        self.ell = np.logspace(log10(lmin), log10(lmax), num=self.lbin, endpoint=True) 
        validate_and_setup_lmax(self, scale_cuts_info, lmin, lmax, zz_mod_wl, zz_mod_gg)
        self.lum_func = get_luminosity_func(dirname+'/scaledmeanlum_E2Sa.dat')

        
    def galaxy_distribution(self, z):
        """
        Calculate the unnormalized galaxy distribution at a given redshift.

        Parameters:
        -----------
        z : float
            The redshift at which to calculate the galaxy distribution.

        Returns:
        --------
        galaxy_dist : float
            The unnormalized galaxy distribution at the given redshift.
        """
        zmean = 0.9
        z0 = zmean/np.sqrt(2)
        galaxy_dist = (z/z0)**2*exp(-(z/z0)**(1.5))
        return galaxy_dist
    
    def photo_z_distribution(self, z, bin):
        """
        Calculate the photo-z error distribution for a given redshift and bin.
        It follows Eqs. (112, 113, 115) in https://arxiv.org/pdf/1910.09273

        Parameters:
        -----------
        z : float
            The redshift at which to calculate the photo-z error.
        bin : int
            The redshift bin index.

        Returns:
        --------
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
        -------
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
        

class LSSTSetUp:
    """
    A class to set up the LSST survey parameters and configurations.

    Attributes:
    -----------
    survey_name : str
        Name of the survey, default is 'LSST'.
    fsky : float
        Fraction of the sky covered by the survey.
    gal_per_sqarcmn : float
        Number of galaxies per square arcminute.
    rms_shear : float
        Root mean square shear.
    nbin : int
        Number of redshift bins.
    zmin : float
        Minimum redshift.
    zmax : float
        Maximum redshift.
    z_bin_center_s : np.ndarray
        Array of redshift corresponding to the mean value of n(z) for source galaxies in each bin.
    z_bin_center_l : np.ndarray
        Array of redshift corresponding to the mean value of n(z) for lens galaxies in each bin.
    eta_z_s : np.ndarray
        Normalized source galaxy distribution.
    eta_z_l : np.ndarray
        Normalized lens galaxy distribution.
    n_bar : float
        Noise parameter.
    noise : dict
        Dictionary containing noise parameters for different correlations.
    aa_integr : np.ndarray
        Array for integration over redshift.
    zbin_integr : int
        Length of the redshift integration array.
    lmin : int
        Minimum multipole moment.
    lbin : int
        Number of multipole bins.
    lmax : float
        Maximum multipole moment.
    ell : np.ndarray
        Array of multipole moments.

    Methods:
    --------
    get_norm_galaxy_distrib(file='forecast_fid.sacc'):
        Reads the source and lens galaxy normalised distributions from a sacc file.
    """
    def __init__(self, survey_info, scale_cuts_info):
        self.survey_name = 'LSST'
        self.fsky = 0.5
        self.gal_per_sqarcmn = 20.0
        self.rms_shear = 0.26
        # redshift setup 
        self.nbin = 5
        self.zmin = 0.01 
        self.zmax = 3.0 #actually 4.0,  but changed because of bacco-emu 
        if survey_info == 'LSST_Y1':
            self.nbin = 5 
            nz_sacc_file = dirname+'/forecast_fid.sacc'
            self.z_bin_center_s = np.array([2.253273944137912099e-01, 4.490567958326107112e-01, 6.545664037512012312e-01, 9.258332934738261466e-01, 1.590547464182611836e+00]) #from DESC github
            self.z_bin_center_l = np.array([3.083232685515536753e-01, 5.011373846158426737e-01, 6.981257392184466726e-01, 8.964730913147888058e-01, 1.095412294727920344e+00])  #from DESC github
        elif survey_info == 'LSST_Y10':   
            self.nbin = 5  # change later
            nz_sacc_file = dirname+'/forecast_fid.sacc' # change later
            self.z_bin_center_s = np.array([2.253273944137912099e-01, 4.490567958326107112e-01, 6.545664037512012312e-01, 9.258332934738261466e-01, 1.590547464182611836e+00]) #change later
            self.z_bin_center_l = np.array([3.083232685515536753e-01, 5.011373846158426737e-01, 6.981257392184466726e-01, 8.964730913147888058e-01, 1.095412294727920344e+00])  #change later
        # normalized galaxy distribution and noise
        self.eta_z_s, self.eta_z_l = self.get_norm_galaxy_distrib(nz_sacc_file) 
        zz_mod_wl = [self.zz_integr[np.argmax(self.eta_z_s[:, i])] for i in range (self.nbin)]
        zz_mod_gg = [self.zz_integr[np.argmax(self.eta_z_l[:, i])] for i in range (self.nbin)]
        self.n_bar = get_noise(self.nbin, self.gal_per_sqarcmn)
        self.noise = {
        'LL': self.rms_shear**2./self.n_bar,
        'LG': 0.,
        'GL': 0.,
        'GG': 1./self.n_bar}
        self.aa_integr = np.array(1./(1.+self.zz_integr[::-1])) 
        self.zbin_integr = len(self.zz_integr)        
        # \ell bins setup
        lmin = 20
        self.lmin = lmin
        self.lbin = 100 
        lmax = 5e3
        self.lmax = lmax
        self.ell = np.logspace(log10(lmin), log10(lmax), num=self.lbin, endpoint=True) 
        validate_and_setup_lmax(self, scale_cuts_info, lmin, lmax, zz_mod_wl, zz_mod_gg)
        self.lum_func = get_luminosity_func(dirname+'/scaledmeanlum_E2Sa.dat')


    def get_norm_galaxy_distrib(self, file_name=dirname+'/forecast_fid.sacc'):
        """
        Reads the source and lens galaxy normalised distributions from a sacc file.
        Compresses to the required number of redshifts.

        Parameters:
        -----------
        file : str
            Name of the sacc file.

        Returns:
        --------
        norm_nz_s : np.ndarray
            Normalized source galaxy distribution as a function of redshift. Shape: (zbin_integr, nbin).
        norm_nz_l : np.ndarray
            Normalized lens galaxy distribution as a function of redshift. Shape: (zbin_integr, nbin).
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
    

class CustomSetUp:
    def __init__(self, file_path):
        with open(file_path, "r") as file_in:
            config_dic_specs = yaml.safe_load(file_in)
        raise NotImplementedError("Custom survey is not implemented yet")

