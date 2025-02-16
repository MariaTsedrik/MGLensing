import numpy as np
from scipy.integrate import simpson, quad, trapezoid
from scipy.special import erf

class Survey:
    def galaxy_distribution(self, z):
        '''
        Returns the (unnormalized) galaxy distribution at redshift z'''
        zmean = 0.9
        z0 = zmean/np.sqrt(2)
        galaxy_dist = (z/z0)**2*np.exp(-(z/z0)**(1.5))
        return galaxy_dist
    def photo_z_distribution(self, z, bin):
        '''
        Returns the error on galaxy distribution at redshift z'''
        c0, z0, sigma_0 = 1.0, 0.1, 0.05
        cb, zb, sigma_b = 1.0, 0.0, 0.05
        f_out = 0.1
        term1 = f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin - 1]))/(sigma_0*(1+z)))/(2.*c0)
        term2 =-f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin    ]))/(sigma_0*(1+z)))/(2.*c0)
        term3 = c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin - 1]))/(sigma_b*(1+z)))/(2.*cb)
        term4 =-c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin    ]))/(sigma_b*(1+z)))/(2.*cb)
        return term1+term2+term3+term4

    def get_norm_galaxy_distrib(self):
        '''
        Returns he normalized source and lenses distribution
        for each z value in each redshift bin '''
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
    def load_n_of_z(self, n_of_z_model):
        if n_of_z_model==0:
            return self.get_norm_galaxy_distrib()
        #if n_of_z_model==1:
        #LSST setup
         
    def __init__(self, config_dic_specs):
        self.fsky = config_dic_specs['fsky'] 
        self.lbin = config_dic_specs['lbin'] 
        lmin = config_dic_specs['lmin'] 
        lmax_WL = config_dic_specs['lmax_WL'] 
        lmax_GC = config_dic_specs['lmax_GC'] 
        self.l_WL = np.logspace(np.log10(lmin), np.log10(lmax_WL), num=self.lbin, endpoint=True) #binned like Euclid
        idx_lmax = int(np.argwhere(self.l_WL >= lmax_GC)[0]) #only the case implemented where lmax_GC<lmax_WL
        self.l_GC = self.l_WL[:idx_lmax+1]
        self.l_XC = self.l_WL[:idx_lmax+1]
        self.ells_WL = np.array(range(lmin,lmax_WL+1)) #all integer ells
        self.ell_jump = lmax_GC - lmin +1
        self.ells_GC = self.ells_WL[:self.ell_jump]
        self.ells_XC = self.ells_GC
        self.nell_WL = len(self.l_WL)
        self.nell_GC = len(self.l_GC)
        self.nell_XC = len(self.l_XC)

        self.k_min_h_by_Mpc = 0.001
        self.k_max_h_by_Mpc = 50.0 #limit of bacco's linear emulator

        self.nbin = config_dic_specs['nbin']
        zmin = config_dic_specs['zmin']
        zmax  = config_dic_specs['zmax'] 

        self.z_bin_edge = np.array(config_dic_specs['z_bin_edge']) #works only for 10 bins
        self.z_bin_center = np.array([(self.z_bin_edge[i]+self.z_bin_edge[i+1])/2 for i in range(self.nbin)])

        ###for integration###
        self.zbin_integr = config_dic_specs['zbin_integr'] 
        self.zz_integr = np.linspace(zmin, zmax, num=self.zbin_integr)
        self.aa_integr = np.array(1./(1.+self.zz_integr[::-1])) ##should be increasing

        gal_per_sqarcmn = config_dic_specs['gal_per_sqarcmn'] 
        self.n_bar = gal_per_sqarcmn * (60.*180./np.pi)**2 #in 1/radians^2=1/sr
        self.n_bar /= self.nbin  
        self.rms_shear = config_dic_specs['rms_shear']

        #self.eta_z = np.zeros((zbin_integr, self.nbin), 'float64')
        #self.photoerror_z = np.zeros((zbin_integr, self.nbin), 'float64')

        self.eta_z_s, self.eta_z_l = self.load_n_of_z(config_dic_specs['n_of_z'])
        self.noise = {
        'LL': self.rms_shear**2./self.n_bar,
        'LG': 0.,
        'GL': 0.,
        'GG': 1./self.n_bar}
