from scipy.integrate import simpson, quad
from scipy import interpolate as itp
from scipy.interpolate import interp1d, RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.special import erf
import numpy as np
from multiprocessing import Pool
import os
from scipy.stats import norm
import math
################################################
#               SETUP (Euclid-like)            #
################################################
#bias_model = 'interpld'

fsky = 0.4
lmax_WL = 3000
lmax_GC = 1000#1500
lmax_XC = 1000#1500

lmin = 10
lbin = 100

l_WL = np.logspace(np.log10(lmin), np.log10(lmax_WL), num=lbin, endpoint=True) #binned like Euclid
idx_lmax = int(np.argwhere(l_WL >= lmax_GC)[0])
l_GC = l_WL[:idx_lmax+1]
l_XC = l_WL[:idx_lmax+1]
l_array = 'WL'

ells_WL = np.array(range(lmin,lmax_WL+1)) #all integer ells
ell_jump = lmax_GC - lmin +1
ells_GC = ells_WL[:ell_jump]
ells_XC = ells_GC

nell_WL = len(l_WL)
nell_GC = len(l_GC)
nell_XC = len(l_XC)

gal_per_sqarcmn = 30.0
nbin = 10
zmin = 0.001
zmax  = 2.5 #4
z_bin_edge = np.array([zmin, 0.418, 0.560, 0.678, 0.789, 0.900, 1.019, 1.155, 1.324, 1.576, 2.5])
z_bin_center = np.array([(z_bin_edge[i]+z_bin_edge[i+1])/2 for i in range(nbin)])


rms_shear = 0.30

###for integration###
zbin_integr = 200 #400
zz_integr = np.linspace(zmin, zmax, num=zbin_integr)
aa_integr = np.array(1./(1.+zz_integr[::-1])) ##should be increasing

###for interpolation of the power spectrum###
zz_Pk = np.array([0., 0.01,  0.12, 0.24, 0.38, 0.52, 0.68, 0.86, 1.05, 1.27, 1.5, 1.76, 2.04, 2.36, 2.5])
aa_Pk = np.array(1./(1.+zz_Pk[::-1])) ##should be increasing
k_min_h_by_Mpc = 0.001
k_max_h_by_Mpc = 50.0 #limit of bacco's linear emulator
nz_Pk = len(zz_Pk)

kLbin = 512
kL = np.logspace(-4, np.log10(k_max_h_by_Mpc), kLbin)

###galaxy distribution per bin###
def galaxy_distribution(z):
    zmean = 0.9
    z0 = zmean/np.sqrt(2)
    galaxy_dist = (z/z0)**2*np.exp(-(z/z0)**(1.5))
    return galaxy_dist


def photo_z_distribution(z, bin):
    c0, z0, sigma_0 = 1.0, 0.1, 0.05
    cb, zb, sigma_b = 1.0, 0.0, 0.05
    f_out = 0.1

    term1 = f_out*    erf((0.707107*(z-z0-c0*z_bin_edge[bin - 1]))/(sigma_0*(1+z)))/(2.*c0)
    term2 =-f_out*    erf((0.707107*(z-z0-c0*z_bin_edge[bin    ]))/(sigma_0*(1+z)))/(2.*c0)
    term3 = c0*(1-f_out)*erf((0.707107*(z-zb-cb*z_bin_edge[bin - 1]))/(sigma_b*(1+z)))/(2.*cb)
    term4 =-c0*(1-f_out)*erf((0.707107*(z-zb-cb*z_bin_edge[bin    ]))/(sigma_b*(1+z)))/(2.*cb)
    return term1+term2+term3+term4
eta_z = np.zeros((zbin_integr, nbin), 'float64')
photoerror_z = np.zeros((zbin_integr, nbin), 'float64')

for Bin in range(nbin):
        for nz in range(zbin_integr):
            z = zz_integr[nz]
            photoerror_z[nz,Bin] = photo_z_distribution(z,Bin+1)
            eta_z[nz, Bin] = photoerror_z[nz,Bin] * galaxy_distribution(z)
for Bin in range(nbin):
    eta_z[:,Bin] /= trapz(eta_z[:,Bin],zz_integr[:]) 
###noise###      
n_bar = gal_per_sqarcmn * (60.*180./np.pi)**2 #in 1/radians^2=1/sr
n_bar /= nbin  

###luminosioty function for IA###
lum_file = open('scaledmeanlum_E2Sa.dat', 'r')
content = lum_file.readlines()
zlum = np.zeros((len(content)))
lum = np.zeros((len(content)))
for index in range(len(content)):
    line = content[index]
    zlum[index] = line.split()[0]
    lum[index] = line.split()[1]
lum_func = interp1d(zlum, lum,kind='linear')

################################################
#               PRIORS                         #
################################################

MasterPriors = {
    'IA':
    {    
    'aIA': {'type':'U', 'p0': 1.72, 'p1': 0., 'p2': 12.1},
    'etaIA': {'type':'U', 'p0': -0.41, 'p1': -7., 'p2': 6.17},
    'betaIA': {'type':'U', 'p0': 0.0, 'p1': -10., 'p2': 10.}
    },
    'HMcode':
    {
    'Omega_m':  {'type':'U', 'p0':0.3085640138408304, 'p1': 0.15,     'p2': 0.6},    
    'Omega_b':  {'type':'U', 'p0':0.04904844290657439, 'p1':0.03,    'p2': 0.07},  
    'Ombh2': {'type':'G', 'p0': 0.02268, 'p1': 0.02268, 'p2': 0.00038},  
    'h'      :  {'type':'U', 'p0': 0.68, 'p1': 0.5,     'p2': 0.9},      
    'log10As' : {'type':'U', 'p0': 3.044, 'p1': 1.7, 'p2': 4.},
    #'log10As' : {'type':'G', 'p0': 3.044, 'p1': 3.044, 'p2': 0.042}, #3 sigma Planck
    #'ns': {'type':'U', 'p0': 0.97, 'p1': 0.6,'p2': 1.2},
    'ns': {'type':'G', 'p0': 0.97, 'p1': 0.97,'p2': 0.004}, #1 sigma Planck
    'Mnu': {'type':'U', 'p0': 0.06, 'p1': 0.0, 'p2': 0.5},
    'w0': {'type':'U', 'p0': -1., 'p1': -3., 'p2': -0.3},    
    'wa': {'type':'U', 'p0': 0., 'p1': -3., 'p2': 3.},  
    #'w0': {'type':'U', 'p0': -1., 'p1': -1.5, 'p2': -0.3},    
    #'wa': {'type':'U', 'p0': 0., 'p1': -0.5, 'p2': 0.5},  
    },

    'MGemus':
    {
    'gamma'   : {'type':'U', 'p0': 0.55,  'p1': 0.,     'p2': 1.},
    'gamma0'   : {'type':'U', 'p0': 0.55,  'p1': 0.,     'p2': 1.},
    'gamma1'   : {'type':'U', 'p0': 0.,  'p1': -0.7,     'p2': 0.7},
    'q1'   : {'type':'U', 'p0': 0.,  'p1': -1.,     'p2': 2.},
    'log10omegarc': {'type':'U', 'p0': np.log10(0.25), 'p1': -3, 'p2': 2.},
    'Omega_m':  {'type':'U', 'p0':0.3085640138408304, 'p1': 0.2899,     'p2': 0.3392},    
    'Omega_b':  {'type':'U', 'p0':0.04904844290657439, 'p1':0.04044,    'p2': 0.05686},  
    'Ombh2': {'type':'G', 'p0': 0.02268, 'p1': 0.02268, 'p2': 0.00038},  #BBN
    'h'   : {'type':'U', 'p0': 0.68, 'p1': 0.629,     'p2': 0.731},            
    'log10As' : {'type':'U', 'p0': 3.044, 'p1': 2.7081, 'p2': 3.2958},
    ##'log10As' : {'type':'G', 'p0': 3.044, 'p1': 3.044, 'p2': 0.042}, #3 sigma Planck
    ##'ns': {'type':'U', 'p0': 0.97, 'p1': 0.9432,'p2': 0.9862},
    'ns': {'type':'G', 'p0': 0.97, 'p1': 0.97,'p2': 0.004}, #1 sigma Planck
    #'Mnu': {'type':'U', 'p0': 0.06, 'p1': 0.0, 'p2': 0.1576},
    'Mnu': {'type':'U', 'p0': 0.06, 'p1': 0.0, 'p2': 0.5}, #for nDGP_v2

    #'Omega_m':  {'type':'G', 'p0':0.3085640138408304, 'p1': 0.3085640138408304,     'p2': 0.0073}, #1 Planck    
    #'h'   : {'type':'G', 'p0': 0.68, 'p1': 0.68,     'p2': 0.0054},  #1 Planck                    
    #'log10As' : {'type':'G', 'p0': 3.044, 'p1': 3.044, 'p2': 0.014}, #1 sigma Planck


    'mu1'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 2.},
    'mu2'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.9},
    'mu3'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.8},
    'mu4'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.8},
    'mu5'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.8},
    'mu6'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.8},
    'mu7'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.5},
    'mu8'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.5},
    'mu9'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.5},
    'mu10'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.5},
    'mu11'   : {'type':'U', 'p0': 1.,  'p1': 0.9,     'p2': 1.3},
    },
    'BCemu':
    {
    'log10Mc': {'type':'U', 'p0': 13.32, 'p1': 11.4, 'p2': 14.6}, #broad
    #'log10Mc': {'type':'U', 'p0': 13.32, 'p1': 13.12, 'p2': 13.52}, #tight
    #'log10Mc': {'type':'G', 'p0': 13.32, 'p1': 13.32, 'p2': 0.1},
    #'log10Mc': {'type':'U', 'p0': 13.32, 'p1': 12.00, 'p2': 12.4}, #tight wrong test
    'thej': {'type':'U', 'p0': 4.235, 'p1': 2.6, 'p2': 7.4},
    'mu': {'type':'U', 'p0': 0.93, 'p1': 0.2, 'p2': 1.8},
    'gamma_bc': {'type':'U', 'p0': 2.25, 'p1': 1.3, 'p2': 3.7},
    'delta': {'type':'U', 'p0': 6.4, 'p1': 3.8, 'p2': 10.2},
    'eta': {'type':'U', 'p0': 0.15, 'p1': 0.085, 'p2': 0.365},
    'deta': {'type':'U', 'p0': 0.14, 'p1': 0.085, 'p2': 0.365}
    },
    'b': {'type':'U', 'p0': 0.68, 'p1': 0., 'p2': 8.},
    'b2': {'type':'U', 'p0': 0., 'p1': -50., 'p2': 50.},
    'Delta_z': {'type':'G', 'p0': 0., 'p1': 0., 'p2': 0.001} #0.002*(1+z_i)
}