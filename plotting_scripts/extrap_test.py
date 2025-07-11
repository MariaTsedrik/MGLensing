import numpy as np
from scipy.interpolate import RectBivariateSpline
from math import log10, log
import matplotlib.pyplot as plt

import baccoemu 

heftemulator = baccoemu.Lbias_expansion()
params_bacco = {
                'ns'            :  0.97,
                'hubble'        :  0.68,
                'sigma8_cold'   :  0.83,
                'omega_baryon'  :  0.05,
                'omega_cold'    :  0.315, 
                'neutrino_mass' :  0., 
                'w0'            :  -1.,
                'wa'            :  0.,
                'expfactor'     :   1. 
            }
k, pnn = heftemulator.get_nonlinear_pnn(**params_bacco)
bL1 = 0.75
bL2 = 0. #0.25
bs2 = 0. #0.1
blapl = 0. #1.4
bias_params = [bL1, bL2, bs2, blapl] # b1, b2, bs2, blaplacian
k, p_gg, p_gm = heftemulator.get_galaxy_real_pk(bias=bias_params, k=k, **params_bacco)



p_dmdm = pnn[0]        
p_dmd1 = pnn[1]
p_dmd2 = pnn[2]
p_dms2 = pnn[3]      
p_dmk2 = pnn[4]     
p_d1d1 = pnn[5]    
p_d1d2 = pnn[6]      
p_d1s2 = pnn[7]    
p_d1k2 = pnn[8]       
p_d2d2 = pnn[9]       
p_d2s2 = pnn[10] 
p_d2k2 = pnn[11]
p_s2s2 = pnn[12]
p_s2k2 = pnn[13] 
p_k2k2 = pnn[14] 

pgg = (p_dmdm +
        (bL1+bL1) * p_dmd1 +
        (bL1*bL1) * p_d1d1 +
        (bL2 + bL2) * p_dmd2 +
        (bs2 + bs2) * p_dms2 +
        (bL1*bL2 + bL1*bL2) * p_d1d2 +
        (bL1*bs2 + bL1*bs2) * p_d1s2 +
        (bL2*bL2) * p_d2d2 +
        (bL2*bs2 + bL2*bs2) * p_d2s2 +
        (bs2*bs2)* p_s2s2 +
        (blapl + blapl) * p_dmk2 +
        (bL1 * blapl + bL1 * blapl) * p_d1k2 +
        (bL2 * blapl + bL2 * blapl) * p_d2k2 +
        (bs2* blapl + bs2 * blapl) * p_s2k2+
        (blapl * blapl) * p_k2k2)

emulator = baccoemu.Matter_powerspectrum()
k, pk_nl_cold = emulator.get_nonlinear_pk(k=k, cold=True, **params_bacco)
k, pk_lin_cold = emulator.get_linear_pk(k=k, cold=True, **params_bacco)
fig, ax = plt.subplots(figsize=(5, 3), facecolor='w')
ax.loglog(k, p_gg, color='tab:blue', label='get_galaxy_real_pk')
ax.loglog(k, pgg, color='tab:orange', label='get_nonlinear_pnn', linestyle='--')
ax.loglog(k, pk_nl_cold*(1+bL1)**2, color='tab:green', label='get_nonlinear_pk $\\times b_1^2$')
ax.loglog(k, pk_lin_cold*(1+bL1)**2, color='tab:red', label='get_linear_pk $\\times b_1^2$', linestyle='--')
ax.legend(loc='lower left', fontsize=8, title=r'bacco with $b_2=b_{s^2}=b_{\nabla}=0$')
ax.set_xlabel(r'$k \, [h \, \mathrm{Mpc}^{-1}]$')
ax.set_ylabel(r'$P_{\rm gg}(k) \, [h^{-3} \, \mathrm{Mpc}^3]$')
plt.tight_layout()
plt.show() 

'''
k_extrap = np.logspace(log10(k[-1]+0.01), log10(1.5), 100)        
log_k = log(k[-1]/k[-2])
m = np.array([log(np.abs(pnn[i][-1] / pnn[i][-2])) / log_k for i in range(15)])


fig, ax = plt.subplots(figsize=(5, 5), facecolor='w')
for i in range(15):
    #ax.loglog(k, pnn[i], label=str(i))
    #if all(pnn[i]>0):
    #    ax.loglog(k, pnn[i], label=str(i))
    #    ax.loglog(k_extrap, pnn[i][-1] * (k_extrap/k[-1])**m[i], linestyle='--', color=ax.get_lines()[-1].get_color())
    
    if any(pnn[i] < 0):
        ax.semilogx(k, pnn[i], label=str(i))
        ax.semilogx(k_extrap, pnn[i][-1] * (k_extrap/k[-1])**m[i], linestyle='--', color=ax.get_lines()[-1].get_color())
#ax.loglog(k, pnn[0]+pnn[1]+pnn[5], color='k')    
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show() 
'''