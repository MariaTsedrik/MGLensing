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
                'omega_cold'    :  0.315, #0.3086053923645651,
                'neutrino_mass' :  0., #0.06,
                'w0'            :  -1.,
                'wa'            :  0.,
                'expfactor'     :   1. #0.8076923076923077
            }
k, pnn = heftemulator.get_nonlinear_pnn(**params_bacco)
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
