import numpy as np
import MGLensing
import matplotlib.pyplot as plt


MGLtest = MGLensing.MGL("config.yaml")


zz = MGLtest.Survey.zz_integr
nbin = MGLtest.Survey.nbin
b0 = 0.68
bias1_arr = np.array([b0*(1.+z_bin_center_i) for z_bin_center_i in MGLtest.Survey.z_bin_center ])
bias2_arr = bias1_arr*2. #0.9*(bias1_arr-1.)+0.5
print(bias1_arr , bias2_arr)

biasL1_arr = bias1_arr-1
#Lagrangian co-evolution 
biasL2_arr = 0.5*(0.9*biasL1_arr**2+0.5-8./21*biasL1_arr)#np.zeros(nbin)#
#local-in-matter-density (LIMD) Lagrangian bias:
biasLs2_arr = np.zeros(nbin)
biasLlapl_arr = np.zeros(nbin) 
print(biasL1_arr, biasL2_arr)

params = {
    'Omega_m' :  0.315,
    'Omega_nu':  0.,
    'As'      :  np.exp(3.07)*1.e-10,
    'Omega_b' :  0.05,
    'ns'      :  0.96,
    'h'       :  0.67,
    'Mnu'     :  0.0,
    'w0'      :  -1.0,
    'wa'      :  0.0,
    'b1_1': bias1_arr[0],      
    'b1_2': bias1_arr[1], 
    'b1_3': bias1_arr[2], 
    'b1_4': bias1_arr[3],             
    'b1_5': bias1_arr[4], 
    'b2_1': bias2_arr[0],      
    'b2_2': bias2_arr[1], 
    'b2_3': bias2_arr[2], 
    'b2_4': bias2_arr[3],             
    'b2_5': bias2_arr[4], 
    'aIA':   1., #1.72,
    'etaIA': 0.25, #-0.9, #-0.41,
    'b1L_1': biasL1_arr[0],      
    'b1L_2': biasL1_arr[1], 
    'b1L_3': biasL1_arr[2], 
    'b1L_4': biasL1_arr[3],             
    'b1L_5': biasL1_arr[4], 
    'b2L_1': biasL2_arr[0],      
    'b2L_2': biasL2_arr[1], 
    'b2L_3': biasL2_arr[2], 
    'b2L_4': biasL2_arr[3],             
    'b2L_5': biasL2_arr[4],   
    'bs2L_1': biasLs2_arr[0],      
    'bs2L_2': biasLs2_arr[1], 
    'bs2L_3': biasLs2_arr[2], 
    'bs2L_4': biasLs2_arr[3],             
    'bs2L_5': biasLs2_arr[4],   
    'blaplL_1': biasLlapl_arr[0],      
    'blaplL_2': biasLlapl_arr[1], 
    'blaplL_3': biasLlapl_arr[2], 
    'blaplL_4': biasLlapl_arr[3],             
    'blaplL_5': biasLlapl_arr[4]   
}     

W_LL = MGLtest.get_wl_kernel(params, NL_model=0, IA_model=100)
W_LL_tot = MGLtest.get_wl_kernel(params, NL_model=0, IA_model=0)
W_IA = MGLtest.get_ia_kernel(params, NL_model=0, IA_model=0)
print(W_LL.shape, W_IA.shape)





def plot_kernel_of_z(show=True):
    fig, ax = plt.subplots(1,2, figsize=(8,5), sharex=True, sharey=True)
    for i in range(5):
        ax[0].plot(zz, W_LL[0, :, i], label='bin '+str(i+1))
    for i in range(5):
        ax[1].plot(zz, W_IA[0, :, i], label='bin '+str(i+1))    
    for i in range(5):
        if i == 0:
            ax[0].plot(zz, W_LL_tot[0, :, i], linestyle='--', label='total', color='k')    
        else:   
            ax[0].plot(zz, W_LL_tot[0, :, i], linestyle='--', color='k')     
    ax[0].set_xlabel("$z$");ax[1].set_xlabel("$z$")
    ax[0].set_ylabel("$W(z)$")
    ax[0].legend(loc='upper right', title='WL kernel')
    ax[1].legend(loc='upper right', title='IA kernel: $A_{\\rm IA} = $'+str(params['aIA'])+', $\eta_{\\rm IA} = $'+str(params['etaIA']))
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('figs/modelling/wl_kernels_of_z_'+MGLtest.Survey.survey_name+'.png')    



plot_kernel_of_z(show=False)





