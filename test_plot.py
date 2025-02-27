import numpy as np
import MGLensing
import matplotlib.pyplot as plt


MGLtest = MGLensing.MGL("config.yaml")


zz = MGLtest.Survey.zz_integr
nbin = MGLtest.Survey.nbin
b0 = 0.68
bias1_arr = np.array([b0*(1.+z_bin_center_i) for z_bin_center_i in MGLtest.Survey.z_bin_center_l ])
bias2_arr = bias1_arr*2. 
print('bias1_arr , bias2_arr: ', bias1_arr , bias2_arr)
bias1_arr = np.array([1.239, 1.378, 1.525, 1.677, 1.832])
biasL1_arr = bias1_arr-1
#Lagrangian co-evolution 
b2_arr = np.array([-0.258, -0.062, 0.107, 0.267, 0.462])
#biasL2_arr = b2_arr-8./21*biasL1_arr
biasL2_arr = (0.9*biasL1_arr**2+0.5)-8./21*biasL1_arr
#local-in-matter-density (LIMD) Lagrangian bias:
biasLs2_arr = np.zeros(nbin)
biasLlapl_arr = np.zeros(nbin) 
print('biasL1_arr, biasL2_arr: ', biasL1_arr, biasL2_arr)

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
    'aIA':   0.42/1.62**2.21, #1., #0.42, #1.72,
    'etaIA': 2.21, #0., #2.21, #-0.9, #-0.41,
    'betaIA': 0., #2.17,
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






def plot_n_of_z(show=True):
    etaz_s, etaz_l  = MGLtest.Survey.eta_z_s, MGLtest.Survey.eta_z_l
    fig, ax = plt.subplots(1,2, figsize=(8,5), sharex=True, sharey=True)
    for i in range(etaz_s.shape[1]):
        ax[0].plot(zz, etaz_s[:, i], label='bin '+str(i+1))
    for i in range(etaz_l.shape[1]):
        ax[1].plot(zz, etaz_l[:, i], label='bin '+str(i+1))    
    ax[0].set_xlabel("$z$");ax[1].set_xlabel("$z$")
    ax[0].set_ylabel("$\\eta(z)$")
    ax[0].legend(loc='upper right', title='sources')
    ax[1].legend(loc='upper right', title='lenses')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('figs/modelling/etas_of_z_'+MGLtest.Survey.survey_name+'.png')    

def plot_p_k_fields(Pk, bPgg, bPgm, z_int_pick, bini, binj, show=True, name=""):
    def k2ell(x, ind=z_int_pick[0]):
        return x*rcom[ind]-0.5
    def ell2k(x, ind=z_int_pick[0]):
        return (x+0.5)/rcom[ind]
    def k2ell1(x, ind=z_int_pick[1]):
        return x*rcom[ind]-0.5
    def ell2k1(x, ind=z_int_pick[1]):
        return (x+0.5)/rcom[ind]
    def k2ell2(x, ind=z_int_pick[2]):
        return x*rcom[ind]-0.5
    def ell2k2(x, ind=z_int_pick[2]):
        return (x+0.5)/rcom[ind]

    f_arr = [k2ell, k2ell1, k2ell2]
    f_inv_arr = [ell2k, ell2k1, ell2k2]

    fig, ax = plt.subplots(1,3, figsize=(12,5))
    for i in range(3):
        if i==0:
            label_m='matter'
            label_g_ij='galaxy bin '+str(bini)+'-'+str(binj)
            label_x_i='cross bin '+str(bini)
            label_x_j='cross bin '+str(binj)
        else:
            label_m=label_g_ij=label_x_i=label_x_j=None   
        ax[i].loglog(k_ell[:, z_int_pick[i]], Pk[:, z_int_pick[i]], label=label_m)
        ax[i].loglog(k_ell[:, z_int_pick[i]], bPgg[:, z_int_pick[i],  bini-1, binj-1], label=label_g_ij)
        ax[i].loglog(k_ell[:, z_int_pick[i]], bPgm[:, z_int_pick[i], bini-1], label=label_x_i)
        ax[i].loglog(k_ell[:, z_int_pick[i]], bPgm[:, z_int_pick[i], binj-1], label=label_x_j)
        ax[i].legend(title='$z=$'+str(round(zz[z_int_pick[i]],2)), loc='lower left')
        ax[i].set_xlabel("$k$ [$h$/Mpc]")
        secax = ax[i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
        secax.set_xlabel('$\ell$')
    ax[0].set_ylabel("$P^{\\rm NL}(k(\ell, z), z)$")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('figs/modelling/p_of_k'+name+'.png')  

def plot_cells(type, Cl_LL_list, Cl_LG_list, Cl_GG_list, show=True, names=""):
    types = ['LL', 'LG', 'GG']
    ells = [l_WL, l_XC, l_GC]
    lmax = [l_WL[-1], l_XC[-1], l_GC[-1]]
    lmax_ij = [l_WL_max, l_GC_max, l_GC_max]
    Cells = [Cl_LL_list, Cl_LG_list, Cl_GG_list]
    fig, ax = plt.subplots(figsize=(8, 8), nrows = nbin, ncols=nbin, sharex=True, sharey=True,  facecolor='w')
    for ind in range(len(names)):
        for i in range(nbin):
            for j in range(nbin):
                if i<j:
                    ax[i, j].axis('off')
                else:
                    if i==0 and j==0:
                        ax[i, j].loglog(ells[type], Cells[type][ind][:, i, j], label=names[ind])
                        ax[i, j].axvspan(xmin=lmax_ij[type][i, j],xmax=lmax[type],color='grey',alpha=0.1)
                    else:    
                        ax[i, j].loglog(ells[type], Cells[type][ind][:, i, j])
                        ax[i, j].axvspan(xmin=lmax_ij[type][i, j],xmax=lmax[type],color='grey',alpha=0.1)
                    ax[i, j].legend(loc='lower left', title_fontsize=10, title='bin '+str(i+1)+'-'+str(j+1))
        for i in range(nbin):
            ax[nbin-1][i].set_xlabel('$\ell$')
        
    ax[int(nbin/2)][0].set_ylabel('$C^{\\rm '+types[type]+'}_{\ell}$')

    #ax[1, 1].annotate('LSST kernels with DES IA parameters: \n $A_{\\rm IA} = $'+str(round(params['aIA'], 2))+', $\eta_{\\rm IA} = $'+str(params['etaIA'])+', $\\beta_{\\rm IA} = $'+str(params['betaIA']), (1.1, 0.05), xycoords='axes fraction', clip_on=False) 
    ax[1, 1].annotate('WL k_eff: '+str(MGLtest.config_dic['specs']['scale_cuts']['max_WL'])+'\n GC k_eff: '+str(MGLtest.config_dic['specs']['scale_cuts']['max_GC']), (1.1, 0.05), xycoords='axes fraction', clip_on=False) 
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('figs/modelling/cells_'+types[type]+'_'+MGLtest.Survey.survey_name+'_cut_in_kmax.png')  

def plot_pk_bias(z_int_pick, bini, binj, show=True):
    def k2ell(x, ind=z_int_pick[0]):
        return x*rcom[ind]-0.5
    def ell2k(x, ind=z_int_pick[0]):
        return (x+0.5)/rcom[ind]
    def k2ell1(x, ind=z_int_pick[1]):
        return x*rcom[ind]-0.5
    def ell2k1(x, ind=z_int_pick[1]):
        return (x+0.5)/rcom[ind]
    def k2ell2(x, ind=z_int_pick[2]):
        return x*rcom[ind]-0.5
    def ell2k2(x, ind=z_int_pick[2]):
        return (x+0.5)/rcom[ind]

    f_arr = [k2ell, k2ell1, k2ell2]
    f_inv_arr = [ell2k, ell2k1, ell2k2]

    fig, ax = plt.subplots(1,3, figsize=(12,5))
    for i in range(3):
        if i==0:
            label_1='b1'
            label_2='b1+b2'
            label_3='bacco'
        else:
            label_1=label_2=label_3=None   
        ax[i].loglog(k_ell[:, z_int_pick[i]], bPgg[:, z_int_pick[i], bini-1, binj-1], label=label_1)
        ax[i].loglog(k_ell[:, z_int_pick[i]], bPgg2[:, z_int_pick[i], bini-1, binj-1], label=label_2)
        ax[i].loglog(k_ell[:, z_int_pick[i]], bPgg_bacco[:, z_int_pick[i], bini-1, binj-1], label=label_3)
        ax[i].legend(title='$z=$'+str(round(zz[z_int_pick[i]],2))+' in  bin '+str(bini)+'-'+str(binj), loc='lower left')
        ax[i].set_xlabel("$k$ [$h$/Mpc]")
        secax = ax[i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
        secax.set_xlabel('$\ell$')
    ax[0].set_ylabel("$P^{\\rm gg}(k(\ell, z), z)$")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('figs/modelling/pgg_of_k.png')  

def plot_pmm(z_int_pick, show=True):
    def k2ell(x, ind=z_int_pick[0]):
        return x*rcom[ind]-0.5
    def ell2k(x, ind=z_int_pick[0]):
        return (x+0.5)/rcom[ind]
    def k2ell1(x, ind=z_int_pick[1]):
        return x*rcom[ind]-0.5
    def ell2k1(x, ind=z_int_pick[1]):
        return (x+0.5)/rcom[ind]
    def k2ell2(x, ind=z_int_pick[2]):
        return x*rcom[ind]-0.5
    def ell2k2(x, ind=z_int_pick[2]):
        return (x+0.5)/rcom[ind]

    f_arr = [k2ell, k2ell1, k2ell2]
    f_inv_arr = [ell2k, ell2k1, ell2k2]

    fig, ax = plt.subplots(1,3, figsize=(12,5))
    for i in range(3):
        if i==0:
            label_1='hmcode'
            label_2='hmcode'
            label_3='bacco'
        else:
            label_1=label_2=label_3=None   
        ax[i].loglog(k_ell[:, z_int_pick[i]], Pk[:, z_int_pick[i]], label=label_1)
        ax[i].loglog(k_ell[:, z_int_pick[i]], Pk[:, z_int_pick[i]], label=label_2)
        ax[i].loglog(k_ell[:, z_int_pick[i]], Pk_bacco[:, z_int_pick[i]], label=label_3)
        ax[i].legend(title='$z=$'+str(round(zz[z_int_pick[i]],2)), loc='lower left')
        ax[i].set_xlabel("$k$ [$h$/Mpc]")
        secax = ax[i].secondary_xaxis('top', functions=(f_arr[i], f_inv_arr[i]))
        secax.set_xlabel('$\ell$')
    ax[0].set_ylabel("$P^{\\rm mm}(k(\ell, z), z)$")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('figs/modelling/pmm_of_k.png')  

_, rcom = MGLtest.get_expansion_and_rcom(params)

#plot power spectra (with/without baryons, with/without galaxy bias)
k_ell, Pk =  MGLtest.get_Pmm(params, NL_model=0, baryon_model=0)
_, bPgg =  MGLtest.get_bPgg(params, NL_model=0, bias_model=0)
_, bPgm =  MGLtest.get_bPgm(params, NL_model=0, bias_model=0)

_, bPgg2 =  MGLtest.get_bPgg(params, NL_model=0, bias_model=1)
_, bPgm2 =  MGLtest.get_bPgm(params, NL_model=0, bias_model=1)



z_int_pick = [10, 100, 180] #goes from 0 to 199
bini=2
binj=3


_, Pk_bacco =  MGLtest.get_Pmm(params, NL_model=1, baryon_model=0)
_, bPgg_bacco =  MGLtest.get_bPgg(params, NL_model=1, bias_model=2)
_, bPgm_bacco =  MGLtest.get_bPgm(params, NL_model=1, bias_model=2)


#plot_n_of_z(False)
#plot_p_k_fields(Pk_bacco, bPgg_bacco, bPgm_bacco, z_int_pick, bini, binj, False, "_bacco")
#plot_p_k_fields(Pk, bPgg, bPgm, z_int_pick, bini, binj, False, "_hmcode")
#plot_pk_bias(z_int_pick, bini, binj, False)
#plot_pmm(z_int_pick, False)




#plot c_ells 
l_WL, Cl_LL = MGLtest.get_cell_shear(params, NL_model=0, baryon_model=0, IA_model=0)
l_GC, Cl_GG = MGLtest.get_cell_galclust(params, NL_model=0, bias_model=0, baryon_model=0)   
l_XC, Cl_LG, Cl_GL = MGLtest.get_cell_galgal(params, NL_model=0, bias_model=0, baryon_model=0, IA_model=0)   

_, Cl_LL_bacco = MGLtest.get_cell_shear(params, NL_model=1, baryon_model=0, IA_model=0)
_, Cl_GG_bacco = MGLtest.get_cell_galclust(params, NL_model=1, bias_model=2,  baryon_model=0)   
_, Cl_LG_bacco, Cl_GL_bacco = MGLtest.get_cell_galgal(params, NL_model=1, bias_model=2, baryon_model=0, IA_model=0)   

l_WL_max, l_GC_max = MGLtest.Survey.ells_WL_max, MGLtest.Survey.ells_GC_max

for type_i in [0, 1, 2]:
    plot_cells(type_i, [Cl_LL, Cl_LL_bacco], [Cl_LG, Cl_LG_bacco], [Cl_GG, Cl_GG_bacco], False, names=['hmcode', 'bacco'])

