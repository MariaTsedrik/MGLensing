import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
import baccoemu
import MGLensing
import os

folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder)


def read_cosmosis_Cls(cosmosispath, tracer='gg', nbins=5):
    if tracer == 'gg':
        folder = 'galaxy_cl/'
    elif tracer == 'll':
        folder = 'shear_cl/'
    elif tracer == 'gl':
        folder = 'galaxy_shear_cl/'
    else:
        raise ValueError('Wrong tracer name! Must be gg, ll or gl')     
    # read ell values 
    ell = np.loadtxt(cosmosispath + folder + 'ell.txt', unpack=True)
    # read Cl values bin by bin
    cl = np.zeros((nbins, nbins, len(ell)))
    for i in range(nbins):
        for j in range(nbins):
            if i >= j:
                cl[i,j] = np.loadtxt(cosmosispath + folder + 'bin_%s_%s.txt'%(str(i+1),str(j+1)), unpack=True)
    return ell, cl


def plot_Cl_vs_CLL(ell, ell3, Cl1, Cl2, Cl3, title, filename, color, annotation_text=''):
    fig, ax = plt.subplots(nbin, nbin, figsize=(10, 10), sharey='row', sharex=True)
    for i in range(nbin):
        for j in range(nbin):
            if i >= j:
                ax[i,j].plot(ell, Cl1[:,i,j], color=color[0], linewidth=2)
                ax[i,j].plot(ell, Cl2[i,j], color=color[1], linestyle='--', linewidth=2)
                ax[i,j].plot(ell3, Cl3[i,j], color=color[2], linestyle=':', linewidth=2)
                ax[i,j].set_xscale('log')
                ax[i,j].set_yscale('log')
                # ax[i, j].set_ylim(1e-10, 1e-6)
                ax[i,j].text(.4, .9, 'bin %s, %s'%(str(i),(j)), fontsize="medium", horizontalalignment='center', transform=ax[i,j].transAxes)  
                ax[i,j].tick_params(axis='both', which='major', labelsize=12)        
            else:
                ax[i,j].axis('off')
    fig.text(0.06, 0.5, r'$C_\ell$', ha='center', va='center', rotation='vertical', fontsize=20)
    fig.text(0.5, 0.04, r'$\ell$', ha='center', va='center', fontsize=20)   
    fig.suptitle(MGLtest.Survey.survey_name + ' - ' + title, fontsize=20)
    fig.legend(['MGL', 'CCL', 'CosmoSIS'], loc='upper right', ncol=1, bbox_to_anchor=(1, 0.93), bbox_transform=fig.transFigure, fontsize=20)
    fig.text(0.8, 0.65, annotation_text, ha='center', va='center', fontsize=12)   
    plt.savefig(f"/home/s2561233/Documents/lss/nonlinear-bias-3x2-MG/new-MGlensing/MGlensing/figs/modelling/{filename}-{MGLtest.Survey.survey_name}.png")
    # plt.show()
                


# ------------------------- #
# MGL initialization and Cls
# ------------------------- #
MGLtest = MGLensing.MGL("config.yaml")

zz = MGLtest.Survey.zz_integr
nbin = MGLtest.Survey.nbin
b0 = 0.68
bias1_arr = np.ones(nbin)
# bias1_arr = np.array([b0*(1.+z_bin_center_i) for z_bin_center_i in MGLtest.Survey.z_bin_center_l ])
bias2_arr = bias1_arr*2. 
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
    'Omega_c' :  0.315-0.05,
    'Omega_cb' :  0.315,
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
    'aIA':   0.,
    'etaIA': 0., #0., #2.21, #-0.9, #-0.41,
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

# MGL Cells 
l_WL, Cl_LL = MGLtest.get_cell_shear(params, nl_model=0, baryon_model=0, ia_model=0, photoz_err_model=0)
l_GC, Cl_GG = MGLtest.get_cell_galclust(params, nl_model=0, bias_model=0, baryon_model=0, photoz_err_model=0) 
l_XC, Cl_LG, Cl_GL = MGLtest.get_cell_galgal(params, nl_model=0, bias_model=0, baryon_model=0, ia_model=0, photoz_err_model=0)   



# ----------- #
# Cls from CCL
# ----------- #
# NL matter power spectrum with Bacco emu
emulator = baccoemu.Matter_powerspectrum()
k_range_lin = emulator.emulator['linear']['k']
k_range_nl = emulator.emulator['nonlinear']['k']

kmin, kmax, nk = min(k_range_nl), max(k_range_nl), len(l_WL)
print('kmin, kmax, nk:', kmin, kmax, nk)
k_bemu = np.logspace(np.log10(kmin*0.67), np.log10(kmax*0.67), nk) # Wavenumber [Mpc]^-1

# non-linear matter power spectrum
bemu_nl = ccl.BaccoemuNonlinear()
cosmo_nl = ccl.Cosmology(Omega_c=params['Omega_c'], Omega_b=params['Omega_b'], h=params['h'], n_s=params['ns'], A_s=params['As'],
                      m_nu=params['Mnu'], transfer_function='boltzmann_camb',
                      matter_power_spectrum=bemu_nl)


# tracers
lens = []
cluster = []
bias_ia = np.zeros(len(MGLtest.Survey.zz_integr)) 
for i in range(MGLtest.Survey.nbin):
    lens.append(ccl.WeakLensingTracer(cosmo_nl, dndz=(MGLtest.Survey.zz_integr, MGLtest.Survey.eta_z_s[:,i]), ia_bias=(MGLtest.Survey.zz_integr,bias_ia))) #CCL automatically normalizes dNdz
    bias_gal = bias1_arr[i]*np.ones(len(MGLtest.Survey.zz_integr))
    cluster.append(ccl.NumberCountsTracer(cosmo_nl, has_rsd=False, dndz=(MGLtest.Survey.zz_integr, MGLtest.Survey.eta_z_l[:,i]), bias=(MGLtest.Survey.zz_integr, bias_gal)))

cl_ll_ccl = np.zeros((MGLtest.Survey.nbin, MGLtest.Survey.nbin, len(l_WL)))
cl_gg_ccl = np.zeros((MGLtest.Survey.nbin, MGLtest.Survey.nbin, len(l_GC)))
cl_lg_ccl = np.zeros((MGLtest.Survey.nbin, MGLtest.Survey.nbin, len(l_XC)))
cl_gl_ccl = np.zeros((MGLtest.Survey.nbin, MGLtest.Survey.nbin, len(l_XC)))

# Calculate CCL Cls
for i in range(MGLtest.Survey.nbin):
    for j in range(MGLtest.Survey.nbin):
        cl_ll_ccl[i,j] = ccl.angular_cl(cosmo_nl, lens[i], lens[j], l_WL) 
        cl_gg_ccl[i,j] = ccl.angular_cl(cosmo_nl, cluster[i], cluster[j], l_GC) 
        cl_lg_ccl[i,j] = ccl.angular_cl(cosmo_nl, lens[i], cluster[j], l_XC) 
        cl_gl_ccl[i,j] = ccl.angular_cl(cosmo_nl, cluster[i], lens[j], l_XC) 

# add noise 
for i in range(MGLtest.Survey.nbin):
    cl_ll_ccl[i,i] += MGLtest.Survey.noise['LL']
    cl_gg_ccl[i,i] += MGLtest.Survey.noise['GG']
    cl_lg_ccl[i,i] += MGLtest.Survey.noise['LG']
    cl_gl_ccl[i,i] += MGLtest.Survey.noise['GL']



# ---------------- #
# Cls from CosmoSIS
# ---------------- #
cosmosispath = '/home/s2561233/Documents/lss/cosmosis-standard-library/output/my_lsst_forecast/'
l_gg_cosmosis, cl_gg_cosmosis = read_cosmosis_Cls(cosmosispath, tracer='gg', nbins=5)
l_ll_cosmosis, cl_ll_cosmosis = read_cosmosis_Cls(cosmosispath, tracer='ll', nbins=5)
l_xx_cosmosis, cl_gl_cosmosis = read_cosmosis_Cls(cosmosispath, tracer='gl', nbins=5)

# add noise 
for i in range(MGLtest.Survey.nbin):
    cl_ll_cosmosis[i,i] += MGLtest.Survey.noise['LL']
    cl_gg_cosmosis[i,i] += MGLtest.Survey.noise['GG']
    cl_gl_cosmosis[i,i] += MGLtest.Survey.noise['GL']



# ---- # 
# Plots
# ---- #
print('CLL plotting...')
plot_Cl_vs_CLL(l_GC, l_gg_cosmosis, Cl_GG, cl_gg_ccl, cl_gg_cosmosis, title='Cl_GG', filename='compare-Cells_GG', color=['C0','r','k'], annotation_text='no bias, no IA')
plot_Cl_vs_CLL(l_WL, l_ll_cosmosis, Cl_LL, cl_ll_ccl, cl_ll_cosmosis, title='Cl_LL', filename='compare-Cells_LL', color=['C2','r','k'], annotation_text='no bias, no IA')
plot_Cl_vs_CLL(l_XC, l_xx_cosmosis, Cl_GL, cl_gl_ccl, cl_gl_cosmosis, title='Cl_GL', filename='compare-Cells_GL', color=['C1','r','k'], annotation_text='no bias, no IA')

