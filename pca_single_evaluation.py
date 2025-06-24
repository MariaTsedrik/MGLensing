import numpy as np
import MGLensing
import matplotlib.pyplot as plt
import seaborn as sns

# Perform PCA with numpy.linalg.svd - find rotation matrix
def findPCA(M_data, B_data, L_ch_inv):
    Delta = np.array(np.matmul(L_ch_inv, (B_data - M_data).T).T)
    Usvd, s, vh = np.linalg.svd(Delta.T, full_matrices=True)
    Usvd = Usvd.T
    return Usvd, Delta



MGL_mu_lin = MGLensing.MGL("ini_files/pca/config_muSigma_lin.yaml")
ells_wl_bins = []
ells_gc_bins = []
ells_xc_bins = []
#print(MGL_mu_lin.Survey.masked_data_vector_3x2pt_ells)
for i in range(MGL_mu_lin.Survey.mask_ells_wl.shape[0]):
    #print('bin ', i, ' number of ells_wl: ', sum(MGL_mu_lin.Survey.mask_ells_wl[i]))
    ells_wl_bins.append(sum(MGL_mu_lin.Survey.mask_ells_wl[i]))
for i in range(MGL_mu_lin.Survey.mask_ells_gc.shape[0]):
    #print('bin ', i, ' number of ells_gc: ', sum(MGL_mu_lin.Survey.mask_ells_gc[i]))
    ells_gc_bins.append(sum(MGL_mu_lin.Survey.mask_ells_gc[i]))
for i in range(MGL_mu_lin.Survey.mask_ells_xc.shape[0]):
    #print('bin ', i, ' number of ells_xc: ', sum(MGL_mu_lin.Survey.mask_ells_xc[i]))
    ells_xc_bins.append(sum(MGL_mu_lin.Survey.mask_ells_xc[i]))

ells_wl_bins = np.array(ells_wl_bins)
ells_gc_bins = np.array(ells_gc_bins)
ells_xc_bins = np.array(ells_xc_bins)

MGL_GR_nl = MGLensing.MGL("ini_files/pca/config_GR.yaml")
MGL_GR_lin = MGLensing.MGL("ini_files/pca/config_GR_lin.yaml")
MGL_nDGP_nl = MGLensing.MGL("ini_files/pca/config_nDGP.yaml")
MGL_nDGP_lin = MGLensing.MGL("ini_files/pca/config_nDGP_lin.yaml")
MGL_gamma_nl = MGLensing.MGL("ini_files/pca/config_gamma.yaml")
MGL_gamma_lin = MGLensing.MGL("ini_files/pca/config_gamma_lin.yaml")

cov = MGL_mu_lin.Data.data_covariance
D_data = MGL_mu_lin.Data.data_vector

L_choleski_uncut = np.linalg.cholesky(np.matrix(cov))
L_choleski_inv_uncut = np.linalg.inv(L_choleski_uncut)
L_ch_inv = L_choleski_inv_uncut

dic_test = {par_i: MGL_mu_lin.params_priors[par_i]['p0'] for par_i in MGL_mu_lin.params_priors.keys()}
dic_test = MGL_mu_lin.params_fixed | dic_test
param_dic_all, status = MGL_mu_lin.Like.Theo.check_pars(dic_test)
if status:
        D_theory = MGL_mu_lin.Like.compute_data_vector(param_dic_all) 
else:
        print("Parameters are not valid, cannot compute the data vector.")        
  
Diff = (D_data - D_theory)
print('Diff: ', Diff.shape)
# Find Choleski scaled data vector
Diff_ch = np.array(np.matmul(L_ch_inv, Diff.T))[0]

### COMBINE
# 1: find C_ell for non-linear matter power spectrum

B1 = MGL_nDGP_nl.Like.compute_data_vector(param_dic_all)  
B2 = MGL_gamma_nl.Like.compute_data_vector(param_dic_all)  
B3 = MGL_GR_nl.Like.compute_data_vector(param_dic_all)  
# 2: find C_ell for linear matter power spectrum
M1 = MGL_nDGP_lin.Like.compute_data_vector(param_dic_all)  
M2 = MGL_gamma_lin.Like.compute_data_vector(param_dic_all)  
M3 = MGL_GR_lin.Like.compute_data_vector(param_dic_all)  
print('B: ', B1.shape, B2.shape, B3.shape)
print('M: ', M1.shape, M2.shape, M3.shape)
       
B_data =np.array([B1,B2,B3])
M_data =np.array([M1,M2,M3])

# EXTRACT PCA MATRIX
Usvd, Delta = findPCA(M_data, B_data, L_ch_inv)
print('Usvd: ', Usvd.shape)
print('Delta: ', Delta.shape)
print(len(M_data))


col = sns.color_palette("colorblind", 6)
# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, sharex=True, figsize=(10, 6))

# Generate the new log-spaced x-axis

colors1 = [col[1],col[2],col[0]]
labels1 = ['nDGP','gamma','GR']
markers1 = ['o','D','P']

colors2 = [col[3],col[4],col[5]]
labels2 = ['PC1','PC2','PC3']
markers2 = ['x','.','^']
l_wl_max, l_gc_max = MGL_mu_lin.Survey.ells_wl_max, MGL_mu_lin.Survey.ells_gc_max
l_wl, l_gc, l_xc = MGL_mu_lin.Survey.l_wl, MGL_mu_lin.Survey.l_gc, MGL_mu_lin.Survey.l_xc

#print(MGL_mu_lin.Survey.mask_ells_wl.shape,MGL_mu_lin.Survey.mask_ells_gc.shape, MGL_mu_lin.Survey.mask_ells_xc.shape)


# Interpolate and plot on each subplot
for i in range(3):
   

    # Scatter points
    axs[0, 0].scatter(l_wl[:ells_wl_bins[0]], Delta[i][:ells_wl_bins[0]], color=colors1[i], label=labels1[i], marker=markers1[i])
    axs[0, 1].scatter(l_xc[:ells_xc_bins[0]], Delta[i][sum(ells_wl_bins):sum(ells_wl_bins)+ells_xc_bins[0]], color=colors1[i], label=labels1[i], marker=markers1[i])
    axs[0, 2].scatter(l_gc[:ells_gc_bins[0]], Delta[i][sum(ells_wl_bins)+sum(ells_xc_bins): sum(ells_wl_bins)+sum(ells_xc_bins)+ells_gc_bins[0]], color=colors1[i], label=labels1[i], marker=markers1[i])

axs[0, 0].set_title('Shear', fontsize=18)
axs[0, 1].set_title('Cross', fontsize=18)
axs[0, 2].set_title('Clustering', fontsize=18)

for i in range(3):

    # Scatter points
    axs[1, 0].scatter(l_wl[:ells_wl_bins[0]], -Usvd[i][:ells_wl_bins[0]], color=colors2[i], label=labels2[i], marker=markers2[i])
    axs[1, 1].scatter(l_xc[:ells_xc_bins[0]], -Usvd[i][sum(ells_wl_bins):sum(ells_wl_bins)+ells_xc_bins[0]], color=colors2[i], label=labels2[i], marker=markers2[i])
    axs[1, 2].scatter(l_gc[:ells_gc_bins[0]], -Usvd[i][sum(ells_wl_bins)+sum(ells_xc_bins): sum(ells_wl_bins)+sum(ells_xc_bins)+ells_gc_bins[0]], color=colors2[i], label=labels2[i], marker=markers2[i])

# Plot horizontal lines at y=0
axs[1, 0].axhline(y=0, linestyle="--", color="k")
axs[1, 1].axhline(y=0, linestyle="--", color="k")
axs[1, 2].axhline(y=0, linestyle="--", color="k")

# Add labels and set x-axis limits and scale
axs[1, 0].set_xlabel(r'$\ell$', fontsize=18)
axs[1, 1].set_xlabel(r'$\ell$', fontsize=18)
axs[1, 2].set_xlabel(r'$\ell$', fontsize=18)


axs[0, 0].legend(fontsize=12, frameon=False)
axs[1, 0].legend(fontsize=12, frameon=False)


axs[0, 0].set_ylabel(r'$\Delta \mathbf{M}_{\text{ch}}$', fontsize=18)
axs[1, 0].set_ylabel('Principal Components' + '\n' + r'(columns of $\mathbf{U}_{\text{ch}}$)', fontsize=18)

# Add x-axis label shared across all subplots
for ax in axs.flat:
    ax.set_xscale("log")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

# Display the plot
plt.show()
