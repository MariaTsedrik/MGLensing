import MGLensing
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
fsky = 0.436
ell = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/galaxy_cl/ell.txt')
d_ell_bin = np.concatenate(([ell[0]], np.diff(ell)))
noise = {'LL': 2.86001911e-10, 'GG': 4.70088611e-09}
def get_cov_diag_ijkl(theory_spectra, name1, name2, bin_ij, bin_kl): 
    c_ij_12 = theory_spectra[name1]
    c_kl_34 = theory_spectra[name2]

    type_1, type_2 = c_ij_12['types']
    type_3, type_4 = c_kl_34['types']

    i,j = bin_ij
    k,l = bin_kl

    bin_pairs = [ (i,k), (j,l), (i,l), (j,k) ]
    type_pairs = [ (type_1,type_3), (type_2,type_4), (type_1,type_4), (type_2,type_3) ]
    #print('type_pairs: ', type_pairs)
    c_ells = []
    for bin_pair,type_pair in zip(bin_pairs, type_pairs):
        bin1, bin2 = bin_pair
        t1, t2 = type_pair
        types = (t1, t2)
        if types not in types_all:
            #If we don't have a spectra with these types, we probably 
            #have an equivalent one with the types the other way round.
            #In this case, we also need to swap bin1 and bin2, because
            #we are accessing e.g. C^{ik}_{13} via C^{ki}_{31}.
            types = (types[1], types[0])
            bin1, bin2 = bin2, bin1
        #print(types_all.index( types ) )    
        s = theory_spectra[ types_all.index( types ) ]
        #c_ells.append(s['cls'][:, bin1, bin2])
        if bin1==bin2:
            if t1==t2==0:
                cl = s['cls'][(bin1, bin2)]+noise['LL']
            elif t1==t2==1:
                cl = s['cls'][(bin1, bin2)]+noise['GG']   
            else:
                cl = s['cls'][(bin1, bin2)]        
        else:
            cl = s['cls'][(bin1, bin2)]          
        c_ells.append(cl)

    cl2_sum = c_ells[0]*c_ells[1] + c_ells[2]*c_ells[3]
    denom = ((2*ell+1)*fsky*d_ell_bin)
    return ( cl2_sum ) / denom

nbin = 5 #3 #5
nbin_flat = 15 #6 #15
cls_gc = {}
cls_wl = {}
cls_xc = {}
wl_guess = []
for bini in range(0,nbin):
    for binj in range(bini+1):
        print(bini, binj)
        if bini!=binj:
            cls_gc[(bini, binj)] = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/galaxy_cl/bin_'+str(bini+1)+'_'+str(binj+1)+'.txt')
            cls_wl[(bini, binj)] =np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/shear_cl/bin_'+str(bini+1)+'_'+str(binj+1)+'.txt')
            cls_gc[(binj, bini)] = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/galaxy_cl/bin_'+str(bini+1)+'_'+str(binj+1)+'.txt')
            cls_wl[(binj, bini)] =np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/shear_cl/bin_'+str(bini+1)+'_'+str(binj+1)+'.txt')
        else:
            cls_gc[(bini, binj)] = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/galaxy_cl/bin_'+str(bini+1)+'_'+str(binj+1)+'.txt')
            cls_wl[(bini, binj)] =np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/shear_cl/bin_'+str(bini+1)+'_'+str(binj+1)+'.txt')
        wl_guess.append(cls_wl[(binj, bini)])    
            
    for bink in range(0,nbin):    
        #print('bink, bini: ', bink, bini)
        #print('galaxy_shear_cl/bin_'+str(bini+1)+'_'+str(bink+1))
        cls_xc[(bink, bini)] =np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/galaxy_shear_cl/bin_'+str(bini+1)+'_'+str(bink+1)+'.txt')
        
n_ell = 20 #50
#ell_lims = self.Survey.ell_bin_edges
#ell_vals = np.arange(20, 5001).astype(int)
cl_ll_dic = {'bin_pairs': [(i, j) for i in range(nbin) for j in range(i, nbin)], 
            'cls': cls_wl,
            'is_auto': True,
            'name': 'LL',
            'types': (0, 0)
            }
cl_gg_dic = {'bin_pairs': [(i, j) for i in range(nbin) for j in range(i, nbin)], 
            'cls': cls_gc,
            'is_auto': True,
            'name': 'GG',
            'types': (1, 1)
            }
cl_xc_dic = {'bin_pairs': [(j, i) for i in range(nbin) for j in range(nbin)], 
            'cls': cls_xc,
            'is_auto': False,
            'name': 'LG',
            'types': (0, 1)
            }

print([(j, i) for i in range(nbin) for j in range(nbin)])
theory_spectra = [cl_ll_dic, cl_xc_dic, cl_gg_dic]
types_all = [cl_ll_dic['types'], cl_xc_dic['types'], cl_gg_dic['types']]
# Get the starting index in the full datavector for each spectrum
# this will be used later for adding covariance blocks to the full matrix.
cl_lengths = [n_ell*nbin_flat, n_ell*nbin**2, n_ell*nbin_flat]
cl_starts = []
for i in range(3):
    cl_starts.append( int(sum(cl_lengths[:i])) )
covmat = np.zeros((2*n_ell*nbin_flat+n_ell*nbin**2, 2*n_ell*nbin_flat+n_ell*nbin**2), 'float64')

# Now loop through pairs of Cls and pairs of bin pairs filling the covariance matrix
for i_cl in range(3):
    cl_spec_i = theory_spectra[i_cl]
    for j_cl in range(i_cl, 3):
        cl_spec_j = theory_spectra[j_cl]
        cov_blocks = {} #collect cov_blocks in this dictionary
        for i_bp, bin_pair_i in enumerate(cl_spec_i['bin_pairs']):
                for j_bp, bin_pair_j in enumerate(cl_spec_j['bin_pairs']):
                    #print(f"Computing covariance {i_cl},{j_cl} pairs <{bin_pair_i} {bin_pair_j}>")
                    # First check if we've already calculated this
                    if (i_cl == j_cl) and cl_spec_i['is_auto'] and ( j_bp < i_bp ):
                        cl_var_binned = cov_blocks[j_bp, i_bp]
                    else:    
                        cl_var_binned = get_cov_diag_ijkl(theory_spectra, i_cl, 
                                j_cl, bin_pair_i, bin_pair_j)

                        '''
                        #First calculate the unbinned Cl covariance
                        cl_var_unbinned = self.get_cov_diag_ijkl( theory_spectra, i_cl, 
                                j_cl, bin_pair_i, bin_pair_j)
                        #Now bin this diaginal covariance
                        #Var(binned_cl) = \sum_l Var(w_l^2 C(l)) / (\sum_l w_l)^2
                        #where w_l = 2*l+1
                        cl_var_binned = np.zeros(n_ell)
                        for ell_bin, (ell_low, ell_high) in enumerate(zip(ell_lims[:-1], ell_lims[1:])):
                            #Get the ell values for this bin:
                            ell_vals_bin = np.arange(ell_low, ell_high).astype(int)
                            #Get the indices in cl_var_binned these correspond to:
                            ell_vals_bin_inds = ell_vals_bin - int(ell_lims[0])
                            cl_var_unbinned_bin = cl_var_unbinned[ell_vals_bin_inds]
                            cl_var_binned[ell_bin] = np.sum((2*ell_vals_bin+1)**2 * 
                                cl_var_unbinned_bin) / np.sum(2*ell_vals_bin+1)**2'
                        '''
                        cov_blocks[i_bp, j_bp] = cl_var_binned
                        

                    # Now work out where this goes in the full covariance matrix
                    # and add it there.
                    inds_i = np.arange( cl_starts[i_cl] + n_ell*i_bp, 
                        cl_starts[i_cl] + n_ell*(i_bp+1) )
                    inds_j = np.arange( cl_starts[j_cl] + n_ell*j_bp, 
                        cl_starts[j_cl] + n_ell*(j_bp+1) )
                    cov_inds = np.ix_( inds_i, inds_j )
                    covmat[ cov_inds ] = np.diag(cl_var_binned)
                    cov_inds_T = np.ix_( inds_j, inds_i )
                    covmat[ cov_inds_T ] = np.diag(cl_var_binned)    

cmap = 'seismic'
cov = covmat
ndata = cov.shape[0]
pp_norm = np.zeros((ndata,ndata))
for i in range(ndata):
    for j in range(ndata):
        pp_norm[i][j] = cov[i][j]/np.sqrt(cov[i][i]*cov[j][j])   
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
im3 = ax.imshow(pp_norm, cmap=cmap, vmin=-1, vmax=1)
fig.colorbar(im3, orientation='vertical')
plt.show()


data_covariance = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/output/lsst_forecast/data_vector/lsst_covariance.txt')
print(covmat/data_covariance)

data_vector = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/output/lsst_forecast/data_vector/lsst_data.txt')
start_lg = 15*n_ell
start_gc = start_lg + 25*n_ell
wl_vector = data_vector[:start_lg]
gc_vector = data_vector[start_gc:]
xc_vector = data_vector[start_lg:start_gc]

wl_guess = np.concatenate(wl_guess)
print(wl_guess/wl_vector)