import MGLensing
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def for_plots(nell, n, flat_array):
    matrix_errs = np.zeros((nell, n, n))
    idx_start = 0
    for bin1 in range(n):
        for bin2 in range(bin1, n):
            matrix_errs[:, bin1, bin2] = flat_array[idx_start*nell:(idx_start+1)*nell]
            matrix_errs[:, bin2, bin1] = matrix_errs[:, bin1, bin2]
            idx_start += 1
    return matrix_errs

MGL = MGLensing.MGL("config.yaml")
data_vector = MGL.Data.data_vector


l_wl_max, l_gc_max = MGL.Survey.ells_wl_max, MGL.Survey.ells_gc_max
l_wl, l_gc, l_xc = MGL.Survey.l_wl, MGL.Survey.l_gc, MGL.Survey.l_xc
types = ['LL', 'LG', 'GG']
ells = [l_wl, l_xc, l_gc]
lmax = [l_wl[-1], l_xc[-1], l_gc[-1]]
lmax_ij = [l_wl_max, l_gc_max, l_gc_max]
nbin= 5 
start_lg = 15*len(l_wl)
start_gc = start_lg + 25*len(l_wl)
cell_wl = for_plots(len(l_wl), nbin, data_vector[:start_lg])
cell_gc = for_plots(len(l_wl), nbin, data_vector[start_gc:])

l_wl_cosmosis = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/shear_cl/ell.txt')
l_gc_cosmosis = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/galaxy_cl/ell.txt')
def plot_cells_ratio(type, show=True, names="", annotation=""):
    fig, ax = plt.subplots(figsize=(8, 8), nrows=nbin, ncols=nbin, sharex=True, sharey='row', facecolor='w')
    for i in range(nbin):
        for j in range(nbin):
            if i < j:
                ax[i, j].axis('off')
            else:
                #cls_cosmosis_ = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/shear_cl/bin_'+str(i+1)+'_'+str(j+1)+'.txt')
                #cls_cosmosis_itp =itp.interp1d(l_wl_cosmosis, cls_cosmosis_, kind='linear', fill_value='extrapolate')
                #cls_cosmosis = cls_cosmosis_itp(l_wl)+MGL.Survey.noise['LL'] if i==j else  cls_cosmosis_itp(l_wl)
                #ax[i, j].loglog(l_wl, cell_wl[:, i, j], label='MGL' if i == 0 and j == 0 else "")
                #ax[i, j].loglog(l_wl, cls_cosmosis, label='cosmosis' if i == 0 and j == 0 else "")


                cls_cosmosis_ = np.genfromtxt('../../Work/cosmosis_stuff/cosmosis-standard-library/lsst1/galaxy_cl/bin_'+str(i+1)+'_'+str(j+1)+'.txt')
                cls_cosmosis_itp =itp.interp1d(l_gc_cosmosis, cls_cosmosis_, kind='linear', fill_value='extrapolate')
                cls_cosmosis = cls_cosmosis_itp(l_gc)+MGL.Survey.noise['GG'] if i==j else  cls_cosmosis_itp(l_gc)
                #ax[i, j].semilogx(l_gc, cell_gc[:, i, j]/cls_cosmosis)
                #ax[i, j].set_ylim(0.98, 1.02) 
                ax[i, j].loglog(l_gc, cell_gc[:, i, j], label='MGL' if i == 0 and j == 0 else "")
                ax[i, j].loglog(l_gc, cls_cosmosis, label='cosmosis' if i == 0 and j == 0 else "")
                ax[i, j].axvspan(xmin=lmax_ij[type][i, j], xmax=lmax[type], color='grey', alpha=0.1)
                ax[i, j].legend(loc='upper left', title_fontsize=10, title='bin ' + str(i + 1) + '-' + str(j + 1))

    for i in range(nbin):
        ax[nbin - 1][i].set_xlabel('$\ell$')
    ax[int(nbin / 2)][0].set_ylabel('$C^{\\rm ' + types[type] + '}_{\ell}$ ratio')
    ax[1, 1].annotate(annotation, (1.1, 0.05), xycoords='axes fraction', clip_on=False)
    plt.tight_layout()
    plt.show() if show else plt.savefig('c_ells_' + types[type] + '_' + MGL.Survey.survey_name + '_vs_cosmosis.png')



plot_cells_ratio(1, True, names=['MGL data'])
# compare p_of_k from cosmosis output using mgl-interpolators:
# MGL.Data.DataModel.StructureEmu.get_pk_interp(MGL.Data.params_data_dic)