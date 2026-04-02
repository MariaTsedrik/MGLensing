import numpy as np
import getdist.plots
import matplotlib.pyplot as plt
import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
with open("plotting_scripts/params_names.yaml", "r") as file:
    params_dic = yaml.safe_load(file)
#with open("params_data_zennaro_y10.yaml", "r") as file:
with open("params_data_hmcode.yaml", "r") as file:
    fiducials = yaml.safe_load(file)



def read_last_header_line(file_path):
    last_header = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                last_header = line.strip('#').strip()  
            else:
                break  

    if last_header:
        return last_header.split() 
    else:
        return []

file_paths = [ #'chains/chain_lsst_y1_fix_cosmo_zennaro_data_zennaro_model.txt',  
              #'chains/chain_lsst_y1_fix_cosmo_zennaro_data_heft_model.txt',
              #'chains/chain_lsst_y1_fix_cosmo_zennaro_data_des_model.txt',
              #'chains/chain_lsst_y1_fix_cosmo_zennaro_data_positiv_b2_zennaro_model.txt',
              #'chains/chain_lsst_y1_fix_cosmo_zennaro_data_positiv_b2_zennaro_model_nobar.txt'
              #'chains/chain_euclid_5bins_linscales_mukdep_fixnuisance_new.txt'

              #'chains/chain_lsst_y1_test_gr_3x2pt.txt',
              #'chains/chain_lsst_y10_test_gr_3x2pt.txt'

            #'chains/chain_2x2pt_lsst_y1_fix_cosmo_hobnobar_data_nohobbar_model.txt',
              #'chains/chain_2x2pt_lsst_y1_fix_cosmo_hobnobar_data_nohobbar_model_v2.txt',
              #'chains/chain_2x2pt_lsst_y1_fix_cosmo_hobnobar_data_nohobbar_model_v3.txt',
              ##'chains/chain_2x2pt_lsst_y1_fix_cosmo_hobnobar_data_nohobbar_model_v4.txt',
              #'chains/chain_2x2pt_lsst_y1_fix_cosmo_hobnobar_data_nohobbar_model_v5.txt',
              #'chains/chain_2x2pt_lsst_y1_fix_cosmo_hobnobar_data_nohobbar_model_v6.txt',
              #'chains/chain_2x2pt_lsst_y1_fix_cosmo_hob10bar_data_nohobbar_model_v6.txt'

               # 'chains/chain_2x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v6.txt',
              #'chains/chain_3x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v6.txt',
              #'chains/chain_3x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v3.txt',
              #'chains/chain_3x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v2.txt',
              #'chains/chain_3x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v7.txt',
                #'chains/chain_3x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v7_withb2.txt',

                # for the plot in the paper:
                #'chains/chain_3x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v6.txt',
                #'chains/chain_3x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v7.txt',
                #'chains/chain_3x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v2.txt',
                #'chains/chain_2x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v6.txt',
                #'chains/chain_2x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v7.txt',
                #'chains/chain_2x2pt_lsst_y1_vary_Ommsigma8_hob10bar_data_nohobbar_model_v2.txt',
            
            
            #'chains/chain_lsst_y1_fix_cosmo_zennaro_data_zennaro_fixbs2blapl_model_mnu.txt',
            #'chains/chain_lsst_y1_fix_cosmo_zennaro_data_zennaro_minimal_model_mnu.txt'
                # 'chains/chain_lsst_y10_fix_cosmo_zennaro_data_zennaro_b1b2bs2model_mnu.txt',
                # 'chains/chain_Y10_heft_heft_kmax0.7_CMBpriors_ns_Ob_fix_bs2_only_09Sep.txt',
                # 'chains/chain_lsst_y10_fix_cosmo_zennaro_data_zennaro_minimalmodel_mnu.txt'

                'chains/chain_3x2pt_lsst_y1_forecast_w0wacdm.txt'
 ]  

file_name = 'lsst_y1_w0wacdm_all'
legend_labels = [
#'with baryons fixed',
#'no baryons'
#'model: heft',
#'model: DES-like'
#'Y1',
#'Y10'
#'(2x2pt) data: no baryons+$b_{s^2}, b_{\\nabla^2} \\neq 0$, model: baryons+$b_{s^2}=b_{\\nabla^2}=0$'
#'b1 fid, bs2=0.2, blapl=0.35',
#'b1 fid, bs2=0.15, blapl=0.2',
#'b1 fid, bs2=0.1, blapl=0.05',
#'b1 fid, bs2=0.0, blapl=[0.1, 0.1, 0.07, 0.05, 0.05]',
#'b1 fid, bs2=0.0, blapl=[0.08, 0.07, 0.07, 0.05, 0.05]',
#'b1 fid, bs2=0.0, blapl=[0.08, 0.07, 0.07, 0.05, 0.05]',
#'2x2pt',
#'3x2pt',
#'3x2pt',
#'3x2pt',
#'3x2pt'
#'HOB fiducial',
# 'only $b_{\\nabla^2}=0$',
# 'only $b_{s^2}=0$ \n(+vary cosmology)',
# 'minimal'

#'$b_{s^2}=0$, $b_{\\nabla^2}=0.1$',
#'$b_{s^2}=0.15$, $b_{\\nabla^2}=0.1$',
#'$b_{s^2}=0.15$, $b_{\\nabla^2}=0.2$',
]
#annotation_text = 'LSST Y1 data \n 3x2pt-analysis\n data: GR\n fixed $\Sigma_0=0$, $\Omega_{\\rm b}$, $n_{\\rm s}$'
#annotation_text = 'LSST Y1 data \n 2x2pt-analysis\n fixed cosmology\n data (Pandey): no baryons, b2=0, bs2,blapl$\\neq 0$ \n model (Pandey): baryons, b2=bs2=blapl=0'
#annotation_text = 'LSST Y1 data \n data (Pandey): log10Mc=10, bs2,blapl$\\neq 0$ \n model (Pandey): baryons, bs2=blapl=0'
annotation_text = 'LSST Y1 data \n 3x2pt-analysis on DESI fiducial cosmology\n $\ell^{\\rm WL, GC}_{\\rm max}=2000, 100$'
# annotation square
num = 1

n_samples = len(file_paths)
chains_info = {}
for i in range(n_samples):
    file_path = file_paths[i]  
    chain = np.genfromtxt(file_path)
    chain_pars = read_last_header_line(file_path)
    chain_pars = chain_pars[:-2]
    chains_info[i] = {}
    chains_info[i]['chain'] = chain
    chains_info[i]['pars'] = chain_pars


samples = []    
for i in range(n_samples):
    samples.append(
        getdist.MCSamples(samples = chains_info[i]['chain'][:,:len(chains_info[i]['pars'])],
                                    names = [i for i in chains_info[i]['pars']],
                                    weights=np.exp(chains_info[i]['chain'][:, -2]),
                                    labels = [params_dic[p]
                                                for p in chains_info[i]['pars']],
                                    settings={'smooth_scale_2D':0.35, 'smooth_scale_1D':0.35},
                                    ) 
    )


ModelPars = chains_info[0]['pars'] #+['blaplL_'+str(i+1) for i in range(10)] 
# add parameters that are not present in the first chain:
#ModelPars = ModelPars[:7]
#ModelPars = ModelPars[:3]
#ModelPars = ['Omega_m', 'sigma8_cb', 'b1L_1', 'log10Mc_bc']

#colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan']
colors = [ '#18698F', '#7AC011', '#FF7083', '#99D4E5']
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
g = getdist.plots.getSubplotPlotter(subplot_size=1.5)

plt.rcParams.update({'font.size':16})
g.settings.legend_fontsize=20#52#20#26#36
g.settings.axes_fontsize=18#25
g.settings.axes_labelsize=20#32
g.settings.linewidth=4   
g.settings.figure_legend_frame = False


g.triangle_plot(samples,
    ModelPars,
legend_labels = legend_labels,
title_limit=1,  # first title limit (for 1D plots) is 68% by default
legend_loc = 'upper right',
contour_args = [{'filled':True, 'color': colors[0]}, {'filled':True, 'color': colors[1], 'ls': '-'}, {'filled':True, 'color': colors[2], 'ls': '-'},
                #{'filled':True, 'color': colors[3], 'ls': '-'},  {'filled':True, 'color': colors[4], 'ls': '-'},  {'filled':True, 'color': colors[5], 'ls': '-'}
                 {'filled':False, 'color': colors[0], 'ls': '--'}, {'filled':False, 'color': colors[1], 'ls': '--'}, {'filled':False, 'color': colors[2], 'ls': '--'},
                ], 
line_args=[ {'color': colors[0], 'ls': '-'}, {'color': colors[1], 'ls': '-'}, {'color': colors[2], 'ls': '-'}, 
            #{'color': colors[3], 'ls': '-'}, {'color': colors[4], 'ls': '-'}, {'color': colors[5], 'ls': '-'}
             {'color': colors[0], 'ls': '--'}, {'color': colors[1], 'ls': '--'}, {'color': colors[2], 'ls': '--'}, 
            ])



if fiducials != None:
    for i in range(len(ModelPars)):
        for j in range(i+1):
            ax = g.subplots[i,j]
            if i != j and ModelPars[i] in fiducials and fiducials[ModelPars[i]] != None:
                ax.axhline(fiducials[ModelPars[i]],lw=1.5,color='tab:gray')
            if ModelPars[j] in fiducials and fiducials[ModelPars[j]] != None:
                ax.axvline(fiducials[ModelPars[j]],lw=1.5,color='tab:gray')
                #if i==1 and j==1:
                #    ax.axvline(fiducials[ModelPars[i]],lw=1.5,color='tab:gray', label='$3 \\times 2$pt')
                #    ax.axvline(fiducials[ModelPars[i]],lw=1.5,color='tab:gray', linestyle='--', label='$2 \\times 2$pt')
num = 3
ax = g.subplots[num, num]
ax.annotate(annotation_text, (3.5, 0.05), xycoords='axes fraction', clip_on=False, fontsize=20) 
#g.subplots[1,1].legend(loc='best', fontsize=g.settings.legend_fontsize-4, bbox_to_anchor=(1., 1.), frameon=False)
          

#plt.savefig('figs/posteriors/'+file_name+'.pdf', bbox_inches='tight')  
plt.savefig('figs/posteriors/'+file_name+'.png', bbox_inches='tight')  