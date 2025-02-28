import numpy as np
import getdist.plots
import matplotlib.pyplot as plt
import yaml
with open("params_names.yaml", "r") as file:
    params_dic = yaml.safe_load(file)
with open("params_data.yaml", "r") as file:
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

file_paths = [
    'chains/chain_lsst_test_bacco_3x2pt_nobar_heft_w0wa_vary_cosmo_ia_b1L_and_b2L.txt',
    'chains/chain_lsst_test_bacco_3x2pt_nobar_heft_w0wa_vary_cosmo_ia_and_b1L.txt',
    'chains/chain_lsst_test_bacco_3x2pt_nobar_heft_w0wa_vary_cosmo_and_ia.txt', 
    'chains/chain_lsst_test_bacco_3x2pt_nobar_heft_w0wa.txt', ]  

file_name = 'lsst_test_posterior_cosmo_vars'
legend_labels = [
'cosmo+ia+b1+b2: 2:07:11',
'cosmo+ia+b1: 1:38:32',
'cosmo+ia: 0:50:57',
'cosmo: 0:35:15']
annotation_text = 'LSST 3x2pt bacco+heft\n Cuillin 14 cores'



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
                                                for p in chains_info[i]['pars']]) 
    )


ModelPars = chains_info[0]['pars']
colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
g = getdist.plots.getSubplotPlotter(subplot_size=1.3)

plt.rcParams.update({'font.size':16})
g.settings.legend_fontsize=36
g.settings.axes_fontsize=25
g.settings.axes_labelsize=32
g.settings.linewidth=3   
g.settings.figure_legend_frame = False


g.triangle_plot(samples,
    ModelPars,
legend_labels = legend_labels,
legend_loc = 'upper right',
contour_args = [{'filled':True, 'color': colors[0]}, 
{'filled':True, 'color': colors[1], 'ls': '-'}, {'filled':True, 'color': colors[2], 'ls': '-'}, {'filled':True, 'color': colors[3], 'ls': '-'}], 
line_args=[ {'color': colors[0]}, 
{'color': colors[1], 'ls': '-'}, {'color': colors[2], 'ls': '-'},  {'color': colors[3], 'ls': '-'}])



if fiducials != None:
    for i in range(len(ModelPars)):
        for j in range(i+1):
            ax = g.subplots[i,j]
            if i != j and ModelPars[i] in fiducials and fiducials[ModelPars[i]] != None:
                ax.axhline(fiducials[ModelPars[i]],lw=2.,color='tab:gray')
            if ModelPars[j] in fiducials and fiducials[ModelPars[j]] != None:
                ax.axvline(fiducials[ModelPars[j]],lw=2.,color='tab:gray')

ax = g.subplots[2, 2]
ax.annotate(annotation_text, (2.5, 0.05), xycoords='axes fraction', clip_on=False, fontsize=30) 
                

plt.savefig('figs/posteriors/'+file_name+'.png')  

