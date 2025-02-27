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

file_path = 'chains/chaintest_hmcode_3x2pt_nobar.txt'  
chain1 = np.genfromtxt(file_path)
chain1_pars = read_last_header_line(file_path)
chain1_pars = chain1_pars[:-2]

file_path = 'chains/chaintest_hmcode_3x2pt_b1b2_nobar.txt'  
chain2 = np.genfromtxt(file_path)
chain2_pars = read_last_header_line(file_path)
chain2_pars = chain2_pars[:-2]

file_path = 'chains/chain_lsst_test_bacco_3x2pt_nobar_heft_fix_sigma8cold.txt'  
chain = np.genfromtxt(file_path)
chain_nuis_pars = read_last_header_line(file_path)
chain_nuis_pars = chain_nuis_pars[:-2]

colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
g = getdist.plots.getSubplotPlotter(subplot_size=1.3)

plt.rcParams.update({'font.size':16})

sample1 =  getdist.MCSamples(samples = chain1[:,:len(chain1_pars)],
                                    names = [i for i in chain1_pars],
                                    weights=np.exp(chain1[:, -2]),
                                    labels = [params_dic[p]
                                                for p in chain1_pars]) 
sample2 = getdist.MCSamples(samples = chain2[:,:len(chain2_pars)],
                                    names = [i for i in chain2_pars],
                                    weights=np.exp(chain2[:, -2]),
                                    labels = [params_dic[p]
                                                for p in chain2_pars])
sample = getdist.MCSamples(samples = chain[:,:len(chain_nuis_pars)],
                                    names = [i for i in chain_nuis_pars],
                                    weights=np.exp(chain[:, -2]),
                                    labels = [params_dic[p]
                                                for p in chain_nuis_pars])
g.settings.legend_fontsize=36
g.settings.axes_fontsize=25
g.settings.axes_labelsize=32
g.settings.linewidth=3   
g.settings.figure_legend_frame = False

ModelPars = chain_nuis_pars
g.triangle_plot(sample, #[sample2, sample1], 
    ModelPars,
legend_labels = [
#'3x2pt hmcode b1+b2',
#'3x2pt hmcode b1'
'LSST 3x2pt bacco heft fix cosmo'
],
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
            

#plt.show()
plt.savefig('figs/lsst_test_posterior_nuisances.png')  

