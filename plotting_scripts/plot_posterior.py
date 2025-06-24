import numpy as np
import getdist.plots
import matplotlib.pyplot as plt
import yaml
import os

folder = os.path.dirname(os.path.abspath(__file__))
# my_colors = ['#FFDD03', "#78206E", '#035177', '#F58300', '#8ECAE6', '#78206E', '#ED91DC']
my_colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan']


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


def plot_corner(list_of_files, legend_labels, annotation_text, file_name, plotbias=True, delete_pars=None, nbins=5, derive=False):
    '''
    list_of_files: list of chain files to plot
    bias_model: list of strings specifying the bias model to use for the plot for each list_of_files element
    legend_labels: list of labels for the legend
    annotation_text: text to annotate the plot
    file_name: name of the file to save the plot
    plotbias: if True, plot all parameters, if False doesn't plot bias parameters
    delete_pars: list of parameters to not include in the plot
    nbins: number of redshift bins
    ---
    Create corner plot for parameters in chain files (and saves it)'''
    with open(folder+"/params_names.yaml", "r") as file:
        params_dic = yaml.safe_load(file)
    num = 1
    with open(folder+"/../params_data.yaml", "r") as file:
        fiducials = yaml.safe_load(file)

    n_samples = len(list_of_files)
    chains_info = {}
    bias_params_name = ['b1', 'b2', 'b1L', 'b2L', 'bs2L', 'blaplL']
    heft_params_name = ['b1L', 'b2L', 'bs2L', 'blaplL']
    for i in range(n_samples):
        file_path = folder + '/../' + list_of_files[i]  
        chain = np.genfromtxt(file_path)
        chain_pars = read_last_header_line(file_path)
        chain_pars = chain_pars[:-2]
        chains_info[i] = {}
        chains_info[i]['chain'] = chain
        chains_info[i]['pars'] = chain_pars#[:7]# + chain_pars[-1:]
        
        # plot derived parameter b*sigma_8 (b = all heft param in each bin)
        if derive:
            # create derived parameters
            for j in range(1, nbins+1):
                for b_name in heft_params_name:
                    param_name = b_name + '_' + str(j)
                    index = chain_pars.index(param_name)
                    sigma8_index = chain_pars.index('sigma8_cb')
                    derived_name = b_name + '_sigma8_' + str(j)
                    derived_value = chain[:, index] * chain[:, sigma8_index]
                    # add to chain as 3rd to last column
                    chain = np.insert(chain, -2, derived_value, axis=1)
                    # add name to chain_pars as 3rd to last
                    chain_pars.insert(-2, derived_name)
                chains_info[i]['pars'] = chain_pars

        if not plotbias:
            for b_name in bias_params_name:
                for j in range(1, nbins+1):
                    param_name = b_name + '_' + str(j)
                    if param_name in chain_pars:
                        index = chain_pars.index(param_name)
                        chain_pars.remove(param_name)
                        chain = np.delete(chain, index, axis=1)
            chains_info[i]['pars'] = chain_pars
            chains_info[i]['chain'] = chain

        if delete_pars != None:
            for p_name in delete_pars:
                if p_name in chain_pars:
                    index = chain_pars.index(p_name)
                    chain_pars.remove(p_name)
                    chain = np.delete(chain, index, axis=1)
            chains_info[i]['pars'] = chain_pars
            chains_info[i]['chain'] = chain
                
            
    samples = []    
    for i in range(n_samples):
        samples.append(
            getdist.MCSamples(samples=chains_info[i]['chain'][:,:len(chains_info[i]['pars'])],
                              names=[p for p in chains_info[i]['pars']],
                              weights=np.exp(chains_info[i]['chain'][:, -2]),
                              labels=[params_dic[p] for p in chains_info[i]['pars']],
                              settings={'smooth_scale_2D':0.35, 'smooth_scale_1D':0.35},
                              )
        )

    ModelPars = chains_info[0]['pars']

    from matplotlib import rc
    rc('text', usetex=False)
    rc('font',**{'family':'serif','serif':['Times']})
    g = getdist.plots.getSubplotPlotter(subplot_size=1.3)

    plt.rcParams.update({'font.size':16})
    g.settings.legend_fontsize = 20
    g.settings.axes_fontsize = 20
    g.settings.axes_labelsize = 20
    g.settings.linewidth = 5
    g.settings.linewidth_contour = 5
    g.settings.figure_legend_frame = False
    fill = [True, False, False]
    g.triangle_plot(samples,
        ModelPars,
        legend_labels=legend_labels,
        legend_loc='upper right',
        contour_args=[{'filled':True, 'color': my_colors[i]} for i in range(n_samples)],
        line_args=[{'color': my_colors[i], 'ls': '-'} for i in range(n_samples)]
    )

    if fiducials is not None:
        for i in range(len(ModelPars)):
            for j in range(i+1):
                ax = g.subplots[i, j]
                if i != j and ModelPars[i] in fiducials and fiducials[ModelPars[i]] is not None:
                    ax.axhline(fiducials[ModelPars[i]], lw=2., color='tab:gray')
                if ModelPars[j] in fiducials and fiducials[ModelPars[j]] is not None:
                    ax.axvline(fiducials[ModelPars[j]], lw=2., color='tab:gray')

    ax = g.subplots[num, num]
    # ax.annotate(annotation_text, (2.5, 0.5), xycoords='axes fraction', clip_on=False, fontsize=12)     
    # ax.annotate(annotation_text, (15.8, -2.5), xycoords='axes fraction', clip_on=False, fontsize=40) 
    plt.tight_layout()
    if not plotbias:
        file_name += '_noBias'
    plt.savefig(folder+'/../figs/posteriors/'+file_name+'.png')
    # plt.show()



file_paths = [
    # ----- test for bias and baryons ---- #
    # 'chains/chain_data-b1_model-heft-noBaryons_kmax0.1_30Apr.txt',
    # 'chains/chain_data-b1_model-heft-noBaryons_kmax0.7_30Apr.txt',
    # 'chains/chain_TEST1_data-b1_model-heft-WLlogMc_kmax0.1_30Apr.txt',
    # 'chains/chain_TEST1_data-b1_model-heft-WLlogMc_kmax0.7_30Apr.txt',
    # 'chains/chain_b1_data_b1model_kmax0.7_withBaryons_28Apr.txt',
    # 'chains/chain_test_masc_3x2_kmax0.7_27May.txt',
    # 'chains/chain_modHEFT_data-GCb1boost_model-GCnoboost_kmax0.7_02Jun.txt',
    #
    # 'chains/chain_WL_HEFTdata_HEFTmodel_kmax0.7_18Apr.txt',
    # 'chains/chain_GC_HEFTdata_HEFTmodel_kmax0.7_28Apr.txt',
    # 'chains/chain_GC_heft_heft_kmax0.7_logMc_b1LsymmetricPrior_01May.txt',
    # 'chains/chain_test_GC_kmax0.7_blin_sigma8_21May.txt',
    # 'chains/chain_test_GC_kmax0.7_heft_sigma8_21May.txt',
    # 'chains/chain_GC_heft_heft_kmax0.7_logMc_fixed_bs2_blap_bsigma8_23May.txt',
    # 'chains/chain_GC_heft_heft_kmax0.7_logMc_fixed_bs2_blap_23May.txt',
    # 'chains/chain_GC_heft_heft_kmax0.7_logMc_bsigma_fix_bs2_blap_02Jun.txt',
    #
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.1_noSystematics-fcfs_03Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.2_logMc-10Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_logMc-w87_09Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_largerlogMc_-11Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_noSystematics-fcfs_07Apr.txt',
    # 'chains/chain_heft_heft_kmax0.5_SRD_20Jun.txt', 
    'chains/chain_heft_heft_kmax0.7_SRD_20Jun.txt', 
    # 'chains/chain_HEFTdata_bextramodel_kmax0.2_noSystematics_15Apr.txt',
    # 'chains/chain_HEFTdata_b1model_kmax0.1_noSystematics-fcfs_03Apr.txt',
    # 'chains/chain_HEFTdata_b1model_kmax0.2_noSystematics_21Apr.txt',
    # 'chains/chain_HEFTdata_b1model_kmax0.7_noSystematics_18Apr.txt',
    # 'chains/chain_HEFTdata_b1model_kmax0.7_noSystematics-fcfs_03Apr.txt'
    # 'chains/chain_heft_b1_kmax0.5_SRD_23Jun.txt',
    # 'chains/chain_heft_b1_kmax0.7_SRD_20Jun.txt',
    # ----- fixing 2 bias pars or 1 baryonic par not fixed (fixing pars at fiducial values) ----- #
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b2_bLap_21Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b2_bs2_21Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_bs2_bLap_21Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b1_b2_21Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b1_bLap_21Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b1_bs2_21Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_bs2_bLap_22Apr_nonZero_bLap_bs2.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b2_bs2_22Apr_nonZero_bLap_bs2.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b2_bLap_22Apr_nonZero_bLap_bs2.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b1_bLap_22Apr_nonZero_bLap_bs2.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b1_bs2_22Apr_nonZero_bLap_bs2.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixed_b1_b2_22Apr_nonZero_bLap_bs2.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_full_bias_baryons_23Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixedbias_logMc_23Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixedbias_etabc_23Apr.txt',
    # 'chains/chain_HEFTdata_HEFTmodel_kmax0.7_fixedbias_betabc_23Apr.txt',
    # ----- fixing 2 bias pars or 1 baryonic par not fixed (fixing pars NOT at fiducial values) ----- #
    # 'chains/chain_heft_heft_kmax0.1_fixed_bs2_bLap_toZero_15May.txt',
    # 'chains/chain_heft_heft_kmax0.3_fixed_bs2_bLap_toZero_13May.txt',
    # 'chains/chain_heft_heft_kmax0.4_fixed_bs2_bLap_toZero_15May.txt',
    # 'chains/chain_HEFT_HEFT_kmax0.4_fix_blap_toZero_29May.txt',
    # 'chains/chain_HEFT_HEFT_kmax0.4_fix_bs2_toZero_30May.txt',
    # 'chains/chain_heft_heft_kmax0.4_fixed_bs2_blap_toZero_nlive_23May.txt',
    # 'chains/chain_heft_heft_kmax0.4_fixed_bs2_blap_toZero_22May.txt',
    # 'chains/chain_heft_heft_kmax0.5_fixed_bs2_bLap_toZero_15May.txt',
    # 'chains/chain_heft_heft_kmax0.7_fixed_bs2_bLap_toZero_13May.txt',
    # 'chains/chain_heft_heft_kmax0.7_fixed_b2_bs2_toZero_08May.txt',
    # 'chains/chain_heft_heft_kmax0.7_fixed_b2_bLap_toZero_08May.txt',
    # 'chains/chain_heft_heft_kmax0.7_fixed_b1_bLap_toZero_08May.txt',
    # 'chains/chain_heft_heft_kmax0.7_fixed_b1_bs2_toZero_08May.txt',
    # 'chains/chain_heft_heft_kmax0.7_fixed_b1_b2_toZero_08May.txt',
    # 'chains/chain_heft_heft_kmax0.7_eta_fixed_logMc_to11.8_09May.txt',
    # 'chains/chain_heft_heft_kmax0.7_logMC_fixed_betabc_toZero_09May.txt',
    # 'chains/chain_heft_heft_kmax0.7_logMC_fixed_etabc_toZero_09May.txt'
    # 'chains/chain_HEFT_HEFT_kmax0.4_fix_bs2_blap_to_fid_zero_29May.txt',
    # 'chains/chain_heft_heft_kmax0.4_fixing_bs2_bLap_fix_Ob_ns_toZero_06Jun.txt',
    # 'chains/chain_heft_heft_kmax0.4_fixing_bs2_bLap_quasiZero_06Jun.txt',
    # ----- fixing 2 bias pars or 1 baryonic par not fixed (fixing pars NOT at fiducial values) SRD fiducials ----- #
    # 'chains/chain_heft_heft_kmax0.4_SRD_fixing_bs2_bLap_toZero_19Jun.txt',
    # 'chains/chain_heft_heft_kmax0.4_SRD_high_bLap_fixing_bs2_bLap_toZero_19Jun.txt',
    # 'chains/chain_heft_heft_kmax0.4_SRD_high_bs2_fixing_bs2_bLap_toZero_20Jun.txt',
    # 'chains/chain_heft_heft_kmax0.5_SRD_fixing_bs2_bLap_toZero_19Jun.txt',
    'chains/chain_heft_heft_kmax0.7_SRD_fixing_bs2_bLap_toZero_19Jun.txt',
    ]  

file_name = 'fixing_bs2_bLap_toZero_SRD_kmax0.5'
legend_labels = [
    # 'data: b1, model: heft, no baryons, kmax = 0.1',
    # 'data: b1, model: heft, no baryons, kmax = 0.7',
    # 'data: b1+baryons, model: heft+WLbaryons, kmax = 0.1 \n vary only logMc (13.8)',  # test 1
    # 'data: b1+baryons, model: heft+WLbaryons, kmax = 0.7 \n vary only logMc (13.8)',  # test 1
    # 'GC, data: b1+baryons, model: heft, kmax = 0.3',
    # 'data: mod Pnl+heft+baryons, model: mod Pnl+heft+WLlogMc, \n kmax=0.1, bs2=0, bLap=0',  # test 2a
    # 'data: mod Pnl+heft+baryons, model: mod Pnl+heft+WLlogMc, \n kmax=0.7, bs2=0, bLap=0',  # test 2a
    # 'data: mod Pnl+heft+baryons, model: mod Pnl+heft+WLlogMc, \n kmax=0.1, bs2, bLap!=0',  # test 2b
    # 'data: mod Pnl+heft+baryons, model: mod Pnl+heft+WLlogMc, \n kmax=0.7, bs2, bLap!=0',  # test 2b
    # 'data: mod Pnl+heft+baryons, model: mod Pnl+heft+WLlogMc, \n no baryon boost in GC pmm \n kmax=0.7, bs2, bLap!=0',  # test 3
    # 'data: b1+baryons, model: b1+baryons'
    # 'test 3 with mask',
    #
    # 'WL, HEFT-HEFT, kmax=0.7, logMc=14.8',
    # 'GC, HEFT-HEFT, kmax=0.7, logMc=14.8',
    # 'GC, HEFT-HEFT, kmax=0.7, logMc=14.8 \n b1L symmetric prior',
    # 'GC sampled blin*sigma8, kmax=0.7 (wrong b fiducials)',
    # 'GC kmax=0.7, fixed bs2 blap to fid \n sampled heft*sigma8',
    # 'GC kmax=0.7, fixed bs2 blap to fid',
    #
    # 'HEFTdata, HEFTmodel, kmax=0.1',
    # 'HEFTdata, HEFTmodel, kmax=0.2, logMc',
    # 'data: heft+baryons, model: heft+baryons, kmax = 0.7, \n vary only logMc (13.8)',
    # 'HEFTdata, HEFTmodel, kmax=0.7, logMc=14.8',
    # 'HEFTdata, HEFTmodel, kmax=0.7',
    # 'HEFTdata, HEFTmodel, kmax=0.5, SRD fiducials',
    'HEFTdata, HEFTmodel, SRD fiducials, \n kmax=0.7',
    # 'HEFTdata, b1+betra*k+b2*k^2, kmax=0.2'
    # 'HEFTdata, b1model, kmax=0.1',
    # 'HEFTdata, b1model, kmax=0.2',
    # 'HEFTdata, b1model, kmax=0.7',
    # 'HEFTdata, b1model, kmax=0.5, SRD fiducials',
    # 'HEFTdata, b1model, kmax=0.7, SRD fiducials',
    #
    # 'fixed b2, bLap',
    # 'fixed b2, bs2',
    # 'fixed bs2, bLap',
    # 'fixed b1, b2',
    # 'fixed b1, bLap',
    # 'fixed b1, bs2',
    # 'fixed bs2, bLap (to fiducials, !=0)',
    # 'fixed b2, bs2 (to fiducials, !=0)',
    # 'fixed b2, bLap (to fiducials, !=0)',
    # 'fixed b1, bLap (to fiducials, !=0)',
    # 'fixed b1, bs2 (to fiducials, !=0)',
    # 'fixed b1, b2 (to fiducials, !=0)',
    # 'HEFT-HEFT, kmax=0.7, vary bias + baryons',
    # 'HEFT-HEFT, kmax=0.7, fixed bias (vary logMc)',
    # 'HEFT-HEFT, kmax=0.7, fixed bias (vary eta)',
    # 'HEFT-HEFT, kmax=0.7, fixed bias (vary beta)',
    #
    # 'fixed bs2, bLap to 0 (not fiducial) \n kmax = 0.1',
    # 'fixed bs2, bLap to 0 (not fiducial) \n kmax = 0.3',
    # 'fixed bs2, bLap to 0 (not fiducial) \n kmax = 0.4',
    # 'fixed bLap to 0 (not fiducial) \n kmax = 0.4',
    # 'fixed bs2 to 0 (not fiducial) \n kmax = 0.4',
    # 'fixed bs2, bLap to 0 (not fiducial) \n kmax = 0.4, more nlive',
    # 'fixed bs2, bLap to 0 (not fiducial) \n kmax = 0.4, wrong b fid',
    # 'fixed bs2, bLap to 0 (not fiducial) \n kmax = 0.5',
    # 'fixed bs2, bLap to 0 (not fiducial) \n kmax = 0.7',
    # 'fixed b2, bs2 to 0 (not fiducial)',
    # 'fixed b2, bLap to 0 (not fiducial)',
    # 'fixed b1, bLap to 0 (not fiducial)',
    # 'fixed b1, bs2 to 0 (not fiducial)',
    # 'fixed b1, b2 to 0 (not fiducial)',
    # 'fixed logMc = 11.8 (vary etabc)',
    # 'fixed etabc = 0',
    # 'fixed betabc = 0',
    # 'fixed bs2, bLap to 0 (fiducial) \n kmax = 0.4',
    # 'fixed bLap to quasi 0 (not fiducial) \n kmax = 0.4',
    #
    # 'fixed bLap to 0, SRD fiducials \n kmax = 0.4',
    # 'fixed bLap to 0, SRD fiducials \n high bLap, kmax = 0.4',
    # 'fixed bLap to 0, SRD fiducials \n high bs2, kmax = 0.4',
    # 'fixed bLap to 0, SRD fiducials \n kmax = 0.5',
    'fixed bLap to 0, SRD fiducials \n kmax = 0.7',
    ]
annotation_text = '' #'LSST Y1 kmax=0.7 h/Mpc \n HEFT data, HEFT model'

b1 = ['b1L_1', 'b1L_2', 'b1L_3', 'b1L_4', 'b1L_5']
b2 = ['b2L_1', 'b2L_2', 'b2L_3', 'b2L_4', 'b2L_5']
bs2 = ['bs2L_1', 'bs2L_2', 'bs2L_3', 'bs2L_4', 'bs2L_5']
bLap = ['blaplL_1', 'blaplL_2', 'blaplL_3', 'blaplL_4', 'blaplL_5']
plot_corner(file_paths, legend_labels=legend_labels, annotation_text=annotation_text, file_name=file_name, plotbias=False, derive=False)

