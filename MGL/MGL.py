import numpy as np
import yaml
from MGL.theory import *
from MGL.likelihood import *
from MGL.powerspectra import *
from MGL.specs import *

class mgl(Survey, Theory, MGLike, Sampling):
    def load_emulators(self):
        print('loading emulators...')
        self.hmcode_emu = HMcode2020()
        self.bcemu = BCemulator(self.zz_integr[-1])
        self.bacco_emu = BaccoEmu()

    def compute_data(self):
        if self.Probe=='3x2pt':
            self.Cov_observ, self.Cov_observ_high = self.compute_covariance(self.params_data_dic, self.data_model_dic)
            self.d_obs = np.linalg.det(self.Cov_observ) 
            self.d_obs_high = np.linalg.det(self.Cov_observ_high) 
        elif self.Probe=='WL':
            self.Cov_observ = self.compute_covariance(self.params_data_dic, self.data_model_dic)
            self.d_obs = np.linalg.det(self.Cov_observ) 
            self.ells_one_probe = self.ells_WL
        elif self.Probe=='GC':   
            self.Cov_observ = self.compute_covariance(self.params_data_dic, self.data_model_dic)
            self.d_obs = np.linalg.det(self.Cov_observ) 
            self.ells_one_probe = self.ells_GC 
 
        
    def __init__(self, config_file):
        """
        Initialization requires a config yaml-file.
        Here we save the specs, compute mock data etc.
        """
        with open(config_file, "r") as file:
            self.config_dic = yaml.safe_load(file)
        #initialise the Survey#
        super().__init__(self.config_dic['specs'])    


        try: self.PATH = self.config_dic['PATH']
        except: self.PATH = "./"
        self.chain_name = self.config_dic['output']['chain_name']
        try: self.hdf5_name = self.config_dic['sampler']['emcee']['hdf5_name']
        except: self.hdf5_name = "nautilus_test"
   
        self.data_model_dic = self.config_dic['data']
        self.theo_model_dic = self.config_dic['theory']

        

        with open(self.config_dic['data']['params'], "r") as file:
            self.params_data_dic = yaml.safe_load(file)
        with open(self.config_dic['theory']['params'], "r") as file:
            params_dic = yaml.safe_load(file)
        
        self.params_data_dic['Omega_b'] = self.params_data_dic['Ombh2']/self.params_data_dic['h']**2    
        self.params_data_dic['Omega_nu'] = self.params_data_dic['Mnu']/93.14/self.params_data_dic['h']**2
        self.params_data_dic['fb'] = self.params_data_dic['Omega_b']/self.params_data_dic['Omega_m']
        self.MasterPriors = params_dic
        self.ModelParsFix = {par: params_dic[par]['p0'] for par in params_dic if params_dic[par]['type'] == 'F'}
        self.ModelPars = [par for par in params_dic if params_dic[par]['type'] != 'F']
        self.Probe = self.config_dic['observable']


        self.load_emulators()

        print('start computing mock data')
        self.compute_data()
        print('finish computing mock data')



