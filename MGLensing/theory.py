import numpy as np
import MGrowth as mg
from scipy.integrate import trapezoid,simpson, quad
from scipy import interpolate as itp
from .powerspectra import HMcode2020, BCemulator, BaccoEmu
from numpy import nan, isnan, allclose
from math import sqrt, log, exp, pow, log10
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

try: import baccoemu
except: print("Bacco not installed!")

H0_h_c = 1./2997.92458 #=100/c in Mpc/h

EmulatorRanges = {
    'HMcode':
        {
        'Omega_c':      {'p1': 0.1,         'p2': 0.8},    
        'Omega_b':      {'p1':0.01,         'p2': 0.1},  
        'h':            {'p1': 0.4,         'p2': 1.},      
        'As':           {'p1': 0.495e-9,    'p2': 5.459e-9},
        'ns':           {'p1': 0.6,         'p2': 1.2}, 
        'Mnu':          {'p1': 0.0,         'p2': 0.5},
        'w0':           {'p1': -3.,         'p2': -0.3},    
        'wa':           {'p1': -3.,         'p2': 3.},  
        },
    'bacco':
        {
        'Omega_cb':     {'p1': 0.23,         'p2': 0.4},    
        'Omega_b':      {'p1':0.04,          'p2': 0.06},  
        'h':            {'p1': 0.6,          'p2': 0.8},      
        'sigma8_cb':    {'p1': 0.73,         'p2': 0.9},
        'ns':           {'p1': 0.92,         'p2': 1.01}, 
        'Mnu':          {'p1': 0.0,          'p2': 0.4},
        'w0':           {'p1': -1.15,        'p2': -0.85},    
        'wa':           {'p1': -0.3,         'p2': 0.3}, 
        },    
    'bacco_lin':
        {
        'Omega_cb':     {'p1': 0.06,         'p2': 0.7},    
        'Omega_b':      {'p1':0.03,          'p2': 0.07},  
        'h':            {'p1': 0.5,          'p2': 0.9},      
        #'sigma8_cb':    {'p1': 0.4,         'p2': 1.2}, #not sure about this value
        'ns':           {'p1': 0.6,         'p2': 1.2}, 
        'Mnu':          {'p1': 0.0,          'p2': 1.},
        'w0':           {'p1': -2.,        'p2': -0.5},    
        'wa':           {'p1': -0.5,         'p2': 0.5}, 
        },        
    'baryons':
        {
        'log10Mc_bcemu':    {'p1': 11.4,         'p2': 14.6}, 
        'thej_bcemu':       {'p1': 2.6,          'p2': 7.4},
        'mu_bcemu':         {'p1': 0.2,          'p2': 1.8},
        'gamma_bcemu':      {'p1': 1.3,          'p2': 3.7},
        'delta_bcemu':      {'p1': 3.8,          'p2': 10.2},
        'eta_bcemu':        {'p1': 0.085,        'p2': 0.365},
        'deta_bcemu':       {'p1': 0.085,        'p2': 0.365},
        'log10Tagn':    {'p1': 7.6,         'p2': 8.3},
        'log10Mc_bc':   {'p1': 9.,           'p2': 15.}, 
        'eta_bc':       {'p1': -0.69,        'p2': 0.69},
        'beta_bc':      {'p1': -1.,          'p2': 0.69},
        'log10Mz0_bc':  {'p1': 9.,           'p2': 13.},
        'thetaout_bc':  {'p1': 0.,           'p2': 0.47},
        'thetainn_bc':  {'p1': -2.,          'p2': -0.52},
        'log10Minn_bc': {'p1': 9.,           'p2': 13.5}
        }

}

COSMOLOGY_CONSISTENCY_RELATIONS = {
    "Omega_m": ["Ommh2/h/h", "Omega_b+Omega_c+Omega_nu"],
    "Omega_b": ["Ombh2/h/h", "Omega_m-Omega_c-Omega_nu"],
    "Omega_c": ["Omch2/h/h", "Omega_m-Omega_b-Omega_nu"],
    "Omega_nu": ["Omnuh2/h/h", "Mnu * ((nnu / 3.0) ** 0.75 / 94.06410581217612 * (TCMB/2.7255)**3)/h/h",  "Omega_m-Omega_b-Omega_c"],
    "Omega_cb": ["Omega_m-Omega_nu", "(Omch2+Ombh2)/h/h", "Omega_c+Omega_b"],
    "Ommh2": ["Omega_m*h*h"],
    "Ombh2": ["Omega_b*h*h"],
    "Omch2": ["Omega_c*h*h"],
    "Omnuh2": ["Omega_nu*h*h", "Mnu * ((nnu / 3.0) ** 0.75 / 94.06410581217612 * (TCMB/2.7255)**3)"],
    "Mnu": ["Omnuh2 / ((nnu / 3.0) ** 0.75 / 94.06410581217612 * (TCMB/2.7255)**3)", "Omega_nu*h*h / ((nnu / 3.0) ** 0.75 / 94.06410581217612 * (TCMB/2.7255)**3)"]}


cosmo_names = ["Omega_m", "Omega_b", "Omega_c", "Omega_cb", "Omega_nu", "Ombh2", "Omnuh2", "fb", "h", "Mnu", "ns", "w0", "wa"]

class Theory:
    def __init__(self, SurveyClass):
        self.Survey = SurveyClass
        ####
        self.HMcode2020emulator = HMcode2020()
        self.BCemulator = BCemulator(self.Survey.zz_integr[-1])
        self.baccoemulator = BaccoEmu()


    def check_consistency(self, params): 
        if 'h' not in params.keys():
            raise KeyError("h not found in dictionary")
        conditions = [
        'Omega_m',
        ('Omega_c', 'Omch2'),
        ('Omega_b', 'Ombh2'),
        ('Omega_nu', 'Mnu', 'Omnuh2')]
        all_present = all(any(key in params for key in condition) if isinstance(condition, tuple) else condition in params for condition in conditions)
        if all_present:
            raise ValueError("Invalid parameter set: too many densities are present in the dictionary.")
        conditions_amp = [
        'log10As',
        'As',
        'sigma8',
        'S8']
        amplitude_params = [param for param in conditions_amp if param in params]
        if len(amplitude_params) > 1:
            raise ValueError(f"Invalid parameter set: multiple amplitude parameters present: {amplitude_params}")
        elif len(amplitude_params) == 0:  
            raise ValueError(f"Invalid parameter set: no amplitude parameters present")
        params_new = self.apply_relations(params)
        still_unspecified = [param for param in cosmo_names if param not in params_new]
        if len(still_unspecified) > 1:
            raise ValueError(f"Invalid parameter set: missing parameters: {still_unspecified}")
        return params_new
        
    
    def apply_relations(self, params): 
        params['TCMB'] = params.get('TCMB', 2.7255)
        params['nnu'] = params.get( 'nnu', 3.044)
        if 'Omega_b' in params.keys():
            params['Ombh2'] = params['Omega_b']*params['h']*params['h']
        elif 'Ombh2' in params.keys():
            params['Omega_b'] = params['Ombh2']/params['h']/params['h']
        if 'Omch2' in params.keys():
            params['Omega_c'] = params['Omch2']/params['h']/params['h']
        elif 'Omega_c' in params.keys():
            params['Omch2'] = params['Omega_c']*params['h']*params['h']
        if 'Omega_nu' in params.keys():
            params['Omnuh2'] = params['Omega_nu']*params['h']*params['h']   
            params['Mnu'] = params['Omnuh2'] / ((params['nnu'] / 3.0) ** 0.75 / 94.06410581217612 * (params['TCMB']/2.7255)**3)    
        elif 'Omnuh2' in params.keys():
            params['Omega_nu'] = params['Omnuh2'] / params['h']/params['h'] 
            params['Mnu'] = params['Omnuh2'] / ((params['nnu'] / 3.0) ** 0.75 / 94.06410581217612 * (params['TCMB']/2.7255)**3)
        elif 'Mnu' in params.keys():
            params['Omnuh2'] = params['Mnu'] * ((params['nnu'] / 3.0) ** 0.75 / 94.06410581217612 * (params['TCMB']/2.7255)**3)
            params['Omega_nu'] = params['Omnuh2'] / params['h']/params['h']
        if 'Omega_m' not in params.keys():
            params['Omega_m'] = params['Omega_b'] + params['Omega_c'] + params['Omega_nu']    
        params['Omega_cb'] = params['Omega_m'] - params['Omega_nu']
        params['Omega_c'] = params['Omega_cb'] - params['Omega_b']
        params['fb'] = params['Omega_b']/params['Omega_m']
        if 'As' in params.keys():
            params['log10As'] = np.log(1e10*params['As'])
        elif 'log10As' in params.keys():
            params['As'] = np.exp(params['log10As'])*1e-10
        if 'S8' in params.keys():
            params['sigma8'] = params['S8']/sqrt(params['Omega_m']/0.3)   
        return params

    def _get_emu_status(self, coordinates, which_emu, flag_once=False):
        """
        Function that checking the relevant boundaries.
        Parameters:
        coordinates (dict): a set of coordinates in parameter space
        which_emu (str) : kind of emulator: options are 'HMcode', 'bacco',
                                            'baryons'
        flag_once (boolean): set to false within a chain, set to true for a single data computation
        Returns:
        boolean: status of the ranges check
        Raises:
        KeyError: If the coordinates do not contain the necessary parameters
        """
        # parameters currently available
        avail_pars = coordinates.keys()
        # parameters strictly needed to evaluate the emulator
        eva_pars = EmulatorRanges[which_emu].keys()
        if flag_once:
            # parameters needed for a computation
            comp_pars = list(set(eva_pars)-set(avail_pars))
            miss_pars = list(set(comp_pars))

            if miss_pars:
                print(f"{which_emu} emulator:")
                print(f"  Please add the parameter(s) {miss_pars}"
                    f" to your coordinates!")
                raise KeyError(f"{which_emu} emulator: coordinates need the"
                            f" following parameters: ", miss_pars)
        pp = [coordinates[p] for p in eva_pars]
        status = True
        for i, par in enumerate(eva_pars):
            val = pp[i]
            if flag_once:
                message = 'Param {}={} out of bounds [{}, {}]'.format(
                par, val, EmulatorRanges[which_emu][par]['p1'],
                EmulatorRanges[which_emu][par]['p2'])
                assert (np.all(val >= EmulatorRanges[which_emu][par]['p1'])
                    & np.all(val <= EmulatorRanges[which_emu][par]['p2'])
                    ), message
            else:    
                status = (np.all(val >= EmulatorRanges[which_emu][par]['p1'])
                    & np.all(val <= EmulatorRanges[which_emu][par]['p2']))
        return status
        
    def get_As(self, params, flag_once=False):
        if 'As' in params:
            return True, params['As']
        else:
            params['sigma8_cb'] = params['sigma8'] #from_sigma8_to_sigma8_cb
            status = self._get_emu_status(params, 'bacco_lin', flag_once)
            if not status:
                return False, nan
            else:
                params_bacco = {
                'ns'            :  params['ns'],
                'sigma8_cold'   :  params['sigma8_cb'], # for future make distinction between sigma8_cold and sigma8
                'hubble'        :  params['h'],
                'omega_baryon'  :  params['Omega_b'],
                'omega_cold'    :  params['Omega_cb'], 
                'neutrino_mass' :  params['Mnu'],
                'w0'            :  params['w0'],
                'wa'            :  params['wa'],
                'expfactor'     :  1    
                }
                As = self.baccoemulator.baccoemulator.get_A_s(**params_bacco)
                return True, As
    
    def get_sigma8_cb(self, params, flag_once=False):
        if 'sigma8' in params:
            params['sigma8_cb'] = params['sigma8']     #from_sigma8_to_sigma8_cb
            return True, params['sigma8_cb'] # for future make distinction between sigma8_cold and sigma8
        else: 
            status = self._get_emu_status(params, 'bacco_lin', flag_once)
            if not status:
                return False, nan
            else:
                params_bacco = {
                'ns'            :  params['ns'],
                'A_s'           :  params['As'], 
                'hubble'        :  params['h'],
                'omega_baryon'  :  params['Omega_b'],
                'omega_cold'    :  params['Omega_cb'], 
                'neutrino_mass' :  params['Mnu'],
                'w0'            :  params['w0'],
                'wa'            :  params['wa'],
                'expfactor'     :  1    
                }
                sigma8_cb = self.baccoemulator.baccoemulator.get_sigma8(cold=True, **params_bacco)
                return True, sigma8_cb
    
    def get_sigma8_from_As(self, params):
        #which_emu='HMcode'
        raise NotImplementedError("get_sigma8_from_As is not implemented yet")
    
    def check_ranges(self, params, model, flag_once=False):
        """
        Check the parameter ranges for the model.
        Parameters:
        params (dict): Dictionary containing the parameters to be checked.                  
        model (dict): Dictionary containing the model-specific parameters.
        flag_once (boolean): set to false within a chain, set to true for a single data computation
        Returns:
        boolean: status of the parameter range check.
        """
        status = True
        if  model['NL_model']==0:
            status, params['As'] = self.get_As(params, flag_once)
            if not status:
                return status
            status = self._get_emu_status(params, 'HMcode', flag_once)
        elif  model['NL_model']==1:       
            status, params['sigma8_cb'] = self.get_sigma8_cb(params)
            if not status:
                return status
            status = self._get_emu_status(params, 'bacco', flag_once)
        if  model['baryon_model']!=0:
            status = self._get_emu_status(params, 'baryons', flag_once)
        if params['w0'] + params['wa'] >= 0:
            status = False
        return  status 

    def check_ranges_simple(self, params, model):
        status = True
        if  model['NL_model']==0:
            if not all(EmulatorRanges['HMcode'][par_i]['p1'] <= params[par_i] <= EmulatorRanges['HMcode'][par_i]['p2'] for par_i in EmulatorRanges['HMcode']):
                #raise ValueError("Not all parameters are within HMcode2020Emu ranges!")
                status = False
        elif  model['NL_model']==1:   
            params['sigma8_cb'] = params['sigma8']    
            if not all(EmulatorRanges['bacco'][par_i]['p1'] <= params[par_i] <= EmulatorRanges['bacco'][par_i]['p2'] for par_i in EmulatorRanges['bacco']):
                #raise ValueError("Not all parameters are within baccoemu ranges!")
                status = False
        if  model['baryon_model']!=0:
            if any( params[par_i] < EmulatorRanges['baryons'][par_i]['p1'] or 
                    params[par_i] > EmulatorRanges['baryons'][par_i]['p2']
                    for par_i in params if par_i in EmulatorRanges['baryons']):
                #raise ValueError("Not all baryonic parameters are within allowed ranges!")
                status = False
        if params['w0'] + params['wa'] >= 0:
            status = False        
        return  status     

    def check_pars_ini(self, param_dic, model_dic, flag_once=True):
        """
        Initial check and validate the parameters for the model and data.
        Parameters:
        param_dic (dict): Dictionary containing the parameters to be checked.
        model_dic (dict): Dictionary containing the model-specific parameters.
        Returns:
        tuple: A tuple containing the updated parameter dictionary and the status
               of the parameter range check.
        Raises:
        KeyError: If the required parameter 'h' is not found in the parameter dictionary.
        KeyError: If some cosmological parameters are missing from the parameter dictionary.
        """
        param_dic_all = self.check_consistency(param_dic)
        status = self.check_ranges(param_dic_all, model_dic, flag_once)
        return param_dic_all, status
    
    def check_pars(self, param_dic, model_dic):
        """
        Check and validate the parameters for the model within a chain.
        Parameters:
        param_dic (dict): Dictionary containing the parameters to be checked.
        model_dic (dict): Dictionary containing the model-specific parameters.
        Returns:
        tuple: A tuple containing the updated parameter dictionary and the status
               of the parameter range check.
        """
        param_dic_all = self.apply_relations(param_dic)
        #status = self.check_ranges(param_dic_all, model_dic)
        status = self.check_ranges_simple(param_dic_all, model_dic)
        return param_dic_all, status

    def get_Ez_rz_k(self, params_dic, zz):
        Omega_m = params_dic['Omega_m']
        w0 = params_dic['w0']
        wa = params_dic['wa']
        omegaL_func = lambda z: (1.-Omega_m) * pow(1.+z, 3.*(1.+w0+wa)) * np.exp(-3.*wa*z/(1.+z))
        E_z_func = lambda z: np.sqrt(Omega_m*pow(1.+z, 3) + omegaL_func(z))
        E_z_grid = np.array([E_z_func(zz_i) for zz_i in zz])
        r_z_int = lambda z: 1./np.sqrt(Omega_m*pow(1.+z, 3) + omegaL_func(z))
        r_z_func = lambda z_in: quad(r_z_int, 0, z_in)[0]
        r_z_grid = np.array([r_z_func(zz_i) for zz_i in zz])/H0_h_c #Mpc/h
        k_grid =(self.Survey.ell[:,None]+0.5)/r_z_grid
        return E_z_grid, r_z_grid, k_grid
    
    def get_growth(self, params_dic, zz_integr,  MGmodel=0):
        aa_integr =  np.array(1./(1.+zz_integr[::-1]))
        background ={
            'Omega_m': params_dic['Omega_m'],
            'h' : params_dic['h'],
            'w0': params_dic['w0'],
            'wa': params_dic['wa'],
            'a_arr': np.hstack((aa_integr, 1.))
            }
        if MGmodel==0 or MGmodel==1:
            cosmo = mg.w0waCDM(background)   
            Da, _ = cosmo.growth_parameters() 
        elif MGmodel==2:
            cosmo = mg.nDGP(background)
            log10omegarc = params_dic['log10omegarc'] #if 'log10omegarc' in params_dic else self.ModelParsFix['log10omegarc']  
            Da, _ = cosmo.growth_parameters(omegarc=10**log10omegarc)  
        elif MGmodel==3:
            cosmo = mg.Linder_gamma_a(background)
            gamma0 = params_dic['gamma0'] #if 'gamma0' in params_dic else self.ModelParsFix['gamma0']  
            gamma1 = params_dic['gamma1'] #if 'gamma1' in params_dic else self.ModelParsFix['gamma1']  
            Da, _ = cosmo.growth_parameters(gamma=gamma0, gamma1=gamma1)  
        else:
            raise ValueError("Invalid MG_model option.")    
        Dz = Da[::-1] #should be normalised to z=0
        Dz = Dz[1:]/Dz[0]
        return Dz
    
    def get_ia_kernel(self, Omega_m, params_dic, Ez, Dz, eta_z_s, zz_integr, IAmodel=0):
        if IAmodel==0:
            W_IA_p = eta_z_s * Ez[:,None] * H0_h_c
            C_IA = 0.0134
            A_IA = params_dic['aIA'] 
            eta_IA = params_dic['etaIA'] 
            beta_IA = params_dic['betaIA']
            F_IA = (1.+zz_integr)**eta_IA * (self.Survey.lum_func(zz_integr))**beta_IA
            Dz = Dz[None,:] #can be scale-dependent for f(R)
            W_IA = - A_IA*C_IA*Omega_m*F_IA[None,:,None]/Dz[:,:,None] * W_IA_p[None,:,:]
        elif IAmodel==100:
            W_IA = np.zeros((self.Survey.zbin_integr, self.Survey.nbin))
        return W_IA
    
    def get_wl_kernel(self, Omega_m, params_dic, Ez, rz, Dz, IAmodel):
        eta_z_s =  self.Survey.eta_z_s #later change to a function with varying photo-z errors
        integrand = 3./2.*H0_h_c**2. * Omega_m * rz[None,:,None]*(1.+self.Survey.zz_integr[None,:,None])*eta_z_s.T[:,None,:]*(1.-rz[None,:,None]/rz[None,None,:])
        W_gamma  = trapezoid(np.triu(integrand), self.Survey.zz_integr,axis=-1).T
        W_L = W_gamma[None,:,:] + self.get_ia_kernel(Omega_m, params_dic, Ez, Dz, eta_z_s, self.Survey.zz_integr, IAmodel)
        return W_L
    
    def get_cell_shear(self, params_dic, Ez, rz, Dz, Pk, IAmodel=0):
        Omega_m = params_dic['Omega_m']
        W_L = self.get_wl_kernel(Omega_m, params_dic, Ez, rz, Dz, IAmodel)
        Cl_LL_int = W_L[:,:,:,None] * W_L[:,:,None,:] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        Cl_LL     = trapezoid(Cl_LL_int,self.Survey.zz_integr,axis=1)[:self.Survey.nell_WL,:,:]
        for i in range(self.Survey.nbin):
            Cl_LL[:,i,i] += self.Survey.noise['LL']
        return Cl_LL, W_L
    
    def get_bPgg(self, params_dic, k, Pgg, Pgg_extr, nbin, bias_model):
        if bias_model == 0:
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(nbin)])
            bPgg = bias1[None, None, :, None] * bias1[None, None, None, :] * Pgg[:,:,None, None]
        elif bias_model == 1:  
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(nbin)])
            bias2 = np.array([params_dic['b2_'+str(i+1)] for i in range(nbin)])  
            bPgg = ( bias1[None, None,:, None] + bias2[None, None, :, None] * k[:, :, None, None]**2 )*( bias1[None, None, None, :] + bias2[None, None, None, :] * k[:, :, None, None]**2 )* Pgg[:,:,None,None]
        elif bias_model == 2:
            bL1 = np.array([params_dic['b1L_'+str(i+1)] for i in range(nbin)])
            bL2 = np.array([params_dic['b2L_'+str(i+1)] for i in range(nbin)])
            bs2 = np.array([params_dic['bs2L_'+str(i+1)] for i in range(nbin)])
            blapl = np.array([params_dic['blaplL_'+str(i+1)] for i in range(nbin)])
            Pdmdm = Pgg[0, :, :]        
            Pdmd1 = Pgg[1, :, :]    
            Pdmd2 = Pgg[2, :, :]        
            Pdms2 = Pgg[3, :, :]        
            Pdmk2 = Pgg[4, :, :]      
            Pd1d1 = Pgg[5, :, :]     
            Pd1d2 = Pgg[6, :, :]        
            Pd1s2 = Pgg[7, :, :]         
            Pd1k2 = Pgg[8, :, :]         
            Pd2d2 = Pgg[9, :, :]        
            Pd2s2 = Pgg[10, :, :]         
            Pd2k2 = Pgg[11, :, :]
            Ps2s2 = Pgg[12, :, :] 
            Ps2k2 = Pgg[13, :, :] 
            Pk2k2 = Pgg[14, :, :] 
            bPgg = (Pdmdm[:,:,None,None]  +
               (bL1[None,None,:,None]+bL1[None,None, None, :]) * Pdmd1[:,:,None,None] +
               (bL1[None,None, :,None]*bL1[None,None, None, :]) * Pd1d1[:,:,None,None] +
               (bL2[None,None, :,None] + bL2[None,None, None, :]) * Pdmd2[:,:,None,None] +
               (bs2[None,None, :,None] + bs2[None,None, None, :]) * Pdms2[:,:,None,None] +
               (bL1[None,None, :,None]*bL2[None,None, None, :] + bL1[None,None, None, :]*bL2[None,None, :,None]) * Pd1d2[:,:,None,None] +
               (bL1[None,None, :,None]*bs2[None,None, None, :] + bL1[None,None, None, :]*bs2[None,None, :,None]) * Pd1s2[:,:,None,None] +
               (bL2[None,None, :,None]*bL2[None,None, None, :]) * Pd2d2[:,:,None,None] +
               (bL2[None,None, :,None]*bs2[None,None, None, :] + bL2[None,None, None, :]*bs2[None,None, :,None]) * Pd2s2[:,:,None,None] +
               (bs2[None,None, :,None]*bs2[None,None, None, :])* Ps2s2[:,:,None,None] +
               (blapl[None,None, :,None] + blapl[None,None, None, :]) * Pdmk2[:,:,None,None] +
               (bL1[None,None, None, :] * blapl[None,None, :,None] + bL1[None,None, :,None] * blapl[None,None, None, :]) * Pd1k2[:,:,None,None] +
               (bL2[None,None, None, :] * blapl[None,None, :,None] + bL2[None,None, :,None] * blapl[None,None, None, :]) * Pd2k2[:,:,None,None] +
               (bs2[None,None, None, :] * blapl[None,None, :,None] + bs2[None,None, :,None] * blapl[None,None, None, :]) * Ps2k2[:,:,None,None] +
               (blapl[None,None, :,None] * blapl[None,None, None, :]) * Pk2k2[:,:,None,None])
            bPgg_extr = (1.+bL1[None,None, :,None])*(1.+bL1[None,None, None, :]) * Pgg_extr[:,:,None,None]
            bPgg += bPgg_extr 
        else:
            raise ValueError("Invalid bias_model option.")
        return bPgg

    def get_gg_kernel(self, Ez, bias_model):
        eta_z_l = self.Survey.eta_z_l #later change to a function with varying photo-z errors
        if bias_model == 0:
            W_G = np.zeros((self.Survey.zbin_integr, self.Survey.nbin), 'float64')
            W_G = Ez[:,None] * H0_h_c * eta_z_l
            W_G = W_G[None, :, :]
        elif bias_model == 1:
            W_G = np.zeros((self.Survey.lbin, self.Survey.zbin_integr, self.Survey.nbin), 'float64')
            W_G = Ez[None, :,None] * H0_h_c * eta_z_l[None, :, :]
        elif bias_model ==2:
            W_G = np.zeros((self.Survey.zbin_integr, self.Survey.nbin), 'float64')
            W_G = Ez[:,None] * H0_h_c * eta_z_l
            W_G = W_G[None, :, :]
        else:
            raise ValueError("Invalid bias_model option.")    
        return W_G    
    
    def get_cell_galclust(self, params_dic, Ez, rz, k, Pgg, Pgg_extr, bias_model=0):
        W_G = self.get_gg_kernel(Ez, bias_model)
        bPgg = self.get_bPgg(params_dic,  k, Pgg, Pgg_extr, self.Survey.nbin, bias_model)
        #ell, z_integr, bin_i, bin_j
        Cl_GG_int = W_G[:,:,:,None] * W_G[:,: , None, :] * bPgg / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c    
        Cl_GG     = trapezoid(Cl_GG_int,self.Survey.zz_integr,axis=1)[:self.Survey.nell_GC,:,:]
        for i in range(self.Survey.nbin):
            Cl_GG[:,i,i] += self.Survey.noise['GG']
        return Cl_GG, W_G    
    
    def get_bPgm(self,params_dic, k, Pgm, Pgm_extr, nbin, bias_model):
        if bias_model == 0:
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(nbin)])
            bPgm = bias1[None,None, :] * Pgm[:,:,None]
        elif bias_model == 1:
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(nbin)])
            bias2 = np.array([params_dic['b2_'+str(i+1)] for i in range(nbin)])  
            bPgm = ( bias1[None, None,:] + bias2[None, None, :] * k[:, :, None]**2 )*Pgm[:,:,None]
        elif bias_model == 2:
            Pdmdm = Pgm[0, :, :]        
            Pdmd1 = Pgm[1, :, :]    
            Pdmd2 = Pgm[2, :, :]        
            Pdms2 = Pgm[3, :, :]        
            Pdmk2 = Pgm[4, :, :]      
            bL1 = np.array([params_dic['b1L_'+str(i+1)] for i in range(nbin)])
            bL2 = np.array([params_dic['b2L_'+str(i+1)] for i in range(nbin)])
            bs2 = np.array([params_dic['bs2L_'+str(i+1)] for i in range(nbin)])
            blapl = np.array([params_dic['blaplL_'+str(i+1)] for i in range(nbin)])
            bPgm = (Pdmdm[:,:,None]  +
                bL1[None,None,:] * Pdmd1[:,:,None] +
                bL2[None,None,:] * Pdmd2[:,:,None] +
                bs2[None,None,:] * Pdms2[:,:,None] +
                blapl[None,None,:] * Pdmk2[:,:,None])   
            bPgm_extr = (1.+bL1[None,None,:]) * Pgm_extr[:,:,None]
            bPgm += bPgm_extr
        else:
            raise ValueError("Invalid bias_model option.")         
        return bPgm

    def get_cell_galgal(self, params_dic, Ez, rz, k, Pgm, Pgm_extr, W_L, W_G, bias_model=0):
        bPgm = self.get_bPgm(params_dic, k, Pgm, Pgm_extr, self.Survey.nbin, bias_model)
        Cl_LG_int = W_L[:,:,:,None] * W_G[:, :, None, :] * bPgm[:,:,None,:] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        Cl_LG     = trapezoid(Cl_LG_int,self.Survey.zz_integr,axis=1)[:self.Survey.nell_XC,:,:]
        Cl_GL     = np.transpose(Cl_LG,(0,2,1))  
        return Cl_LG, Cl_GL
    
    def get_Pmm(self, params_dic, k, lbin, zz_integr, NL_model=0, baryon_model=0):
        if NL_model==0:
            Pk = self.HMcode2020emulator.get_pk(params_dic, k, lbin, zz_integr)
        elif NL_model==1:
            Pk = self.baccoemulator.get_pk(params_dic, k, lbin, zz_integr) 
        else:
            raise ValueError("Invalid nonlin_model option.")    
   
        if baryon_model!=0:
            if baryon_model==1:
                boost_bar = self.HMcode2020emulator.get_barboost(params_dic, k, lbin, zz_integr)
            elif baryon_model==2:
                boost_bar = self.BCemulator.get_barboost(params_dic, k, lbin, zz_integr)    
            elif baryon_model==3:
                boost_bar = self.baccoemulator.get_barboost(params_dic, k, lbin, zz_integr)    
            else:
                raise ValueError("Invalid baryon_model option.")
            Pk *= boost_bar 
        return Pk

    def compute_covariance_WL(self, params_dic, model_dic):
        Ez, rz, k = self.get_Ez_rz_k(params_dic, self.Survey.zz_integr)
        Dz = self.get_growth(params_dic, self.Survey.zz_integr, model_dic['NL_model'])
        Pk = self.get_Pmm(params_dic, k, self.Survey.lbin, self.Survey.zz_integr, model_dic['NL_model'], model_dic['baryon_model'])
        Cl_LL, _ = self.get_cell_shear(params_dic, Ez, rz, Dz, Pk, model_dic['IA_model'])
        spline_LL = np.empty((self.Survey.nbin, self.Survey.nbin),dtype=(list,3))
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_LL[bin1,bin2] = list(itp.splrep(
                    self.Survey.l_WL[:], Cl_LL[:,bin1,bin2]))    
        cov_theory = np.zeros((len(self.Survey.ells_WL), self.Survey.nbin, self.Survey.nbin), 'float64')
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):  
                cov_theory[:,bin1,bin2] = itp.splev(self.Survey.ells_WL[:], spline_LL[bin1,bin2]) 
        #cov_theory = np.where(self.Survey.mask_ells_WL, cov_theory, 0)
        return cov_theory    

    def compute_covariance_GC(self, params_dic, model_dic):
        Ez, rz, k = self.get_Ez_rz_k(params_dic, self.Survey.zz_integr)
        Pk = self.get_Pmm(params_dic, k, self.Survey.lbin, self.Survey.zz_integr, model_dic['NL_model'], model_dic['baryon_model'])
        Pgg = Pk
        Pgg_extr = None
        if model_dic['bias_model']==2:
            Pgg, Pgg_extr = self.baccoemulator.get_heft(params_dic, k, self.Survey.lbin, self.Survey.zz_integr) 
        
        Cl_GG, _ = self.get_cell_galclust(params_dic, Ez, rz, k, Pgg, Pgg_extr, model_dic['bias_model'])    
        spline_GG = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list,3))
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_GG[bin1,bin2] = list(itp.splrep(
                    self.Survey.l_GC[:], Cl_GG[:,bin1,bin2]))

        cov_theory = np.zeros((len(self.Survey.ells_GC), self.Survey.nbin, self.Survey.nbin), 'float64')
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):  
                cov_theory[:,bin1,bin2] = itp.splev(self.Survey.ells_GC[:], spline_GG[bin1,bin2]) 
        #cov_theory = np.where(self.Survey.mask_ells_GC, cov_theory, 0)
        return cov_theory   

    def compute_covariance_3x2pt(self, params_dic, model_dic):
        Ez, rz, k = self.get_Ez_rz_k(params_dic, self.Survey.zz_integr)
        Dz = self.get_growth(params_dic, self.Survey.zz_integr, model_dic['NL_model'])
        Pk = self.get_Pmm(params_dic, k, self.Survey.lbin, self.Survey.zz_integr, model_dic['NL_model'], model_dic['baryon_model'])
        Pmm = Pk
        Pgg = Pk
        Pgg_extr = None
        Pgm = Pk 
        Pgm_extr = None
        if model_dic['bias_model']==2:
            Pgg, Pgg_extr = Pgm, Pgm_extr = self.baccoemulator.get_heft(params_dic, k, self.Survey.lbin, self.Survey.zz_integr) 
        ###Window functions W_xx(l,z,bin) in units of [W] = h/Mpc
        Cl_LL, W_L = self.get_cell_shear(params_dic, Ez, rz, Dz, Pmm, model_dic['IA_model'])
        spline_LL = np.empty((self.Survey.nbin, self.Survey.nbin),dtype=(list,3))
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_LL[bin1,bin2] = list(itp.splrep(
                    self.Survey.l_WL[:], Cl_LL[:,bin1,bin2]))    
        Cl_GG, W_G = self.get_cell_galclust(params_dic, Ez, rz, k, Pgg, Pgg_extr, model_dic['bias_model'])    
        spline_GG = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list,3))
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_GG[bin1,bin2] = list(itp.splrep(
                    self.Survey.l_GC[:], Cl_GG[:,bin1,bin2]))

        Cl_LG, Cl_GL = self.get_cell_galgal(params_dic, Ez, rz, k, Pgm, Pgm_extr, W_L, W_G, model_dic['bias_model'])   
        spline_LG = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list,3))
        spline_GL = np.empty((self.Survey.nbin, self.Survey.nbin), dtype=(list,3))
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                spline_LG[bin1,bin2] = list(itp.splrep(
                    self.Survey.l_XC[:], Cl_LG[:,bin1,bin2]))
                spline_GL[bin1,bin2] = list(itp.splrep(
                    self.Survey.l_XC[:], Cl_GL[:,bin1,bin2]))
            
        cov_theory = np.zeros((len(self.Survey.ells_GC), 2*self.Survey.nbin, 2*self.Survey.nbin), 'float64')
        cov_theory_high = np.zeros(((len(self.Survey.ells_WL)-self.Survey.ell_jump), self.Survey.nbin, self.Survey.nbin), 'float64')    
        for bin1 in range(self.Survey.nbin):
            for bin2 in range(self.Survey.nbin):
                cov_theory[:,bin1,bin2] = itp.splev(
                    self.Survey.ells_GC[:], spline_LL[bin1,bin2])
                cov_theory[:,self.Survey.nbin+bin1,bin2] = itp.splev(
                    self.Survey.ells_GC[:], spline_GL[bin1,bin2])
                cov_theory[:,bin1,self.Survey.nbin+bin2] = itp.splev(
                    self.Survey.ells_GC[:], spline_LG[bin1,bin2])
                cov_theory[:,self.Survey.nbin+bin1,self.Survey.nbin+bin2] = itp.splev(
                    self.Survey.ells_GC[:], spline_GG[bin1,bin2])
                cov_theory_high[:,bin1,bin2] = itp.splev(
                    self.Survey.ells_WL[self.Survey.ell_jump:], spline_LL[bin1,bin2])
        #cov_theory[self.Survey.mask_ells_3x2pt] = np.nan       
        #cov_theory_high[self.Survey.mask_ells_high] = np.nan  
        return cov_theory, cov_theory_high
    

 