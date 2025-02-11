import numpy as np
import MGrowth as mg
from scipy.integrate import trapezoid,simpson, quad
from scipy import interpolate as itp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

try: import baccoemu
except: print("Bacco not installed!")

H0_h_c = 1./2997.92458

class Theory:
    def get_Ez_rz(self, Omega_m, params_dic, zz):
        w0 = params_dic['w0']
        wa = params_dic['wa']
        H0_h_c = 1./2997.92458 #=100/c in Mpc/h
        omegaL_func = lambda z: (1.-Omega_m) * pow(1.+z, 3.*(1.+w0+wa)) * np.exp(-3.*wa*z/(1.+z))
        E_z_func = lambda z: np.sqrt(Omega_m*pow(1.+z, 3) + omegaL_func(z))
        E_z_grid = np.array([E_z_func(zz_i) for zz_i in zz])
        r_z_int = lambda z: 1./np.sqrt(Omega_m*pow(1.+z, 3) + omegaL_func(z))
        r_z_func = lambda z_in: quad(r_z_int, 0, z_in)[0]
        r_z_grid = np.array([r_z_func(zz_i) for zz_i in zz])/H0_h_c #Mpc/h
        return E_z_grid, r_z_grid
    
    def get_growth(self, Omega_m, params_dic, MGmodel=0):
        background ={
            'Omega_m': Omega_m,
            'h' : params_dic['h'],
            'w0': params_dic['w0'],
            'wa': params_dic['wa'],
            'a_arr': np.hstack((self.aa_integr, 1.))
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
        Dz = Da[::-1] #should be normalised to z=0
        Dz = Dz[1:]/Dz[0]
        return Dz
    
    def get_cell_shear(self, Omega_m, params_dic, noise, Ez, rz, Dz, Pk, IAmodel=0):
        integrand = 3./2.*H0_h_c**2. * Omega_m * rz[None,:,None]*(1.+self.zz_integr[None,:,None])*self.eta_z_s.T[:,None,:]*(1.-rz[None,:,None]/rz[None,None,:])
        W_gamma  = trapezoid(np.triu(integrand),self.zz_integr,axis=-1).T
        # Compute contribution from IA (Intrinsic Alignement)
        if IAmodel==0:
            # - compute window function W_IA
            W_IA_p = self.eta_z_s * Ez[:,None] * H0_h_c
            # - IA contribution depends on a few parameters assigned here
            # fiducial values {a, eta, beta} = {1.72, -0.41, 0.0}
            C_IA = 0.0134
            A_IA = params_dic['aIA'] #if 'aIA' in params_dic else self.ModelParsFix['aIA'] 
            eta_IA = params_dic['etaIA'] #if 'etaIA' in params_dic else self.ModelParsFix['etaIA'] 
            F_IA = (1.+self.zz_integr)**eta_IA 
            Dz = Dz[None,:]
            W_IA = - A_IA*C_IA*Omega_m*F_IA[None,:,None]/Dz[:,:,None] * W_IA_p[None,:,:]

        W_L = W_gamma[None,:,:] + W_IA 
        Cl_LL_int = W_L[:,:,:,None] * W_L[:,:,None,:] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        Cl_LL     = trapezoid(Cl_LL_int,self.zz_integr,axis=1)[:self.nell_WL,:,:]
        for i in range(self.nbin):
            Cl_LL[:,i,i] += noise['LL']
        return Cl_LL, W_L
    
    def get_cell_galclust(self, params_dic, noise, Ez, rz, k, Pk, bias_model=0):
        if bias_model == 0:
            #bias1 = np.array([(params_dic['b1_'+str(i+1)] if 'b1_'+str(i+1) in params_dic else self.ModelParsFix['b1_'+str(i+1)] )for i in range(self.nbin)])
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.nbin)])
            W_G = np.zeros((self.zbin_integr, self.nbin), 'float64')
            W_G = bias1[None,:] * Ez[:,None] * H0_h_c * self.eta_z_l
            Cl_GG_int = W_G[None,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
            W_G_newdim = W_G[None,: , None, :]
        elif bias_model == 1:
            #bias1 = np.array([(params_dic['b1_'+str(i+1)] if 'b1_'+str(i+1) in params_dic else self.ModelParsFix['b1_'+str(i+1)] )for i in range(self.nbin)])
            #bias2 = np.array([(params_dic['b2_'+str(i+1)] if 'b2_'+str(i+1) in params_dic else self.ModelParsFix['b2_'+str(i+1)] )for i in range(self.nbin)])
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.nbin)])
            bias2 = np.array([params_dic['b2_'+str(i+1)] for i in range(self.nbin)])
            W_G = np.zeros((self.lbin, self.zbin_integr, self.nbin), 'float64')
            W_G = ( bias1[None, None,:] + bias2[None, None, :] * k[:, :, None]**2 ) * Ez[None, :,None] * H0_h_c * self.eta_z_l[None, :, :]
            Cl_GG_int = W_G[:,:,:,None] * W_G[:,: , None, :] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
            W_G_newdim = W_G[:,: , None, :]

        Cl_GG     = trapezoid(Cl_GG_int,self.zz_integr,axis=1)[:self.nell_GC,:,:]
        for i in range(self.nbin):
            Cl_GG[:,i,i] += noise['GG']
        return Cl_GG, W_G_newdim     
    
    def get_cell_galgal(self, params_dic, noise, Ez, rz, k, Pk, W_L, W_G):
        Cl_LG_int = W_L[:,:,:,None] * W_G * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        Cl_LG     = trapezoid(Cl_LG_int,self.zz_integr,axis=1)[:self.nell_XC,:,:]
        Cl_GL     = np.transpose(Cl_LG,(0,2,1))  
        return Cl_LG, Cl_GL

    def compute_covariance(self, params_dic, model_dic):
        Omega_m =  params_dic['Omega_m']
        Ez, rz = self.get_Ez_rz(Omega_m, params_dic, self.zz_integr)
        Dz = self.get_growth(Omega_m, params_dic, model_dic['NL_model'])
        k =(self.l_WL[:,None]+0.5)/rz
        

        pk_m_l  = np.zeros((self.lbin, self.zbin_integr), 'float64')
        index_pknn = np.array(np.where((k > self.k_min_h_by_Mpc) & (k<self.k_max_h_by_Mpc))).transpose()
        
        if model_dic['NL_model']==0:
            Pk_l_interp = self.hmcode_emu.get_pk_interp(params_dic)

        
    
        for index_l, index_z in index_pknn:
            pk_m_l[index_l, index_z] = Pk_l_interp(self.zz_integr[index_z], k[index_l,index_z])
        Pk = pk_m_l     
    
        ###Add baryonic feedback 
        if model_dic['baryon_model']!=0:
            boost_bar = np.zeros((self.lbin, self.zbin_integr), 'float64')
            if model_dic['baryon_model']==1:
                boost_bar_interp = self.hmcode_emu.get_barboost_interp(params_dic)
            for index_l, index_z in index_pknn:
                boost_bar[index_l, index_z] = boost_bar_interp(min(self.zz_integr[index_z], 2), k[index_l,index_z])
            Pk *= boost_bar

    

        noise = {
        'LL': self.rms_shear**2./self.n_bar,
        'LG': 0.,
        'GL': 0.,
        'GG': 1./self.n_bar}

    

        ###Window functions W_xx(l,z,bin) in units of [W] = h/Mpc
        if self.Probe=='WL' or self.Probe=='3x2pt':
            Cl_LL, W_L = self.get_cell_shear(Omega_m, params_dic, noise, Ez, rz, Dz, Pk, model_dic['IA_model'])
            spline_LL = np.empty((self.nbin, self.nbin),dtype=(list,3))
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    spline_LL[Bin1,Bin2] = list(itp.splrep(
                        self.l_WL[:], Cl_LL[:,Bin1,Bin2]))    


        if self.Probe=='GC' or self.Probe=='3x2pt':
            Cl_GG, W_G = self.get_cell_galclust(params_dic, noise, Ez, rz, k, Pk, model_dic['bias_model'])    
            spline_GG = np.empty((self.nbin, self.nbin), dtype=(list,3))
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    spline_GG[Bin1,Bin2] = list(itp.splrep(
                        self.l_GC[:], Cl_GG[:,Bin1,Bin2]))

        if self.Probe=='3x2pt':
            Cl_LG, Cl_GL = self.get_cell_galgal(params_dic, noise, Ez, rz, k, Pk, W_L, W_G)   
            spline_LG = np.empty((self.nbin, self.nbin), dtype=(list,3))
            spline_GL = np.empty((self.nbin, self.nbin), dtype=(list,3))
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    spline_LG[Bin1,Bin2] = list(itp.splrep(
                        self.l_XC[:], Cl_LG[:,Bin1,Bin2]))
                    spline_GL[Bin1,Bin2] = list(itp.splrep(
                        self.l_XC[:], Cl_GL[:,Bin1,Bin2]))
            
            Cov_theory = np.zeros((self.ell_jump, 2*self.nbin, 2*self.nbin), 'float64')
            Cov_theory_high = np.zeros(((len(self.ells_WL)-self.ell_jump), self.nbin, self.nbin), 'float64')    
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    Cov_theory[:,Bin1,Bin2] = itp.splev(
                        self.ells_GC[:], spline_LL[Bin1,Bin2])
                    Cov_theory[:,self.nbin+Bin1,Bin2] = itp.splev(
                        self.ells_GC[:], spline_GL[Bin1,Bin2])
                    Cov_theory[:,Bin1,self.nbin+Bin2] = itp.splev(
                        self.ells_GC[:], spline_LG[Bin1,Bin2])
                    Cov_theory[:,self.nbin+Bin1,self.nbin+Bin2] = itp.splev(
                        self.ells_GC[:], spline_GG[Bin1,Bin2])
                    Cov_theory_high[:,Bin1,Bin2] = itp.splev(
                        self.ells_WL[self.ell_jump:], spline_LL[Bin1,Bin2])
       
            return Cov_theory, Cov_theory_high
    
        elif self.Probe=='WL':
            Cov_theory = np.zeros((len(self.ells_WL), self.nbin, self.nbin), 'float64')
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):  
                    Cov_theory[:,Bin1,Bin2] = itp.splev(self.ells_WL[:], spline_LL[Bin1,Bin2]) 
            return Cov_theory  
            
        elif self.Probe=='GC':
            Cov_theory = np.zeros((len(self.ells_GC), self.nbin, self.nbin), 'float64')
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):  
                    Cov_theory[:,Bin1,Bin2] = itp.splev(self.ells_GC[:], spline_GG[Bin1,Bin2]) 
            return Cov_theory   
 