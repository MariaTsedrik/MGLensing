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
    def get_Ez_rz_k(self, params_dic, zz):
        Omega_m = params_dic['Omega_m']
        w0 = params_dic['w0']
        wa = params_dic['wa']
        H0_h_c = 1./2997.92458 #=100/c in Mpc/h
        omegaL_func = lambda z: (1.-Omega_m) * pow(1.+z, 3.*(1.+w0+wa)) * np.exp(-3.*wa*z/(1.+z))
        E_z_func = lambda z: np.sqrt(Omega_m*pow(1.+z, 3) + omegaL_func(z))
        E_z_grid = np.array([E_z_func(zz_i) for zz_i in zz])
        r_z_int = lambda z: 1./np.sqrt(Omega_m*pow(1.+z, 3) + omegaL_func(z))
        r_z_func = lambda z_in: quad(r_z_int, 0, z_in)[0]
        r_z_grid = np.array([r_z_func(zz_i) for zz_i in zz])/H0_h_c #Mpc/h
        k_grid =(self.l_WL[:,None]+0.5)/r_z_grid
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
    
    def get_ia_kernel(self, Omega_m, params_dic, Ez, Dz, IAmodel):
        if IAmodel==0:
            W_IA_p = self.eta_z_s * Ez[:,None] * H0_h_c
            C_IA = 0.0134
            A_IA = params_dic['aIA'] 
            eta_IA = params_dic['etaIA'] 
            F_IA = (1.+self.zz_integr)**eta_IA 
            Dz = Dz[None,:]
            W_IA = - A_IA*C_IA*Omega_m*F_IA[None,:,None]/Dz[:,:,None] * W_IA_p[None,:,:]
        return W_IA
    
    def get_wl_kernel(self, Omega_m, params_dic, Ez, rz, Dz, IAmodel):
        integrand = 3./2.*H0_h_c**2. * Omega_m * rz[None,:,None]*(1.+self.zz_integr[None,:,None])*self.eta_z_s.T[:,None,:]*(1.-rz[None,:,None]/rz[None,None,:])
        W_gamma  = trapezoid(np.triu(integrand),self.zz_integr,axis=-1).T
        W_L = W_gamma[None,:,:] + self.get_ia_kernel(Omega_m, params_dic, Ez, Dz, IAmodel) 
        return W_L
    
    def get_cell_shear(self, params_dic, Ez, rz, Dz, Pk, IAmodel=0):
        Omega_m = params_dic['Omega_m']
        W_L = self.get_wl_kernel(Omega_m, params_dic, Ez, rz, Dz, IAmodel)
        Cl_LL_int = W_L[:,:,:,None] * W_L[:,:,None,:] * Pk[:,:,None,None] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        Cl_LL     = trapezoid(Cl_LL_int,self.zz_integr,axis=1)[:self.nell_WL,:,:]
        for i in range(self.nbin):
            Cl_LL[:,i,i] += self.noise['LL']
        return Cl_LL, W_L
    
    def get_bPgg(self, params_dic, k, Pgg, bias_model):
        if bias_model == 0:
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.nbin)])
            bPgg = bias1[None, None, :, None] * bias1[None, None, None, :] * Pgg[:,:,None, None]
        elif bias_model == 1:  
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.nbin)])
            bias2 = np.array([params_dic['b2_'+str(i+1)] for i in range(self.nbin)])  
            bPgg = ( bias1[None, None,:, None] + bias2[None, None, :, None] * k[:, :, None, None]**2 )*( bias1[None, None, None, :] + bias2[None, None, None, :] * k[:, :, None, None]**2 )* Pgg[:,:,None,None]
        elif bias_model == 2:
            bL1 = np.array([params_dic['b1L_'+str(i+1)] for i in range(self.nbin)])
            bL2 = np.array([params_dic['b2L_'+str(i+1)] for i in range(self.nbin)])
            bs2 = np.array([params_dic['bs2L_'+str(i+1)] for i in range(self.nbin)])
            blapl = np.array([params_dic['blaplL_'+str(i+1)] for i in range(self.nbin)])
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
        else:
            raise ValueError("Invalid bias_model option.")
        return bPgg

    def get_gg_kernel(self, Ez, bias_model):
        if bias_model == 0:
            W_G = np.zeros((self.zbin_integr, self.nbin), 'float64')
            W_G = Ez[:,None] * H0_h_c * self.eta_z_l
            W_G = W_G[None, :, :]
        elif bias_model == 1:
            W_G = np.zeros((self.lbin, self.zbin_integr, self.nbin), 'float64')
            W_G = Ez[None, :,None] * H0_h_c * self.eta_z_l[None, :, :]
        elif bias_model ==2:
            W_G = np.zeros((self.zbin_integr, self.nbin), 'float64')
            W_G = Ez[:,None] * H0_h_c * self.eta_z_l
            W_G = W_G[None, :, :]
        else:
            raise ValueError("Invalid bias_model option.")    
        return W_G    
    
    def get_cell_galclust(self, params_dic, Ez, rz, k, Pgg, bias_model=0):
        W_G = self.get_gg_kernel(Ez, bias_model)
        bPgg = self.get_bPgg(params_dic,  k, Pgg, bias_model)
        #ell, z_integr, bin_i, bin_j
        Cl_GG_int = W_G[:,:,:,None] * W_G[:,: , None, :] * bPgg / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c    
        Cl_GG     = trapezoid(Cl_GG_int,self.zz_integr,axis=1)[:self.nell_GC,:,:]
        for i in range(self.nbin):
            Cl_GG[:,i,i] += self.noise['GG']
        return Cl_GG, W_G    
    
    def get_bPgm(self,params_dic, k, Pgm, bias_model):
        if bias_model == 0:
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.nbin)])
            bPgm = bias1[None,None, :] * Pgm[:,:,None]
        elif bias_model == 1:
            bias1 = np.array([params_dic['b1_'+str(i+1)] for i in range(self.nbin)])
            bias2 = np.array([params_dic['b2_'+str(i+1)] for i in range(self.nbin)])  
            bPgm = ( bias1[None, None,:] + bias2[None, None, :] * k[:, :, None]**2 )*Pgm[:,:,None]
        elif bias_model == 2:
            Pdmdm = Pgm[0, :, :]        
            Pdmd1 = Pgm[1, :, :]    
            Pdmd2 = Pgm[2, :, :]        
            Pdms2 = Pgm[3, :, :]        
            Pdmk2 = Pgm[4, :, :]      
            bL1 = np.array([params_dic['b1L_'+str(i+1)] for i in range(self.nbin)])
            bL2 = np.array([params_dic['b2L_'+str(i+1)] for i in range(self.nbin)])
            bs2 = np.array([params_dic['bs2L_'+str(i+1)] for i in range(self.nbin)])
            blapl = np.array([params_dic['blaplL_'+str(i+1)] for i in range(self.nbin)])
            bPgm = (Pdmdm[:,:,None]  +
                bL1[None,None,:] * Pdmd1[:,:,None] +
                bL2[None,None,:] * Pdmd2[:,:,None] +
                bs2[None,None,:] * Pdms2[:,:,None] +
                blapl[None,None,:] * Pdmk2[:,:,None])   
        else:
            raise ValueError("Invalid bias_model option.")         
        return bPgm

    def get_cell_galgal(self, params_dic, Ez, rz, k, Pgm, W_L, W_G, bias_model=0):
        bPgm = self.get_bPgm(params_dic, k, Pgm, bias_model)
        Cl_LG_int = W_L[:,:,:,None] * W_G[:, :, None, :] * bPgm[:,:,None,:] / Ez[None,:,None,None] / rz[None,:,None,None] / rz[None,:,None,None] /H0_h_c
        Cl_LG     = trapezoid(Cl_LG_int,self.zz_integr,axis=1)[:self.nell_XC,:,:]
        Cl_GL     = np.transpose(Cl_LG,(0,2,1))  
        return Cl_LG, Cl_GL
    
    def get_Pmm(self, params_dic, k, NL_model=0, baryon_model=0):
        if NL_model==0:
            Pk = self.hmcode_emu.get_pk(params_dic, k, self.lbin, self.zz_integr)
        elif NL_model==1:
            Pk = self.bacco_emu.get_pk(params_dic, k, self.lbin, self.zz_integr) 
        else:
            raise ValueError("Invalid nonlin_model option.")    
   
        if baryon_model!=0:
            if baryon_model==1:
                boost_bar = self.hmcode_emu.get_barboost(params_dic, k, self.lbin, self.zz_integr)
            elif baryon_model==2:
                boost_bar = self.bcemu.get_barboost(params_dic, k, self.lbin, self.zz_integr)    
            elif baryon_model==3:
                boost_bar = self.bacco_emu.get_barboost(params_dic, k, self.lbin, self.zz_integr)    
            else:
                raise ValueError("Invalid baryon_model option.")
            Pk *= boost_bar 
        return Pk

    def compute_covariance(self, params_dic, model_dic):
        Ez, rz, k = self.get_Ez_rz_k(params_dic, self.zz_integr)
        Dz = self.get_growth(params_dic, self.zz_integr, model_dic['NL_model'])
        

        Pk = self.get_Pmm(params_dic, k, model_dic['NL_model'], model_dic['baryon_model'])
        Pmm = Pk
        Pgg = Pk
        Pgm = Pk 
        if model_dic['bias_model']==2:
            Pgg = Pgm = self.bacco_emu.get_heft(params_dic, k, self.lbin, self.zz_integr) 
        

        ###Window functions W_xx(l,z,bin) in units of [W] = h/Mpc
        if self.Probe=='WL' or self.Probe=='3x2pt':
            Cl_LL, W_L = self.get_cell_shear(params_dic, Ez, rz, Dz, Pmm, model_dic['IA_model'])
            spline_LL = np.empty((self.nbin, self.nbin),dtype=(list,3))
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    spline_LL[Bin1,Bin2] = list(itp.splrep(
                        self.l_WL[:], Cl_LL[:,Bin1,Bin2]))    


        if self.Probe=='GC' or self.Probe=='3x2pt':
            Cl_GG, W_G = self.get_cell_galclust(params_dic, Ez, rz, k, Pgg, model_dic['bias_model'])    
            spline_GG = np.empty((self.nbin, self.nbin), dtype=(list,3))
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    spline_GG[Bin1,Bin2] = list(itp.splrep(
                        self.l_GC[:], Cl_GG[:,Bin1,Bin2]))

        if self.Probe=='3x2pt':
            Cl_LG, Cl_GL = self.get_cell_galgal(params_dic, Ez, rz, k, Pgm, W_L, W_G, model_dic['bias_model'])   
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
 