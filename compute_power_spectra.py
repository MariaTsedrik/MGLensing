from scipy.integrate import trapz,simpson, quad, simps
from scipy import interpolate as itp
from scipy.interpolate import interp1d, RectBivariateSpline, InterpolatedUnivariateSpline
from scipy.special import erf
import numpy as np
import baccoemu as bacco
import pyhmcode 
import cosmopower as cp
from cosmopower import cosmopower_NN
import math
from setup_and_priors import *
import MGrowth as mg



###for gammaz and nDGP emulator###
kminp = 0.01
kmaxp = 5.
Nkp = 300
k_modes = np.array([ kminp * np.exp(i*np.log(kmaxp/kminp)/(Nkp-1)) for i in range(Nkp)])
idx_emu = (k_modes>=0.01)
k_modes = k_modes[idx_emu]
###for gamma emulator###
kminp_g = 0.001
kmaxp_g = 5.
Nkp_g = 350
k_modes_g = np.array([ kminp_g * np.exp(i*np.log(kmaxp_g/kminp_g)/(Nkp_g-1)) for i in range(Nkp_g)])
idx_emu_g = (k_modes_g>=0.01)
k_modes_g = k_modes_g[idx_emu_g]
###for HMcode###
kmin_NL = 0.01 
kmax_NL = 50. 
npoints_NL = 350 
kh_hmcode = np.logspace(np.log10(kmin_NL), np.log10(kmax_NL), npoints_NL)
###############
x = kL*8.
W = 3./pow(x, 3)* (np.sin(x) - x*np.cos(x))
###############

hmc = pyhmcode.Cosmology()
hmod = pyhmcode.Halomodel(pyhmcode.HMcode2020, verbose=False)
emulator = bacco.Matter_powerspectrum()
print('load emulators...')
cp_nn_nDGP = cosmopower_NN(restore=True, 
                      restore_filename='emulators/Boost_nDGP_emu_v2',#'emulators/Boost_nDGP_emu',
                      )
cp_nn_gamma = cosmopower_NN(restore=True, 
                      restore_filename='emulators/Boost_gLEMURS',
                      )
cp_nn_gammaz = cosmopower_NN(restore=True, 
                      restore_filename='emulators/Boost_gLEMURS_z',
                      )
cp_nn_HMcode = cosmopower_NN(restore=True, 
                      restore_filename='emulators/log10_total_matter_nonlinear_emu',
                      )
cp_nn_muspline = cosmopower_NN(restore=True, 
                      restore_filename='emulators/Boost_MuSpline_fixcosmo_emu',
                    )
print('emulators are loaded')


index_b_lin = kL < kminp
index_b_nonlin = (kL > kmaxp) 
index_b_interp = (kL >= kminp) & (kL <= kmaxp)
k_interp_last = kL[index_b_interp][-1]
k_interp_lastlast = kL[index_b_interp][-2]
log_k = math.log(k_interp_last / k_interp_lastlast)
def call_hmcode_with_boost_nDGP(CosmoDict):
    ns   = CosmoDict['ns']
    As   = CosmoDict['As']
    h    = CosmoDict['h']
    Omega_b = CosmoDict['Omega_b']
    Omega_m = CosmoDict['Omega_m']
    Omega_nu  = CosmoDict['Omega_nu']
    omegarc    = 10**CosmoDict['log10omegarc']
    params_cp = {'ns': ns*np.ones(nz_Pk), 
            'H0': h*100*np.ones(nz_Pk), 
            'Omega_b': Omega_b*np.ones(nz_Pk), 
            'Omega_m': Omega_m*np.ones(nz_Pk), 
            'Omega_nu': Omega_nu*np.ones(nz_Pk),
            'As': As*np.ones(nz_Pk), 
            'omegarc': omegarc*np.ones(nz_Pk), 
            'z': zz_Pk
            }
    Boost = cp_nn_nDGP.predictions_np(params_cp) #k from the training
    Boost = Boost[:, idx_emu]
    ###power law extrapolation###
    Boost_itp = [interp1d(k_modes, Boost[i], bounds_error=False,
                kind='cubic',
                fill_value=(Boost[i][0], Boost[i][-1])) for i in range(nz_Pk)]
    Boost_interp_values = np.array([Boost_itp[z](kL[index_b_interp]) for z in range(nz_Pk)], dtype=np.float64)
    Boost_last_entry = Boost_interp_values[:, -1]
    Boost_lastlast_entry = Boost_interp_values[:, -2]
    m = np.array([math.log(Boost_last_entry[i] / Boost_lastlast_entry[i]) / log_k for i in range(nz_Pk)])
    Boost_k = np.outer(Boost[:, 0], np.ones(kLbin, 'float64') )
    Boost_k[:, index_b_interp] = Boost_interp_values
    Boost_k[:, index_b_nonlin] = Boost_last_entry[:, np.newaxis] *(kL[np.newaxis, index_b_nonlin]/k_interp_last)**m[:, np.newaxis]
    params_bacco = {
            'ns'            :  ns,
            'A_s'           :  As,
            'hubble'        :  h,
            'omega_baryon'  :  Omega_b,
            'omega_cold'    :  Omega_m,
            'neutrino_mass' :  0.,
            'w0'            :  -1.,
            'wa'            :  0.,
            'expfactor'     :  aa_Pk
        }
    _, PL = emulator.get_linear_pk(k=kL, cold=False, **params_bacco)
    integrand = W**2*PL[-1]*kL**2/(2.*np.pi**2)
    sigma8 = np.sqrt(simpson(integrand, x=kL))
    # Set HMcode internal cosmological parameters
    hmc.om_m = Omega_m
    hmc.om_b = Omega_b
    hmc.om_v = 1-Omega_m
    hmc.h = h
    hmc.ns = ns
    hmc.sig8 = sigma8
    hmc.m_nu = 0. #no neutrinos!!!
    hmc.set_linear_power_spectrum(kL, zz_Pk, np.flip(PL,axis=0))
    PNL_LCDM = pyhmcode.calculate_nonlinear_power_spectrum(hmc, hmod, verbose=False)
    PNL = Boost_k*PNL_LCDM
    PNL_interp = RectBivariateSpline(zz_Pk,
                                kL,
                                PNL ,
                                kx=1, ky=1)
    return PNL_interp





index_b_lin_g = kL < kminp_g
index_b_nonlin_g = (kL > kmaxp_g) 
index_b_interp_g = (kL >= kminp_g) & (kL <= kmaxp_g)
k_interp_last_g = kL[index_b_interp_g][-1]
k_interp_lastlast_g = kL[index_b_interp_g][-2]
log_k_g = math.log(k_interp_last_g / k_interp_lastlast_g)

def call_hmcode_with_boost_gamma(CosmoDict):
    ns   = CosmoDict['ns']
    As   = CosmoDict['As']
    h    = CosmoDict['h']
    gamma    = CosmoDict['gamma']
    q1    = CosmoDict['q1']
    Omm = CosmoDict['Omega_m']
    Omb = CosmoDict['Omega_b']
    Omnu = CosmoDict['Omega_nu']
    params_cp = {'ns': ns*np.ones(nz_Pk), 
            'H0': h*100*np.ones(nz_Pk), 
            'Omega_b': Omb*np.ones(nz_Pk), 
            'Omega_m': Omm*np.ones(nz_Pk), 
            'Omega_nu': Omnu*np.ones(nz_Pk),
            'As': As*np.ones(nz_Pk), 
            'gamma': gamma*np.ones(nz_Pk), 
            'q1': q1*np.ones(nz_Pk),
            'q2': np.zeros(nz_Pk),
            'q3': np.zeros(nz_Pk),
            'z': zz_Pk
            }
    Boost = cp_nn_gamma.predictions_np(params_cp) #k from the training
    Boost = Boost[:, idx_emu_g]
    ###power law extrapolation###
    Boost_itp = [interp1d(k_modes_g, Boost[i], bounds_error=False,
                kind='cubic',
                fill_value=(Boost[i][0], Boost[i][-1])) for i in range(nz_Pk)]
    Boost_interp_values = np.array([Boost_itp[z](kL[index_b_interp_g]) for z in range(nz_Pk)], dtype=np.float64)
    Boost_last_entry = Boost_interp_values[:, -1]
    Boost_lastlast_entry = Boost_interp_values[:, -2]
    m = np.array([math.log(Boost_last_entry[i] / Boost_lastlast_entry[i]) / log_k_g for i in range(nz_Pk)])
    Boost_k = np.outer(Boost[:, 0], np.ones(kLbin, 'float64') )
    Boost_k[:, index_b_interp_g] = Boost_interp_values
    Boost_k[:, index_b_nonlin_g] = Boost_last_entry[:, np.newaxis] *(kL[np.newaxis, index_b_nonlin_g]/k_interp_last_g)**m[:, np.newaxis]   
    params_bacco = {
            'ns'            :  ns,
            'A_s'           :  As,
            'hubble'        :  h,
            'omega_baryon'  :  Omb,
            'omega_cold'    :  Omm,
            'neutrino_mass' :  0.,
            'w0'            :  -1.,
            'wa'            :  0.,
            'expfactor'     :  aa_Pk
        }
    _, PL = emulator.get_linear_pk(k=kL, cold=False, **params_bacco)
    integrand = W**2*PL[-1]*kL**2/(2.*np.pi**2)
    sigma8 = np.sqrt(simpson(integrand, x=kL))
    # Set HMcode internal cosmological parameters
    hmc.om_m = Omm
    hmc.om_b = Omb
    hmc.om_v = 1-Omm
    hmc.h = h
    hmc.ns = ns
    hmc.sig8 = sigma8
    hmc.m_nu = 0.
    hmc.set_linear_power_spectrum(kL, zz_Pk, np.flip(PL,axis=0))
    PNL_LCDM = pyhmcode.calculate_nonlinear_power_spectrum(hmc, hmod, verbose=False)
    PNL = Boost_k*PNL_LCDM
    PNL_interp = RectBivariateSpline(zz_Pk,
                                kL,
                                PNL ,
                                kx=1, ky=1)
    return PNL_interp

def call_hmcode_with_boost_gamma_z(CosmoDict):
    ns   = CosmoDict['ns']
    As   = CosmoDict['As']
    h    = CosmoDict['h']
    gamma0    = CosmoDict['gamma0']
    gamma1    = CosmoDict['gamma1']
    q1    = CosmoDict['q1']
    Omm = CosmoDict['Omega_m']
    Omb = CosmoDict['Omega_b']
    Omnu = CosmoDict['Omega_nu']

    params_cp = {'ns': ns*np.ones(nz_Pk), 
            'H0': h*100*np.ones(nz_Pk), 
            'Omega_b': Omb*np.ones(nz_Pk), 
            'Omega_m': Omm*np.ones(nz_Pk), 
            'Omega_nu': Omnu*np.ones(nz_Pk),
            'As': As*np.ones(nz_Pk), 
            'gamma0': gamma0*np.ones(nz_Pk), 
            'gamma1': gamma1*np.ones(nz_Pk), 
            'q1': q1*np.ones(nz_Pk),
            'z': zz_Pk
            }
    Boost = cp_nn_gammaz.predictions_np(params_cp) #k from the training
    Boost = Boost[:, idx_emu]
    ###power law extrapolation###
    Boost_itp = [interp1d(k_modes, Boost[i], bounds_error=False,
                kind='cubic',
                fill_value=(Boost[i][0], Boost[i][-1])) for i in range(nz_Pk)]
    Boost_interp_values = np.array([Boost_itp[z](kL[index_b_interp]) for z in range(nz_Pk)], dtype=np.float64)
    Boost_last_entry = Boost_interp_values[:, -1]
    Boost_lastlast_entry = Boost_interp_values[:, -2]
    m = np.array([math.log(Boost_last_entry[i] / Boost_lastlast_entry[i]) / log_k for i in range(nz_Pk)])
    Boost_k = np.outer(Boost[:, 0], np.ones(kLbin, 'float64') )
    Boost_k[:, index_b_interp] = Boost_interp_values
    Boost_k[:, index_b_nonlin] = Boost_last_entry[:, np.newaxis] *(kL[np.newaxis, index_b_nonlin]/k_interp_last)**m[:, np.newaxis]   
    params_bacco = {
            'ns'            :  ns,
            'A_s'           :  As,
            'hubble'        :  h,
            'omega_baryon'  :  Omb,
            'omega_cold'    :  Omm,
            'neutrino_mass' :  0.,
            'w0'            :  -1.,
            'wa'            :  0.,
            'expfactor'     :  aa_Pk
        }
    _, PL = emulator.get_linear_pk(k=kL, cold=False, **params_bacco)
    integrand = W**2*PL[-1]*kL**2/(2.*np.pi**2)
    sigma8 = np.sqrt(simpson(integrand, x=kL))
    # Set HMcode internal cosmological parameters
    hmc.om_m = Omm
    hmc.om_b = Omb
    hmc.om_v = 1.-Omm
    hmc.h = h
    hmc.ns = ns
    hmc.sig8 = sigma8
    hmc.m_nu = 0.
    hmc.set_linear_power_spectrum(kL, zz_Pk, np.flip(PL,axis=0))
    PNL_LCDM = pyhmcode.calculate_nonlinear_power_spectrum(hmc, hmod, verbose=False)
    PNL = Boost_k*PNL_LCDM
    PNL_interp = RectBivariateSpline(zz_Pk,
                                kL,
                                PNL ,
                                kx=1, ky=1)
    return PNL_interp

def call_hmcode_with_boost_mu_spline(CosmoDict):
    ns   = CosmoDict['ns']
    As   = CosmoDict['As']
    h    = CosmoDict['h']
    q1    = CosmoDict['q1']
    Omm = CosmoDict['Omega_m']
    Omb = CosmoDict['Omega_b']
    #Omnu = CosmoDict['Omega_nu']

    mu1    = CosmoDict['mu1']
    mu2    = CosmoDict['mu2']
    mu3    = CosmoDict['mu3']
    mu4    = CosmoDict['mu4']
    mu5    = CosmoDict['mu5']
    mu6    = CosmoDict['mu6']
    mu7    = CosmoDict['mu7']
    mu8    = CosmoDict['mu8']
    mu9    = CosmoDict['mu9']
    mu10    = CosmoDict['mu10']
    mu11    = CosmoDict['mu11']
    q1    = CosmoDict['q1']

    params_cp = {
            'mu1': mu1*np.ones(nz_Pk), 
            'mu2': mu2*np.ones(nz_Pk), 
            'mu3': mu3*np.ones(nz_Pk), 
            'mu4': mu4*np.ones(nz_Pk), 
            'mu5': mu5*np.ones(nz_Pk), 
            'mu6': mu6*np.ones(nz_Pk), 
            'mu7': mu7*np.ones(nz_Pk), 
            'mu8': mu8*np.ones(nz_Pk), 
            'mu9': mu9*np.ones(nz_Pk), 
            'mu10': mu10*np.ones(nz_Pk), 
            'mu11': mu11*np.ones(nz_Pk), 
            'q1': q1*np.ones(nz_Pk),
            'z': zz_Pk
            }
    Boost = cp_nn_muspline.predictions_np(params_cp) #k from the training
    Boost = Boost[:, idx_emu]
    ###power law extrapolation###
    Boost_itp = [interp1d(k_modes, Boost[i], bounds_error=False,
                kind='cubic',
                fill_value=(Boost[i][0], Boost[i][-1])) for i in range(nz_Pk)]
    Boost_interp_values = np.array([Boost_itp[z](kL[index_b_interp]) for z in range(nz_Pk)], dtype=np.float64)
    Boost_last_entry = Boost_interp_values[:, -1]
    Boost_lastlast_entry = Boost_interp_values[:, -2]
    m = np.array([math.log(Boost_last_entry[i] / Boost_lastlast_entry[i]) / log_k for i in range(nz_Pk)])
    Boost_k = np.outer(Boost[:, 0], np.ones(kLbin, 'float64') )
    Boost_k[:, index_b_interp] = Boost_interp_values
    Boost_k[:, index_b_nonlin] = Boost_last_entry[:, np.newaxis] *(kL[np.newaxis, index_b_nonlin]/k_interp_last)**m[:, np.newaxis]   
    params_bacco = {
            'ns'            :  ns,
            'A_s'           :  As,
            'hubble'        :  h,
            'omega_baryon'  :  Omb,
            'omega_cold'    :  Omm,
            'neutrino_mass' :  0.,
            'w0'            :  -1.,
            'wa'            :  0.,
            'expfactor'     :  aa_Pk
        }
    _, PL = emulator.get_linear_pk(k=kL, cold=False, **params_bacco)
    integrand = W**2*PL[-1]*kL**2/(2.*np.pi**2)
    sigma8 = np.sqrt(simpson(integrand, x=kL))
    # Set HMcode internal cosmological parameters
    hmc.om_m = Omm
    hmc.om_b = Omb
    hmc.om_v = 1.-Omm
    hmc.h = h
    hmc.ns = ns
    hmc.sig8 = sigma8
    hmc.m_nu = 0.
    hmc.set_linear_power_spectrum(kL, zz_Pk, np.flip(PL,axis=0))
    PNL_LCDM = pyhmcode.calculate_nonlinear_power_spectrum(hmc, hmod, verbose=False)
    PNL = Boost_k*PNL_LCDM
    PNL_interp = RectBivariateSpline(zz_Pk,
                                kL,
                                PNL ,
                                kx=1, ky=1)
    return PNL_interp


def call_cambhmcode(CosmoDict):
    ns   = CosmoDict['ns']
    As   = CosmoDict['As']
    h    = CosmoDict['h']
    w0    = CosmoDict['w0'] if 'w0' in CosmoDict else -1.,
    wa    = CosmoDict['wa'] if 'wa' in CosmoDict else 0.,
    Omm = CosmoDict['Omega_m']
    Omb = CosmoDict['Omega_b']
    Omnu = CosmoDict['Omega_nu']
    Mnu  = CosmoDict['Mnu'] if 'Mnu' in CosmoDict else Omnu*(93.14*h**2),
    Omc = Omm-Omb-Omnu
    params_hmcode = {
            'ns'            :  ns*np.ones(nz_Pk),
            'As'            :  As*np.ones(nz_Pk),
            'hubble'        :  h*np.ones(nz_Pk),
            'omega_baryon'  :  Omb*np.ones(nz_Pk),
            'omega_cdm'     :  Omc*np.ones(nz_Pk),
            'neutrino_mass' :  Mnu*np.ones(nz_Pk),
            'w0'            :  w0*np.ones(nz_Pk),
            'wa'            :  wa*np.ones(nz_Pk),
            'z'             :  zz_Pk
        }
    PNL  = cp_nn_HMcode.ten_to_predictions_np(params_hmcode)
    PNL_interp = RectBivariateSpline(zz_Pk,
                                kh_hmcode,
                                PNL,
                                kx=1, ky=1)
    return PNL_interp

def bes_j_1(x):
    return np.sin(x)/x - np.cos(x)

cp_nn = cosmopower_NN(restore=True, 
                      restore_filename='emulators/sigma8_emu',
                      )
def compute_sigma8(points, ModelParsCosmo, Model):
    Omega_m_s8 = points[:, 0]
    h_s8 = points[:, 2]
    Omega_b_s8 = points[:, 1]/h_s8**2 
    As_s8 = np.exp(points[:, 3])*1e-10
    ns_s8 = points[:, 4]
    if 'Mnu' in ModelParsCosmo:
        Mnu_s8 = points[:, len(ModelParsCosmo)-1]
        Omega_cdmb = Omega_m_s8-Mnu_s8/93.14/h_s8**2
    else:
        Omega_cdmb = Omega_m_s8
    if ('w0' and 'wa') in ModelParsCosmo:
        w0_s8 = points[:, len(ModelParsCosmo)-3]
        wa_s8 = points[:, len(ModelParsCosmo)-2]
    elif ('wa' not in ModelParsCosmo) and ('w0' in ModelParsCosmo):
        print('implement wCDM later')    

    N = len(h_s8)
    D_t_arr = np.ones(N)
    params_bacco = {
            'ns'            :  ns_s8,
            'A_s'           :  As_s8,
            'hubble'        :  h_s8,
            'omega_baryon'  :  Omega_b_s8,
            'omega_cold'    :  Omega_cdmb,
            'neutrino_mass' :  Mnu_s8 if 'Mnu' in ModelParsCosmo else 0.,
            'w0'            :  -1.,
            'wa'            :  0.,
            'expfactor'     :  1.
        }
    sigma8_total = emulator.get_sigma8(cold=False, **params_bacco)
    #sigma8_cdb = emulator.get_sigma8(cold=True, **params_bacco)
    #k, pk_t = emulator.get_linear_pk(cold=False, **params_bacco)
    #sigma8_total_int = np.sqrt(simps((3*bes_j_1(k*8)/(k*8)**2)**2*k**2*pk_t/2/np.pi**2,k))
    #k, pk_t = emulator.get_linear_pk(cold=True, **params_bacco)
    #sigma8_cdb_int = np.sqrt(simps((3*bes_j_1(k*8)/(k*8)**2)**2*k**2*pk_t/2/np.pi**2,k))

    if Model=='HMcode':
        params_hmcode = {
            'ns'            :  ns_s8,
            'As'            :  As_s8,
            'hubble'        :  h_s8,
            'omega_baryon'  :  Omega_b_s8,
            'omega_cdm'     :  Omega_cdmb,
            'neutrino_mass' :  Mnu_s8 if 'Mnu' in ModelParsCosmo else np.ones(N),
            'w0'            :  w0_s8 if 'w0' in ModelParsCosmo else -np.ones(N),
            'wa'            :  wa_s8 if 'wa' in ModelParsCosmo else np.zeros(N),
            'z'             :  np.zeros(N)
        }
        results = cp_nn.predictions_np(params_hmcode)
        sigma8 =results[:, 0]
        return sigma8
    else:
        for i in range(N):
            #if i % 1000: print(i, '/', N)
            background ={'Omega_m': Omega_m_s8[i],#Omega_cdmb[i], 
                    'h' : h_s8[i],
                    'w0': -1.,
                    'wa': 0.,
                    'a_arr': [1.]}
            if Model=='GQ':
                cosmo1 = mg.Linder_gamma(background)   
                Da, _ = cosmo1.growth_parameters(gamma=points[i][5]) 
            elif Model=='GzQ':    
                cosmo1 = mg.Linder_gamma_a(background)   
                #Da, _ = cosmo1.growth_parameters(gamma0=points[:, 5][i], gamma1=-0.2) 
                Da, _ = cosmo1.growth_parameters(gamma0=points[i][5], gamma1=points[i][6]) 
            elif Model=='nDGP':
                cosmo1 = mg.nDGP(background)   
                Da, _ = cosmo1.growth_parameters(omegarc=10**(points[i][5])) 
            cosmo0 = mg.LCDM(background)   
            Da_LCDM, _ = cosmo0.growth_parameters() 
            D_t = Da[0]/Da_LCDM[0]
            D_t_arr[i] = D_t

        return D_t_arr * sigma8_total

    #return sigma8_cdb, sigma8_total, sigma8_cdb_int, sigma8_total_int
