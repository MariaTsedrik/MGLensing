import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import MGLensing


MGLtest = MGLensing.MGL("config.yaml")
head = []
def read_last_header_line(file_path):
    last_header = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                last_header = line.strip('#').strip() 
                head.append(last_header+'\n') 
            else:
                break  
    if last_header:
        return last_header.split(), head 
    else:
        return []
# read the file
file_name = 'chains/chaintest_hmcode_3x2pt_b1b2_nobar'
file_path = file_name+'.txt'
chain = np.genfromtxt(file_path)
chain = np.delete(chain, np.where(chain[:, -1]==-np.inf), axis=0)
len_chain = len(chain)
chain_pars, head = read_last_header_line(file_path)
chain_pars_new = chain_pars[:-2] + ["sigma8", "S8"] + chain_pars[-2:]
head[-1] = "    ".join(chain_pars_new)


chain_info = {}
for i in range(len(chain_pars)):
    chain_info[chain_pars[i]] = chain[:, i]
# check that all required parameters are present
# for sigma8-emulator we need 'omega_baryon', 'omega_cdm', 'neutrino_mass', 'w0',  'wa'   
# 'ns',  'As', and 'h'.   
pars = {
        'Omega_m': chain_info['Omega_m'],
        'Omega_c': chain_info['Omega_m']-chain_info['Ombh2']/chain_info['h']**2,
        'Omega_b': chain_info['Ombh2']/chain_info['h']**2, 
        'h':chain_info['h'],
        'As': np.exp(chain_info['log10As'])*1e-10,
        'ns':chain_info['ns'],
        'Mnu': np.zeros(len_chain),
        'w0': np.full(len_chain, -1.),
        'wa': np.zeros(len_chain)
        # modified gravity and dark energy parameters
        # insert: 
        # ...
        }
#get sigma8
sigma8 = MGLtest.get_sigma8_from_a_s_from_chain(pars, nl_model=0)
S8 = sigma8*np.sqrt(chain_info['Omega_m']/0.3)
new_chain = np.hstack((chain[:, :-2], sigma8[:, np.newaxis], S8[:, np.newaxis], chain[:, -2:]))
np.savetxt(file_name+'_w_S8.txt', new_chain, fmt="%s", header="".join(head))
