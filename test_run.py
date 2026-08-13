import MGLensing
import os 

folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder)



#MGLtest = MGLensing.MGL("ini_files/config_files/config_hmcode.yaml")
#MGLtest = MGLensing.MGL("ini_files/pca/config_muSigma_lin.yaml")
#MGLtest = MGLensing.MGL("ini_files/config_files/config_qg.yaml")
MGLtest = MGLensing.MGL("ini_files/config_files/config_spaceborne.yaml")
MGLtest.test()

