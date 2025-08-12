import MGLensing
import os 

folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder)


MGLtest = MGLensing.MGL("config_lssty10.yaml")
MGLtest.test()

