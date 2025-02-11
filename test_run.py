import numpy as np
import yaml
import MGL

with open("config.yaml", "r") as file:
    config_dic = yaml.safe_load(file)
print(config_dic)

MGLtest = MGL.mgl("config.yaml")
MGLtest.test()
  