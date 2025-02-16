import numpy as np
import yaml
import MGL
import matplotlib.pyplot as plt
import os

os.environ["OMP_NUM_THREADS"] = "1"
MGLtest = MGL.mgl("config.yaml")
MGLtest.test()

#MGLtest.nautilus(n_live_points=1000,n_pools=1)

