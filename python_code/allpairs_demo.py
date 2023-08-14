import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join('code'))
sys.path.append(os.path.join('data'))

from MeshClass import MeshClass

filename = 'cat_rr'

# load mesh
Mm = MeshClass(filename)

alpha_hat = 0.03
U = rdg_allpairs_admm(Mm, alpha_hat)

# Figures
umin = np.min(U)
umax = np.max(U)
nlines = 15

x0 = 494
Mm.visualizeDistances(U[x0, :], x0, nlines, [umin, umax])

x0 = 242
Mm.visualizeDistances(U[x0, :], x0, nlines, [umin, umax])