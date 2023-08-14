

import os
import numpy as np
import MeshClass

# add paths
os.chdir('code/')
os.system('addpath(genpath(''code/''))')
os.chdir('../data/')
os.system('addpath(''data/'')')

# load mesh
filename = 'spot_rr'
Mm = MeshClass(filename)

# Regularized - Dirichlet Energy
alpha_hat0 = 0.05 # scale invariant, represents the weight of the regularizer
                  # for Dirichlet regularizer - the size of the smoothing area
x0 = 2273 # source point / set

u_D1 = rdg_ADMM(Mm, x0, alpha_hat = alpha_hat0)
u0 = rdg_ADMM(Mm, x0, alpha_hat = 0) # No regularization
u_D2 = rdg_ADMM(Mm, x0, alpha_hat = 3*alpha_hat0) # Higher Regularization - Dirichlet Energy

# Regularized - Vector Field Alignment
# given directions:
given_vf_faces = [4736, 2703]
given_vf_vals = [[1.6256, -0.3518, -0.6234], [1.6952, 0.3193, 0.0335]]

vf = np.zeros((Mm.nf,3))
vf[given_vf_faces,:] = given_vf_vals

# interpolate vf to mesh
vf_int = smooth_vf(Mm, vf, 2)

# Optionally, scale the interpolated line field with a geodesic Gaussian
localize_vf = 1
if localize_vf:
    vf_faces_v = Mm.faces[given_vf_faces,:]
    vf_faces_v = vf_faces_v.reshape(-1)
    dist_to_vf_faces = rdg_ADMM(Mm, vf_faces_v, alpha_hat = 0)
    sigma2 = sum(Mm.ta)/10**2
    dist_vf_gaus = np.exp(-dist_to_vf_faces**2/(2*sigma2))

    vf_int = Mm.interpulateVertices2Face(dist_vf_gaus)*vf_int

# regularizers
alpha_hat = 0.05
beta_hat = 100

u_vfa = rdg_ADMM(Mm, x0, reg = 'vfa', alpha_hat = alpha_hat, beta_hat = beta_hat, vf = vf_int)

# Figures
cam = np.load('spot_rr_cam.mat')
cam = cam['cam']
u_all = np.concatenate((u0.reshape(-1), u_D1.reshape(-1), u_D2.reshape(-1), u_vfa.reshape(-1)))
umin = np.min(u_all)
umax = np.max(u_all)
nlines = 15

Mm.visualizeDistances(u0, x0, nlines, [umin, umax], cam)
Mm.visualizeDistances(u_D1, x0, nlines, [umin, umax], cam)
Mm.visualizeDistances(u_D2, x0, nlines, [umin, umax], cam)

Mm.visualizeDistances(u_vfa, x0, nlines, [umin, umax], cam)
br = Mm.baryCentersCalc
plt.quiver(br[:,0],br[:,1],br[:,2], vf[:,0],vf[:,1],vf[:,2],2, color='k', linewidth=2, arrow_length_ratio=0)
plt.quiver(br[:,0],br[:,1],br[:,2], -vf[:,0],-vf[:,1],-vf[:,2],2, color='k', linewidth=2, arrow_length_ratio=0)