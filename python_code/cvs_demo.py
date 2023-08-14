import os
import sys
import numpy as np
import cvxpy as cp

sys.path.append('code/')
sys.path.append('data/')

# add cvx to path
cvx_folder = '/path/to/cvx' # update to cvx_folder
sys.path.append(cvx_folder)

filename = 'bent_pipe_closed_lr'
x0 = 4883 # source point / set

# load mesh
Mm = MeshClass(filename)
Mm.center_mesh()

alpha_hat = 0.03 # regularizer weight

alpha = alpha_hat*np.sqrt(np.sum(Mm.va))

# Regularized - Dirichlet Energy
u_D = cp.Variable(Mm.nv)
grad = cp.reshape(Mm.G*u_D, Mm.nf, 3)

objective = cp.Maximize(cp.sum(Mm.Va * u_D) - alpha*cp.quad_form(u_D, Mm.Ww))
constraints = [u_D[x0] == 0, cp.norms(grad, 2, 2) <= 1]

cp.Problem(objective, constraints).solve()

# Regularized - Linf
u_Linf = cp.Variable(Mm.nv)
grad = cp.reshape(Mm.G*u_Linf, Mm.nf, 3)

objective = cp.Maximize(cp.sum(Mm.Va * u_Linf) - alpha*cp.sum(Mm.ta * cp.pow_pos(cp.norms(grad, np.inf, 2), 2)))
constraints = [u_Linf[x0] == 0, cp.norms(grad, 2, 2) <= 1]

cp.Problem(objective, constraints).solve()

# rotate mesh
A = np.rotz(45)
VV = Mm.vertices*A
Mm_rot = MeshClass(VV, Mm.faces)

u_Linf_rot = cp.Variable(Mm.nv)
grad = cp.reshape(Mm_rot.G*u_Linf_rot, Mm_rot.nf, 3)

objective = cp.Maximize(cp.sum(Mm_rot.Va * u_Linf_rot) - alpha*cp.sum(Mm_rot.ta * cp.pow_pos(cp.norms(grad, np.inf, 2), 2)))
constraints = [u_Linf_rot[x0] == 0, cp.norms(grad, 2, 2) <= 1]

cp.Problem(objective, constraints).solve()

# Figures
u_all = np.concatenate((u_D.value, u_Linf.value, u_Linf_rot.value))
umin = np.min(u_all)
umax = np.max(u_all)
nlines = 15

Mm.visualizeDistances(u_D.value, x0, nlines, [umin, umax])
Mm.visualizeDistances(u_Linf.value, x0, nlines, [umin, umax])
Mm_rot.visualizeDistances(u_Linf_rot.value, x0, nlines, [umin, umax])