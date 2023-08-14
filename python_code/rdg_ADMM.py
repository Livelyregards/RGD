def rdg_ADMM(Mm, x0, **kwargs):
    """
    ADMM algorithm for computing regularized geodesic distances
    
    Inputs:
    Mm - MeshClass
    x0 - source set - vertex indices
    Optional:
        reg - which regularizer to use.
            'D' - dirichlet
            'vfa' - vector field alignment
            'H' - dirichlet
        alpha_hat - regularizer weight, scale invariant
        beta_hat - vector field alignment weight, scale-invariant, relevant for reg = 'vfa'
        vf - |F|x3 - vector field to align to, relevant for reg = 'vfa'
    
    Outputs:
    u - the computed regularized distance
    """
    
    # Parse optional arguments
    reg = kwargs.get('reg', 'D')
    alpha_hat = kwargs.get('alpha_hat', 0.1)
    beta_hat = kwargs.get('beta_hat', 0)
    vf = kwargs.get('vf', 0)
    
    # Mesh data
    vertices = Mm.vertices
    faces = Mm.faces
    nv = Mm.nv
    nf = Mm.nf
    va = Mm.va
    ta = Mm.ta
    G = Mm.G
    Ww = Mm.Ww
    tasq = np.repeat(np.sqrt(ta), 3, axis=1)
    
    # Set parameters according to the desired regularizer
    if reg == 'D':
        alpha = alpha_hat * np.sqrt(np.sum(va))
        varRho = 1 # Determine whether to use a varying penalty parameter
    elif reg == 'H':
        alpha = alpha_hat * np.sqrt(np.sum(va)**3)
        
        # This code uses the Hessian computed by 
        # "A Smoothness Energy without Boundary Distortion for Curved Surfaces" by [Stein et al., 2020]
        # Ensure that the mex `curved_hessian` is available.
        # If not available, please follow the instructions in
        # https://github.com/odedstein/ASmoothnessEnergyWithoutBoundaryDistortionForCurvedSurfaces
        Ww_s = curved_hessian(vertices, faces)
        varRho = 0 # Determine whether to use a varying penalty parameter
    elif reg == 'vfa':
        alpha = alpha_hat * np.sqrt(np.sum(va))
        beta = beta_hat * np.sqrt(np.sum(va))
        if np.max(Mm.normv(vf)) < 1e-10:
            raise ValueError('Vector field for alignment is empty')
        Vmat = np.array([[spdiags(vf[:,0]*vf[:,0], 0, nf, nf), spdiags(vf[:,0]*vf[:,1], 0, nf, nf), spdiags(vf[:,0]*vf[:,2], 0, nf, nf)],
                         [spdiags(vf[:,1]*vf[:,0], 0, nf, nf), spdiags(vf[:,1]*vf[:,1], 0, nf, nf), spdiags(vf[:,1]*vf[:,2], 0, nf, nf)],
                         [spdiags(vf[:,2]*vf[:,0], 0, nf, nf), spdiags(vf[:,2]*vf[:,1], 0, nf, nf), spdiags(vf[:,2]*vf[:,2], 0, nf, nf)]])
        Ww_s = G.T @ spdiags(np.repeat(ta, 3, axis=1), 0, 3*nf, 3*nf) @ (np.eye(3*nf, 3*nf) + beta * Vmat) @ G
        varRho = 0 # Determine whether to use a varying penalty parameter
    else:
        raise ValueError('Unrecognized regularizer')
        
    # Compute regularized distance
    u = compute_regularized_distance(Mm, x0, reg, alpha, beta, vf, varRho, Ww_s)
    
    return u


# ADMM parameters
rho = 2*np.sqrt(np.sum(va))
niter = 10000
QUIET = 1
ABSTOL = 1e-5/2
RELTOL = 1e-2
mu = 10 # >1
tauinc = 2 # >1
taudec = 2 # >1
alphak = 1.7 # over-relaxation

if reg == 'H':
    ABSTOL = ABSTOL/20
    RELTOL = RELTOL/20

thresh1 = np.sqrt(3*nf)*ABSTOL*np.sqrt(np.sum(va))
thresh2 = np.sqrt(nv)*ABSTOL*(np.sum(va))

u_p = np.zeros(nv-len(x0),1) # distance to x in M \ x0
y = np.zeros(3*nf,1) # dual variable
z = np.zeros(3*nf,1) # auxiliary variable
div_y = np.zeros(nv-len(x0),1)
div_z = np.zeros(nv-len(x0),1)

history = {'r_norm': np.zeros(niter,1),
           's_norm': np.zeros(niter,1),
           'eps_pri': np.zeros(niter,1),
           'eps_dual': np.zeros(niter,1)}

# Eliminating x0 (b.c):
nv_p = np.arange(1,nv+1)
nv_p[x0] = []
va_p = va
va_p[x0] = []
Ww_p = Ww
Ww_p[:,x0] = []
Ww_p[x0,:] = []
G_p = G
G_p[:,x0] = []
G_pt = G_p.T
div_p = G_pt*np.repeat(ta,3,1).T

if reg == 'vfa' or reg == 'H':
    Ww_s_p = Ww_s
    Ww_s_p[:,x0] = []
    Ww_s_p[x0,:] = []

# Pre-factorization
if reg == 'D':
    L,_,P = np.linalg.cholesky(Ww_p,lower=True)
else: # 'H', 'vfa'
    if not varRho:
        L,_,P = np.linalg.cholesky(alpha*Ww_s_p+rho*Ww_p,lower=True)

if not QUIET:
    print('{:3s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t'.format('iter','r norm','eps pri','s norm','eps dual'))

for ii in range(1,niter+1):
    # step 1 - u-minimization
    b = va_p-div_y+rho*div_z

    if reg == 'D':
        u_p = P * (L.T  \ (L \ (P.T * b ))) /(alpha+rho)
    else: # 'H', 'vfa'
        if not varRho:
            u_p = P * (L.T  \ (L \ (P.T * b )))
        else: # reg == 'H' and varRho
            u_p = (alpha*Ww_s_p+rho*Ww_p) \ b
    Gx = G_p*u_p

    # step 2 - z-minimization
    zold = z
    div_zold = div_z
    z = (1/rho)*y + Gx
    z = z.reshape(nf,3).T
    norm_z = np.sqrt(np.sum(z**2))
    norm_z[norm_z<1] = 1
    z = z/norm_z
    z = z.T
    z = z.flatten()
    div_z = div_p*z

    # step 3 - dual variable update
    y = y + rho*(alphak*Gx+(1-alphak)*zold-z)
    div_y = div_p*y

    # residuals update
    tasqGx = tasq*Gx
    tasqZ = tasq*z
    history['r_norm'][ii] = np.linalg.norm(tasqGx-tasqZ,'fro')
    history['s_norm'][ii] = rho*np.linalg.norm(div_z-div_zold,'fro')
    history['eps_pri'][ii] = thresh1 + RELTOL*max(np.linalg.norm(tasqGx,'fro'), np.linalg.norm(tasqZ,'fro'))
    history['eps_dual'][ii] = thresh2 + RELTOL*np.linalg.norm(div_y,'fro')

    if not QUIET:
        print('{:3d}\t{:10.4f}\t{:10.4f}\t{:10.4f}\t{:10.4f}\n'.format(ii,
            history['r_norm'][ii], history['eps_pri'][ii],
            history['s_norm'][ii], history['eps_dual'][ii]))

    # stopping criteria
    if ii>1 and (history['r_norm'][ii] < history['eps_pri'][ii] and
       history['s_norm'][ii] < history['eps_dual'][ii]):
        break

    # varying penalty parameter
    if varRho:
        if history['r_norm'][ii] > mu*history['s_norm'][ii]:
            rho = tauinc*rho
        elif history['s_norm'][ii] > mu*history['r_norm'][ii]:
            rho =  rho/taudec

u = np.zeros(nv,1)
u[nv_p] = u_p