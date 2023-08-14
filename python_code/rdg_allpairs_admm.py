def rdg_allpairs_admm(Mm, alpha_hat):

    # ADMM algorithm for computing regularized all-pairs geodesic distances
    #   regularized with the Dirichlet regularizer
    #
    # Inputs:
    #   Mm - MeshClass
    #   alpha_hat - regularizer weight
    #
    # Outputs:
    #   U - the computed all-pairs regularized distance


    # mesh data
    nv = Mm.nv
    nf = Mm.nf
    Va = Mm.Va
    ta = Mm.ta
    G = Mm.G
    Ww = Mm.Ww
    vasq = np.sqrt(Va)
    VaMat = np.diag(Va)
    vainv = 1/Va
    vavatMat = Va*Va.T
    tasq = np.tile(np.sqrt(ta), (3,1))
    taMat = np.diag(np.tile(ta, (3,1)))
    div = G.T*taMat
        
    alpha = alpha_hat*np.sqrt(np.sum(Va))


    # ADMM parameters
    rho     = 2*np.sqrt(np.sum(Va))
    rho2    = 10/np.sqrt(np.sum(Va))
    niter   = 20000
    QUIET   = 0
    ABSTOL  = 1e-6
    RELTOL  = 2e-4
    mu      = 10  # >1
    tauinc  = 2   # >1
    taudec  = 2   # >1
    alphak  = 1.7 # over-relaxation

    thresh1 = np.sqrt(3*nf)*ABSTOL*(np.sum(Va))
    thresh2 = np.sqrt(nv)*ABSTOL*(np.sum(Va))**2
    thresh3 = np.sqrt(nv)*ABSTOL*np.sqrt(np.sum(Va))**3
    thresh4 = np.sqrt(nv)*ABSTOL*(np.sum(Va))

    varRho = 1 # determine whether to use a varying penalty parameter
    rho1or2changed = 1


    # initialization:
    X = np.zeros((nv,nv))       # all-pairs distances, represent the gradient along the columns
    R = np.zeros((nv,nv))       # all-pairs distances, represent the gradient along the rows
    U = np.zeros((nv,nv))       # dual consensus variable, all-pairs distance matrix
    Z = np.zeros((3*nf,nv))     # auxiliary variable, GX = Z
    Q = np.zeros((3*nf,nv))     # auxiliary variable, GR = Q
    Y = np.zeros((3*nf,nv))     # dual variable
    S = np.zeros((3*nf,nv))     # dual variable
    H = np.zeros((nv,nv))       # dual consensus variable
    K = np.zeros((nv,nv))       # dual consensus variable
    div_Z = sparse.csr_matrix((nv,nv))
    div_Q = sparse.csr_matrix((nv,nv))
    div_Y = sparse.csr_matrix((nv,nv))
    div_S = sparse.csr_matrix((nv,nv))


    history.r_norm = np.zeros(niter,1)
    history.s_norm = np.zeros(niter,1)
    history.eps_pri = np.zeros(niter,1)
    history.eps_dual = np.zeros(niter,1)
    history.r_norm2 = np.zeros(niter,1)
    history.s_norm2 = np.zeros(niter,1)
    history.eps_pri2 = np.zeros(niter,1)
    history.eps_dual2 = np.zeros(niter,1)
    history.r_xr1 = np.zeros(niter,1)
    history.r_xr2 = np.zeros(niter,1)
    history.s_xr = np.zeros(niter,1)
    history.eps_pri_xr = np.zeros(niter,1)
    history.eps_dual_xr = np.zeros(niter,1)

    if not QUIET:
        print('{:3s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}'.format(
            'iter', 'r norm', 'eps pri', 's norm', 'eps dual','r2 norm', 'eps2 pri', 's2 norm', 'eps2 dual', 'xr1 r norm', 'xr1 eps pri', 'xr2 r norm', 'xr2 eps pri', 'xr s norm', 'xr eps dual'))

    # Pre-factorization
    if not varRho:
        A2in_p = (alpha+rho)*Ww+rho2*VaMat
        L,_,P = np.linalg.cholesky(A2in_p, lower=True)

    for ii in range(niter):

        # step 1 - X,R-minimization
        bx = (0.5*vavatMat*vainv - div_Y + rho*div_Z - va*H + rho2*va*U )
        br = (0.5*vavatMat*vainv - div_S + rho*div_Q - va*K + rho2*va*U.T )

        if varRho and rho1or2changed:
            A2in_p = (alpha+rho)*Ww+rho2*vaMat
            L,g,P = np.linalg.cholesky(A2in_p, lower=True)
        
        X = P @ (np.linalg.solve(L.T, np.linalg.solve(L, P.T @ bx )))
        R = P @ (np.linalg.solve(L.T, np.linalg.solve(L, P.T @ br )))

        Gx = G@X
        Gr = G@R
        
        # step 2 - Z,Q,U-minimization:
        Zold = Z
        div_Zold = div_Z
        Z = (1/rho)*Y + Gx
        Z = Z.reshape(nf,3,nv) 
        Z = Z/np.maximum(1,np.sqrt(np.sum(Z**2,axis=1)))
        Z = Z.reshape(3*nf,nv)
        div_Z = div@Z
        
        Qold = Q
        div_Qold = div_Q
        Q = (1/rho)*S + Gr
        Q = Q.reshape(nf,3,nv) 
        Q = Q/np.maximum(1,np.sqrt(np.sum(Q**2,axis=1)))
        Q = Q.reshape(3*nf,nv) 
        div_Q = div@Q

        Uold = U
        U1 = 0.5*((1/rho2)*(H+K.T) + X+R.T)
        U = U1-np.diag(np.diag(U1))
        U[U<0] = 0

        # step 3 - dual variable update
        Y = Y + rho*(alphak*Gx+(1-alphak)*Zold-Z)
        S = S + rho*(alphak*Gr+(1-alphak)*Qold-Q)
        H = H + rho2*(alphak*X+(1-alphak)*Uold-U)
        K = K + rho2*(alphak*R+(1-alphak)*Uold.T-U.T)
        div_Y = div@Y
        div_S = div@S

        # residuals update
        GxW = tasq*Gx*vasq.T
        ZW =  tasq*Z*vasq.T
        history['r_norm'][ii]  = np.linalg.norm(GxW-ZW, 'fro')
        history['eps_pri'][ii]  = thresh1 + RELTOL*max(np.linalg.norm(GxW, 'fro'), np.linalg.norm(ZW, 'fro'))
        history['s_norm'][ii]  = rho*np.linalg.norm((div_Z - div_Zold)*va.T, 'fro')
        history['eps_dual'][ii] = thresh2 + RELTOL*np.linalg.norm(div_Y*va.T, 'fro')
        
        GrW = tasq*Gr*vasq.T
        QW =  tasq*Q*vasq.T
        history['r_norm2'][ii]  = np.linalg.norm(GrW-QW, 'fro')
        history['eps_pri2'][ii]  = thresh1 + RELTOL*max(np.linalg.norm(GrW, 'fro'), np.linalg.norm(QW, 'fro'))
        history['s_norm2'][ii]  = rho*np.linalg.norm((div_Q - div_Qold)*va.T, 'fro')
        history['eps_dual2'][ii] = thresh2 + RELTOL*np.linalg.norm(div_S*va.T, 'fro')
        
        history['r_xr1'][ii] = np.linalg.norm(vasq*(X-U)*vasq.T, 'fro')
        history['r_xr2'][ii] = np.linalg.norm(vasq*(R-U.T)*vasq.T, 'fro')
        history['eps_pri_xr1'][ii] = thresh3 + RELTOL*min(np.linalg.norm(vasq*X*vasq.T, 'fro'), np.linalg.norm(vasq*U*vasq.T, 'fro'))
        history['eps_pri_xr2'][ii] = thresh3 + RELTOL*min(np.linalg.norm(vasq*R*vasq.T, 'fro'), np.linalg.norm(vasq*U.T*vasq.T, 'fro'))
        history['s_xr'][ii] = np.sqrt(2)*rho2*np.linalg.norm(vasq*(U-Uold)*vasq.T, 'fro')
        history['eps_dual_xr'][ii] = thresh4 + RELTOL*0.5*(np.linalg.norm(vasq*H*vasq.T, 'fro')+np.linalg.norm(vasq*K*vasq.T, 'fro'))

    if not QUIET and not ii % 10:
            print('{:3d} |{:10.4f}  {:10.4f} |{:10.4f}  {:10.4f} |{:10.4f}  {:10.4f} |{:10.4f}  {:10.4f} |{:10.4f}  {:10.4f} |{:10.4f}  {:10.4f} |{:10.4f}  {:10.4f}'.format(ii, 
                history.r_norm[ii], history.eps_pri[ii], 
                history.s_norm[ii], history.eps_dual[ii], 
                history.r_norm2[ii], history.eps_pri2[ii], 
                history.s_norm2[ii], history.eps_dual2[ii], 
                history.r_xr1[ii], history.eps_pri_xr1[ii], 
                history.r_xr2[ii], history.eps_pri_xr2[ii], 
                history.s_xr[ii], history.eps_dual_xr[ii]))
                
        if (history.r_norm[ii] < history.eps_pri[ii] and 
        history.s_norm[ii] < history.eps_dual[ii] and 
        history.r_norm2[ii] < history.eps_pri2[ii] and 
        history.s_norm2[ii] < history.eps_dual2[ii] and 
        history.r_xr1[ii] < history.eps_pri_xr1[ii] and 
        history.r_xr2[ii] < history.eps_pri_xr2[ii] and 
        history.s_xr[ii] < history.eps_dual_xr[ii]):
            break
            
        # varying penalty parameter
        if varRho:
            if (history.r_norm[ii]/history.eps_pri[ii] > mu*history.s_norm[ii]/history.eps_dual[ii]) and \
                (history.r_norm2[ii]/history.eps_pri2[ii] > mu*history.s_norm2[ii]/history.eps_dual2[ii]):
                rho = tauinc*rho
                rho1or2changed = 1
            elif (history.s_norm[ii]/history.eps_dual[ii] > mu*history.r_norm[ii]/history.eps_pri[ii]) and \
                    (history.s_norm2[ii]/history.eps_dual2[ii] > mu*history.r_norm2[ii]/history.eps_pri2[ii]):
                rho =  rho/taudec
                rho1or2changed = 1
            if (history.r_xr1[ii]/history.eps_pri_xr1[ii] > mu*history.s_xr[ii]/history.eps_dual_xr[ii]) and \
                    (history.r_xr2[ii]/history.eps_pri_xr2[ii] > mu*history.s_xr[ii]/history.eps_dual_xr[ii]):
                rho2 = tauinc*rho2
                rho1or2changed = 1
            elif (history.s_xr[ii]/history.eps_dual_xr[ii] > mu*history.r_xr1[ii]/history.eps_pri_xr1[ii]) and \
                    (history.s_xr[ii]/history.eps_dual_xr[ii] > mu*history.r_xr2[ii]/history.eps_pri_xr2[ii]):
                rho2 =  rho2/taudec
                rho1or2changed = 1

