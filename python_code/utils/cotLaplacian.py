

def cotLaplacian(mesh, L23 = None, L13 = None, L12 = None):
    X = mesh.vertices
    T = mesh.faces
    nv = len(X)
    
    inputL = 0
    if L23 is not None:
        inputL += 1
    if L13 is not None:
        inputL += 1
    if L12 is not None:
        inputL += 1
        
    if inputL < 3:
        # Find orig edge lengths and angles
        L1 = np.linalg.norm(X[T[:,2]] - X[T[:,3]], axis = 1)
        L2 = np.linalg.norm(X[T[:,1]] - X[T[:,3]], axis = 1)
        L3 = np.linalg.norm(X[T[:,1]] - X[T[:,2]], axis = 1)
    else:
        L1 = L23
        L2 = L13
        L3 = L12
    
    EL = np.array([L1, L2, L3])
    A1 = (L2**2 + L3**2 - L1**2) / (2 * L2 * L3)
    A2 = (L1**2 + L3**2 - L2**2) / (2 * L1 * L3)
    A3 = (L1**2 + L2**2 - L3**2) / (2 * L1 * L2)
    A = np.array([A1, A2, A3])
    A = np.arccos(A)
    
    # The Cot Laplacian 
    I = np.concatenate([T[:,1], T[:,2], T[:,3]])
    J = np.concatenate([T[:,2], T[:,3], T[:,1]])
    S = 0.5 * np.cot([A[:,3], A[:,1], A[:,2]])
    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([-S, -S, S, S])
    W = sparse.csr_matrix((Sn, (In, Jn)), shape = (nv, nv))
    
    if inputL < 3:
        # Use the Barycentric areas
        M = mass_matrix_barycentric(mesh)
        A = np.sum(M, axis = 1)
    else:
        M = mass_matrix_barycentric(mesh, L1, L2, L3)
        A = np.sum(M, axis = 1)
        
    return W, A

def normv(V):
    nn = np.sqrt(np.sum(V**2, axis = 1))
    return nn

def mass_matrix_barycentric(mesh, L1 = None, L2 = None, L3 = None):
    T = mesh.faces
    inputL = 0
    if L1 is not None:
        inputL += 1
    if L2 is not None:
        inputL += 1
    if L3 is not None:
        inputL += 1
        
    if inputL < 3:
        Ar = mesh.ta
    else:
        s = (L1 + L2 + L3) / 2
        Ar = np.sqrt(s * (s - L1) * (s - L2) * (s - L3))
        
    nv = mesh.nv
    
    I = np.concatenate([T[:,1], T[:,2], T[:,3]])
    J = np.concatenate([T[:,2], T[:,3], T[:,1]])
    Mij = 1/12 * np.concatenate([Ar, Ar, Ar])
    Mji = Mij
    Mii = 1/6 * np.concatenate([Ar, Ar, Ar])
    In = np.concatenate([I, J, I])
    Jn = np.concatenate([J, I, I])
    Mn = np.concatenate([Mij, Mji, Mii])
    M = sparse.csr_matrix((Mn, (In, Jn)), shape = (nv, nv))
    
    return M