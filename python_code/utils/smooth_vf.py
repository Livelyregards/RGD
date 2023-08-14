import numpy as np
from scipy import sparse

from MeshClass import MeshClass


def smooth_vf(Mm, vf, n):
    nf = Mm.nf
    we = []
    
    locs = np.where(MeshClass.normv(vf) > 1e-5)
    nl = locs.shape[0]
    
    Aeq = sparse.csr_matrix((np.ones(2*nl), (np.arange(2*nl), np.concatenate((locs, nf+locs)))), shape=(2*nl, 2*nf))
    
    # -> local basis -> power n -> locs
    beq = np.reshape(ff(Mm.EB*vf.flatten(),n), (-1,2))[locs,:]
    beq = beq.flatten()
    
    C = Mm.godf(n)
    vf = np.zeros(nf*2)
    
    # If there are no constraints, use the eigenvector
    if Aeq.shape[0] > 0:
        x = lsqlin(C, vf, [], [], Aeq, beq)
    else:
        x,_ = eigs(C, 1, 'SM')
    
    # -> sqrt n -> extrinsic
    w = np.reshape(Mm.EBI*ff(x,1/n), (-1,3))
    # Normalize
    w = MeshClass.normalize_vf(w)
    
    return w
    
    # in: 2n x 1
    # out: 2n x 1, f(x,n) = x.^n
    # grad: 2n x 2n

def ff(input,n):
    s = input.shape[0]/2
    a = input[:s]
    b = input[s:]
    c = a+1j*b
    cn = c**n
    e = np.concatenate((np.real(cn), np.imag(cn)))
    return e