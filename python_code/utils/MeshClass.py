
import numpy as np
from scipy import sparse


class MeshClass:
    def __init__(self, *args):
        if len(args[0]) >= 3 and len(args[0][0]) == 3:
            X = args[0]
            T = args[1]
            if len(args) > 2:
                self.name = args[2]
        else:
            try:
                X, T = readOff(args[0] + '.off')
            except:
                raise Exception('Problem reading file')
            T = double(T)
            if args[0].rfind('/') == -1:
                self.name = args[0]
            else:
                self.name = args[0][args[0].rfind('/')+1:]
        self.vertices = X
        self.faces = T
        self.compute_all()

    def compute_all(obj):
            obj.nv = len(obj.vertices)
            obj.nf = len(obj.faces)
            Nf = np.cross(obj.vertices[obj.faces[:,1],:] - obj.vertices[obj.faces[:,2],:], 
                          obj.vertices[obj.faces[:,1],:] - obj.vertices[obj.faces[:,3],:])
            obj.Nf = MeshClass.normalize_vf(Nf)
            obj.ta = np.sqrt(np.sum(Nf**2, axis=1))/2
            obj.Nv = vertex_normals(obj)
            obj.Va = obj.calculatefvConnectivity().T * obj.ta / 3
            obj.E1 = obj.vertices[obj.faces[:,2],:] - obj.vertices[obj.faces[:,3],:]
            obj.E2 = obj.vertices[obj.faces[:,1],:] - obj.vertices[obj.faces[:,3],:]
            obj.E3 = obj.vertices[obj.faces[:,1],:] - obj.vertices[obj.faces[:,2],:]
            obj.R = rot(obj)
            EE = np.sort([obj.faces[:,1], obj.faces[:,2], obj.faces[:,2], obj.faces[:,3], obj.faces[:,3], obj.faces[:,1]], axis=1)
            obj.E = np.unique(EE, axis=0)
            obj.edges, obj.e2t, obj.t2e, obj.e2t1, obj.e2t2, obj.v2e, obj.ie, obj.ne, obj.nie, obj.inner_edges, obj.bv, obj.bf = nc_data(obj)
            obj.ea = edge_areas(obj)
            obj.edge_basis()
            obj.compute_LB()
            obj.GG()
            obj.DD()
            
    def interpulateFace2Vertices(mesh, fF=None):
        Af = mesh.ta
        Av = mesh.Va
        
        I_F2V = sparse([mesh.faces[:,1], mesh.faces[:,2], mesh.faces[:,3]], 
                        [(i for i in range(mesh.nf)), (i for i in range(mesh.nf)), (i for i in range(mesh.nf))], 
                        [(1/3)*Af/Av[mesh.faces[:,1]], (1/3)*Af/Av[mesh.faces[:,2]], (1/3)*Af/Av[mesh.faces[:,3]]])
        
        if fF is not None:
            fv = I_F2V*fF
        else:
            fv = []
        return fv, I_F2V
    
    def interpulateVertices2Face(mesh, fV=None):
        Afinv = sparse(range(mesh.nf), range(mesh.nf), 1/mesh.ta)
        Av = sparse(range(mesh.nv), range(mesh.nv), mesh.va)
        _, I_F2V = mesh.interpulateFace2Vertices(np.ones(mesh.nf,1))
        I_V2F = Afinv*I_F2V.T*Av
        if fV is not None:
            fF = I_V2F*fV
        else:
            fF = []
        return fF, I_V2F


    def edge_basis(mesh):
        NE1 = MeshClass.normalize_vf(mesh.E1)
        NE2 = np.reshape(mesh.R * NE1.flatten(), [], 3)
        
        I = np.repeat(np.arange(1, mesh.nf+1), 3)
        J = np.concatenate((np.arange(1, mesh.nf+1), np.arange(mesh.nf+1, 2*mesh.nf+1), np.arange(2*mesh.nf+1, 3*mesh.nf+1)))
        
        B1 = sparse.coo_matrix((NE1, (I, J)), shape=(mesh.nf, 3*mesh.nf))
        B2 = sparse.coo_matrix((NE2, (I, J)), shape=(mesh.nf, 3*mesh.nf))
        
        EB = np.vstack((B1, B2))
        EBI = EB.transpose()
        
        mesh.F1 = NE1
        mesh.F2 = NE2
        mesh.EB = EB
        mesh.EBI = EBI
        
        
    def rot(mesh):
        sf = mesh.nf
        n = mesh.Nf
        
        II = np.repeat(np.arange(1, sf+1), 6)
        JJ1 = np.arange(1, sf+1)
        JJ2 = JJ1 + sf
        JJ3 = JJ2 + sf
        JJ = np.concatenate((JJ2, JJ3, JJ1, JJ3, JJ1, JJ2))
        SS = np.concatenate((-n[:,3], n[:,2], n[:,3], -n[:,1], -n[:,2], n[:,1]))
        
        R = sparse.coo_matrix((SS, (II, JJ)), shape=(3*sf, 3*sf))
        
        
    def calculatefvConnectivity(obj):
        fvConnectivity = sparse.coo_matrix((np.ones(size(obj.faces)), (np.repeat(np.arange(1, obj.nf+1), 3), obj.faces)), shape=(obj.nf, obj.nv))
        
        
    def baryCentersCalc(mesh):
        v1 = mesh.vertices[mesh.faces[:,0],:]
        v2 = mesh.vertices[mesh.faces[:,1],:]
        v3 = mesh.vertices[mesh.faces[:,2],:]
        baryCenters = (1/3)*(v1+v2+v3)


    def edge_areas(mesh):
        T = np.double(mesh.faces)
        I = np.concatenate((T[:,1], T[:,2], T[:,3]))
        J = np.concatenate((T[:,2], T[:,3], T[:,1]))
        S = np.tile(mesh.ta/3, 3)
        In = np.concatenate((I, J))
        Jn = np.concatenate((J, I))
        Sn = np.concatenate((S, S))
        W = sparse.coo_matrix((Sn, (In, Jn)), shape=(mesh.nv, mesh.nv)).tocsr()
        ea = np.zeros(len(mesh.edges))
        for i in range(len(ea)):
            ea[i] = W[mesh.edges[i,0], mesh.edges[i,1]]
            ea[i] = ea[i] + W[mesh.edges[i,1], mesh.edges[i,0]]
        return ea
    
    def rotate_vf(mesh, vf):
        vf = np.reshape(vf, (mesh.nf, 3))
        rvf = np.cross(mesh.Nf, vf)
        return rvf
    
    def compute_LB(mesh):
        mesh.Ww = cotLaplacian(mesh)
        laplacian = sparse.diags(1/mesh.va, 0) * mesh.Ww
        mesh.Lap = laplacian
        mesh.Aa = sparse.diags(mesh.va, 0)
    
    def GG(mesh):
        I = np.tile(np.arange(mesh.nf), 3)
        II = np.concatenate((I, I+mesh.nf, I+2*mesh.nf))
        J = np.double(mesh.faces.T)
        JJ = np.concatenate((J, J, J))
        RE1 = rotate_vf(mesh, mesh.E1)
        RE2 = rotate_vf(mesh, mesh.E2)
        RE3 = rotate_vf(mesh, mesh.E3)
        TA = mesh.ta
        S = np.concatenate((-RE1.flatten(), RE2.flatten(), -RE3.flatten()))
        G = sparse.coo_matrix((S, (II, JJ)), shape=(3*mesh.nf, mesh.nv)).tocsr()
        ITA = sparse.diags(.5*np.tile(1/TA, 3), 0)
        grad_op = ITA*G
        if np.any(np.isnan(grad_op.data)):
            print('Grad: NANs exist')
        grad_op.data[np.isnan(grad_op.data)] = 0
        mesh.G = grad_op
        return grad_op
    
    def DD(mesh):
      IVA = spdiags(1./mesh.va,0,mesh.nv,mesh.nv)
      TA = spdiags([mesh.ta; mesh.ta; mesh.ta],0,3*mesh.nf,3*mesh.nf)
      D = - IVA * mesh.G' * TA
      mesh.D = D
      
    def vertex_normals(mesh):
        I = np.repeat(mesh.faces(:),3,1)
        J = np.repeat(np.arange(1,3),3*mesh.nf,1)
        J = J(:)
        
        TA = spdiags([mesh.ta; mesh.ta; mesh.ta],0,3*mesh.nf,3*mesh.nf)
        S = np.repeat(TA(1:mesh.nf,1:mesh.nf)*mesh.Nf,3,1)
        S = S(:)
        
        Nv = full(sparse(I,J,S,mesh.nv,3))
        Nv = MeshClass.normalize_vf(Nv)
        
    def nc_data(mesh):
        T = np.double(mesh.faces)
        
        I = [T(:,2);T(:,3);T(:,1)]
        J = [T(:,3);T(:,1);T(:,2)]
        S = [1:mesh.nf,1:mesh.nf,1:mesh.nf]
        E = sparse(I,J,S,mesh.nv,mesh.nv)
        
        Elisto = [I,J]
        sElist = np.sort(Elisto,2)
        s = (MeshClass.normv(Elisto - sElist) > 1e-12)
        t = S'.*(-1).^s
        [edges,une] = np.unique(sElist, 'rows')
        
        ne = np.size(edges,1)
        e2t = np.zeros((ne,4))
        t2e = np.zeros((mesh.nf,3))
        ie = np.zeros((ne,1))
        for m in range(len(edges)):
            i = edges(m,1)
            j = edges(m,2)
            t1 = t(une(m))
            t2 = -(E(i,j) + E(j,i) - abs(t1))*np.sign(t1)
            e2t(m,1:2) = [t1, t2]
            f = T(abs(t1),:)
            loc = np.find(f == (np.sum(f) - i - j))
            t2e(abs(t1),loc) = m*np.sign(t1)
            e2t(m,3) = loc
            if t2 != 0:
                f = T(abs(t2),:)
                loc = np.find(f == (np.sum(f) - i - j))
                t2e(abs(t2),loc) = m*np.sign(t2)
                e2t(m,4) = loc
                ie(m) = 1
                
        v2e = sparse(edges(:,1),edges(:,2),1:length(edges),mesh.nv,mesh.nv)
        
        ne = np.size(edges,1)
        nie = np.sum(ie)
        inner_edges = np.find(ie)
        bv = np.zeros((mesh.nv,1))
        bv(edges(ie == 0,:)) = 1
        bf = np.zeros((mesh.nf,1))
        bf(np.sum(np.ismember(mesh.faces, np.find(bv==1)),2)>0) = 1
        
        t1 = np.abs(e2t(inner_edges,1))
        t2 = np.abs(e2t(inner_edges,2))
        
        I = np.arange(1,2*nie+1)
        S = np.ones((2*nie,1))
        e2t1 = sparse(I, [t1; t1+mesh.nf], S, 2*nie, 2*mesh.nf)
        e2t2 = sparse(I, [t2; t2+mesh.nf], S, 2*nie, 2*mesh.nf)

    def normalize_mesh(mesh, bbdO=1):
            xx = mesh.vertices - mean(mesh.vertices)
            bbd = norm(max(mesh.vertices) - min(mesh.vertices))
            xx = mesh.vertices / bbd * bbdO
            mesh.vertices = xx
            mesh.compute_all()
            
    def center_mesh(mesh, aa=None):
        if aa is not None:
            xx = mesh.vertices - aa
        else:
            xx = mesh.vertices - mean(mesh.vertices)
            aa = mean(mesh.vertices)
        mesh.vertices = xx
        mesh.compute_all()
        
    def scale_mesh(mesh, scale_fac):
        xx = mesh.vertices * scale_fac
        mesh.vertices = xx
        mesh.compute_all()
        
    def visualizeMesh(mesh, f_vertices=[], f_faces=[], edgeColorFlag=0, figFlag=1):
        if figFlag:
            figure()
        p = patch('Faces',mesh.faces,'Vertices',mesh.vertices)
        if f_vertices and not f_faces:
            p.FaceVertexCData = f_vertices
            p.FaceColor = 'interp'
            colorbar()
        elif f_faces and not f_vertices:
            p.FaceVertexCData = f_faces
            p.FaceColor = 'flat'
            colorbar()
        else:
            p.FaceColor = 'w'
            p.FaceAlpha = 0.5
            title(mesh.name)
        if edgeColorFlag == 1:
            p.EdgeColor = 'none'
        p.FaceAlpha = 1
        axis('equal')
        axis('off')
        cameratoolbar()
        cameratoolbar('SetCoordSys','none')


    def vectorFieldVisualization(mesh, vectorField, vectorFieldPos=[], f_vertices=[], f_faces=[], edgeColorFlag=0):
        if len(vectorField[1]) == 1:
            vectorField = reshape(vectorField, [], 3)
        if len(vectorFieldPos) == 0:
            vectorFieldPos = mesh.baryCentersCalc
        p = mesh.visualizeMesh(f_vertices, f_faces, edgeColorFlag)
        hold on
        quiver3(vectorFieldPos[:,1], vectorFieldPos[:,2], vectorFieldPos[:,3], vectorField[:,1], vectorField[:,2], vectorField[:,3])
        
    def vectorFieldVisualization2(mesh, vectorField1, vectorField2, vectorFieldPos=[], f_vertices=[], f_faces=[], edgeColorFlag=0):
        p = mesh.visualizeMesh(f_vertices, f_faces, edgeColorFlag)
        hold on
        quiver3(vectorFieldPos[:,1], vectorFieldPos[:,2], vectorFieldPos[:,3], vectorField1[:,1], vectorField1[:,2], vectorField1[:,3], 'b')
        quiver3(vectorFieldPos[:,1], vectorFieldPos[:,2], vectorFieldPos[:,3], vectorField2[:,1], vectorField2[:,2], vectorField2[:,3], 'r')
        
    def visualizeDistances(mesh, u, x0, nisolines=0, urange=[], cam=[]):
        if len(urange) == 0:
            urange = [min(u), max(u)]
        p = mesh.visualizeMesh(u, [], 1, 1)
        caxis([urange[1], urange[2]])
        hold on
        scatter3(mesh.vertices[x0,1], mesh.vertices[x0,2], mesh.vertices[x0,3], 'r', 'filled')
        if nisolines > 0:
            colormap(lines(2*(nisolines-1)))
            colormap jet
        else:
            colormap jet
        if len(cam) != 0:
            MeshClass.set_camera(gca, cam)
        l = camlight
        lighting phong
        material dull

    def godf(mesh, n):
        M = mesh
        
        inner = M.inner_edges
        
        t1 = abs(M.e2t(inner,1))
        t2 = abs(M.e2t(inner,2))
        oe = -np.ones(M.nie,1)
        ze = np.zeros(M.nie,1)
        
        # Connection-specific computations
        EV = M.vertices[M.edges[inner,2],:] - M.vertices[M.edges[inner,1],:]
        EV = MeshClass.normalize_vf(EV)
        
        IN1 = np.arctan2(np.dot(EV,M.F2[t1,:],2),np.dot(EV,M.F1[t1,:],2))
        IN2 = np.arctan2(np.dot(EV,M.F2[t2,:],2),np.dot(EV,M.F1[t2,:],2))
        PT = n*(IN2-IN1)
        
        II = np.repeat((np.arange(1,M.nie+1)),2,axis=0)
        II = np.repeat(np.vstack((II,II+M.nie)),2,axis=0)
        JJ = np.hstack((t1,t1+M.nf,t1,t1+M.nf,t2,t2+M.nf,t2,t2+M.nf))
        SS = np.hstack((np.cos(PT),-np.sin(PT),np.sin(PT),np.cos(PT),oe,ze,ze,oe))
        CovD = sparse.coo_matrix((SS,(II,JJ)),shape=(2*M.nie,2*M.nf))
        
        Ws = sparse.diags(np.repeat(np.sqrt(M.ea[inner]),2,axis=0))
        oph = Ws * CovD
        op = oph.T * oph
        
        return op, oph
    

    @staticmethod
    def normv(vf):
        nv = np.sqrt(np.sum(np.square(vf), axis=1))
        return nv

    @staticmethod
    def normalize_vf(vf):
        nnv = vf / np.reshape(MeshClass.normv(vf), (-1, 1))
        nnv[MeshClass.normv(vf) < 1e-15, :] = 0
        return nnv

    @staticmethod
    def get_camera(ca=None):
        if ca is None:
            ca = plt.gca()
        cam = {
            'pba': ca.get_plotboxaspectratio(),
            'dar': ca.get_dataaspectratio(),
            'cva': ca.get_cameraviewangle(),
            'cuv': ca.get_cameraupvector(),
            'ct': ca.get_cameratarget(),
            'cp': ca.get_cameraposition()
        }
        return cam

    @staticmethod
    def set_camera(ca, cam):
        ca.set_plotboxaspectratio(cam['pba'])
        ca.set_dataaspectratio(cam['dar'])
        ca.set_cameraviewangle(cam['cva'])
        ca.set_cameraupvector(cam['cuv'])
        ca.set_cameratarget(cam['ct'])
        ca.set_cameraposition(cam['cp'])

