
def readOff(filename):
    # Open the file
    fid = open(filename, 'r')
    if fid == -1:
        error('Can''t open the file.')
        return
    
    # Read the header
    firstLine = fid.readline()
    if not firstLine[:3] == 'OFF':
        error('The file is not a valid OFF file.')
        return
    
    # Read the first line
    N = fid.readline().split()
    nv = int(N[0])
    nf = int(N[1])
    
    # Read the vertices
    Dim = 3
    vertices = []
    for i in range(nv):
        vertices.append(fid.readline().split())
    vertices = np.array(vertices).astype(float)
    
    # Read the faces
    faces = []
    for i in range(nf):
        faces.append(fid.readline().split())
    faces = np.array(faces).astype(int)
    faces = faces[:, 1:] + 1
    
    # Close the file
    fid.close()
    
    return vertices, faces