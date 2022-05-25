# File to test out little pieces of code before using them in main.py
import time
# alternatively, import cupy as np if len(points)>1e7 and GPU
import numpy as np
from util import quad_eval, quad_sample_points
import math


def foo(x, y, z):
    """
    Test function for quadrature
    :param x: The x value to evaluate the function at.
    :param y: The y value to evaluate the function at.
    :param z: The z value to evaluate the function at.
    :return: The value of the function at a particular point.
    """
    return 2 - x - 2*y


x1, y1, z1 = 1, 2, 3
x2, y2, z2 = 1, 2, 3
x3, y3, z3 = 1, 2, 3
x4, y4, z4 = 1, 2, 3
init_array = np.array([[1, x1, y1, z1],
                       [1, x2, y2, z2],
                       [1, x3, y3, z3],
                       [1, x4, y4, z4]], dtype=float)

all_cofactors = np.zeros([4, 4])
# Iterate over each row
for row in range(4):
    cofactors = np.zeros([4])
    # Iterate over each column, computing the cofactor determinant of the row + column combination
    for col in range(4):
        # Compute the cofactor (remove the proper row and column and compute the determinant)
        cofactors[col] = np.linalg.det(np.delete(np.delete(init_array, row, axis=0), col, axis=1))
    all_cofactors[row] = cofactors

n = 3
# p1 = [-1, 0, 0]
# p2 = [1, 0, 0]
# p3 = [0, 1, 0]
p1 = [0, 0, 0]
p2 = [1, 1/2, 0]
p3 = [0, 1, 0]
sample_points = quad_sample_points(n, p1, p2, p3)
values = np.zeros([len(sample_points), 1])
for i in range(len(sample_points)):
    values[i, 0] = foo(sample_points[i, 0], sample_points[i, 1], sample_points[i, 2])
# The result of this should be 1/3 (and has been verified to be this)
result = quad_eval(p1, p2, p3, values)

# waveguide_2d = Waveguide("rect_mesh_two_epsilons_coarse.inp", 2, [1, 1])

# betas, all_eigenvectors, k0s = waveguide_2d.solve()
# waveguide_2d.plot_dispersion(k0s, betas)


def Tetrahedron(vertices):
    """
    Given a list of the xyz coordinates of the vertices of a tetrahedron,
    return tetrahedron coordinate system
    """
    origin, *rest = vertices
    mat = (np.array(rest) - origin).T
    tetra = np.linalg.inv(mat)
    return tetra, origin


def point_inside(point, tetra, origin):
    """
    Takes a single point or array of points, as well as tetra and origin objects returned by
    the Tetrahedron function.
    Returns a boolean or boolean array indicating whether the point is inside the tetrahedron.
    """
    newp = np.matmul(tetra, (point-origin).T).T
    return np.all(newp>=0, axis=-1) & np.all(newp <=1, axis=-1) & (np.sum(newp, axis=-1) <=1)


npt=100
points = np.random.rand(npt, 3)
# Coordinates of vertices A, B, C and D
A=np.array([0.1, 0.1, 0.1])
B=np.array([0.9, 0.2, 0.1])
C=np.array([0.1, 0.9, 0.1])
D=np.array([0.3, 0.3, 0.9])
# A point that is inside the above tet:
pts = np.array([[0.2, 0.2, 0.09], [0.2, 0.2, 0.15]])

start_time = time.time()
vertices = [A, B, C, D]
tetra, origin = Tetrahedron(vertices)
inTet = point_inside(pts, tetra, origin)
print("--- %s seconds ---" % (time.time() - start_time))
# print(point_inside(pt, tetra, origin))


def where(node_coordinates, node_ids, p):
    ori=node_coordinates[node_ids[:,0],:]
    v1=node_coordinates[node_ids[:,1],:]-ori
    v2=node_coordinates[node_ids[:,2],:]-ori
    v3=node_coordinates[node_ids[:,3],:]-ori
    n_tet=len(node_ids)
    v1r=v1.T.reshape((3,1,n_tet))
    v2r=v2.T.reshape((3,1,n_tet))
    v3r=v3.T.reshape((3,1,n_tet))
    mat = np.concatenate((v1r,v2r,v3r), axis=1)
    inv_mat = np.linalg.inv(mat.T).T    # https://stackoverflow.com/a/41851137/12056867
    if p.size==3:
        p=p.reshape((1,3))
    n_p=p.shape[0]
    orir=np.repeat(ori[:,:,np.newaxis], n_p, axis=2)
    newp=np.einsum('imk,kmj->kij',inv_mat,p.T-orir)
    val=np.all(newp>=0, axis=1) & np.all(newp <=1, axis=1) & (np.sum(newp, axis=1)<=1)
    id_tet, id_p = np.nonzero(val)
    res = -np.ones(n_p, dtype=id_tet.dtype) # Sentinel value
    res[id_p]=id_tet
    return res


all_nodes = np.array(vertices)
node_ids_test = np.array([[0, 1, 2, 3]])
output = where(all_nodes, node_ids_test, pts)


# Test the barycentric constants are working properly

# Compute the simplex (barycentric) constants for the nodes of this TetrahedralElement
# Each row is for a node. Each column is for a, b, c, and d (in order) from NASA paper eq. 162
# These are stored in the same order as self.nodes (i.e. simplex_consts[0] are the constants for self.nodes[0])
all_cofactors = np.zeros([4, 4])
points = np.array([A, B, C, D])
# Iterate over each row
negate = 1
for row in range(4):
    cofactors = np.zeros([4])
    # Iterate over each column, computing the cofactor determinant of the row + column combination
    for col in range(4):
        # Compute the cofactor (remove the proper row and column and compute the determinant)
        if (row + col) % 2 == 0:
            negate = 1
        else:
            negate = -1
        cofactors[col] = negate * np.linalg.det(np.delete(np.delete(np.append(np.ones([4, 1]), points, 1), row, axis=0), col, axis=1))
    all_cofactors[row] = cofactors
simplex_consts = all_cofactors

full_mat = np.append(np.ones([4, 1]), points, 1)
volume = abs(1/6 * np.linalg.det(full_mat))
# This point lies in the tetrahedron, test it
test_point = [0.2, 0.2, 0.15]
full_mat[0, 1:] = test_point
v1 = abs(1/6 * np.linalg.det(full_mat))

for i in range(4):
    alpha1 = (simplex_consts[i, 0] + simplex_consts[i, 1]*test_point[0] + simplex_consts[i, 2]*test_point[1] + simplex_consts[i, 3]*test_point[2]) / 6 / volume
    print(alpha1)

print("actual alpha1:")
print(v1/volume)

a, b, d = 1, 0.5, 0.75
m, n, p = 1, 0, 1
k0 = math.sqrt((m*math.pi/a)**2 + (n*math.pi/b)**2 + (p*math.pi/d)**2)
