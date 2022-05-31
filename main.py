from util import load_mesh, Edge, quad_eval, quad_sample_points, where
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
import math
from scipy.linalg import eig
from scipy import sparse
from scipy.sparse.linalg import eigs
import time


# Turn on interactive plotting
plt.ion()
# print(load_mesh_block("rectangular_waveguide_3d.inp", "ALLNODES"))
# print(load_mesh_block("rectangular_waveguide_3d.inp", "InputPort"))
print("Loading data from mesh")
start_time = time.time()
all_nodes, tetrahedrons, tets_node_ids, all_edges, boundary_pec_edge_numbers, remap_edge_nums, all_edges_map, = load_mesh("rectangular_waveguide_3d_less_coarse_pec.inp")
# all_nodes, tetrahedrons, tets_node_ids, all_edges, boundary_pec_edge_numbers, remap_edge_nums, all_edges_map, = load_mesh("rectangular_waveguide_3d_fine_pec.inp", pec_walls_name="EB2")
# all_nodes, tetrahedrons, tets_node_ids, all_edges, boundary_pec_edge_numbers, remap_edge_nums, all_edges_map, = load_mesh("test_somewhat_fine_mesh.inp")
print(f"Finished loading data from mesh in {time.time() - start_time} seconds")
# Initialize the K and b matrices
S = np.zeros([len(remap_edge_nums), len(remap_edge_nums)])
T = np.zeros([len(remap_edge_nums), len(remap_edge_nums)])

print("Begin constructing equation matrix")
start_time = time.time()
# Iterate over the tetrahedrons and construct the S and T matrices (this concept comes from Jin page 454)
for tet in tetrahedrons:

    # Compute the x-mean, y-mean, and z-mean of the points that make up the tetrahedron (needed later)
    x_mean = np.average(tet.points[:, 0])
    y_mean = np.average(tet.points[:, 1])
    z_mean = np.average(tet.points[:, 2])

    # Iterate over the edges of the tetrahedron
    for edgei in tet.edges:
        # Skip over PEC walls
        if edgei in boundary_pec_edge_numbers:
            continue
        # Get a hold of the Edge object
        edge1 = all_edges[edgei]
        # Get the nodes that make up this edge
        node_il, node_jl = all_nodes[edge1.node1], all_nodes[edge1.node2]

        indices_l = [np.argwhere(tet.nodes == edge1.node1)[0][0], np.argwhere(tet.nodes == edge1.node2)[0][0]]
        # The simplex constants for nodes i and j of edge l
        # Necessary constants from NASA paper eqs. 163-172
        a_il, a_jl = tet.simplex_consts[indices_l]
        Axl = a_il[0]*a_jl[1] - a_il[1]*a_jl[0]
        Bxl = a_il[2]*a_jl[1] - a_il[1]*a_jl[2]
        # Change from a_jl[2] to a_jl[1], a_il[2] to a_il[1]
        Cxl = a_il[3]*a_jl[1] - a_il[1]*a_jl[3]
        Ayl = a_il[0]*a_jl[2] - a_il[2]*a_jl[0]
        Byl = a_il[1]*a_jl[2] - a_il[2]*a_jl[1]
        Cyl = a_il[3]*a_jl[2] - a_il[2]*a_jl[3]
        Azl = a_il[0]*a_jl[3] - a_il[3]*a_jl[0]
        Bzl = a_il[1]*a_jl[3] - a_il[3]*a_jl[1]
        Czl = a_il[2]*a_jl[3] - a_il[3]*a_jl[2]

        # Iterate over the edges of the tetrahedron
        for edgej in tet.edges:
            # Skip over PEC walls
            if edgej in boundary_pec_edge_numbers:
                continue
            edge2 = all_edges[edgej]
            node_ik, node_jk = all_nodes[edge1.node1], all_nodes[edge1.node2]

            # Find the indices of the edge of interest
            indices_k = [np.argwhere(tet.nodes == edge2.node1)[0][0], np.argwhere(tet.nodes == edge2.node2)[0][0]]
            # The simplex constants for nodes i and j of edge l
            a_ik, a_jk = tet.simplex_consts[indices_k]
            # Necessary constants from NASA paper eqs. 163-172
            Axk = a_ik[0] * a_jk[1] - a_ik[1] * a_jk[0]
            Bxk = a_ik[2] * a_jk[1] - a_ik[1] * a_jk[2]
            # Cxk = a_ik[3] * a_jk[2] - a_ik[2] * a_jk[3]
            # Change from a_jk[2] to a_jk[1], a_ik[2] to a_ik[1]
            Cxk = a_ik[3] * a_jk[1] - a_ik[1] * a_jk[3]
            Ayk = a_ik[0] * a_jk[2] - a_ik[2] * a_jk[0]
            Byk = a_ik[1] * a_jk[2] - a_ik[2] * a_jk[1]
            Cyk = a_ik[3] * a_jk[2] - a_ik[2] * a_jk[3]
            Azk = a_ik[0] * a_jk[3] - a_ik[3] * a_jk[0]
            Bzk = a_ik[1] * a_jk[3] - a_ik[3] * a_jk[1]
            Czk = a_ik[2] * a_jk[3] - a_ik[3] * a_jk[2]

            # Compute the curl(N_i) \dot curl(N_j) part of K_ij
            curl_dot_curl_part = edge1.length*edge2.length / 324 / tet.volume**3 * (Czl*Czk + Cxl*Cxk + Byl*Byk)
            S[remap_edge_nums[all_edges_map[edge1]], remap_edge_nums[all_edges_map[edge2]]] += curl_dot_curl_part

            # Build constants for N_i \dot N_j integral (part of K_ij and b_i)
            # These constants come from NASA paper eq. 182
            I1 = Axl*Axk + Ayl*Ayk + Azl*Azk
            I2 = (Ayl*Byk + Ayk*Byl + Azl*Bzk + Azk*Bzl) * x_mean
            I3 = (Axl*Bxk + Axk*Bxl + Azl*Czk + Azk*Czl) * y_mean
            I4 = (Axl*Cxk + Axk*Cxl + Ayl*Cyk + Ayk*Cyl) * z_mean
            I5 = 1/20 * (Bzl*Czk + Bzk*Czl) * (sum([xi*yi for xi, yi, zi in tet.points]) + 16*x_mean*y_mean)
            I6 = 1/20 * (Bxl*Cxk + Bxk*Cxl) * (sum([yi*zi for xi, yi, zi in tet.points]) + 16*y_mean*z_mean)
            I7 = 1/20 * (Byl*Cyk + Byk*Cyl) * (sum([xi*zi for xi, yi, zi in tet.points]) + 16*x_mean*z_mean)
            I8 = 1/20 * (Byl*Byk + Bzl*Bzk) * (sum([xi*xi for xi, yi, zi in tet.points]) + 16*x_mean*x_mean)
            I9 = 1/20 * (Bxl*Bxk + Czl*Czk) * (sum([yi*yi for xi, yi, zi in tet.points]) + 16*y_mean*y_mean)
            I10 = 1/20 * (Cxl*Cxk + Cyl*Cyk) * (sum([zi*zi for xi, yi, zi in tet.points]) + 16*z_mean*z_mean)
            i_sum = I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10
            dot_part = tet.permittivity * edge1.length * edge2.length / 1296 / tet.volume**3 * i_sum
            T[remap_edge_nums[all_edges_map[edge1]], remap_edge_nums[all_edges_map[edge2]]] += dot_part

print(f"Finished constructing equation matrix in {time.time() - start_time} seconds")
print("Solving eigenvalue problem")
start_time = time.time()
sS = sparse.csr_matrix(S)
sT = sparse.csr_matrix(T)
eigenvalues, eigenvectors = eigs(sS, k=100, M=sT, which='SR')
# eigenvalues, eigenvectors = eig(S, T, right=True)
# Take the transpose such that each row of the matrix now corresponds to an eigenvector (helpful for sorting)
eigenvectors = eigenvectors.transpose()
# Prepare to sort the eigenvalues and eigenvectors by propagation constant
p = np.argsort(eigenvalues)
# All the eigenvalues should have no imaginary component. If they do, something is wrong. Still need this.
eigenvalues = np.real(eigenvalues[p])
eigenvectors = np.real(eigenvectors[p])
# Find the first positive propagation constant
first_pos = np.argwhere(eigenvalues >= 0)[0, 0]
k0s, eigenvectors = np.sqrt(eigenvalues[first_pos:]), eigenvectors[first_pos:]
print(f"Finished solving eigenvalue problem in {time.time() - start_time} seconds")
# ----------------------GET FIELDS--------------------------
print("Calculating field data")
start_time = time.time()
# Compute the bounds of the waveguide
x_min = np.amin(all_nodes[:, 0])
x_max = np.amax(all_nodes[:, 0])
y_min = np.amin(all_nodes[:, 1])
y_max = np.amax(all_nodes[:, 1])
z_min = np.amin(all_nodes[:, 2])
z_max = np.amax(all_nodes[:, 2])
# Create a cuboid grid of points that the geometry is inscribed in
num_x_points = 100
x_points = np.linspace(x_min, x_max, num_x_points)
num_y_points = 100
y_points = np.linspace(y_min, y_max, num_y_points)
num_z_points = 1
z_points = np.linspace(z_min+(0.75-0.1), z_max, num_z_points)
# For now, just get the fields at z_min
# z_points = np.array([z_min+0.2])
Ex = np.zeros([num_y_points, num_x_points, num_z_points])
Ey = np.zeros([num_y_points, num_x_points, num_z_points])
Ez = np.zeros([num_y_points, num_x_points, num_z_points])

field_points = np.zeros([num_x_points * num_y_points * num_z_points, 3])
# Iterate over the points
for i in range(num_z_points):
    pt_z = z_points[i]
    for j in range(num_y_points):
        pt_y = y_points[j]
        for k in range(num_x_points):
            pt_x = x_points[k]
            field_points[k + j*num_y_points + i*num_z_points] = np.array([pt_x, pt_y, pt_z])
            # field_points[k + j*num_x_points + i*num_y_points*num_x_points] = np.array([pt_x, pt_y, pt_z])

tet_indices = where(all_nodes, tets_node_ids, field_points)
first = np.where(k0s >= 0.1)[0][0]

# The mode of interest, in increasing k0 value (4/5 -> TM111/TE111)
mode = 4

# Compute the field at each of the points
for i, tet_index in enumerate(tet_indices):
    tet = tetrahedrons[tet_index]
    phis = [eigenvectors[first + mode, remap_edge_nums[edge]] if edge in remap_edge_nums else 0 for edge in tet.edges]
    z_i = math.floor(i / (num_x_points * num_y_points)) % num_z_points
    y_i = math.floor(i / num_x_points) % num_y_points
    x_i = i % num_x_points
    # Note the indexing here is done with y_i first and x_i second. If we consider a grid being indexed, the first
    # index corresponds to the row (vertical control), hence y_i first and x_i second
    Ex[y_i, x_i, z_i], Ey[y_i, x_i, z_i], Ez[y_i, x_i, z_i] = tet.interpolate(phis, field_points[i])
print(f"Finished calculating field data in {time.time() - start_time} seconds")

plt.figure()
color_image = plt.imshow(Ez[:, :, 0], extent=[x_min, x_max, y_min, y_max], cmap="cividis")
# color_image = plt.imshow(Ex[0, :, :], extent=[y_min, y_max, z_min, z_max], cmap="cividis")
plt.colorbar(label="Ez")
X, Y = np.meshgrid(x_points, y_points)
# Y, Z = np.meshgrid(y_points, z_points)
skip = (slice(None, None, 5), slice(None, None, 5))
field_skip = (slice(None, None, 5), slice(None, None, 5), 0)
# field_skip = (0, slice(None, None, 5), slice(None, None, 5))
plt.quiver(X[skip], Y[skip], Ex[field_skip], Ey[field_skip], color="black")

# Try a 3d quiver plot:
# ax = plt.figure().add_subplot(projection='3d')
# x, y, z = np.meshgrid(x_points, y_points, z_points)
# ax.quiver(x, y, z, Ex, Ey, Ez, length=0.05, normalize=True)
