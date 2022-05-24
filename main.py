from util import load_mesh, Edge, quad_eval, quad_sample_points, where
import numpy as np
from iwaveguide.waveguide import Waveguide
from scipy.linalg import inv
import matplotlib.pyplot as plt
import math

# Turn on interactive plotting
plt.ion()
# print(load_mesh_block("rectangular_waveguide_3d.inp", "ALLNODES"))
# print(load_mesh_block("rectangular_waveguide_3d.inp", "InputPort"))
all_nodes, tetrahedrons, tets_node_ids, all_edges, boundary_pec_edge_numbers, boundary_input_edge_numbers, boundary_output_edge_numbers, remap_edge_nums, all_edges_map, boundary_input_triangles, boundary_output_triangles = load_mesh("rectangular_waveguide_3d_less_coarse.inp")
# Initialize the K and b matrices
K = np.zeros([len(remap_edge_nums), len(remap_edge_nums)], dtype=complex)
b = np.zeros([len(remap_edge_nums)], dtype=complex)

# Get a Waveguide object
waveguide = Waveguide("rect_mesh_two_epsilons_coarse.inp", 2, [1, 1])
waveguide.solve_k0(4)
k0 = waveguide.k0
beta = waveguide.get_selected_beta()

print("Begin constructing equation matrix")
# Iterate over the tetrahedrons and construct the K and b matrices (this concept comes from Jin page 454)
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
        Cxl = a_il[3]*a_jl[2] - a_il[2]*a_jl[3]
        Ayl = a_il[0]*a_jl[2] - a_il[2]*a_jl[0]
        Byl = a_il[1]*a_jl[2] - a_il[2]*a_jl[1]
        Cyl = a_il[3]*a_jl[2] - a_il[2]*a_jl[3]
        Azl = a_il[0]*a_jl[3] - a_il[3]*a_jl[0]
        Bzl = a_il[1]*a_jl[3] - a_il[3]*a_jl[1]
        Czl = a_il[2]*a_jl[3] - a_il[3]*a_jl[2]
        # If we have an input port edge, we might need to perform the N_i integral (if this tet has a face on the port)
        if edgei in boundary_input_edge_numbers:
            # Need the third point that makes up the face of the input port
            found_edge_no = -1
            # Search for a different edge on the input port from this tetrahedral element
            for edge in tet.edges:
                if edge == edgei:
                    continue
                elif edge in boundary_input_edge_numbers:
                    # We found an edge containing the third node, note it
                    found_edge_no = edge
                    break
            # It is possible we do not find another edge that lies on the input port. We are checking for this here.
            if found_edge_no != -1:
                found_edge = all_edges[found_edge_no]
                # nodes = np.unique(np.array([node_il, node_jl, all_nodes, all_nodes[found_edge.node1], all_nodes[found_edge.node2]]))

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
            Cxk = a_ik[3] * a_jk[2] - a_ik[2] * a_jk[3]
            Ayk = a_ik[0] * a_jk[2] - a_ik[2] * a_jk[0]
            Byk = a_ik[1] * a_jk[2] - a_ik[2] * a_jk[1]
            Cyk = a_ik[3] * a_jk[2] - a_ik[2] * a_jk[3]
            Azk = a_ik[0] * a_jk[3] - a_ik[3] * a_jk[0]
            Bzk = a_ik[1] * a_jk[3] - a_ik[3] * a_jk[1]
            Czk = a_ik[2] * a_jk[3] - a_ik[3] * a_jk[2]
            # If working with input port edges, need to do different integral
            if edgei in boundary_input_edge_numbers and edgej in boundary_input_edge_numbers:
                # TODO: Implement this integral and logic
                # Perform the (n_hat x N_i) \dot (n_hat x N_j) integral
                # For n_hat = z_hat, this turns out to be the same as N_i \dot N_j, so integral is same as inhomo wg
                # Need all 3 nodes first
                # Necessary constants from NASA paper eqs. 68-72.
                pass
            # If working with output port edges, need to do different integral
            elif edgei in boundary_output_edge_numbers and edgej in boundary_output_edge_numbers:
                # TODO: Implement this integral and logic
                pass
            # Otherwise we are working with an inner edge (not on boundary). Do necessary integral.
            else:
                # Compute the curl(N_i) \dot curl(N_j) part of K_ij
                curl_dot_curl_part = edge1.length*edge2.length / 324 / tet.volume**3 * (Czl*Czk + Cxl*Cxk + Byl*Byk)

                # Build constants for N_i \dot N_j integral (part of K_ij and b_i)
                # These constants come from NASA paper eq. 182
                I1 = Axl*Axk + Ayl*Ayk + Azl*Azk
                I2 = (Ayl*Byk + Ayk*Byl + Azl*Bzk + Azk*Bzl) * x_mean
                I3 = (Axl*Bxk + Axk*Bxl + Azl*Czk + Azk*Czl) * y_mean
                I4 = (Axl*Cxk + Axk*Cxl + Ayl*Cyk + Ayk*Cyl) * z_mean
                I5 = 1/20 * (Bzl*Czk + Bzk*Czl) * (sum(xi*yi for xi, yi, zi in tet.points) + 16*x_mean*y_mean)
                I6 = 1/20 * (Bxl*Cxk + Bxk*Cxl) * (sum(yi*zi for xi, yi, zi in tet.points) + 16*y_mean*z_mean)
                I7 = 1/20 * (Byl*Cyk + Byk*Cyl) * (sum(xi*zi for xi, yi, zi in tet.points) + 16*x_mean*z_mean)
                I8 = 1/20 * (Byl*Byk + Bzl*Bzk) * (sum(xi*xi for xi, yi, zi in tet.points) + 16*x_mean*x_mean)
                I9 = 1/20 * (Bxl*Bxk + Czl*Czk) * (sum(yi*yi for xi, yi, zi in tet.points) + 16*y_mean*y_mean)
                I10 = 1/20 * (Cxl*Cxk + Cyl*Cyk) * (sum(zi*zi for xi, yi, zi in tet.points) + 16*z_mean*z_mean)
                i_sum = I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8 + I9 + I10
                dot_part = -k0**2 * tet.permittivity * edge1.length * edge2.length / 1296 / tet.volume**3 * i_sum
                K[remap_edge_nums[all_edges_map[edge1]], remap_edge_nums[all_edges_map[edge2]]] += curl_dot_curl_part + dot_part


# Iterate over the input port TriangleElement objects
for element in boundary_input_triangles:
    nodes = element.nodes
    area = element.area()
    x_mean = (all_nodes[nodes[0]][0] + all_nodes[nodes[1]][0] + all_nodes[nodes[2]][0]) / 3
    y_mean = (all_nodes[nodes[0]][1] + all_nodes[nodes[1]][1] + all_nodes[nodes[2]][1]) / 3
    # Iterate over the 3 edges of the element
    for l, k in ((0, 1), (1, 2), (2, 0)):
        # The first edge whose basis vector will be integrated against another edge (edge2)
        edge1 = Edge(nodes[l], nodes[k])
        # Skip edges on the boundary
        if all_edges_map[edge1] in boundary_pec_edge_numbers:
            continue
        # Index of the third node (the one not making up the edge) going in a CCW fashion
        m = (k + 1) % 3
        # Create the ccw node list started from the first node of the edge
        nodes_lk = (all_nodes[nodes[l]], all_nodes[nodes[k]], all_nodes[nodes[m]])
        # These constants are for the first edge (call it edge l)
        a_i_l, a_j_l = nodes_lk[1][0] * nodes_lk[2][1] - nodes_lk[2][0] * nodes_lk[1][1], nodes_lk[2][0] * nodes_lk[0][
            1] - nodes_lk[0][0] * nodes_lk[2][1]
        b_i_l, b_j_l = nodes_lk[1][1] - nodes_lk[2][1], nodes_lk[2][1] - nodes_lk[0][1]
        c_i_l, c_j_l = nodes_lk[2][0] - nodes_lk[1][0], nodes_lk[0][0] - nodes_lk[2][0]
        # These come from the edge basis function definitions (see NASA paper eqs. 51 and 56-60)
        A_l = a_i_l * b_j_l - a_j_l * b_i_l
        B_l = c_i_l * b_j_l - c_j_l * b_i_l
        C_l = a_i_l * c_j_l - a_j_l * c_i_l
        D_l = b_i_l * c_j_l - b_j_l * c_i_l

        # Perform the N_i \dot K_N integral
        sample_points = quad_sample_points(3, nodes_lk[0], nodes_lk[1], nodes_lk[2])
        # Compute the x component of the edge interpolating function for each of the sample points
        n_ix = np.zeros([len(sample_points), 1])
        n_ix[:, 0] = A_l + B_l*sample_points[:, 0]
        # Compute the y component of the edge interpolating function for each of the sample points
        n_iy = np.zeros([len(sample_points), 1])
        n_iy[:, 0] = C_l + D_l*sample_points[:, 1]
        # Get the E_inc field at each of the sample points
        field_x = np.zeros([len(sample_points), 1])
        field_y = np.zeros([len(sample_points), 1])
        for n, point in enumerate(sample_points):
            field_x[n, 0], field_y[n, 0], _ = waveguide.get_field_at(point[0], point[1])
        # Compute the dot product at each point
        values = n_ix * field_x + n_iy * field_y
        integral = quad_eval(nodes_lk[0], nodes_lk[1], nodes_lk[2], values)
        b[remap_edge_nums[all_edges_map[edge1]]] += integral

        # Iterate over the 3 edges of the element (p stands for prime here, see Jin's book pg 455)
        # This is being done for the neumann B.C. integral
        for l_p, k_p in ((0, 1), (1, 2), (2, 0)):
            # The second edge whose basis vector will be integrated against the other edge (edge1)
            edge2 = Edge(nodes[l_p], nodes[k_p])
            # Skip edges on the boundary
            if all_edges_map[edge2] in boundary_pec_edge_numbers:
                continue
            # Index of the third node (the one not making up the edge) going in a CCW fashion
            m_p = (k_p + 1) % 3
            # Create the ccw node list started from the first node of the edge
            nodes_lpkp = (all_nodes[nodes[l_p]], all_nodes[nodes[k_p]], all_nodes[nodes[m_p]])
            # Constants needed to calculate the integral involving the 2 edges
            # These come from the nodal basis function definitions (see Jin's book pg 441)
            # These constants are for the second edge (call it edge k)
            a_i_k, a_j_k = nodes_lpkp[1][0] * nodes_lpkp[2][1] - nodes_lpkp[2][0] * nodes_lpkp[1][1], nodes_lpkp[2][0] * \
                           nodes_lpkp[0][1] - nodes_lpkp[0][0] * nodes_lpkp[2][1]
            b_i_k, b_j_k = nodes_lpkp[1][1] - nodes_lpkp[2][1], nodes_lpkp[2][1] - nodes_lpkp[0][1]
            c_i_k, c_j_k = nodes_lpkp[2][0] - nodes_lpkp[1][0], nodes_lpkp[0][0] - nodes_lpkp[2][0]
            # These come from the edge basis function definitions (see NASA paper eqs. 51 and 56-60)
            A_k = a_i_k * b_j_k - a_j_k * b_i_k
            B_k = c_i_k * b_j_k - c_j_k * b_i_k
            C_k = a_i_k * c_j_k - a_j_k * c_i_k
            D_k = b_i_k * c_j_k - b_j_k * c_i_k
            # See An Introduction to the finite element method 3rd edition page 426 and NASA paper eqs. 73-77
            I1 = A_l * A_k + C_l * C_k
            I2 = (C_l * D_k + C_k * D_l) * x_mean
            I3 = (A_l * B_k + A_k * B_l) * y_mean
            I4 = B_l * B_k * (sum([node[1] ** 2 for node in nodes_lk]) + 9 * y_mean ** 2) / 12
            I5 = D_l * D_k * (sum([node[0] ** 2 for node in nodes_lk]) + 9 * x_mean ** 2) / 12
            # Ni_bar dot Nj_bar - NASA paper pg 12
            value = edge1.length * edge2.length * (I1 + I2 + I3 + I4 + I5) / 16 / (area ** 3)
            K[remap_edge_nums[all_edges_map[edge1]], remap_edge_nums[all_edges_map[edge2]]] += value * beta * 1j

# Iterate over the output port TriangleElement objects
for element in boundary_output_triangles:
    nodes = element.nodes
    area = element.area()
    x_mean = (all_nodes[nodes[0]][0] + all_nodes[nodes[1]][0] + all_nodes[nodes[2]][0]) / 3
    y_mean = (all_nodes[nodes[0]][1] + all_nodes[nodes[1]][1] + all_nodes[nodes[2]][1]) / 3
    # Iterate over the 3 edges of the element
    for l, k in ((0, 1), (1, 2), (2, 0)):
        # The first edge whose basis vector will be integrated against another edge (edge2)
        edge1 = Edge(nodes[l], nodes[k])
        # Skip edges on the boundary
        if all_edges_map[edge1] in boundary_pec_edge_numbers:
            continue
        # Index of the third node (the one not making up the edge) going in a CCW fashion
        m = (k + 1) % 3
        # Create the ccw node list started from the first node of the edge
        nodes_lk = (all_nodes[nodes[l]], all_nodes[nodes[k]], all_nodes[nodes[m]])
        # These constants are for the first edge (call it edge l)
        a_i_l, a_j_l = nodes_lk[1][0] * nodes_lk[2][1] - nodes_lk[2][0] * nodes_lk[1][1], nodes_lk[2][0] * nodes_lk[0][
            1] - nodes_lk[0][0] * nodes_lk[2][1]
        b_i_l, b_j_l = nodes_lk[1][1] - nodes_lk[2][1], nodes_lk[2][1] - nodes_lk[0][1]
        c_i_l, c_j_l = nodes_lk[2][0] - nodes_lk[1][0], nodes_lk[0][0] - nodes_lk[2][0]
        # These come from the edge basis function definitions (see NASA paper eqs. 51 and 56-60)
        A_l = a_i_l * b_j_l - a_j_l * b_i_l
        B_l = c_i_l * b_j_l - c_j_l * b_i_l
        C_l = a_i_l * c_j_l - a_j_l * c_i_l
        D_l = b_i_l * c_j_l - b_j_l * c_i_l

        # Iterate over the 3 edges of the element (p stands for prime here, see Jin's book pg 455)
        # This is being done for the neumann B.C. integral
        for l_p, k_p in ((0, 1), (1, 2), (2, 0)):
            # The second edge whose basis vector will be integrated against the other edge (edge1)
            edge2 = Edge(nodes[l_p], nodes[k_p])
            # Skip edges on the boundary
            if all_edges_map[edge2] in boundary_pec_edge_numbers:
                continue
            # Index of the third node (the one not making up the edge) going in a CCW fashion
            m_p = (k_p + 1) % 3
            # Create the ccw node list started from the first node of the edge
            nodes_lpkp = (all_nodes[nodes[l_p]], all_nodes[nodes[k_p]], all_nodes[nodes[m_p]])
            # Constants needed to calculate the integral involving the 2 edges
            # These come from the nodal basis function definitions (see Jin's book pg 441)
            # These constants are for the second edge (call it edge k)
            a_i_k, a_j_k = nodes_lpkp[1][0] * nodes_lpkp[2][1] - nodes_lpkp[2][0] * nodes_lpkp[1][1], nodes_lpkp[2][0] * \
                           nodes_lpkp[0][1] - nodes_lpkp[0][0] * nodes_lpkp[2][1]
            b_i_k, b_j_k = nodes_lpkp[1][1] - nodes_lpkp[2][1], nodes_lpkp[2][1] - nodes_lpkp[0][1]
            c_i_k, c_j_k = nodes_lpkp[2][0] - nodes_lpkp[1][0], nodes_lpkp[0][0] - nodes_lpkp[2][0]
            # These come from the edge basis function definitions (see NASA paper eqs. 51 and 56-60)
            A_k = a_i_k * b_j_k - a_j_k * b_i_k
            B_k = c_i_k * b_j_k - c_j_k * b_i_k
            C_k = a_i_k * c_j_k - a_j_k * c_i_k
            D_k = b_i_k * c_j_k - b_j_k * c_i_k
            # See An Introduction to the finite element method 3rd edition page 426 and NASA paper eqs. 73-77
            I1 = A_l * A_k + C_l * C_k
            I2 = (C_l * D_k + C_k * D_l) * x_mean
            I3 = (A_l * B_k + A_k * B_l) * y_mean
            I4 = B_l * B_k * (sum([node[1] ** 2 for node in nodes_lk]) + 9 * y_mean ** 2) / 12
            I5 = D_l * D_k * (sum([node[0] ** 2 for node in nodes_lk]) + 9 * x_mean ** 2) / 12
            # Ni_bar dot Nj_bar - NASA paper pg 12
            value = edge1.length * edge2.length * (I1 + I2 + I3 + I4 + I5) / 16 / (area ** 3)
            K[remap_edge_nums[all_edges_map[edge1]], remap_edge_nums[all_edges_map[edge2]]] += value * beta * 1j

print("Finished constructing equation matrix")
print("Solving equation matrix")
edge_coefficients = np.dot(inv(K), b)
print("Finished solving equation matrix")

# ----------------------GET FIELDS--------------------------
print("Calculating field data")
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
# z_points = np.linspace(z_min, z_max, num_z_points)
# For now, just get the fields at z_min
z_points = np.array([z_min])
Ex = np.zeros([num_x_points, num_y_points, num_z_points])
Ey = np.zeros([num_x_points, num_y_points, num_z_points])
Ez = np.zeros([num_x_points, num_y_points, num_z_points])

field_points = np.zeros([num_x_points * num_y_points * num_z_points, 3])
# Iterate over the points
for i in range(num_z_points):
    pt_z = z_points[i]
    for j in range(num_y_points):
        pt_y = y_points[j]
        for k in range(num_x_points):
            pt_x = x_points[k]
            field_points[k + j*num_y_points + i*num_z_points] = np.array([pt_x, pt_y, pt_z])

tet_indices = where(all_nodes, tets_node_ids, field_points)

# Compute the field at each of the points
for i, tet_index in enumerate(tet_indices):
    tet = tetrahedrons[tet_index]
    phis = [edge_coefficients[remap_edge_nums[edge]] if edge in remap_edge_nums else 0 for edge in tet.edges]
    ex, ey, ez = tet.interpolate(phis, field_points[i])
    z_i = math.floor(i / (num_x_points * num_y_points)) % num_z_points
    y_i = math.floor(i / num_x_points) % num_y_points
    x_i = i % num_x_points
    # Note the indexing here is done with y_i first and x_i second. If we consider a grid being indexed, the first
    # index corresponds to the row (vertical control), hence y_i first and x_i second
    Ex[y_i, x_i, z_i], Ey[y_i, x_i, z_i], Ez[y_i, x_i, z_i] = tet.interpolate(phis, field_points[i])
print("Finished calculating field data")

plt.figure()
color_image = plt.imshow(Ez[:, :, 0], extent=[x_min, x_max, y_min, y_max], cmap="cividis")
plt.colorbar(label="Ez")
X, Y = np.meshgrid(x_points, y_points)
skip = (slice(None, None, 5), slice(None, None, 5))
field_skip = (slice(None, None, 5), slice(None, None, 5), 0)
plt.quiver(X[skip], Y[skip], Ex[field_skip], Ey[field_skip], color="black")
