from util import load_mesh, Edge, quad_eval, quad_sample_points, where
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from math import floor, e
from scipy.linalg import eig
from scipy import sparse
from scipy.sparse.linalg import eigs
import time

# Turn on interactive plotting
plt.ion()


class Cavity:

    def __init__(self, filename):
        """
        Constructor for Cavity.
        :param filename: The path to the file to load.
        """
        print("Loading data from mesh")
        start_time = time.time()
        self.all_nodes, self.tetrahedrons, self.tets_node_ids, self.all_edges, self.boundary_pec_edge_numbers, self.remap_edge_nums, self.all_edges_map, = load_mesh(filename)
        print(f"Finished loading data from mesh in {time.time() - start_time} seconds")
        self.x_min = np.amin(self.all_nodes[:, 0])
        self.x_max = np.amax(self.all_nodes[:, 0])
        self.y_min = np.amin(self.all_nodes[:, 1])
        self.y_max = np.amax(self.all_nodes[:, 1])
        self.z_min = np.amin(self.all_nodes[:, 2])
        self.z_max = np.amax(self.all_nodes[:, 2])
        self.k0s = np.array([])
        self.eigenvectors = np.array([])

    def solve(self):
        """
        Solve the cavity.
        :return: The sorted, positive k0s and the corresponding eigenvectors.
        """
        print("Begin constructing equation matrix")
        start_time = time.time()
        S = np.zeros([len(self.remap_edge_nums), len(self.remap_edge_nums)])
        T = np.zeros([len(self.remap_edge_nums), len(self.remap_edge_nums)])
        # Iterate over the tetrahedrons and construct the S and T matrices (this concept comes from Jin page 454)
        for tet in self.tetrahedrons:

            # Compute the x-mean, y-mean, and z-mean of the points that make up the tetrahedron (needed later)
            x_mean = np.average(tet.points[:, 0])
            y_mean = np.average(tet.points[:, 1])
            z_mean = np.average(tet.points[:, 2])

            # Iterate over the edges of the tetrahedron
            for edgei in tet.edges:
                # Skip over PEC walls
                if edgei in self.boundary_pec_edge_numbers:
                    continue
                # Get a hold of the Edge object
                edge1 = self.all_edges[edgei]
                # Get the nodes that make up this edge
                node_il, node_jl = self.all_nodes[edge1.node1], self.all_nodes[edge1.node2]

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
                    if edgej in self.boundary_pec_edge_numbers:
                        continue
                    edge2 = self.all_edges[edgej]
                    node_ik, node_jk = self.all_nodes[edge1.node1], self.all_nodes[edge1.node2]

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
                    S[self.remap_edge_nums[self.all_edges_map[edge1]], self.remap_edge_nums[self.all_edges_map[edge2]]] += curl_dot_curl_part

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
                    T[self.remap_edge_nums[self.all_edges_map[edge1]], self.remap_edge_nums[self.all_edges_map[edge2]]] += dot_part

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
        self.k0s, self.eigenvectors = np.sqrt(eigenvalues[first_pos:]), eigenvectors[first_pos:]
        print(f"Finished solving eigenvalue problem in {time.time() - start_time} seconds")
        return self.k0s, self.eigenvectors

    def plot_fields(self, mode, num_x_points=100, num_y_points=100, num_z_points=1, plane="xy", offset=0.1):
        """
        Plot the fields in the selected plane. Note that field plotting is expensive due to needing to locate which
        tetrahedron each point lies in. Finer meshes may need to use fewer sample points.
        :param mode: The mode to plot (0 -> lowest k0 mode, 1 -> next highest k0 mode, and so on).
        :param num_x_points: The number of x points to compute the fields for.
        :param num_y_points: The number of y points to compute the fields for.
        :param num_z_points: The number of z points to compute the fields for.
        :param plane: One of {"xy", "xz", "yz"} to select which plane to take a cut of.
        :param offset: The offset from the edge of the geometry in the direction perpendicular to the plane to calc at.
        :return: The figure containing all the field data (the result of running plt.figure()).
        """
        print("Calculating field data")
        # Compute the bounds of the waveguide
        # Create a cuboid grid of points that the geometry is inscribed in
        offset_x = offset if plane.upper() == "YZ" else 0
        offset_y = offset if plane.upper() == "XZ" else 0
        offset_z = offset if plane.upper() == "XY" else 0
        x_points = np.linspace(self.x_min+offset_x, self.x_max, num_x_points)
        y_points = np.linspace(self.y_min+offset_y, self.y_max, num_y_points)
        z_points = np.linspace(self.z_min+offset_z, self.z_max, num_z_points)
        Ex = np.zeros([num_z_points, num_y_points, num_x_points])
        Ey = np.zeros([num_z_points, num_y_points, num_x_points])
        Ez = np.zeros([num_z_points, num_y_points, num_x_points])

        field_points = np.zeros([num_x_points * num_y_points * num_z_points, 3])
        # Iterate over the points
        for i in range(num_z_points):
            pt_z = z_points[i]
            for j in range(num_y_points):
                pt_y = y_points[j]
                for k in range(num_x_points):
                    pt_x = x_points[k]
                    field_points[k + j*num_x_points + i*num_x_points*num_y_points] = np.array([pt_x, pt_y, pt_z])

        tet_indices = where(self.all_nodes, self.tets_node_ids, field_points)
        first_mode = np.where(self.k0s >= 0.1)[0][0]

        # Compute the field at each of the points
        for i, tet_index in enumerate(tet_indices):
            tet = self.tetrahedrons[tet_index]
            phis = [self.eigenvectors[first_mode+mode, self.remap_edge_nums[edge]] if edge in self.remap_edge_nums else 0 for edge in tet.edges]
            z_i = floor(i / (num_x_points * num_y_points)) % num_z_points
            y_i = floor(i / num_x_points) % num_y_points
            x_i = i % num_x_points
            # Note the indexing here is done with z_i first, y_i second, and x_i third. If we consider a 2D grid being
            # indexed, the first index corresponds to the row (vertical control), hence y_i second and x_i third.
            # Same idea applies to having z_i first.
            Ex[z_i, y_i, x_i], Ey[z_i, y_i, x_i], Ez[z_i, y_i, x_i] = tet.interpolate(phis, field_points[i])

        print("Finished calculating field data")

        fig = plt.figure()
        if plane.upper() == "YZ":
            axis1, axis2 = np.meshgrid(y_points, z_points)
            skip = (slice(None, None, 5), slice(None, None, 5))
            field_skip = (slice(None, None, 5), slice(None, None, 5), 0)
            plt.imshow(Ex[:, :, 0], extent=[self.y_min, self.y_max, self.z_min, self.z_max], cmap="cividis")
            plt.colorbar(label="Ex")
            plt.quiver(axis1[skip], axis2[skip], Ey[field_skip], Ez[field_skip], color="black")
        elif plane.upper() == "XZ":
            axis1, axis2 = np.meshgrid(x_points, z_points)
            skip = (slice(None, None, 5), slice(None, None, 5))
            field_skip = (slice(None, None, 5), 0, slice(None, None, 5))
            plt.imshow(Ey[:, 0, :], extent=[self.x_min, self.x_max, self.z_min, self.z_max], cmap="cividis")
            plt.colorbar(label="Ey")
            plt.quiver(axis1[skip], axis2[skip], Ex[field_skip], Ez[field_skip], color="black")
        elif plane.upper() == "XY":
            axis1, axis2 = np.meshgrid(x_points, y_points)
            skip = (slice(None, None, 5), slice(None, None, 5))
            field_skip = (0, slice(None, None, 5), slice(None, None, 5))
            plt.imshow(Ez[0, :, :], extent=[self.x_min, self.x_max, self.y_min, self.y_max], cmap="cividis")
            plt.colorbar(label="Ez")
            plt.quiver(axis1[skip], axis2[skip], Ex[field_skip], Ey[field_skip], color="black")
        else:
            raise RuntimeError(f"Invalid argument for plane '{plane}'. Must be one of 'xy', 'xz', or 'yz'.")
        return fig


cavity = Cavity("rectangular_waveguide_3d_less_coarse_pec.inp")
cavity.solve()
cavity.plot_fields(5, 1, 100, 100, "yz")
# for z in np.linspace(cavity.z_min, cavity.z_max, 20):
    # cavity.plot_fields(5, offset=z-cavity.z_min)
for x in np.linspace(cavity.x_min, cavity.x_max, 20):
    # cavity.plot_fields(5, offset=z-cavity.z_min)
    cavity.plot_fields(4, 1, 100, 100, "yz", offset=x-cavity.x_min)
