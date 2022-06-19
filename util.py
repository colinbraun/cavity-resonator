import numpy as np
import math


class Edge:
    """Class representing an edge. Immutable (Values should only be read from, not written to)."""
    def __init__(self, node1, node2):
        """
        Constructor
        :param node1: The global node number of the first node of the edge
        :param node2: The global node number of the second node of the edge
        """
        self.node1 = node1
        self.node2 = node2
        # print(len(Element.all_nodes))
        node1_t, node2_t = TriangleElement.all_nodes[node1], TriangleElement.all_nodes[node2]
        # TODO: Evaluate whether the sign should be considered or not (currently does assign it a sign)
        sign_multiplier = 1 if node1 < node2 else -1
        self.length = sign_multiplier * math.sqrt((node2_t[0] - node1_t[0])**2 + (node2_t[1] - node1_t[1])**2 + (node2_t[2] - node1_t[2])**2)

    def flip(self):
        """Return a copy of this object with the node numbers switched"""
        return Edge(self.node2, self.node1)

    def line(self):
        """Return a tuple of matrices, the first containing a list of 2 x points, the second of y points"""
        return [TriangleElement.all_nodes[self.node1][0], TriangleElement.all_nodes[self.node2][0]], [TriangleElement.all_nodes[self.node1][1], TriangleElement.all_nodes[self.node2][1]]

    def __eq__(self, other):
        """Edges are considered equal if they have the same global node numbers"""
        return self.node1 == other.node1 and self.node2 == other.node2 or self.node1 == other.node2 and self.node2 == other.node1

    def __hash__(self):
        """
        Hash the edge object for fast performance on maps/sets.
        Two edges with flipped node numbers produce the same hash value (are considered equal when compared thru map/set)
        """
        if self.node1 < self.node2:
            hash_value = (self.node1, self.node2).__hash__()
        else:
            hash_value = (self.node2, self.node1).__hash__()
        # print(f"Hash value: {hash_value}")
        return hash_value


class TriangleElement:
    """A class representing a triangular element (containing 3 global node numbers and 3 global edge numbers, as well as a permittivity) """
    all_nodes = []
    all_edges = []
    all_edges_map = None

    def __init__(self, nodes, edges, permittivity=1):
        """
        :param nodes: Three global node numbers as a numpy array
        :param edges: Three global edge numbers as a numpy array
        :param permittivity: The relative permittivity associated with this element (relative to vacuum permittivity)
        """
        self.nodes = nodes
        self.edges = edges
        self.permittivity = permittivity

    def area(self):
        """
        Compute the area of this element
        :return: The area of the triangle element
        """
        x1, y1, _ = TriangleElement.all_nodes[self.nodes[0]]
        x2, y2, _ = TriangleElement.all_nodes[self.nodes[1]]
        x3, y3, _ = TriangleElement.all_nodes[self.nodes[2]]
        return area(x1, y1, x2, y2, x3, y3)

    def is_inside(self, x, y):
        """
        Check if a point is inside this triangle element
        :param x: The x coordinate of the point
        :param y: The y coordinate of the point
        :return: True if the point lies in the triangle element, false otherwise
        """
        node1, node2, node3 = TriangleElement.all_nodes[self.nodes[0]], TriangleElement.all_nodes[self.nodes[1]], TriangleElement.all_nodes[self.nodes[2]]
        return is_inside(node1, node2, node3, x, y)

    def is_adjacent_to(self, element):
        """
        Check if this element is adjacent to the passed element
        :param element: The element to check if this is adjacent to
        :return: True if it is, false otherwise
        """
        node_count = 0
        if self.nodes[0] in element.nodes:
            node_count += 1
        if self.nodes[1] in element.nodes:
            node_count += 1
        if self.nodes[2] in element.nodes:
            node_count += 1
        return node_count == 2

    def nodal_interpolate(self, phi1, phi2, phi3, x, y):
        x1, y1 = TriangleElement.all_nodes[self.nodes[0]][0], TriangleElement.all_nodes[self.nodes[0]][1]
        x2, y2 = TriangleElement.all_nodes[self.nodes[1]][0], TriangleElement.all_nodes[self.nodes[1]][1]
        x3, y3 = TriangleElement.all_nodes[self.nodes[2]][0], TriangleElement.all_nodes[self.nodes[2]][1]
        a1 = x2 * y3 - x3 * y2
        a2 = x3 * y1 - x1 * y3
        a3 = x1 * y2 - x2 * y1
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        n1 = phi1 * (a1 + b1 * x + c1 * y)
        n2 = phi2 * (a2 + b2 * x + c2 * y)
        n3 = phi3 * (a3 + b3 * x + c3 * y)
        return (n1 + n2 + n3) / 2 / self.area()

    def edge_interpolate(self, phi1, phi2, phi3, x, y):
        area = self.area()
        x_component = 0
        y_component = 0
        phis = [phi1, phi2, phi3]
        count = 0
        for edge_number in self.edges:
            # Generate the edge from 2 of the nodes (done in a CCW fashion by choice of tuples in for loop)
            edge = TriangleElement.all_edges[edge_number]
            # TODO: Fix the below statement so that it doesn't print TRUE
            # if edge != Element.all_edges[self.edges[count]]:
            #     print("TRUE")
            # Index of the third node (the one not making up the edge) going in a CCW fashion
            node1, node2 = edge.node1, edge.node2
            node3 = set(self.nodes).difference({node1, node2}).pop()
            n1_index = np.where(self.nodes == node1)[0][0]
            n2_index = np.where(self.nodes == node2)[0][0]
            # n2_index = self.nodes.index(node2)
            # print(str(n1_index[0][0]) + str(n2_index[0][0]))
            negate = 1
            # TODO: Change this logic to work for earlier versions of python. Commented out since not used right now.
            n1_index, n2_index, n3_index, negate = 0, 1, 2, 1
            # match str(n1_index) + str(n2_index):
            #     case "01":
            #         n3_index = 2
            #     case "02":
            #         n1_index, n2_index, n3_index, negate = 2, 0, 1, -1
            #     case "10":
            #         n1_index, n2_index, n3_index, negate = 0, 1, 2, -1
            #     case "12":
            #         n1_index, n2_index, n3_index = 1, 2, 0
            #     case "20":
            #         n1_index, n2_index, n3_index = 2, 0, 1
            #     case "21":
            #         n1_index, n2_index, n3_index, negate = 1, 2, 0, -1

            # Create the ccw node list started from the first node of the edge
            nodes_lk = (TriangleElement.all_nodes[self.nodes[n1_index]], TriangleElement.all_nodes[self.nodes[n2_index]], TriangleElement.all_nodes[self.nodes[n3_index]])

            a_i_l, a_j_l = nodes_lk[1][0] * nodes_lk[2][1] - nodes_lk[2][0] * nodes_lk[1][1], nodes_lk[2][0] * nodes_lk[0][1] - nodes_lk[0][0] * nodes_lk[2][1]
            b_i_l, b_j_l = nodes_lk[1][1] - nodes_lk[2][1], nodes_lk[2][1] - nodes_lk[0][1]
            c_i_l, c_j_l = nodes_lk[2][0] - nodes_lk[1][0], nodes_lk[0][0] - nodes_lk[2][0]

            A_l = a_i_l * b_j_l - a_j_l * b_i_l
            B_l = c_i_l * b_j_l - c_j_l * b_i_l
            C_l = a_i_l * c_j_l - a_j_l * c_i_l
            D_l = b_i_l * c_j_l - b_j_l * c_i_l

            x_component += phis[count] * negate * edge.length / 4 / area**2 * (A_l + B_l*y)
            y_component += phis[count] * negate * edge.length / 4 / area**2 * (C_l + D_l*x)
            count += 1
        return x_component, y_component

    def __eq__(self, other):
        if self.nodes[0] in other.nodes and self.nodes[1] in other.nodes and self.nodes[2] in other.nodes:
            return True
        return False


class TetrahedralElement:
    """Class representing a tetrahedral (3D) element consisting of 4 TriangularElement objects"""

    def __init__(self, nodes, permittivity=1):
        """
        Constructor.
        :param nodes: A numpy array containing the 4 nodes (as global node numbers) of the tetrahedral element. The
        nodes must ordered such that the calculated volume is positive (Cubit orders the nodes this way automatically).
        :param permittivity: The permittivity of this element.
        """
        self.nodes = nodes
        # Generate the edges:
        edge1 = TriangleElement.all_edges_map[Edge(nodes[0], nodes[1])]
        edge2 = TriangleElement.all_edges_map[Edge(nodes[0], nodes[2])]
        edge3 = TriangleElement.all_edges_map[Edge(nodes[0], nodes[3])]
        edge4 = TriangleElement.all_edges_map[Edge(nodes[1], nodes[2])]
        edge5 = TriangleElement.all_edges_map[Edge(nodes[1], nodes[3])]
        edge6 = TriangleElement.all_edges_map[Edge(nodes[2], nodes[3])]
        self.edges = np.array([edge1, edge2, edge3, edge4, edge5, edge6])
        self.permittivity = permittivity
        # Store the unique set of points that make up this tetrahedron
        self.points = TriangleElement.all_nodes[self.nodes]
        # Volume calculated using eq (157) in NASA paper
        mat = [[1, self.points[0][0], self.points[0][1], self.points[0][2]],
               [1, self.points[1][0], self.points[1][1], self.points[1][2]],
               [1, self.points[2][0], self.points[2][1], self.points[2][2]],
               [1, self.points[3][0], self.points[3][1], self.points[3][2]]]
        # This could be a method, but is frequently used.
        self.volume = np.linalg.det(mat)
        if self.volume < 0:
            raise RuntimeError("Computed volume was negative but should have been positive")

        # Compute the simplex (barycentric) constants for the nodes of this TetrahedralElement
        # Each row is for a node. Each column is for a, b, c, and d (in order) from NASA paper eq. 162
        # These are stored in the same order as self.nodes (i.e. simplex_consts[0] are the constants for self.nodes[0])
        all_cofactors = np.zeros([4, 4])
        # Iterate over each row
        for row in range(4):
            cofactors = np.zeros([4])
            # Iterate over each column, computing the cofactor determinant of the row + column combination
            for col in range(4):
                # Compute the cofactor (remove the proper row and column and compute the determinant)
                if (row + col) % 2 == 0:
                    negate = 1
                else:
                    negate = -1
                cofactors[col] = negate * np.linalg.det(np.delete(np.delete(np.append(np.ones([4, 1]), self.points, 1), row, axis=0), col, axis=1))
            all_cofactors[row] = cofactors
        self.simplex_consts = all_cofactors * 6

    def interpolate(self, phis, p):
        """
        Get the fields at point p using the given phi values for each edge. The phi values must be passed in the same
        order as the edges field of this object.
        :param phis: An iterable containing the 6 phi values (the edge coefficients), one for each edge.
        :param p: The point to compute the Ex, Ey, and Ez fields at
        :return: The Ex, Ey, and Ez fields.
        """
        x_component = 0
        y_component = 0
        z_component = 0
        # if not self.point_inside(p):
        #     # raise RuntimeError("Point was not inside tetrahedron")
        #     print("Point was not inside tet")
        for i, edge_no in enumerate(self.edges):
            edge = TriangleElement.all_edges[edge_no]
            indices_l = [np.argwhere(self.nodes == edge.node1)[0][0], np.argwhere(self.nodes == edge.node2)[0][0]]
            # The simplex coordinates for nodes i and j of edge l
            # Necessary constants from NASA paper eqs. 163-172
            a_il, a_jl = self.simplex_consts[indices_l]
            Axl = a_il[0]*a_jl[1] - a_il[1]*a_jl[0]
            Bxl = a_il[2]*a_jl[1] - a_il[1]*a_jl[2]
            Cxl = a_il[3]*a_jl[1] - a_il[1]*a_jl[3]
            Ayl = a_il[0]*a_jl[2] - a_il[2]*a_jl[0]
            Byl = a_il[1]*a_jl[2] - a_il[2]*a_jl[1]
            Cyl = a_il[3]*a_jl[2] - a_il[2]*a_jl[3]
            Azl = a_il[0]*a_jl[3] - a_il[3]*a_jl[0]
            Bzl = a_il[1]*a_jl[3] - a_il[3]*a_jl[1]
            Czl = a_il[2]*a_jl[3] - a_il[3]*a_jl[2]
            x_component += phis[i] * edge.length / 36 / self.volume**2 * (Axl + Bxl*p[1] + Cxl*p[2])
            y_component += phis[i] * edge.length / 36 / self.volume**2 * (Ayl + Byl*p[0] + Cyl*p[2])
            z_component += phis[i] * edge.length / 36 / self.volume**2 * (Azl + Bzl*p[0] + Czl*p[1])
        return x_component, y_component, z_component

    def point_inside(self, point):
        """
        Determine if the given point lies in this tetrahedron (expensive).
        :param point: The point to check against.
        :return: True if the point lies in the tetrahedron, false otherwise
        """
        m0 = np.array([[self.points[0][0], self.points[0][1], self.points[0][2], 1],
               [self.points[1][0], self.points[1][1], self.points[1][2], 1],
               [self.points[2][0], self.points[2][1], self.points[2][2], 1],
               [self.points[3][0], self.points[3][1], self.points[3][2], 1]])
        m1 = np.copy(m0)
        m2 = np.copy(m0)
        m3 = np.copy(m0)
        m4 = np.copy(m0)
        m1[0, 0:3] = point
        m2[1, 0:3] = point
        m3[2, 0:3] = point
        m4[3, 0:3] = point
        d0, d1, d2, d3, d4 = np.linalg.det(m0), np.linalg.det(m1), np.linalg.det(m2), np.linalg.det(m3), np.linalg.det(m4)
        total = int(d0 > 0) + int(d1 > 0) + int(d2 > 0) + int(d3 > 0) + int(d4 > 0)
        if total == 5 or total == 0:
            return True
        else:
            return False


def construct_triangles_from_surface(element_to_node_conn, all_edges_map):
    """
    Construct a numpy array of TriangleElement objects for a particular surface.
    :param element_to_node_conn: A numpy array. Each row corresponds to a triangle in the surface (values are global node nums).
    :param all_edges_map: A dictionary mapping Edge objects to global edge numbers.
    :return: A numpy array of TriangleElements and a set of Edge objects that make up the surface.
    """
    # EDGE SHENANIGANS
    # TODO: MAKE SURE NOTHING FUNNY GOING ON WITH CCW VS CW ROTATION
    # The element-to-edge connectivity list
    element_to_edge_conn = []
    # The TriangleElement list
    triangles = []
    # All of the Edge objects that make up this surface
    edges = set()
    # Iterate over each element in the nodal connectivity list
    for element in element_to_node_conn:
        # Construct 3 edges (these have been created before when we went through all the tetrahedrons)
        edge1 = Edge(element[0], element[1])
        edge2 = Edge(element[0], element[2])
        edge3 = Edge(element[1], element[2])
        # Get the global edge numbers for these 3 edges
        edge1_number = all_edges_map[edge1]
        edge2_number = all_edges_map[edge2]
        edge3_number = all_edges_map[edge3]
        # Add the element_to_node_conn global edge numbers to the connectivity list
        element_to_edge_conn.append(np.array([edge1_number, edge2_number, edge3_number]))
        triangle = TriangleElement(element, np.array((edge1_number, edge2_number, edge3_number)), 1)
        triangles.append(triangle)
        edges.add(edge1)
        edges.add(edge2)
        edges.add(edge3)

    # Transform into numpy array
    triangles = np.array(triangles)
    return triangles, edges


def load_mesh_block(filename, block_name):
    """
    A function to load a block of data from a .inp file (abaqus format).
    Note that the node or element numbers are chopped off from the returned data.
    Node numbers contained in blocks of elements should be used as indices into the ALLNODES block.
    :param filename: The file to load the block of mesh data from
    :param block_name: The name of the block in the file that should be loaded
    :return: A 2D numpy array, each row containing a node or element (element can be 2D or 3D)
    """

    with open(filename, 'r') as file:
        lines = file.readlines()
        count = 0
        line = lines[count]
        # Go until we get to named section
        while block_name not in line:
            count += 1
            line = lines[count]
        # Skip over the line with the block_name in it
        count += 1
        line = lines[count]
        skip = count
        while "*" not in line:
            count += 1
            line = lines[count]
        rows = count - skip
        # Load all the node information into a numpy array, slicing off the node numbers (these are implied by index)
        # If we are dealing with elements, subtract off 1 so that the global node numbers work as indices
        if block_name != "ALLNODES":
            data = np.subtract(np.loadtxt(filename, skiprows=skip, max_rows=rows, delimiter=',', dtype=np.int32)[:, 1:], 1)
        else:
            data = np.loadtxt(filename, skiprows=skip, max_rows=rows, delimiter=',')[:, 1:]
        return data


def load_mesh(filename, tet_name="Tetrahedrons", pec_walls_name="PECWalls"):
    """
    Load a mesh from a file. Must have at least 2 blocks, one containing all surface element_to_node_conn, the other the edge ones.
    :param filename: The name of the mesh file (a .inp a.k.a. abaqus file) to load
    :return: A number of items. See comments at bottom of this function, above the return statement.
    """
    # Load the nodes
    all_nodes = load_mesh_block(filename, "ALLNODES")
    # Make the nodes accessible globally (TODO: Do this differently?)
    TriangleElement.all_nodes = all_nodes
    # Load the tetrahedrons
    element_to_node_conn = load_mesh_block(filename, tet_name)
    # All the edges (the index is the global edge number, i.e. all_edges[0] gets edge w/ global edge number 0)
    all_edges = []
    TriangleElement.all_edges = all_edges
    # A temporary map from an edge object to its global edge number (MUCH lower computational complexity)
    all_edges_map = {}
    TriangleElement.all_edges_map = all_edges_map
    # Keep track of what edge number we are on
    edge_count = 0

    # Create a list of tetrahedrons
    tetrahedrons = []
    # Iterate over all the tetrahedrons
    for tet in element_to_node_conn:
        # The list to hold the 4 triangle elements that make up the tetrahedron
        triangle_elements = []
        # The set holding the 6 global edge numbers of this tetrahedron
        tet_edges = set()
        # Iterate over each triangle face of the tetrahedron
        for element in (tet.take((0, 1, 2)), tet.take((0, 1, 3)), tet.take((0, 2, 3)), tet.take((1, 2, 3))):
            # Construct 3 edges
            edge1 = Edge(element[0], element[1])
            edge2 = Edge(element[0], element[2])
            edge3 = Edge(element[1], element[2])
            # For each of the 3 edges in this element, check if we have created this edge before. If not, create it.
            for edge in (edge1, edge2, edge3):
                if edge in all_edges_map:
                    # We do not want duplicate edges in our "all_edges" (global edge) list
                    pass
                else:
                    # Otherwise, we have not come across this edge before, so add it
                    all_edges.append(edge)
                    all_edges_map[edge] = edge_count
                    edge_count += 1
            # Get the global edge numbers for these 3 edges
            edge1_number = all_edges_map[edge1]
            edge2_number = all_edges_map[edge2]
            edge3_number = all_edges_map[edge3]
            tet_edges.add(edge1_number)
            tet_edges.add(edge2_number)
            tet_edges.add(edge3_number)
            # triangle = TriangleElement(np.array(element), np.array((edge1_number, edge2_number, edge3_number)), 1)
            # triangle_elements.append(triangle)
        # Create the tetrahedron object, converting the global edge numbers set to a numpy array
        tetrahedrons.append(TetrahedralElement(tet))

    # Convert tetrahedrons to numpy array
    all_tets = np.array(tetrahedrons)
    # Convert edges to numpy array
    all_edges = np.array(all_edges)
    # Make it globally available
    TriangleElement.all_edges = all_edges

    # Load the PEC Wall triangle elements
    boundary_pec_elements = load_mesh_block(filename, pec_walls_name)
    # boundary_pec_edges = [Edge(element[0], element[1]) for element in boundary_pec_elements]
    _, boundary_pec_edges = construct_triangles_from_surface(boundary_pec_elements, all_edges_map)
    boundary_pec_edge_numbers = set(all_edges_map[edge] for edge in boundary_pec_edges)
    # Load the InputPort triangle elements

    # Get the set of non-boundary global edge numbers
    # inner_edge_numbers = set(np.arange(0, len(all_edges))) - boundary_pec_edge_numbers - boundary_input_edge_numbers - boundary_output_edge_numbers
    # Different version of above:
    edge_nums = set(np.arange(0, len(all_edges))) - boundary_pec_edge_numbers
    # A map that takes one of the non-PEC edge numbers and maps it to a unique integer between [0, num of non-PEC edges]
    remap_edge_nums = {item: i for i, item in enumerate(edge_nums)}

    # Created for convenience if a point lies in a mesh
    tets_node_ids = np.array([tet.nodes for tet in all_tets])

    # Return the results of loading the mesh
    # all_tets: A list of all the TetrahedralElement objects that make up the entire geometry
    # tets_node_ids: A list containing lists of length 4 containing the 4 global node numbers that make up the tets.
    # all_edges: A list of all the global edge numbers that make up the entire geometry
    # boundary_pec_edge_numbers: A set of all the global edge numbers that lie on the PEC wall of the geometry
    # remap_edge_nums: A map that takes one of the inner edge numbers and maps it to a unique integer between [0, number of inner edges]
    # all_edges_map: A map from an Edge object to its global edge number
    return all_nodes, all_tets, tets_node_ids, all_edges, boundary_pec_edge_numbers, remap_edge_nums, all_edges_map


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def is_inside(pt1, pt2, pt3, x, y):
    if sign((x, y), pt1, pt2) < 0.0:
        return False
    if sign((x, y), pt2, pt3) < 0.0:
        return False
    if sign((x, y), pt3, pt1) < 0.0:
        return False
    return True


def quad_sample_points(n, p1, p2, p3):
    """
    Generate gaussian quadrature sample points for a triangle with points p1, p2, and p3.
    :param n: The order of the quadrature.
    :param p1: First point of a triangle.
    :param p2: Second point of a triangle.
    :param p3: Third point of a triangle.
    :return: An nx3 numpy array. Each row contains the x, y, and z values of a single sample point.
    """

    if n > 9 or n < 1:
        print("N must be a value between 1 and 9")
        return None

    loc = None
    x1, y1, z1 = p1[:]
    x2, y2, z2 = p2[:]
    x3, y3, z3 = p3[:]
    if n == 1:
        alpha = 1 / 3
        beta = 1 / 3
        temp1 = x1 * (1 - alpha - beta) + x2 * alpha + x3 * beta
        temp2 = y1 * (1 - alpha - beta) + y2 * alpha + y3 * beta
        temp3 = z1 * (1 - alpha - beta) + z2 * alpha + z3 * beta
        loc = [temp1, temp2, temp3]

    elif n == 2:
        alpha = [1 / 6, 2 / 3, 1 / 6]
        beta = [1 / 6, 1 / 6, 2 / 3]
        loc = np.zeros([3, 3])
        for m in range(len(alpha)):
            temp1 = x1 * (1 - alpha[m] - beta[m]) + x2 * alpha[m] + x3 * beta[m]
            temp2 = y1 * (1 - alpha[m] - beta[m]) + y2 * alpha[m] + y3 * beta[m]
            temp3 = z1 * (1 - alpha[m] - beta[m]) + z2 * alpha[m] + z3 * beta[m]
            loc[m, 0] = temp1
            loc[m, 1] = temp2
            loc[m, 2] = temp3
    elif n == 3:
        alpha = [1 / 3, 1 / 5, 1 / 5, 3 / 5]
        beta = [1 / 3, 3 / 5, 1 / 5, 1 / 5]
        loc = np.zeros([4, 3])
        for m in range(len(alpha)):
            temp1 = x1 * (1 - alpha[m] - beta[m]) + x2 * alpha[m] + x3 * beta[m]
            temp2 = y1 * (1 - alpha[m] - beta[m]) + y2 * alpha[m] + y3 * beta[m]
            temp3 = z1 * (1 - alpha[m] - beta[m]) + z2 * alpha[m] + z3 * beta[m]
            loc[m, 0] = temp1
            loc[m, 1] = temp2
            loc[m, 2] = temp3
    if loc is None:
        print("Error in computing quadrature. Did you choose a proper n value?")
    return loc


def quad_eval(p1, p2, p3, f):
    """
    Compute the integral using gaussian quadrature. The values of function (f(x, y, z)) should be stored in the argument f,
    stored as a column vector with each row corresponding to f(x, y) for a particular sample point. The sample points
    should come from quad_sample_points().
    :param p1: First point of a triangle.
    :param p2: Second point of a triangle.
    :param p3: Third point of a triangle.
    :param f: A numpy array containing f(x, y, z) in each row for each of the corresponding sample points.
    :return: The approximation of the integration (or exact value in some cases).
    """
    num_pts = len(f[:, 0])
    x1, y1, z1 = p1[:]
    x2, y2, z2 = p2[:]
    x3, y3, z3 = p3[:]

    if num_pts == 1:
        weights = [1]
    elif num_pts == 3:
        weights = [1/3,1/3,1/3]
    elif num_pts == 4:
        weights = [-27/48,25/48,25/48,25/48]
    elif num_pts == 6:
        weights = [0.22338158967801,0.22338158967801,0.22338158967801,
                   0.10995174365532,0.10995174365532,0.10995174365532]
    elif num_pts == 7:
        weights = [0.22500000000000,0.13239415278851,0.13239415278851,
                   0.13239415278851,0.12593918054483,0.12593918054483,
                   0.12593918054483]
    elif num_pts == 12:
        weights = [0.11678627572638,0.11678627572638,0.11678627572638,
                   0.05084490637021,0.05084490637021,0.05084490637021,
                   0.08285107561837,0.08285107561837,0.08285107561837,
                   0.08285107561837,0.08285107561837,0.08285107561837]
    elif num_pts == 13:
        weights = [-0.14957004446768,0.17561525743321,0.17561525743321,
                   0.17561525743321,0.05334723560884,0.05334723560884,
                   0.05334723560884,0.07711376089026,0.07711376089026,
                   0.07711376089026,0.07711376089026,0.07711376089026,
                   0.07711376089026]
    elif num_pts == 16:
        weights = [0.14431560767779,0.09509163426728,0.09509163426728,
                   0.09509163426728,0.10321737053472,0.10321737053472,
                   0.10321737053472,0.03245849762320,0.03245849762320,
                   0.03245849762320,0.02723031417443,0.02723031417443,
                   0.02723031417443,0.02723031417443,0.02723031417443,
                   0.02723031417443]
    elif num_pts == 19:
        weights = [0.097135796282799,0.031334700227139,0.031334700227139,
                   0.031334700227139,0.077827541004774,0.077827541004774,
                   0.077827541004774,0.079647738927210,0.079647738927210,
                   0.079647738927210,0.025577675658698,0.025577675658698,
                   0.025577675658698,0.043283539377289,0.043283539377289,
                   0.043283539377289,0.043283539377289,0.043283539377289,
                   0.043283539377289]
    else:
        print('Incorrect number of points used')

    val = 0
    Ak = np.sqrt(((y1-y3)*(z2-z3)-(y2-y3)*(z1-z3))**2 +
                 ((x1-x3)*(z2-z3)-(x2-x3)*(z1-z3))**2 +
                 ((x1-x3)*(y2-y3)-(x2-x3)*(y1-y3))**2)/2
    for m in range(len(weights)):
        val = val+weights[m]*f[m, :]
    val = val*Ak
    return val


def where(node_coordinates, node_ids, p):
    """
    Find where a point lies in a mesh.
    :param node_coordinates: A numpy array containing all the nodes.
    :param node_ids: The TetrahedralElement object numpy array composing the mesh
    :param p: The point or points to be tested
    :return: A numpy array corresponding to the numpy array of TetrahedralElement objects. -1 -> Outside, tet# -> Inside.
    """
    ori=node_coordinates[node_ids[:, 0], :]
    v1=node_coordinates[node_ids[:, 1], :]-ori
    v2=node_coordinates[node_ids[:, 2], :]-ori
    v3=node_coordinates[node_ids[:, 3], :]-ori
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
