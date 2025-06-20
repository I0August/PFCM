import numpy as np
from collections import defaultdict
import openfoamparser as ofp
import os
import warnings

class CMGenerator:
    def __init__(self, path_to_foam=None, time_dir=None):
        foam_mesh = ofp.FoamMesh(path_to_foam)
        self.path_to_foam = path_to_foam
        self.part_to_write_foam = path_to_foam + time_dir
        self.net = {}  # Stores constructed network relationships (e.g., graph)
        self.compartment_to_elements = []
        self.U = ofp.parse_internal_field(self.part_to_write_foam + 'U')
        self.U_norm = np.linalg.norm(self.U, axis=1)
        self.element_to_volume = ofp.parse_internal_field(self.part_to_write_foam + 'Vc')
        self.element_to_coordinates = ofp.parse_internal_field(self.part_to_write_foam + 'C')
        self.point_to_coordinates = foam_mesh.points
        self.face_to_points = foam_mesh.faces
        self.face_to_element_owner = foam_mesh.owner
        self.face_to_element_neighbour = foam_mesh.neighbour
        self.n_elements = len(self.element_to_coordinates)
        self.U_convol = np.zeros((self.n_elements, 3))
        # Build internal connectivity structures from the loaded data
        self.element_to_faces = [set(etf) for etf in foam_mesh.cell_faces]
        self.constructElementToPoint()
        self.constructElementToNeighbours()

    def constructElementToPoint(self):
        self.element_to_points = []
        for faces in self.element_to_faces:
            dummy = set()
            for face in faces:
                dummy |= set(self.face_to_points[face])
            self.element_to_points.append(dummy)

    def constructElementToNeighbours(self):
        """
        Given a list of sets of point IDs for each element, return a list of sets
        of neighbor element IDs. Elements are neighbors if they share at least one point ID.

        Parameters:
        ----------
        element_to_point_ids : List[Set[int]]
            List where each item is a set of point IDs associated with an element.

        Returns:
        -------
        List[Set[int]]
            List where each item is a set of neighboring element indices.
        """
        # Step 1: Build point-to-elements map
        point_to_elements = defaultdict(set)
        for elem_id, point_set in enumerate(self.element_to_points):
            for pt in point_set:
                point_to_elements[pt].add(elem_id)

        # Step 2: Build element-to-neighbors list
        self.element_to_neighbors = []
        for elem_id, point_set in enumerate(self.element_to_points):
            neighbors = set()
            for pt in point_set:
                neighbors.update(point_to_elements[pt])
            neighbors.discard(elem_id)  # Remove self
            self.element_to_neighbors.append(neighbors)

    def constructElementToZone(self):
        self.element_to_compartment = np.zeros(self.n_elements, dtype=int)
        for i in range(self.n_compartments):
            for j in self.compartment_to_elements[i]:
                self.element_to_compartment[j] = i
        return self.element_to_compartment

    def constructNetDict(self):
        """
        Construct a network dictionary representing connectivity between elements.

        The dictionary `self.net` is structured with each element ID as a key.
        Each entry contains:
            - 'elements': a list with the element itself (can later include merged elements).
            - 'shells': the face IDs (or shell IDs) associated with the element.
            - 'volume': the volume or a placeholder (currently using same as 'shells').
            - 'neighbors': a mapping from neighbor element IDs to the face(s) that connect them.
        """

        # Initialize each element's entry in the network dictionary
        for i in range(self.n_elements):
            self.net[i] = {
                'elements': [i],  # Initially, each zone contains just one element
                # 'partition': zone_to_partition[i],  # (Optional) partition index, if available
                'coordinates': self.data['element_to_coordinates'][i],  # List of average coordinate for the zone
                'shells': self.data['element_to_faces'][i],  # List of face IDs (shells) for the element
                'volume': self.data['element_to_volume'][i],  # Volume of the element
                'neighbors': self.data['element_to_neighbors'][i]  # Will be filled in next loop
            }

    def calConvolutionalVelocity(self, u=None, level=1):
        if u is None:
            u = self.U
        u_norm = np.linalg.norm(u, axis=1)
        element_to_neighbors = self.element_to_neighbors
        # Expand neighbors up to the specified level
        for _ in range(level - 1):
            new_element_to_neighbors = {}
            for i in range(self.n_elements):
                expanded_neighbors = set(element_to_neighbors[i])
                for neighbor in element_to_neighbors[i]:
                    expanded_neighbors |= element_to_neighbors[neighbor]
                new_element_to_neighbors[i] = expanded_neighbors
            element_to_neighbors = new_element_to_neighbors

        for i in range(self.n_elements):
            u_i = u[i]
            u_norm_i = u_norm[i]
            vol_i = self.element_to_volume[i]

            # Initialize convolutional quantities
            total_vol = vol_i
            weighted_u = u_i * vol_i

            # Filter neighbors and accumulate volume-weighted velocity
            valid_neighbors = set()
            for neighbor in element_to_neighbors[i]:
                u_n = u[neighbor]
                u_norm_n = u_norm[neighbor]
                if np.dot(u_i / u_norm_i, u_n / u_norm_n) > 0.5:
                    valid_neighbors.add(neighbor)
                    vol_n = self.element_to_volume[neighbor]
                    total_vol += vol_n
                    weighted_u += u_n * vol_n

            # Update neighbor list and convolutional velocity
            self.U_convol[i] = weighted_u / total_vol

    def compartmentalization(self, method='PFC', u=None):
        if u is None:
            u = self.U
        u_norm = np.linalg.norm(u, axis=1)
        if method == 'PFC':
            unprocessed_meshes = set(range(0, self.n_elements))
            while len(unprocessed_meshes) > 0:
                processed_mesh = set()
                it_initial = iter(unprocessed_meshes)
                initial = next(it_initial)
                mesh_set = {initial}
                processing_set = {initial}
                while len(processing_set) > 0:
                    it_plane = iter(processing_set)
                    plane = next(it_plane)
                    processed_mesh |= {plane}
                    neighbors = list(self.element_to_neighbors[plane] & unprocessed_meshes)
                    u_c = u[plane]
                    u_norm_c = u_norm[plane]
                    r_c = self.element_to_coordinates[plane]
                    accepted_neighbors = []
                    eps = 1e-10
                    for neighbor in neighbors:
                        points = self.element_to_points[neighbor]
                        point_coordinates = np.array([self.point_to_coordinates[pt] for pt in points])
                        check_cut_element = np.dot(point_coordinates - r_c, u_c)
                        if np.any(check_cut_element > eps) and np.any(check_cut_element < -eps):
                            accepted_neighbors.append(neighbor)
                    # Filter neighbors whose velocity vectors align with u_c
                    accepted_neighbors_velocity = {
                        nid for nid in accepted_neighbors
                        if np.dot(u_c / u_norm_c, u[nid] / u_norm[nid]) > 0.999
                    }
                    # Append neighbors not yet in processing_set
                    processing_set |= accepted_neighbors_velocity
                    mesh_set |= accepted_neighbors_velocity
                    processing_set -= processed_mesh
                unprocessed_meshes -= mesh_set
                print(len(unprocessed_meshes))

                self.compartment_to_elements.append(mesh_set)
            self.n_compartments = len(self.compartment_to_elements)
        else:
            warnings.warn(f"{method} is not implemented for compartmentalization.")

    def writeOpenFOAMScalarField(self, output_name, scalar, path=None, input_name=None):
        """
        Replace the scalar field in an OpenFOAM field file.

        Parameters:
        - path (str): Directory path where the input and output files are located.
        - input_name (str): Name of the input file (e.g., "V").
        - output_name (str): Name of the output file (e.g., "compartment").
        - scalar (list or array): Scalar values to write into the field block.
        """
        if path is None:
            path = self.part_to_write_foam
        if input_name is None:
            input_name = "Vc"

        input_file_path = os.path.join(path, input_name)
        output_file_path = os.path.join(path, output_name)

        k = 0
        start_write = False
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                if line.strip().startswith("object") and input_name in line:
                    output_file.write("    object      " + output_name + ";\n")
                    continue
                if line.strip() == "(" and not start_write:
                    start_write = True
                    output_file.write(line)
                    continue
                if line.strip() == ")":
                    start_write = False
                    output_file.write(line)
                    continue
                if start_write:
                    output_file.write(str(scalar[k]) + "\n")
                    k += 1
                else:
                    output_file.write(line)

    def writeOpenFOAMVectorField(self, output_name, vector, path=None, input_name=None):
        """
        Replace the vector field in an OpenFOAM field file.

        Parameters:
        - path (str): Directory path where the input and output files are located.
        - input_name (str): Name of the input file (e.g., "U").
        - output_name (str): Name of the output file (e.g., "convolution").
        - vector (2D array-like): Nx3 array with vector values to write.
        """
        if path is None:
            path = self.part_to_write_foam
        if input_name is None:
            input_name = "U"
        input_file_path = os.path.join(path, input_name)
        output_file_path = os.path.join(path, output_name)

        k = 0
        start_write = False
        with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
            for line in input_file:
                if line.strip().startswith("object") and input_name in line:
                    output_file.write("    object      " + output_name + ";\n")
                    continue
                if line.strip() == "(" and not start_write:
                    start_write = True
                    output_file.write(line)
                    continue
                if line.strip() == ")":
                    start_write = False
                    output_file.write(line)
                    continue
                if start_write:
                    output_file.write(f"({vector[k, 0]} {vector[k, 1]} {vector[k, 2]})\n")
                    k += 1
                else:
                    output_file.write(line)