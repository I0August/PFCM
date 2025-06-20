import numpy as np
from collections import defaultdict
import os
import warnings
class CMGenerator:
    def __init__(self, path2CFD=None):
        self.data = {}  # Stores raw and processed CFD data from .npy files
        self.net = {}  # Stores constructed network relationships (e.g., graph)
        self.compartment_to_elements = []

        if path2CFD:
            self.load_cfd_data(path2CFD)  # Load CFD data files if path is given
        else:
            warnings.warn("No CFD path provided; data not loaded.")
        self.n_elements = len(self.data['element_to_coordinates'])
        self.data['U_convol'] = np.zeros((self.n_elements, 3))
        # Build internal connectivity structures from the loaded data
        self.constructFaceToElements()
        self.constructElementToFaces()
        self.constructElementToPoint()
        self.constructElementToNeighbours()


    def load_cfd_data(self, path2CFD):
        # Check if directory exists
        if not os.path.isdir(path2CFD):
            warnings.warn(f"CFD directory '{path2CFD}' does not exist. Data not loaded.")
            return  # Don't store any data

        try:
            self.data['U'] = np.load(os.path.join(path2CFD, 'U.npy'))
            self.data['U_norm'] = np.linalg.norm(self.data['U'], axis=1)
            self.data['element_to_volume'] = np.load(os.path.join(path2CFD, 'V.npy'))
            self.data['element_to_coordinates'] = np.load(os.path.join(path2CFD, 'C.npy'))
            self.data['point_to_coordinates'] = np.load(os.path.join(path2CFD, 'points.npy'))
            self.data['face_to_points'] = np.load(os.path.join(path2CFD, 'faces.npy'))
            self.data['face_to_element_owner'] = np.load(os.path.join(path2CFD, 'owner.npy'))
            self.data['face_to_element_neighbour'] = np.load(os.path.join(path2CFD, 'neighbour.npy'))
        except FileNotFoundError as e:
            warnings.warn(f"Some CFD files are missing in '{path2CFD}': {e}")
            # Optionally clear any partially loaded data
            self.data.clear()

    def constructFaceToElements(self):
        """
        Combine two face-to-element data (owner and neighbour) to create a list of faces to elements.
        The list stores elements that are shared by each face.

        Parameters:
        owner (ndarray): Array containing the owner elements for each face.
        neighbour (ndarray): Array containing the neighbour elements for each face.

        Returns:
        face_to_elements_interFace (ndarray): 2D array where each row contains the owner and neighbour elements for internal faces.
        face_to_elements_boundaryFace (ndarray): 2D array where each row contains the owner element for boundary faces.
        """
        # Declare ndarrays to store data
        self.data['face_to_elements_interFace'] = np.zeros([len(self.data['face_to_element_neighbour']), 2], dtype=int)  # Array for internal faces
        self.data['face_to_elements_boundaryFace'] = np.zeros([len(self.data['face_to_element_owner']) - len(self.data['face_to_element_neighbour']), 1],
                                                 dtype=int)  # Array for boundary faces

        # Copy the data from the original objects
        self.data['face_to_elements_interFace'][:, 0] = self.data['face_to_element_owner'][:len(self.data['face_to_element_neighbour'])]  # Owner elements for internal faces
        self.data['face_to_elements_interFace'][:, 1] = self.data['face_to_element_neighbour']  # Neighbour elements for internal faces
        self.data['face_to_elements_boundaryFace'][:, 0] = self.data['face_to_element_owner'][len(self.data['face_to_element_neighbour']):]  # Owner elements for boundary faces

    def constructElementToFaces(self):
        '''
        Construct the connectivity map between elements and their associated faces
        using the provided face-to-element data (-ref and -neighbour).

        Parameters:
        owner (ndarray): Array where each entry represents the element ID on one side of the faces.
        neighbour (ndarray): Array where each entry represents the element ID on the other side of the faces.

        Returns:
        list_of_sets (list of sets): A list where each set contains the indices of faces associated with each element.
        '''
        element_to_faces = np.zeros([max(self.data['face_to_element_owner']) + 1, 6], dtype=int)
        count = np.zeros(max(self.data['face_to_element_owner']) + 1, dtype=int)
        for i in range(len(self.data['face_to_element_owner'])):
            value = self.data['face_to_element_owner'][i]
            element_to_faces[value, count[value]] = i
            count[value] = count[value] + 1
        for i in range(len(self.data['face_to_element_neighbour'])):
            value = self.data['face_to_element_neighbour'][i]
            element_to_faces[value, count[value]] = i
            count[value] = count[value] + 1

        # Get the number of rows in the NumPy array
        num_rows = element_to_faces.shape[0]

        # Convert each row of the NumPy array to a list of sets
        self.data['element_to_faces'] = []
        for i in range(num_rows):
            row = element_to_faces[i, :]  # Extract the i-th row
            row_set = set(row)  # Convert the row to a set
            self.data['element_to_faces'].append(row_set)
        return self.data['element_to_faces']

    def constructElementToPoint(self):
        self.data['element_to_points'] = []
        for faces in self.data['element_to_faces']:
            dummy = set()
            for face in faces:
                dummy |= set(self.data['face_to_points'][face])
            self.data['element_to_points'].append(dummy)

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
        for elem_id, point_set in enumerate(self.data["element_to_points"]):
            for pt in point_set:
                point_to_elements[pt].add(elem_id)

        # Step 2: Build element-to-neighbors list
        self.data['element_to_neighbors'] = []
        for elem_id, point_set in enumerate(self.data["element_to_points"]):
            neighbors = set()
            for pt in point_set:
                neighbors.update(point_to_elements[pt])
            neighbors.discard(elem_id)  # Remove self
            self.data['element_to_neighbors'].append(neighbors)

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

    def calConvolutionalVelocity(self, velo='U', level=1):
        u = self.data[velo]
        u_norm = np.linalg.norm(u, axis=1)
        element_to_neighbors = self.data['element_to_neighbors']
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
            vol_i = self.data["element_to_volume"][i]

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
                    vol_n = self.data["element_to_volume"][neighbor]
                    total_vol += vol_n
                    weighted_u += u_n * vol_n

            # Update neighbor list and convolutional velocity
            self.data['U_convol'][i] = weighted_u / total_vol

    def compartmentalization(self, method='PFC', velo='U'):
        u = self.data[velo]
        u_norm = np.linalg.norm(u, axis=1)
        if method == 'PFC':
            counter = 0
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
                    neighbors = list(self.data['element_to_neighbors'][plane] & unprocessed_meshes)
                    u_c = u[plane]
                    u_norm_c = u_norm[plane]
                    r_c = self.data["element_to_coordinates"][plane]
                    accepted_neighbors = []
                    eps = 1e-10
                    for neighbor in neighbors:
                        points = self.data["element_to_points"][neighbor]
                        point_coordinates = np.array([self.data["point_to_coordinates"][pt] for pt in points])
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
                counter += 1
        else:
            warnings.warn(f"{method} is not implemented for compartmentalization.")

def writeOpenFOAMScalarField(input_file_name, output_file_name, file_base_name, name, scalar):
	k = 0
	startWrite = False
	with open(input_file_name, 'r') as input_file, open(output_file_name, 'w') as output_file:
		for line in input_file:
			if line == "    object      "+file_base_name+";\n":
				output_file.write("    object      "+ name +";\n")
				continue
			if line == "(\n" and k == 0:
				startWrite = True
				output_file.write(line)
				continue
			if line == ")\n":
				startWrite = False
				output_file.write(line)
				continue
			if startWrite == True:
				output_file.write(str(scalar[k]) + "\n")
				k += 1
			else:
				output_file.write(line)

def constructElementToZone(zone_to_elements, count):
    element_to_zone = np.zeros([count], dtype=int)
    for i in range(len(zone_to_elements)):
        for j in zone_to_elements[i]:
            element_to_zone[j] = i
    return element_to_zone

CM = CMGenerator("./../CFD")
U_orig =  CM.data["U"]
CM.calConvolutionalVelocity(level=1)
for _ in range(5):
    CM.calConvolutionalVelocity(velo='U_convol', level=1)

CM.compartmentalization(method='PFC', velo='U_convol')
# counter = 0
# for i in range(len(CM.compartment_to_elements)):
#     counter += len(CM.compartment_to_elements[i])
#
element_to_zone = constructElementToZone(CM.compartment_to_elements, CM.n_elements)

writeOpenFOAMScalarField("./../pitzDaily/0.35/Vc", "./../pitzDaily/0.35/compartment", 'Vc', 'compartment', element_to_zone)
