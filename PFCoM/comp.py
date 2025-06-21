import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import openfoamparser as ofp
import matplotlib
matplotlib.use('TkAgg')
from multiprocessing import shared_memory
from matplotlib import cm
from matplotlib.colors import Normalize
from joblib import Parallel, delayed
import os
from tqdm import tqdm
import warnings

class CMGenerator:
    def __init__(self, path_to_foam=None, time_dir=None):
        # Load OpenFOAM mesh and associated geometry
        foam_mesh = ofp.FoamMesh(path_to_foam)

        # Save paths
        self.path_to_foam = path_to_foam
        self.part_to_write_foam = path_to_foam + time_dir  # e.g., "OF_case/0.3/"

        # Initialize containers
        self.compartment_to_elements = []  # Maps compartments to sets of elements

        # Parse velocity field (U) and compute its norm
        self.U = ofp.parse_internal_field(self.part_to_write_foam + 'U')
        self.phi = ofp.parse_internal_field(self.part_to_write_foam + 'phi')
        self.U_norm = np.linalg.norm(self.U, axis=1)

        # Load other field data
        self.element_to_volume = ofp.parse_internal_field(self.part_to_write_foam + 'Vc')  # cell volume
        self.element_to_coordinates = ofp.parse_internal_field(self.part_to_write_foam + 'C')  # cell centers

        # Load mesh topology
        self.point_to_coordinates = foam_mesh.points
        self.face_to_points = foam_mesh.faces
        self.face_to_element_owner = foam_mesh.owner
        self.face_to_element_neighbour = [ x for x in foam_mesh.neighbour if x >= 0]
        self.n_elements = len(self.element_to_coordinates)
        self.n_faces = len(self.face_to_element_neighbour)

        # Placeholder for smoothed velocity (convolution result)
        self.U_convol = np.zeros((self.n_elements, 3))

        # Build internal mesh connectivity structures
        self.element_to_faces = [set(etf) for etf in foam_mesh.cell_faces]
        self.constructElementToPoint()
        self.constructElementToNeighbours()

        del foam_mesh

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

    def calConvolutionalVelocity(self, u=None, level=1, n_jobs=-1):
        if u is None:
            u = self.U
        u = np.asarray(u)
        u_norm = np.linalg.norm(u, axis=1)
        u_unit = np.divide(u, u_norm[:, None], out=np.zeros_like(u), where=u_norm[:, None] != 0)

        # === Shared Memory Setup ===
        def create_shared_array(arr, name=None):
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
            shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shared_arr[:] = arr[:]
            return shm, shared_arr

        shm_u, _ = create_shared_array(u, name="shm_u")
        shm_uu, _ = create_shared_array(u_unit, name="shm_uu")

        shape_u = u.shape
        dtype_u = u.dtype
        actual_n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
        chunks = np.array_split(np.arange(self.n_elements), actual_n_jobs)

        element_to_neighbors = self.element_to_neighbors
        element_to_volume = self.element_to_volume

        def compute_chunk(chunk_indices, shm_name_u, shm_name_uu, shape, dtype):
            def attach_shared_array(shm_name, shape, dtype):
                shm = shared_memory.SharedMemory(name=shm_name)
                arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                return shm, arr

            shm_u, u_shared = attach_shared_array(shm_name_u, shape, dtype)
            shm_uu, u_unit_shared = attach_shared_array(shm_name_uu, shape, dtype)

            result = []
            for i in chunk_indices:
                u_i = u_shared[i]
                u_unit_i = u_unit_shared[i]
                vol_i = element_to_volume[i]

                total_vol = vol_i
                weighted_u = u_i * vol_i

                for neighbor in element_to_neighbors[i]:
                    if neighbor != i and np.dot(u_unit_i, u_unit_shared[neighbor]) > 0:
                        vol_n = element_to_volume[neighbor]
                        weighted_u += u_shared[neighbor] * vol_n
                        total_vol += vol_n

                result.append((i, weighted_u / total_vol))

            shm_u.close()
            shm_uu.close()
            return result

        # === Parallel Execution ===
        chunk_results = Parallel(n_jobs=actual_n_jobs, backend='loky')(
            delayed(compute_chunk)(chunk, shm_u.name, shm_uu.name, shape_u, dtype_u)
            for chunk in chunks
        )

        # === Final Merge ===
        self.U_convol = np.zeros_like(self.U)
        for chunk in chunk_results:
            for idx, val in chunk:
                self.U_convol[idx] = val

        # === Cleanup Shared Memory ===
        shm_u.close(); shm_u.unlink()
        shm_uu.close(); shm_uu.unlink()

    def compartmentalization(self, method='PFC', u=None, flow_similarity=0.99):
        if u is None:
            u = self.U
        u_norm = np.linalg.norm(u, axis=1)
        u_unit = np.divide(u, u_norm[:, np.newaxis], out=np.zeros_like(u), where=u_norm[:, np.newaxis] != 0)

        if method == 'PFC':
            unprocessed_meshes = set(range(0, self.n_elements))
            total_elements = len(unprocessed_meshes)
            pbar = tqdm(total=total_elements, desc="Compartmentalizing", unit="mesh")

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
                    u_unit_c = u_unit[plane]
                    r_c = self.element_to_coordinates[plane]
                    accepted_neighbors = []
                    eps = 1e-10
                    for neighbor in neighbors:
                        points = self.element_to_points[neighbor]
                        point_coordinates = np.array([self.point_to_coordinates[pt] for pt in points])
                        check_cut_element = np.dot(point_coordinates - r_c, u_c)
                        if np.any(check_cut_element > eps) and np.any(check_cut_element < -eps):
                            accepted_neighbors.append(neighbor)
                    accepted_neighbors_velocity = {
                        nid for nid in accepted_neighbors
                        if np.dot(u_unit_c, u_unit[nid]) > flow_similarity
                    }
                    processing_set |= accepted_neighbors_velocity
                    mesh_set |= accepted_neighbors_velocity
                    processing_set -= processed_mesh

                unprocessed_meshes -= mesh_set
                pbar.update(len(mesh_set))
                self.compartment_to_elements.append(mesh_set)

            pbar.close()
            self.n_compartments = len(self.compartment_to_elements)
            self.constructElementToZone()
            self.constructCompartmentToVolume()
            self.constructCompartmentToShell()
            self.constructCompartmentToNeighbors()

    def constructCompartmentToVolume(self):
        """
        Compute the total volume of each compartment.

        Uses:
        - self.compartment_to_elements: A list where each entry contains the element indices of a compartment.
        - self.element_to_volume: A NumPy array mapping each element index to its volume.

        Result:
        - self.compartment_to_volume: A NumPy array of total volume per compartment.
        """
        self.compartment_to_volume = np.array([
            sum(self.element_to_volume[element_id] for element_id in compartment)
            for compartment in self.compartment_to_elements
        ])

    def constructCompartmentToShell(self):
        """
        Constructs the set of faces (shells) for each compartment by performing
        a symmetric difference (XOR) of all element face sets within the compartment.
        """
        self.compartment_to_shells = [
            set() for _ in range(self.n_compartments)
        ]

        for i in range(self.n_compartments):
            for element_id in self.compartment_to_elements[i]:
                self.compartment_to_shells[i] ^= self.element_to_faces[element_id]

    def constructCompartmentToNeighbors(self):
        # Declare the neighbour zone list
        self.compartment_to_neighbors = [set() for p in range(self.n_compartments)]

        # list or the shell
        n_cut = self.n_faces
        set_shell = set()
        for shell in self.compartment_to_shells:
            set_shell.update(shell)
        set_shell = list(set_shell)
        set_shell = set([x for x in set_shell if x < n_cut])

        while len(set_shell) > 0:
            shell = set_shell.pop()
            elements = [self.face_to_element_owner[shell], self.face_to_element_neighbour[shell]]
            zones = [self.element_to_compartment[elements[0]], self.element_to_compartment[elements[1]]]
            self.compartment_to_neighbors[zones[0]] |= {zones[1]}
            self.compartment_to_neighbors[zones[1]] |= {zones[0]}
            intersect = self.compartment_to_shells[zones[0]] & self.compartment_to_shells[zones[1]]
            set_shell -= intersect

    def constructNetDict(self, compartment_to_partition=None):
        """
        Construct a network dictionary representing connectivity between elements.

        The dictionary `self.net` is structured with each element ID as a key.
        Each entry contains:
            - 'elements': a list with the element itself (can later include merged elements).
            - 'shells': the face IDs (or shell IDs) associated with the element.
            - 'volume': the volume or a placeholder (currently using same as 'shells').
            - 'neighbors': a mapping from neighbor element IDs to the face(s) that connect them.
        """
        if compartment_to_partition is None:
            compartment_to_partition = np.zeros(self.n_compartments)
        self.net = {}

        owner_side = self.face_to_element_owner[:len(self.face_to_element_neighbour)]

        for i in range(self.n_compartments):
            self.net.update({i: {"elements": self.compartment_to_elements[i],
                                 "volume": self.compartment_to_volume[i],
                                 "shells": self.compartment_to_shells[i],
                                 "neighbors": {},
                                 "partition": compartment_to_partition[i]}})
            for j in self.compartment_to_neighbors[i]:
                common_shells = self.compartment_to_shells[i].intersection(self.compartment_to_shells[j])
                self.net[i]["neighbors"].update({j: {"common_shell": common_shells, "vol_rate": 0}})
        for i in self.net:
            for j in self.net[i]["neighbors"]:
                if self.net[i]["neighbors"][j]["vol_rate"] == 0 and self.net[j]["neighbors"][i]["vol_rate"] == 0:
                    for shell in self.net[i]["neighbors"][j]["common_shell"]:
                        if owner_side[shell] in self.net[i]["elements"]:
                            owner_flag = True
                        else:
                            owner_flag = False
                        if (owner_flag == True and self.phi[shell] < 0):  # flow to the zone i
                            self.net[j]["neighbors"][i]["vol_rate"] += -1 * self.phi[shell]
                        elif (owner_flag == False and self.phi[shell] > 0):  # flow to the zone i
                            self.net[j]["neighbors"][i]["vol_rate"] += self.phi[shell]
                        elif (owner_flag == False and self.phi[shell] < 0):  # flow to the zone j
                            self.net[i]["neighbors"][j]["vol_rate"] += -1 * self.phi[shell]
                        elif (owner_flag == True and self.phi[shell] > 0):  # flow to the zone j
                            self.net[i]["neighbors"][j]["vol_rate"] += self.phi[shell]

    def drawNetDict(self, threshold=0, view='isometric'):
        self.calCompartmentCoordinates()
        self.extractFlowMatrix()

        # Copy and zero diagonal
        Q = self.Q.copy()
        np.fill_diagonal(Q, 0)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        coords = self.compartment_to_coords
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

        # Compute max flow (in or out) per node
        node_flow = np.maximum(Q.max(axis=0), Q.max(axis=1))

        # Normalize node flow for colormap
        nf_min, nf_max = node_flow.min(), node_flow.max()
        if nf_max == nf_min:
            nf_min = 0  # avoid divide-by-zero
        norm_nodes = Normalize(vmin=nf_min, vmax=nf_max)
        cmap_nodes = cm.get_cmap("coolwarm")
        node_colors = cmap_nodes(norm_nodes(node_flow))

        # Plot nodes with colors based on flow strength
        ax.scatter(xs, ys, zs, c=node_colors, s=50, label='Nodes')

        # Normalize flow values for edges
        flow_vals = Q[Q > threshold]
        if len(flow_vals) > 0:
            q_min, q_max = flow_vals.min(), flow_vals.max()
            if q_max == q_min:
                q_min = 0

            norm_edges = Normalize(vmin=q_min, vmax=q_max)
            cmap_edges = cm.get_cmap("coolwarm")

            n = Q.shape[0]
            for i in range(n):
                for j in range(n):
                    if i != j and Q[i, j] > threshold:
                        start, end = coords[i], coords[j]
                        color = cmap_edges(norm_edges(Q[i, j]))

                        ax.plot(
                            [start[0], end[0]],
                            [start[1], end[1]],
                            [start[2], end[2]],
                            color=color,
                            linewidth=2
                        )

        # Set equal scaling
        max_range = np.ptp(coords, axis=0).max() / 2.0
        mid_x, mid_y, mid_z = coords.mean(axis=0)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Bird's eye view
        if view == 'Top-Down':
            ax.view_init(elev=90, azim=-90)
        elif view == 'isometric':
            ax.view_init(elev=30, azim=60)
        elif view == 'Front':
            ax.view_init(elev=0, azim=90)
        elif view == 'Side':
            ax.view_init(elev=0, azim=0)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Flow Network")

        # Optional: colorbar for nodes
        sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
        sm_nodes.set_array([])
        cbar = fig.colorbar(sm_nodes, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Node Max Flow (In/Out)')

        plt.tight_layout()
        plt.show(block=True)

    def calCompartmentCoordinates(self):
        """
        Calculate the volume-weighted centroid coordinates of each compartment (zone),
        and store the result in both self.net[i]['coords'] and
        self.compartment_to_coords as an array of shape (n_compartments, 3).
        """
        self.compartment_to_coords = np.zeros((self.n_compartments, 3))
        for i, compartment in self.net.items():
            element_ids = compartment['elements']
            volumes = np.array([self.element_to_volume[e] for e in element_ids])
            coords = np.array([self.element_to_coordinates[e] for e in element_ids])

            weighted_coords = np.average(coords, axis=0, weights=volumes)
            self.compartment_to_coords[i] = weighted_coords

    def extractFlowMatrix(self):
        zone_ids = list(self.net.keys())
        n_zones = len(zone_ids)
        id_to_index = {zone_id: idx for idx, zone_id in enumerate(zone_ids)}

        self.Q = np.zeros((n_zones, n_zones))

        for i in zone_ids:
            inx_n = id_to_index[i]
            for j, props in self.net[i]["neighbors"].items():
                vol_rate = props.get("vol_rate", 0)
                if vol_rate > 0 and j in id_to_index:
                    inx_m = id_to_index[j]
                    self.Q[inx_n, inx_m] = vol_rate
            self.Q[inx_n, inx_n] = -np.sum(self.Q[inx_n, :])

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
                    if k < len(vector):
                        output_file.write(f"({vector[k, 0]} {vector[k, 1]} {vector[k, 2]})\n")
                    else:
                        output_file.write(line)  # fallback to original line if out of bounds
                    k += 1
                else:
                    output_file.write(line)
