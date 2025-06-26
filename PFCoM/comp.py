import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import openfoamparser as ofp
import matplotlib
from typing import *
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from multiprocessing import shared_memory
from matplotlib import cm
from matplotlib.colors import Normalize
from joblib import Parallel, delayed
import os
from tqdm import tqdm
import warnings

class CMGenerator:
    def __init__(self, path_to_foam: Optional[str] = None, time_dir: Optional[str] = None) -> None:
        """
        Initialize the processor by loading OpenFOAM mesh and field data for further analysis.

        Args:
            path_to_foam (Optional[str]): Path to the base OpenFOAM case directory.
            time_dir (Optional[str]): Time directory containing the simulation snapshot (e.g., '0.3/').

        Attributes:
            path_to_foam (str): The provided path to the OpenFOAM case directory.
            part_to_write_foam (str): Full path to the time-specific simulation data.
            compartment_to_elements (List): Placeholder for compartment-to-element mapping.
            U (np.ndarray): Velocity vectors at each mesh element (Nx3 array).
            phi (np.ndarray): Face flux field.
            U_norm (np.ndarray): Norm (magnitude) of velocity vectors.
            element_to_volume (np.ndarray): Volume of each mesh element.
            element_to_coordinates (np.ndarray): Centroids of each mesh element.
            point_to_coordinates (np.ndarray): Coordinates of mesh points (vertices).
            face_to_points (List[List[int]]): Point indices for each face.
            face_to_element_owner (List[int]): Owner element index for each face.
            face_to_element_neighbour (List[int]): Neighbour element indices (excluding boundaries).
            n_elements (int): Total number of mesh elements.
            n_faces (int): Number of internal faces (with valid neighbours).
            U_convol (np.ndarray): Placeholder array for smoothed velocity field.
            element_to_faces (List[Set[int]]): Faces associated with each element.
            element_to_neighbors (List[Set[int]]): Neighbouring elements for each element.
        """
        # Load OpenFOAM mesh and associated geometry
        foam_mesh = ofp.FoamMesh(path_to_foam)

        # Save paths
        self.path_to_foam: str = path_to_foam
        self.part_to_write_foam: str = path_to_foam + time_dir  # e.g., "OF_case/0.3/"

        # Initialize containers
        self.compartment_to_elements: List = []  # Maps compartments to sets of elements

        # Parse velocity and flux fields
        self.U: np.ndarray = ofp.parse_internal_field(self.part_to_write_foam + 'U')
        self.phi: np.ndarray = ofp.parse_internal_field(self.part_to_write_foam + 'phi')
        self.U_norm: np.ndarray = np.linalg.norm(self.U, axis=1)

        # Load scalar fields
        self.element_to_volume: np.ndarray = ofp.parse_internal_field(self.part_to_write_foam + 'Vc')
        self.element_to_volume_org = []
        self.element_to_coordinates: np.ndarray = ofp.parse_internal_field(self.part_to_write_foam + 'C')
        self.element_to_coordinates_org = []

        # Load mesh topology
        self.point_to_coordinates: np.ndarray = foam_mesh.points
        self.face_to_points: List[List[int]] = foam_mesh.faces
        self.face_to_element_owner: List[int] = foam_mesh.owner
        self.face_to_element_neighbour: List[int] = [x for x in foam_mesh.neighbour if x >= 0]
        self.n_elements: int = len(self.element_to_coordinates)
        self.n_elements_org: int = len(self.element_to_coordinates)
        self.n_faces: int = len(self.face_to_element_neighbour)

        # Placeholder for smoothed velocity (e.g., after convolution)
        #self.U_convol: np.ndarray = np.zeros((self.n_elements, 3))

        # Build internal mesh connectivity structures
        self.element_to_faces: List[Set[int]] = [set(etf) for etf in foam_mesh.cell_faces]
        self.element_to_faces_org = []
        self.element_to_neighbors: List[Set[int]] = [set(np.abs(etf)) for etf in foam_mesh.cell_neighbour]
        self.element_to_neighbors_org = []
        self.constructElementToPoint()
        self.element_to_points_org = []

        self.reverse_map = []
        self.doesMapNeedRecovery = False

        # Clean up
        del foam_mesh

    def selectedElementsByCoordinate(self, axis: float = 0, r_min: float=0, r_max: float=1)->None:
        if len(self.element_to_volume_org) == 0:
            self.element_to_volume_org = self.element_to_volume.copy()
            self.element_to_coordinates_org = self.element_to_coordinates.copy()
            self.element_to_faces_org = self.element_to_faces.copy()
            self.element_to_neighbors_org = self.element_to_neighbors.copy()
            self.element_to_points_org = self.element_to_points.copy()
        selected_elements = (
                (self.element_to_coordinates[:, axis] >= r_min) &
                (self.element_to_coordinates[:, axis] <= r_max)
        )

        # Step 1: Create a mapping from old indices to new ones
        old_map = np.arange(self.n_elements)
        new_map = old_map[selected_elements]

        self.U: np.ndarray = self.U[selected_elements]
        self.U_norm: np.ndarray = self.U_norm[selected_elements]
        self.element_to_volume: np.ndarray = self.element_to_volume[selected_elements]
        self.element_to_coordinates: np.ndarray = self.element_to_coordinates[selected_elements]
        self.n_elements: int = len(self.element_to_coordinates)
        self.element_to_faces: List[Set[int]] = [element for element, selected in zip(self.element_to_faces, selected_elements) if selected]
        self.element_to_points: List[Set[int]] = [element for element, selected in zip(self.element_to_points, selected_elements) if selected]

        # Step 2: Filter element_to_neighbors using selected_elements
        self.element_to_neighbors: List[Set[int]] = [
            neighbors for neighbors, selected in zip(self.element_to_neighbors, selected_elements) if selected
        ]

        # Step 3: Remap neighbor indices to new indices
        # Create a reverse map: old index → new index
        reverse_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_map)}

        # Step 4: Update the neighbor sets with the new indices
        temp_neighbors: List[Set[int]] = []
        for neighbors in self.element_to_neighbors:
            remapped = {reverse_map[j] for j in neighbors if j in reverse_map}
            temp_neighbors.append(remapped)

        # Step 5: Assign the updated neighbors back
        self.element_to_neighbors = temp_neighbors
        self.reverse_map.append(reverse_map)
        self.doesMapNeedRecovery = True

    def constructElementToPoint(self) -> None:
        """
        Constructs a mapping from each mesh element to the set of point (vertex) indices it uses.

        This method iterates over the faces associated with each element and aggregates the
        point indices from those faces, effectively identifying all the vertices that define each element.

        Sets the attribute:
            self.element_to_points (List[Set[int]]): For each element, a set of point indices that define its geometry.
        """
        self.element_to_points: List[Set[int]] = []
        for faces in self.element_to_faces:
            point_indices: Set[int] = set()
            for face in faces:
                point_indices |= set(self.face_to_points[face])
            self.element_to_points.append(point_indices)

    def constructElementToCompartment(self) -> None:
        """
        Constructs a mapping from each element to its corresponding compartment (zone) index.

        This method populates the `element_to_compartment` array, where each entry at index `j`
        indicates the compartment index `i` that the element `j` belongs to.

        Assumes:
            - `self.n_elements` is the total number of elements.
            - `self.n_compartments` is the total number of compartments.
            - `self.compartment_to_elements` is a list where each item is a list (or set)
              of element indices belonging to that compartment.

        Sets the attribute:
            self.element_to_compartment (np.ndarray): An integer array of length `n_elements`, where
            each value is the index of the compartment that the corresponding element belongs to.
        """
        self.element_to_compartment: np.ndarray = np.zeros(self.n_elements, dtype=int)
        for i in range(self.n_compartments):
            for j in self.compartment_to_elements[i]:
                self.element_to_compartment[j] = i

    def mergeToACompartment(self):
        self.compartment_to_elements = [set([x for x in range(self.n_elements)])]

    def manualInputCompartments(self, list_of_compartment):
        for list in list_of_compartment:
            for compartment in list:
                self.compartment_to_elements.append(compartment)

    def plugFlowCompartmentalization(self, u: Optional[np.ndarray] = None, eps: float = 1e-8, theta_deg: float = 10)->None:
        """
        Partition mesh elements into compartments based on local flow direction and magnitude similarity.

        The algorithm performs the following steps:
        1. Detects flow-induced "cuts" between neighboring elements by projecting velocity onto the vector
           connecting cell centers to neighbor face points.
        2. Groups elements into compartments where each element has similar velocity direction (within `theta_deg`)
           and is mutually reachable via detected cuts.

        Parameters:
            u (Optional[np.ndarray]): Velocity vectors at each element (shape: n_elements × 3). If None, uses `self.U`.
            eps (float): Tolerance used to determine if velocity significantly crosses a face. Smaller values
                         make the method more sensitive to flow separation.
            theta_deg (float): Angular threshold (in degrees) for grouping elements. Elements are only grouped
                               if their normalized velocity vectors are within this angle.

        Updates:
            self.compartment_to_elements (List[Set[int]]): Groups of element indices representing compartments.
            self.n_compartments (int): Total number of compartments.
            self.element_to_compartment (np.ndarray): Index of compartment for each element.

        Calls:
            - self.constructElementToZone()
            - self.constructCompartmentToVolume()
            - self.constructCompartmentToShell()
            - self.constructCompartmentToNeighbors()
        """
        if u is None:
            u = self.U
        u = np.asarray(u)

        # Normalize velocities
        u_norm = np.linalg.norm(u, axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            u_unit = np.divide(u, u_norm, out=np.zeros_like(u), where=(u_norm != 0))

        cos_threshold = np.cos(np.deg2rad(theta_deg))

        cut_neighbors = [set() for _ in range(self.n_elements)]

        # Step 1: Geometry-based face cut detection
        for elem_id in range(self.n_elements):
            center_coord = self.element_to_coordinates[elem_id]
            velocity = u[elem_id]
            neighbors = self.element_to_neighbors[elem_id]

            for neighbor_id in neighbors:
                point_ids = self.element_to_points[neighbor_id]
                point_coords = np.array([self.point_to_coordinates[pt] for pt in point_ids])
                projections = np.dot(point_coords - center_coord, velocity)

                if np.any(projections > eps) and np.any(projections < -eps):
                    cut_neighbors[elem_id].add(neighbor_id)

        # Step 2: Grouping based on directional similarity and flow compatibility
        self.compartment_to_elements = []
        unprocessed = set(range(self.n_elements))

        while unprocessed:
            initial_elem = unprocessed.pop()
            compartment = {initial_elem}
            to_process = {initial_elem}

            while to_process:
                current_elem = to_process.pop()
                current_velocity = u[current_elem]
                current_unit = u_unit[current_elem]

                # Optionally skip low-flow elements
                if np.linalg.norm(current_velocity) < 1e-8:
                    continue

                mutual_neighbors = cut_neighbors[current_elem] & unprocessed

                for neighbor in mutual_neighbors:
                    neighbor_velocity = u[neighbor]
                    neighbor_unit = u_unit[neighbor]

                    # Skip neighbors with nearly zero velocity
                    if np.linalg.norm(neighbor_velocity) < 1e-8:
                        continue

                    # Flow direction similarity (angle)
                    direction_check = np.dot(current_unit, neighbor_unit) >= cos_threshold

                    # Require bidirectional cut (mutual visibility)
                    if direction_check and current_elem in cut_neighbors[neighbor]:
                        to_process.add(neighbor)
                        compartment.add(neighbor)
                        unprocessed.remove(neighbor)

            self.compartment_to_elements.append(compartment)
    def constructComparmentToVSN(self):
        if self.doesMapNeedRecovery:
            self.reverse_compartment_mapping()

        total_elements = sum(len(s) for s in self.compartment_to_elements)
        if total_elements != self.n_elements_org:
            self.finalize_compartment_recovery()

        self.n_compartments = len(self.compartment_to_elements)
        self.constructElementToCompartment()
        self.constructCompartmentToVolume()
        self.constructCompartmentToShell()
        self.constructCompartmentToNeighbors()

    def reverse_compartment_mapping(self) -> None:
        """
        Reverses previously applied compartment mappings stored in self.reverse_map
        and updates self.compartment_to_elements accordingly. Also removes any empty compartments.
        """
        compartment_to_elements: List[Set[int]] = self.compartment_to_elements

        for reverse_mapping in reversed(self.reverse_map):
            max_key: int = max(reverse_mapping.keys())
            switched_map: Dict[int, int] = {v: k for k, v in reverse_mapping.items()}
            recovered_c2e: List[Set[int]] = [set() for _ in range(max_key + 1)]

            for comp_idx, elements in enumerate(compartment_to_elements):
                new_elements = {switched_map[element] for element in elements}
                recovered_c2e[switched_map[comp_idx]].update(new_elements)

            compartment_to_elements = recovered_c2e

        # Remove empty compartments
        self.compartment_to_elements = [s for s in compartment_to_elements if s]
        self.doesMapNeedRecovery = False

    def finalize_compartment_recovery(self) -> None:
        """
        Ensures all original elements are present in self.compartment_to_elements.
        Also restores original metadata (element counts, volumes, coordinates, etc.).
        """
        # Fill in any missing elements
        full_elements: Set[int] = set(range(self.n_elements_org))
        processed_elements: Set[int] = set().union(*self.compartment_to_elements)
        unprocessed_elements: Set[int] = full_elements - processed_elements

        for element in unprocessed_elements:
            self.compartment_to_elements.append({element})

        # Update number of compartments
        self.n_compartments = len(self.compartment_to_elements)

        # Restore original metadata
        self.n_elements = self.n_elements_org
        self.element_to_volume = self.element_to_volume_org
        self.element_to_coordinates = self.element_to_coordinates_org
        self.element_to_faces = self.element_to_faces_org
        self.element_to_neighbors = self.element_to_neighbors_org
        self.element_to_points = self.element_to_points_org


    def constructCompartmentToVolume(self) -> None:
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

    def constructCompartmentToShell(self) -> None:
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

    def constructCompartmentToNeighbors(self) -> None:
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

    def constructNetDict(self, compartment_to_partition: Optional[np.ndarray] = None) -> None:
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

    def drawNetDict(self, unidirectional: bool = False, threshold: float = 0,
                    view: Literal['isometric', 'Top-Down', 'Front', 'Side'] = 'isometric') -> None:
        """
        Visualize the compartment flow network as a 3D quiver plot.

        This method uses flow data stored in `self.Q` (flow matrix) and compartment coordinates
        (`self.compartment_to_coords`) to render arrows between compartments and color nodes
        based on flow intensity.

        Parameters:
            unidirectional (bool): If True, uses only unidirectional flow in `self.Q`. If False, uses net flow.
            threshold (float): Minimum flow magnitude for an edge to be visualized.
            view (str): Camera view preset for 3D plot. Options:
                        - 'isometric' (default)
                        - 'Top-Down'
                        - 'Front'
                        - 'Side'

        Preconditions:
            - `self.calCompartmentCoordinates()` must compute `self.compartment_to_coords`.
            - `self.extractFlowMatrix(unidirectional)` must populate `self.Q`.

        Visualization:
            - Nodes (compartments) are colored by their max incoming or outgoing flow.
            - Arrows (edges) represent flow between nodes, colored by flow magnitude.
            - A colorbar shows node flow scale.

        Raises:
            ValueError: If `self.Q` or `self.compartment_to_coords` are not set properly.
        """
        self.calCompartmentCoordinates()
        self.extractFlowMatrix(unidirectional)

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
                        direction = end - start
                        color = cmap_edges(norm_edges(Q[i, j]))

                        ax.quiver(
                            start[0], start[1], start[2],
                            direction[0], direction[1], direction[2],
                            color=color,
                            linewidth=3,
                            arrow_length_ratio=0.2,  # adjust arrowhead size
                            normalize=False
                        )

        # Set equal scaling
        max_range = np.ptp(coords, axis=0).max() / 2.0
        mid_x, mid_y, mid_z = coords.mean(axis=0)

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # View control
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

        # Colorbar for nodes
        sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
        sm_nodes.set_array([])
        cbar = fig.colorbar(sm_nodes, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Node Max Flow (In/Out)')

        plt.tight_layout()
        plt.show(block=True)

    def calCompartmentCoordinates(self) -> None:
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

    def extractFlowMatrix(self, unidirectional: bool = False) -> None:
        """
        Constructs the flow matrix `self.Q` based on inter-compartment volumetric flow rates.

        The flow matrix `Q` is of size (n_compartments × n_compartments), where each entry Q[i, j]
        represents the volumetric flow rate from compartment i to compartment j.

        Parameters:
            unidirectional (bool): If True, uses net unidirectional flow. That is:
                - Q[i, j] = max(q_ij - q_ji, 0)
                - Q[j, i] = max(q_ji - q_ij, 0)
              If False, only forward (positive) flow q_ij is used, regardless of q_ji.

        Sets:
            self.Q (np.ndarray): A square matrix representing flow between compartments.
                                 Diagonal entries are set to negative row sums to enforce conservation.
        """
        zone_ids = list(self.net.keys())
        n_zones = len(zone_ids)
        id_to_index = {zone_id: idx for idx, zone_id in enumerate(zone_ids)}

        self.Q = np.zeros((n_zones, n_zones))

        for i in zone_ids:
            for j, props in self.net[i]["neighbors"].items():
                if j not in id_to_index:
                    continue

                inx_i = id_to_index[i]
                inx_j = id_to_index[j]
                q_ij = props.get("vol_rate", 0)
                q_ji = self.net[j]["neighbors"].get(i, {}).get("vol_rate", 0)

                if unidirectional:
                    net_flow = q_ij - q_ji
                    if net_flow > 0:
                        self.Q[inx_i, inx_j] = net_flow
                    elif net_flow < 0:
                        self.Q[inx_j, inx_i] = -net_flow
                else:
                    if q_ij > 0:
                        self.Q[inx_i, inx_j] = q_ij

        # Set diagonal entries to ensure flow conservation (negative row sum)
        for i in range(n_zones):
            self.Q[i, i] = -np.sum(self.Q[i, :])

    def writeOpenFOAMScalarField(
            self,
            output_name: str,
            scalar: Union[Sequence[float], Sequence[int]],
            path: Optional[str] = None,
            input_name: Optional[str] = None
    ) -> None:
        """
        Write or replace a scalar field in an OpenFOAM field file, preserving header and formatting.

        This method reads an existing OpenFOAM scalar field file (e.g., `Vc`) and creates a new file
        (or overwrites an existing one) with the same structure, but replaces the values inside
        the field block with those provided in `scalar`.

        Parameters:
            output_name (str): Name of the output file to write (e.g., "compartment").
            scalar (Sequence[float] or Sequence[int]): List or array of scalar values to insert.
            path (Optional[str]): Directory path to the OpenFOAM time directory (e.g., "case/0.3/").
                                  If not specified, defaults to `self.part_to_write_foam`.
            input_name (Optional[str]): Name of the input file to use as a template (default: "Vc").

        File Behavior:
            - All header and metadata from the input file are preserved.
            - The `object` entry in the header is renamed to match `output_name`.
            - The data block (between parentheses) is replaced line-by-line with the new `scalar` values.

        Raises:
            IndexError: If the number of values in `scalar` does not match the number of entries in the input file's field block.
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

    def writeOpenFOAMVectorField(
            self,
            output_name: str,
            vector: Union[np.ndarray, Sequence[Sequence[float]]],
            path: Optional[str] = None,
            input_name: Optional[str] = None
    ) -> None:
        """
        Write or replace a vector field in an OpenFOAM field file, preserving header and formatting.

        This method reads an existing OpenFOAM vector field file (e.g., "U") and writes a new one
        with the same structure but updated vector values.

        Parameters:
            output_name (str): Name of the output field file (e.g., "convolution").
            vector (array-like): A 2D array or list of shape (N, 3) containing vector values.
            path (Optional[str]): Directory path to the OpenFOAM time folder. Defaults to `self.part_to_write_foam`.
            input_name (Optional[str]): Name of the input file to use as a format template (default: "U").

        Behavior:
            - Preserves all original lines outside the data block.
            - Replaces the 'object' name with the new `output_name`.
            - Replaces the data block between parentheses with the new vector values in OpenFOAM format.

        Raises:
            ValueError: If `vector` does not have shape (N, 3).
            IndexError: If the number of vectors does not match the number of entries in the data block.
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
