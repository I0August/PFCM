'''
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
                    if neighbor != i and np.dot(u_unit_i, u_unit_shared[neighbor]) > 0.5:
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

    def compartmentalization(self, method='average', u=None, flow_similarity=0.99, eps = 1e-10):
        if u is None:
            u = self.U
        u = np.asarray(u)
        u_norm = np.linalg.norm(u, axis=1)
        u_unit = np.divide(u, u_norm[:, np.newaxis], out=np.zeros_like(u), where=u_norm[:, np.newaxis] != 0)

        unprocessed_meshes = set(range(self.n_elements))
        total_elements = len(unprocessed_meshes)
        pbar = tqdm(total=total_elements, desc="Compartmentalizing", unit="mesh")

        while unprocessed_meshes:
            processed_mesh = set()
            initial = next(iter(unprocessed_meshes))
            mesh_set = {initial}
            processing_set = {initial}

            # Initial velocity info
            u_init = u[initial]
            u_unit_init = u_unit[initial]
            compartment_velocity_sum = np.copy(u_init)
            compartment_volume_sum = self.element_to_volume[initial]

            while processing_set:
                plane = processing_set.pop()
                processed_mesh.add(plane)

                neighbors = list(self.element_to_neighbors[plane] & unprocessed_meshes)
                r_c = self.element_to_coordinates[plane]
                u_c = u[plane]

                # Geometric cut check
                accepted_neighbors = []
                for neighbor in neighbors:
                    points = self.element_to_points[neighbor]
                    point_coordinates = np.array([self.point_to_coordinates[pt] for pt in points])
                    check_cut_element = np.dot(point_coordinates - r_c, u_c)
                    if np.any(check_cut_element > eps) and np.any(check_cut_element < -eps):
                        accepted_neighbors.append(neighbor)

                # Choose the velocity reference for similarity check
                if method == 'local':
                    ref_velocity = u_unit[plane]
                elif method == 'average':
                    avg_velocity = compartment_velocity_sum / (compartment_volume_sum + 1e-12)
                    norm = np.linalg.norm(avg_velocity)
                    ref_velocity = avg_velocity / norm if norm > 1e-12 else np.zeros_like(avg_velocity)
                elif method == 'initial':
                    ref_velocity = u_unit_init
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Flow similarity check
                accepted_neighbors_velocity = {
                    nid for nid in accepted_neighbors
                    if np.dot(ref_velocity, u_unit[nid]) > flow_similarity
                }

                # Update compartment aggregates (only for v2)
                if method == 'average':
                    for nid in accepted_neighbors_velocity:
                        compartment_velocity_sum += u[nid]
                        compartment_volume_sum += self.element_to_volume[nid]

                processing_set.update(accepted_neighbors_velocity - processed_mesh)
                mesh_set.update(accepted_neighbors_velocity)

            unprocessed_meshes -= mesh_set
            pbar.update(len(mesh_set))
            self.compartment_to_elements.append(mesh_set)

        pbar.close()
        self.n_compartments = len(self.compartment_to_elements)
        self.constructElementToZone()
        self.constructCompartmentToVolume()
        self.constructCompartmentToShell()
        self.constructCompartmentToNeighbors()
'''