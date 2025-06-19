# PFCM
Plug Flow Compartmentalization for Compartmental Models
Work in Progress

This repository contains functions and example scripts used for plug flow compartmentalization of velocity fields obtained from OpenFOAM simulations (finite volume data format).

Author: Ittisak Promma, Ph.D.
Contact: itpromma@outlook.com
---
## Main Algorithm: Plug Flow Compartment Construction

1. **Construct a connection map**

2. **Initialize** a set `unprocessed_mesh_id` that includes all mesh IDs.

3. **While** `len(unprocessed_mesh_id) > 0`:

   3.1. Set `i = 0`\
   3.2. Initialize a list `plan_j = [unprocessed_mesh_id[0]]`\
   3.3. **While** `i < len(plan_j)`:

   - 3.3.1. Extract the following from mesh `plan_j[i]`:
     - `u_c`: velocity vector\
     - `r_c`: center coordinate\

   - 3.3.2. Identify all **face neighbors** of the current mesh → `n_f` \
   - 3.3.3. Determine the set of **plane neighbors** from `n_f` → `n_p`\
   - 3.3.4. For each neighbor mesh `k` in `n_f`, compute the directional distance:

     ```
     d_k = | dot(u_c, (r_k - r_c)) | / ||u_c||
     ```

   - 3.3.5. Define `accepted_mesh` as the set of neighbors that satisfy:\
     - Are among the `n_p` closest neighbors to the plane (i.e., smallest `d_k`)\
     - Have velocity aligned with `u_c`, i.e., `dot(u_c, u_k) >= 0\

   - 3.3.6. Append `accepted_mesh` to `plan_j` (if not already present) \
   - 3.3.7. Increment `i` by 1

    3.4. Update the connection map based on the final `plan_j`\
    3.5. Remove all meshes in `plan_j` from `unprocessed_mesh_id`
---
