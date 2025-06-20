# PFCoM (Work in Progress)
Plug Flow Compartmentalization for Compartmental Models


This repository contains functions and example scripts used for plug flow compartmentalization of velocity fields obtained from OpenFOAM simulations (finite volume data format).

Developer: Ittisak Promma, Ph.D.\
Contact: itpromma@outlook.com

---
## Main Algorithm:

1. **Construct a connection map**

2. **Initialize** a set `unprocessed_mesh_id` that includes all mesh IDs.

3. **While** `len(unprocessed_mesh_id) > 0`:

   3.1. Set `i = 0`\
   3.2. Initialize a list `plan_j = [unprocessed_mesh_id[0]]`\
   3.3. **While** `i < len(plan_j)`:

   - 3.3.1. Extract the following from mesh `plan_j[i]`:
     - `u_c`: velocity vector
     - `r_c`: center coordinate

   - 3.3.2. Define accepted_mesh as the subset of neighbors satisfying all the following conditions:
        - At least one corner point of the neighbor satisfies dot(u_c, r_k - r_c) ≥ 0
        - At least one corner point of the neighbor satisfies dot(u_c, r_k - r_c) < 0
        - The velocity vector of the neighbor u_k is aligned with u_c, i.e., dot(u_c, u_k) ≥ 0
        (Here, r_k denotes the coordinates of a corner point of the neighbor)

   - 3.3.3. Append `accepted_mesh` to `plan_j` (if not already present) 
   - 3.3.4. Increment `i` by 1

    3.4. Update the connection map based on the final `plan_j`\
    3.5. Remove all meshes in `plan_j` from `unprocessed_mesh_id`
---

