from src.comp import *

# Initialize compartmental model for the case 'pitzDaily' at time step '0.3'
CM = CMGenerator("pitzDaily/", "0.3/")

# # Compute initial convolutional velocity field (smoothed velocity) at level 2
# CM.calConvolutionalVelocity(level=5)
#
# # Reapply convolutional velocity calculation iteratively to intensify smoothing
# for _ in range(5):
#     CM.calConvolutionalVelocity(u=CM.U_convol, level=5)

# Perform compartmentalization based on flow similarity (PFC: Predefined Flow Clustering)
CM.compartmentalization(method='PFC', u=CM.U)

CM.constructNetDict()

# Write scalar field 'compartment' to an OpenFOAM-compatible file using the zone mapping
CM.writeOpenFOAMScalarField("compartments", CM.element_to_compartment)

# Write the smoothed velocity field to an OpenFOAM-compatible vector field file
#CM.writeOpenFOAMVectorField("U_convol", CM.U_convol)
