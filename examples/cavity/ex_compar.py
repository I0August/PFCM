from PFCoM import CMGenerator
from time import process_time

# === Configuration ===
case_name = "OF_case/"
time_step = "1/"

# === Initialize the Compartmental Model ===
CM = CMGenerator(case_name, time_step)

# === Step 1: Perform Flow-Based Compartmentalization ===
CM.compartmentalization(theta_deg=10)

# === Step 2: Construct and Visualize the Compartment Network ===
CM.constructNetDict()
CM.drawNetDict(unidirectional=False)

# === Step 3: Export Results to OpenFOAM Format ===
CM.writeOpenFOAMScalarField("compartments", CM.element_to_compartment)
