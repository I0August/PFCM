from PFCoM import CMGenerator

# === Configuration ===
case_name = "OF_case/"
time_step = "0.35/"

# === Initialize the Compartmental Model ===
CM = CMGenerator(case_name, time_step)

# === Step 1: Perform Flow-Based Compartmentalization ===
CM.compartmentalization(method='PFC', u=CM.U, flow_similarity=0.99)

# === Step 2: Construct and Visualize the Compartment Network ===
CM.constructNetDict()
CM.drawNetDict()

# === Step 3: Export Results to OpenFOAM Format ===
CM.writeOpenFOAMScalarField("compartments", CM.element_to_compartment)