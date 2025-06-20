from PFCoM import CMGenerator

# === Configuration ===
case_name = "OF_case/"
time_step = "0.35/"
smoothing_level = 3
n_smoothing_iterations = 5

# === Initialize the Compartmental Model ===
CM = CMGenerator(case_name, time_step)

# === Step 1: Initial Smoothing of Velocity Field ===
CM.calConvolutionalVelocity(level=smoothing_level)

# === Step 2: Iteratively Reapply Smoothing to Intensify ===
for _ in range(n_smoothing_iterations):
    CM.calConvolutionalVelocity(u=CM.U_convol, level=smoothing_level)

# === Step 3: Perform Flow-Based Compartmentalization ===
CM.compartmentalization(method='PFC', u=CM.U_convol)

# === Step 4: Construct and Visualize the Compartment Network ===
CM.constructNetDict()
CM.drawNetDict()

# === Step 5: Export Results to OpenFOAM Format ===
CM.writeOpenFOAMScalarField("compartments", CM.element_to_compartment)
CM.writeOpenFOAMVectorField("U_convol", CM.U_convol)
