import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Step 1: Reservoir Type Selection
# --------------------------
st.title("Reservoir Material Balance Simulator")
reservoir_type = st.radio("Select Reservoir Type", ("Oil Reservoir", "Gas Reservoir"))

# --------------------------
# Step 2: Drive Mechanism Selection
# --------------------------
if reservoir_type == "Oil Reservoir":
    drive_mechanism = st.selectbox(
        "Select Drive Mechanism",
        ("Solution Gas Drive", "Water Drive", "Gas Cap Drive")
    )
else:
    drive_mechanism = st.selectbox(
        "Select Drive Mechanism",
        ("Natural Depletion", "Water Drive")
    )

# --------------------------
# Step 3: Input Parameters
# --------------------------
st.header("Reservoir Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    pi = st.number_input("Initial Pressure (psi)", value=3000)
    p = st.number_input("Current Pressure (psi)", value=2500)
    temp = st.number_input("Temperature (°F)", value=200)

with col2:
    if reservoir_type == "Oil Reservoir":
        api = st.number_input("API Gravity", value=35)
        rs = st.number_input("Solution GOR (scf/STB)", value=600)
    else:
        z = st.number_input("Gas Compressibility Factor", value=0.85)

with col3:
    swi = st.number_input("Initial Water Saturation", value=0.2)
    cw = st.number_input("Water Compressibility (psi⁻¹)", 3e-6)
    cf = st.number_input("Formation Compressibility (psi⁻¹)", 4e-6)

# --------------------------
# Step 4: PVT Correlations
# --------------------------
def calculate_rs(api, temp, p):
    """Standing's correlation for solution GOR"""
    return 0.0362 * api * (p ** 1.205) / (10 ** (0.0125 * api) * 10 ** (0.00091 * temp))

def calculate_bo(rs, api, temp, p):
    """Vasquez-Beggs correlation for oil FVF"""
    return 1 + 4.677e-4 * rs + (temp - 60) * (api / 131.5) * (1.751e-5 * rs - 1.811e-8)

if reservoir_type == "Oil Reservoir":
    rs_calculated = calculate_rs(api, temp, p)
    bo = calculate_bo(rs_calculated, api, temp, p)
    st.write(f"Calculated Solution GOR: {rs_calculated:.2f} scf/STB")
    st.write(f"Calculated Oil FVF: {bo:.3f} bbl/STB")

# --------------------------
# Step 5: Material Balance Calculations
# --------------------------
st.header("Material Balance Results")

if reservoir_type == "Oil Reservoir":
    # Oil material balance equation
    n = st.number_input("Oil in Place (STB)", value=1e6)
    n_p = st.number_input("Cumulative Oil Production (STB)", value=1e5)
    w_p = st.number_input("Cumulative Water Production (STB)", value=0)
    
    # Simplified material balance calculation
    delta_p = pi - p
    f = n_p * bo
    e_o = (bo - 1)  # Simplified expansion term
    n_calculated = f / e_o
    
    st.success(f"Calculated Original Oil in Place (N): {n_calculated/1e6:.2f} MMSTB")

else:
    # Gas material balance calculation
    g = st.number_input("Gas in Place (MSCF)", value=1e9)
    g_p = st.number_input("Cumulative Gas Production (MSCF)", value=1e8)
    
    # p/Z vs Gp plot calculation
    p_z = [pi / z, p / z]
    gp = [0, g_p]
    
    st.success(f"Calculated Original Gas in Place (G): {g_p / (1 - p/z * pi/z):.2f} MSCF")

# --------------------------
# Step 6: Plotting
# --------------------------
st.header("Diagnostic Plots")

if reservoir_type == "Oil Reservoir":
    # Create pressure vs production plot
    fig1, ax1 = plt.subplots()
    pressure = np.linspace(pi, pi*0.5, 100)
    production = n * (1 - pressure/pi)
    
    ax1.plot(production, pressure)
    ax1.set_xlabel("Cumulative Production (STB)")
    ax1.set_ylabel("Reservoir Pressure (psi)")
    ax1.set_title("Pressure vs Production")
    st.pyplot(fig1)

else:
    # Create p/Z plot
    fig2, ax2 = plt.subplots()
    ax2.plot(gp, p_z, 'o-')
    ax2.set_xlabel("Cumulative Gas Production (MSCF)")
    ax2.set_ylabel("p/Z (psi)")
    ax2.set_title("p/Z vs Cumulative Gas Production")
    st.pyplot(fig2)

# --------------------------
# Step 7: Additional Features
# --------------------------
st.sidebar.header("Simulation Parameters")
num_points = st.sidebar.slider("Number of Simulation Points", 10, 100, 50)
simulation_time = st.sidebar.number_input("Simulation Time (years)", 10)

if st.button("Run Full Simulation"):
    with st.spinner("Running simulation..."):
        # Add full simulation logic here
        st.balloons()