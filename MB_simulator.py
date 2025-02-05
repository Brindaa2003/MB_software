import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title for the web app
st.title("Material Balance Simulator: Depletion Drive")

# Sidebar inputs for reservoir properties
st.sidebar.header("Input Reservoir Properties")
N = st.sidebar.number_input("OOIP (STB)", min_value=1_000.0, value=1_000_000.0, step=1_000.0)
P_i = st.sidebar.number_input("Initial Pressure (psi)", min_value=100.0, value=3000.0, step=10.0)
P_b = st.sidebar.number_input("Bubble Point Pressure (psi)", min_value=100.0, value=2000.0, step=10.0)
B_oi = st.sidebar.number_input("Initial Oil Formation Volume Factor (rb/stb)", min_value=0.1, value=1.2, step=0.01)
c_o = st.sidebar.number_input("Oil Compressibility (1/psi)", min_value=1e-6, value=1e-5, step=1e-6)
c_r = st.sidebar.number_input("Formation Compressibility (1/psi)", min_value=1e-6, value=1e-5, step=1e-6)

# Pressure decline simulation
st.sidebar.header("Simulation Parameters")
num_steps = st.sidebar.slider("Number of Pressure Steps", min_value=10, max_value=100, value=50)
final_pressure = st.sidebar.number_input("Final Pressure (psi)", min_value=100.0, value=1000.0, step=10.0)
pressures = np.linspace(P_i, final_pressure, num_steps)

# Compute expansion terms
B_o = B_oi * (1 + c_o * (P_i - pressures))  # Oil formation volume factor
E_o = B_o - B_oi  # Oil expansion term
E_f = c_r * (P_i - pressures)  # Formation compressibility term

# Material Balance Equation (Depletion Drive)
F = N * (E_o + E_f)

# Display results
st.subheader("Simulation Results")
st.write("Cumulative Withdrawals (F) vs. Pressure:")
results = {"Pressure (psi)": pressures, "Cumulative Withdrawals (F) [STB]": F}
st.dataframe(results)

# Plot results
st.subheader("Visualization")
fig, ax = plt.subplots()
ax.plot(pressures, F, label="Cumulative Withdrawals (F)", color="blue")
ax.set_xlabel("Pressure (P) [psi]")
ax.set_ylabel("Cumulative Withdrawals (F) [STB]")
ax.set_title("Depletion Drive: Pressure vs. Cumulative Withdrawals")
ax.grid()
ax.legend()
st.pyplot(fig)
