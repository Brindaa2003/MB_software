import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_oil_fvf(pressure, pb, bo_pb, co, api):
    """Calculate oil formation volume factor"""
    if pressure >= pb:
        return bo_pb
    else:
        return bo_pb * np.exp(co * (pb - pressure))

def calculate_gas_fvf(pressure, temp, z_factor):
    """Calculate gas formation volume factor"""
    return 0.00504 * z_factor * temp / pressure

def calculate_z_factor(pressure, temp, gas_sg):
    """Simplified Z-factor calculation using Standing-Katz correlation"""
    tpc = 168 + 325 * gas_sg - 12.5 * gas_sg ** 2
    ppc = 677 + 15.0 * gas_sg - 37.5 * gas_sg ** 2
    tpr = temp / tpc
    ppr = pressure / ppc
    z = 1.0 - (ppr / (4.0 * tpr)) * (1.0 - np.exp(-4.0 * ppr / tpr))
    return z

def calculate_water_influx(reservoir_type, time, pressure_drop, reservoir_params):
    """Calculate water influx based on reservoir type"""
    if reservoir_type == "No Water Drive":
        return 0
    
    wei = reservoir_params.get('wei', 1e6)  # Water influx constant
    
    if reservoir_type == "Strong Water Drive":
        return wei * pressure_drop * time * 0.1
    elif reservoir_type == "Moderate Water Drive":
        return wei * pressure_drop * time * 0.05
    elif reservoir_type == "Weak Water Drive":
        return wei * pressure_drop * time * 0.02
    
    return 0

def calculate_drive_indices(pressure, initial_pressure, pb, reservoir_type):
    """Calculate drive mechanism indices"""
    # Simplified drive indices calculation
    if pressure >= pb:
        # Above bubble point
        if reservoir_type in ["Strong Water Drive", "Moderate Water Drive", "Weak Water Drive"]:
            water_drive_index = 0.7 if "Strong" in reservoir_type else 0.4 if "Moderate" in reservoir_type else 0.2
            fluid_expansion_index = 1 - water_drive_index
            return {
                "Water Drive": water_drive_index,
                "Fluid Expansion": fluid_expansion_index,
                "Solution Gas": 0.0,
                "Gas Cap": 0.0
            }
        elif reservoir_type == "Gas Cap Drive":
            return {
                "Water Drive": 0.0,
                "Fluid Expansion": 0.3,
                "Solution Gas": 0.0,
                "Gas Cap": 0.7
            }
        else:  # Depletion Drive
            return {
                "Water Drive": 0.0,
                "Fluid Expansion": 1.0,
                "Solution Gas": 0.0,
                "Gas Cap": 0.0
            }
    else:
        # Below bubble point
        if reservoir_type == "Solution Gas Drive":
            return {
                "Water Drive": 0.0,
                "Fluid Expansion": 0.2,
                "Solution Gas": 0.8,
                "Gas Cap": 0.0
            }
        elif reservoir_type == "Gas Cap Drive":
            return {
                "Water Drive": 0.0,
                "Fluid Expansion": 0.2,
                "Solution Gas": 0.3,
                "Gas Cap": 0.5
            }
        else:  # Water Drive cases
            water_drive_index = 0.6 if "Strong" in reservoir_type else 0.4 if "Moderate" in reservoir_type else 0.2
            return {
                "Water Drive": water_drive_index,
                "Fluid Expansion": 0.2,
                "Solution Gas": 1 - water_drive_index - 0.2,
                "Gas Cap": 0.0
            }

def material_balance_calculation(reservoir_params, production_data, reservoir_type):
    """Perform material balance calculations"""
    initial_pressure = reservoir_params['initial_pressure']
    reservoir_temp = reservoir_params['reservoir_temp']
    bubble_point = reservoir_params['bubble_point']
    initial_oil_fvf = reservoir_params['initial_oil_fvf']
    oil_compressibility = reservoir_params['oil_compressibility']
    gas_sg = reservoir_params['gas_sg']
    
    results = []
    current_pressure = initial_pressure
    cumulative_we = 0
    
    for time, prod in enumerate(production_data.iterrows(), 1):
        _, prod_data = prod
        
        # Calculate fluid properties
        z_factor = calculate_z_factor(current_pressure, reservoir_temp, gas_sg)
        current_oil_fvf = calculate_oil_fvf(current_pressure, bubble_point, 
                                          initial_oil_fvf, oil_compressibility,
                                          reservoir_params['api_gravity'])
        current_gas_fvf = calculate_gas_fvf(current_pressure, reservoir_temp, z_factor)
        
        # Calculate water influx
        pressure_drop = initial_pressure - current_pressure
        water_influx = calculate_water_influx(reservoir_type, time, pressure_drop, reservoir_params)
        cumulative_we += water_influx
        
        # Calculate drive indices
        drive_indices = calculate_drive_indices(current_pressure, initial_pressure, 
                                              bubble_point, reservoir_type)
        
        # Store results
        results.append({
            'Time': time,
            'Pressure': current_pressure,
            'Oil_FVF': current_oil_fvf,
            'Gas_FVF': current_gas_fvf,
            'Z_Factor': z_factor,
            'Np': prod_data['cumulative_oil'],
            'Gp': prod_data['cumulative_gas'],
            'We': cumulative_we,
            'Water_Drive_Index': drive_indices['Water Drive'],
            'Fluid_Expansion_Index': drive_indices['Fluid Expansion'],
            'Solution_Gas_Index': drive_indices['Solution Gas'],
            'Gas_Cap_Index': drive_indices['Gas Cap']
        })
        
        # Update pressure for next iteration
        if reservoir_type == "Strong Water Drive":
            current_pressure *= 0.98
        elif reservoir_type == "Moderate Water Drive":
            current_pressure *= 0.96
        elif reservoir_type == "Gas Cap Drive":
            current_pressure *= 0.94
        else:
            current_pressure *= 0.92
    
    return pd.DataFrame(results)

# Streamlit UI
st.title('Advanced Reservoir Material Balance Simulator')

# Sidebar for reservoir type selection and input parameters
st.sidebar.header('Reservoir Configuration')
reservoir_type = st.sidebar.selectbox(
    'Select Reservoir Type',
    ['Solution Gas Drive', 'Gas Cap Drive', 'Strong Water Drive', 
     'Moderate Water Drive', 'Weak Water Drive', 'No Water Drive']
)

st.sidebar.header('Reservoir Parameters')
initial_pressure = st.sidebar.number_input('Initial Reservoir Pressure (psia)', value=4000.0)
reservoir_temp = st.sidebar.number_input('Reservoir Temperature (°F)', value=180.0)
bubble_point = st.sidebar.number_input('Bubble Point Pressure (psia)', value=3000.0)
initial_oil_fvf = st.sidebar.number_input('Initial Oil FVF (Bo)', value=1.2)
oil_compressibility = st.sidebar.number_input('Oil Compressibility (1/psi)', value=1.5e-5)
gas_sg = st.sidebar.number_input('Gas Specific Gravity', value=0.65)
api_gravity = st.sidebar.number_input('Oil API Gravity', value=35.0)

if reservoir_type in ["Strong Water Drive", "Moderate Water Drive", "Weak Water Drive"]:
    wei = st.sidebar.number_input('Water Influx Constant (wei)', value=1e6)
else:
    wei = 0

# Production data input
st.header('Production Data')
uploaded_file = st.file_uploader("Upload production data CSV (columns: cumulative_oil, cumulative_gas)", type="csv")

if uploaded_file is not None:
    production_data = pd.read_csv(uploaded_file)
else:
    # Create sample production data
    time_steps = 10
    sample_data = {
        'cumulative_oil': np.linspace(0, 1000000, time_steps),
        'cumulative_gas': np.linspace(0, 800000000, time_steps)
    }
    production_data = pd.DataFrame(sample_data)
    st.write("Using sample production data:")
    st.write(production_data)

# Perform calculations
reservoir_params = {
    'initial_pressure': initial_pressure,
    'reservoir_temp': reservoir_temp + 459.67,  # Convert to Rankine
    'bubble_point': bubble_point,
    'initial_oil_fvf': initial_oil_fvf,
    'oil_compressibility': oil_compressibility,
    'gas_sg': gas_sg,
    'api_gravity': api_gravity,
    'wei': wei
}

results = material_balance_calculation(reservoir_params, production_data, reservoir_type)

# Display results
st.header('Results')
st.write(results)

# Create plots
st.header('Visualization')
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pressure vs Cumulative Oil Production
ax1.plot(results['Np'], results['Pressure'], 'b-')
ax1.set_xlabel('Cumulative Oil Production (STB)')
ax1.set_ylabel('Pressure (psia)')
ax1.set_title('P vs Np')
ax1.grid(True)

# Drive Indices vs Time
ax2.plot(results['Time'], results['Water_Drive_Index'], 'b-', label='Water Drive')
ax2.plot(results['Time'], results['Fluid_Expansion_Index'], 'r-', label='Fluid Expansion')
ax2.plot(results['Time'], results['Solution_Gas_Index'], 'g-', label='Solution Gas')
ax2.plot(results['Time'], results['Gas_Cap_Index'], 'y-', label='Gas Cap')
ax2.set_xlabel('Time')
ax2.set_ylabel('Drive Index')
ax2.set_title('Drive Indices vs Time')
ax2.legend()
ax2.grid(True)

st.pyplot(fig1)

# Additional plots
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

# Water Influx vs Time
ax3.plot(results['Time'], results['We'], 'g-')
ax3.set_xlabel('Time')
ax3.set_ylabel('Cumulative Water Influx (bbl)')
ax3.set_title('Water Influx vs Time')
ax3.grid(True)

# Oil FVF vs Pressure
ax4.plot(results['Pressure'], results['Oil_FVF'], 'r-')
ax4.set_xlabel('Pressure (psia)')
ax4.set_ylabel('Oil FVF (bbl/STB)')
ax4.set_title('Bo vs P')
ax4.grid(True)

st.pyplot(fig2)