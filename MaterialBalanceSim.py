import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------
def calculate_Bw(res_p, t):
    """Calculate water formation volume factor using McCain's correlation."""
    delta_VwP = (-1.95301e-9 * res_p * t - 1.72834e-13 * (res_p**2) * t - 
                 3.58922e-7 * res_p - 2.25341e-10 * (res_p**2))
    delta_VwT = (-1.0001e-2 + 1.33391e-4 * t + 5.50654e-7 * (t**2))
    Bw = (1 + delta_VwP) * (1 + delta_VwT)
    return Bw

def compute_z(Sg, p_val, t):
    """Compute Z-factor using Newton-Raphson method."""
    T_pc = 120.1 + 425 * Sg - 62.9 * (Sg**2)
    P_pc = 671.1 + 14 * Sg - 34.3 * (Sg**2)
    T_pr = (t + 460) / T_pc
    P_pr = p_val / P_pc
    Z = 1.0
    tol = 1e-6
    for _ in range(100):
        rho_r = 0.27 * P_pr / (Z * T_pr)
        term1 = (0.3265 - 1.07/T_pr - 0.5339/(T_pr**3) + 0.01569/(T_pr**4) - 
                 0.05165/(T_pr**5)) * rho_r
        term2 = (0.5475 - 0.7361/T_pr + 0.1844/(T_pr**2)) * (rho_r**2)
        term3 = -0.1056 * ((-0.7361/T_pr) + (0.1844/(T_pr**2))) * (rho_r**5)
        term4 = 0.6134 * (1 + 0.7210*(rho_r**2)) * ((rho_r**2)/(T_pr**3)) * \
                np.exp(-0.7210*(rho_r**2))
        F_val = 1 + term1 + term2 + term3 + term4
        f_Z = Z - F_val
        delta = 1e-6
        Z_plus = Z + delta
        rho_r_plus = 0.27 * P_pr / (Z_plus * T_pr)
        term1_plus = (0.3265 - 1.07/T_pr - 0.5339/(T_pr**3) + 0.01569/(T_pr**4) - 
                      0.05165/(T_pr**5)) * rho_r_plus
        term2_plus = (0.5475 - 0.7361/T_pr + 0.1844/(T_pr**2)) * (rho_r_plus**2)
        term3_plus = -0.1056 * ((-0.7361/T_pr) + (0.1844/(T_pr**2))) * (rho_r_plus**5)
        term4_plus = 0.6134 * (1 + 0.7210*(rho_r_plus**2)) * ((rho_r_plus**2)/(T_pr**3)) * \
                     np.exp(-0.7210*(rho_r_plus**2))
        F_plus = 1 + term1_plus + term2_plus + term3_plus + term4_plus
        f_Z_plus = Z_plus - F_plus
        Z_minus = Z - delta
        rho_r_minus = 0.27 * P_pr / (Z_minus * T_pr)
        term1_minus = (0.3265 - 1.07/T_pr - 0.5339/(T_pr**3) + 0.01569/(T_pr**4) - 
                       0.05165/(T_pr**5)) * rho_r_minus
        term2_minus = (0.5475 - 0.7361/T_pr + 0.1844/(T_pr**2)) * (rho_r_minus**2)
        term3_minus = -0.1056 * ((-0.7361/T_pr) + (0.1844/(T_pr**2))) * (rho_r_minus**5)
        term4_minus = 0.6134 * (1 + 0.7210*(rho_r_minus**2)) * ((rho_r_minus**2)/(T_pr**3)) * \
                      np.exp(-0.7210*(rho_r_minus**2))
        F_minus = 1 + term1_minus + term2_minus + term3_minus + term4_minus
        f_Z_minus = Z_minus - F_minus
        derivative = (f_Z_plus - f_Z_minus) / (2 * delta)
        if abs(derivative) < 1e-12:
            break
        Z_new = Z - f_Z / derivative
        if abs(Z_new - Z) < tol:
            Z = Z_new
            break
        Z = Z_new
    return Z

def calculate_gas_viscosity(p, t, sg_g, Z):
    """Calculate gas viscosity using Lee et al. correlation."""
    M = 28.97 * sg_g
    rho_g = (28.97 * sg_g * p) / (Z * 10.73 * (t + 460))
    K = ((9.4 + 0.02 * M) * ((t + 460)**1.5)) / (209 + 19 * M + t + 460)
    X = 3.5 + (986 / (t + 460)) + 0.01 * M
    Y = 2.4 - 0.2 * X
    viscosity = 1e-4 * K * np.exp(X * ((rho_g / 62.4)**Y))
    return viscosity

# -----------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------
st.set_page_config(page_title="Material Balance Simulator", layout="wide")
st.title("Material Balance Simulator for Oil and Gas Reservoirs")

# Reservoir Type Selection
reservoir_type = st.selectbox("Select Reservoir Type", ["Oil Reservoir", "Gas Reservoir"])

# -----------------------------------------------------
# Oil Reservoir Section
# -----------------------------------------------------
if reservoir_type == "Oil Reservoir":
    with st.sidebar:
        st.header("Reservoir Parameters")
        res_p = st.number_input("Initial Reservoir Pressure (Psi)", min_value=100.0, value=3000.0, step=100.0)
        api = st.number_input("API Gravity", min_value=10, max_value=50, value=30, step=1)
        t = st.number_input("Reservoir Temperature (F)", min_value=100, max_value=300, value=180, step=5)
        sg_g = st.number_input("Gas Specific Gravity", min_value=0.5, max_value=1.5, value=0.7, step=0.01)
        r_s_input = st.number_input("Initial Solution GOR (SCF/STB)", min_value=100, value=500, step=50)
        c_f = st.number_input("Rock Compressibility (1/psi)", min_value=1e-7, max_value=1e-4, value=3e-6, step=1e-7, format="%.2e")
        c_w = st.number_input("Water Compressibility (1/psi)", min_value=1e-7, max_value=1e-4, value=3e-6, step=1e-7, format="%.2e")
        S_wc = st.number_input("Initial Water Saturation (fraction)", min_value=0.0, max_value=0.5, value=0.2, step=0.01)
        m = st.number_input("Gas Cap Ratio (m)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        
        st.header("Water Influx Parameters")
        influx_present = st.checkbox("Water Influx Present?")
        if influx_present:
            W_ei = st.number_input("Aquifer Volume (MMft^3)", min_value=0.0, value=1000.0, step=100.0)
    
    uploaded_file = st.file_uploader("Upload Production Data (Excel/CSV)", type=["xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                history_data = pd.read_csv(uploaded_file)
            else:
                history_data = pd.read_excel(uploaded_file)
            
            required_columns = ['Date', 'Pressure', 'Cum Oil Production', 'Cum Gas Production', 'Cum Water Production']
            if not all(col in history_data.columns for col in required_columns):
                st.error(f"Uploaded file must contain these columns: {', '.join(required_columns)}")
                st.stop()
            
            dates = pd.to_datetime(history_data['Date'])
            pressure_data = history_data['Pressure'].values
            Np_data = history_data['Cum Oil Production'].values
            Gp_data = history_data['Cum Gas Production'].values
            Wp_data = history_data['Cum Water Production'].values
            n_points = len(pressure_data)
            
            ### Fluid Properties Calculation
            # Bubble Point Pressure
            p_b = 18.2 * (((r_s_input / sg_g) ** 0.83) * (10 ** (0.00091 * t - 0.0125 * api)) - 1.4)
            
            # Solution GOR (Rs)
            Rs_data = np.zeros(n_points)
            for i in range(n_points):
                if pressure_data[i] > p_b:
                    Rs_data[i] = r_s_input
                else:
                    Rs_data[i] = sg_g * (((pressure_data[i] + 14.7) * (10 ** (0.0125 * api))) / 
                                       (18 * (10 ** (0.00091 * t)))) ** 1.2048
            
            # Oil Formation Volume Factor (Bo)
            sg_o = 141.5 / (api + 131.5)
            rs_bp = sg_g * ((p_b * (10 ** (0.0125 * api))) / (18 * (10 ** (0.00091 * t)))) ** 1.2048
            bo_b = 0.9759 + 0.00012 * (((rs_bp * (sg_g / sg_o) ** 0.5) + (1.25 * t)) ** 1.2)
            c_o_data = np.zeros(n_points)
            for i in range(n_points):
                c_o_data[i] = ((5 * rs_bp) + (17.2 * t) - (1180 * sg_g) + (12.61 * api) - 1433) / \
                              ((pressure_data[i] + 14.7) * 100000)
            Bo_data = np.zeros(n_points)
            for i in range(n_points):
                if pressure_data[i] <= p_b:
                    Bo_data[i] = 0.9759 + 0.00012 * (((Rs_data[i] * (sg_g / sg_o) ** 0.5) + (1.25 * t)) ** 1.2)
                else:
                    Bo_data[i] = bo_b * np.exp(-c_o_data[i] * (pressure_data[i] - p_b))
            
            # Gas Formation Volume Factor (Bg) and Z-factor
            Z_prod = np.array([compute_z(sg_g, p_val, t) for p_val in pressure_data])
            Bg_data = 0.005035 * Z_prod * (t + 460) / pressure_data
            
            # Water Formation Volume Factor (Bw)
            Bw_data = np.array([calculate_Bw(p_val, t) for p_val in pressure_data])
            
            # Viscosity
            v_od = 10 ** (10 ** (3.0324 - 0.02023 * api) * t ** (-1.163)) - 1
            Viscosity_data = np.zeros(n_points)
            for i in range(n_points):
                if pressure_data[i] <= p_b:
                    Viscosity_data[i] = (10.715 * (Rs_data[i] + 100) ** (-0.515)) * \
                                       v_od ** (5.44 * (Rs_data[i] + 150) ** (-0.338))
                else:
                    v_o_sat = (10.715 * (rs_bp + 100) ** (-0.515)) * \
                              v_od ** (5.44 * (r_s_input + 150) ** (-0.338))
                    Viscosity_data[i] = v_o_sat + 0.001 * (pressure_data[i] - p_b) * \
                                       ((0.024 * v_o_sat ** 1.6) + (0.038 * v_o_sat ** 0.56))
            
            # Initial Conditions
            B_oi = Bo_data[0]
            R_si = Rs_data[0]
            B_gi = Bg_data[0]
            
            ### Production Data Visualization
            st.header("Production Data Visualization")
            col1, col2, col3 = st.columns(3)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(dates, Np_data, 'b-', linewidth=2, label='Cum Oil Production')
                ax.set_xlabel('Time')
                ax.set_ylabel('Cum Oil Production')
                ax.legend()
                ax.grid()
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(dates, Gp_data, 'g-', linewidth=2, label='Cum Gas Production')
                ax.set_xlabel('Time')
                ax.set_ylabel('Cum Gas Production')
                ax.legend()
                ax.grid()
                st.pyplot(fig)
            with col3:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(dates, Wp_data, 'r-', linewidth=2, label='Cum Water Production')
                ax.set_xlabel('Time')
                ax.set_ylabel('Cum Water Production')
                ax.legend()
                ax.grid()
                st.pyplot(fig)
            
            ### Fluid Properties Visualization
            st.header("Fluid Properties")
            tab1, tab2, tab3 = st.tabs(["Solution GOR", "Formation Volume Factors", "Viscosity & Z-factor"])
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(pressure_data, Rs_data, 'b-', linewidth=2, label='Rs (SCF/STB)')
                ax.axvline(p_b, color='r', linestyle='--', linewidth=2, label=f'Pb: {p_b:.2f} psi')
                ax.set_xlabel('Pressure (psi)')
                ax.set_ylabel('Rs (SCF/STB)')
                ax.set_title('Solution Gas-Oil Ratio')
                ax.legend()
                ax.grid()
                st.pyplot(fig)
            with tab2:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                ax1.plot(pressure_data, Bo_data, 'g-', linewidth=2, label='Bo (RB/STB)')
                ax1.axvline(p_b, color='r', linestyle='--', linewidth=2, label=f'Pb: {p_b:.2f} psi')
                ax1.set_title('Oil Formation Volume Factor')
                ax1.set_xlabel('Pressure (psi)')
                ax1.set_ylabel('Bo (RB/STB)')
                ax1.legend()
                ax1.grid()
                ax2.plot(pressure_data, Bg_data, 'm-', linewidth=2, label='Bg (RB/SCF)')
                ax2.axvline(p_b, color='r', linestyle='--', linewidth=2, label=f'Pb: {p_b:.2f} psi')
                ax2.set_title('Gas Formation Volume Factor')
                ax2.set_xlabel('Pressure (psi)')
                ax2.set_ylabel('Bg (RB/SCF)')
                ax2.legend()
                ax2.grid()
                ax3.plot(pressure_data, Bw_data, 'c-', linewidth=2, label='Bw (RB/STB)')
                ax3.axvline(p_b, color='r', linestyle='--', linewidth=2, label=f'Pb: {p_b:.2f} psi')
                ax3.set_title('Water Formation Volume Factor')
                ax3.set_xlabel('Pressure (psi)')
                ax3.set_ylabel('Bw (RB/STB)')
                ax3.legend()
                ax3.grid()
                st.pyplot(fig)
            with tab3:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(pressure_data, Viscosity_data, 'orange', linewidth=2, label='Viscosity (cP)')
                ax1.axvline(p_b, color='r', linestyle='--', linewidth=2, label=f'Pb: {p_b:.2f} psi')
                ax1.set_title('Oil Viscosity')
                ax1.set_xlabel('Pressure (psi)')
                ax1.set_ylabel('Viscosity (cP)')
                ax1.legend()
                ax1.grid()
                ax2.plot(pressure_data, Z_prod, 'k-', linewidth=2, label='Z-factor')
                ax2.axvline(p_b, color='r', linestyle='--', linewidth=2, label=f'Pb: {p_b:.2f} psi')
                ax2.set_title('Z-factor')
                ax2.set_xlabel('Pressure (psi)')
                ax2.set_ylabel('Z-factor')
                ax2.legend()
                ax2.grid()
                st.pyplot(fig)
            
            ### Campbell Plot
            st.header("Campbell Plot Analysis")
            F_data = np.zeros(n_points)
            for i in range(n_points):
                gas_term = (Gp_data[i] - Np_data[i] * Rs_data[i]) * Bg_data[i] if pressure_data[i] < p_b else 0
                F_data[i] = Np_data[i] * Bo_data[i] + gas_term + Wp_data[i] * Bw_data[i]
            c_e = (c_w * S_wc + c_f) / (1 - S_wc)
            E_o_data = np.array([((Bo_data[i] - B_oi) + (R_si - Rs_data[i]) * Bg_data[i]) 
                                if pressure_data[i] < p_b else (Bo_data[i] - B_oi) 
                                for i in range(n_points)])
            E_g_data = B_oi * (Bg_data / B_gi - 1)
            E_fw_data = (1 + m) * B_oi * c_e * (res_p - pressure_data)
            E_t_data = E_o_data + m * E_g_data + E_fw_data
            F_over_Et = np.where(np.abs(E_t_data) > 1e-6, F_data / E_t_data, np.nan)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(F_data, F_over_Et, 'bo-', linewidth=2, label='F / E_t vs F')
            ax.set_xlabel('Cumulative Reservoir Voidage (F)')
            ax.set_ylabel('F / E_t')
            ax.set_title('Campbell Plot')
            ax.legend()
            ax.grid()
            st.pyplot(fig)
            
            ### Material Balance Analysis
            st.header("Material Balance Analysis")
            if influx_present:
                c_t = c_w + c_f
                We_data = (W_ei / 5.615) * c_t * (res_p - pressure_data)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(dates, We_data, 'r--', linewidth=2, label='Water Influx (MMBBL)')
                ax.set_xlabel('Time')
                ax.set_ylabel('Water Influx (MMBBL)')
                ax.set_title('Water Influx Over Time')
                ax.legend()
                ax.grid()
                st.pyplot(fig)
                num = np.sum(E_t_data * (F_data - We_data / 1e6))
                denom = np.sum(E_t_data**2)
                N_intercept = num / denom if denom != 0 else np.nan
                F_minus_We = F_data - We_data / 1e6
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(E_t_data, F_minus_We, 'bo', linewidth=2, label='F - W_e vs E_t')
                ax.plot(E_t_data, N_intercept * E_t_data, 'r-', linewidth=2, 
                        label=f'Fit: N = {N_intercept * 1e6:.2f} STB')
                ax.set_xlabel('Total Expansion Term (E_t, RB/STB)')
                ax.set_ylabel('Adjusted Production Term (F - W_e)')
                ax.set_title('Material Balance Plot (With Influx)')
                ax.legend()
                ax.grid()
                st.pyplot(fig)
                st.success(f"Estimated OOIP (With Influx): {N_intercept * 1e6:,.2f} STB")
            else:
                num = np.sum(E_t_data * F_data)
                denom = np.sum(E_t_data**2)
                N_intercept = num / denom if denom != 0 else np.nan
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(E_t_data, F_data, 'bo', linewidth=2, label='F vs E_t')
                ax.plot(E_t_data, N_intercept * E_t_data, 'r-', linewidth=2, 
                        label=f'Fit: N = {N_intercept * 1e6:.2f} STB')
                ax.set_xlabel('Total Expansion Term (E_t, RB/STB)')
                ax.set_ylabel('Production Term (F)')
                ax.set_title('Material Balance Plot (No Influx)')
                ax.legend()
                ax.grid()
                st.pyplot(fig)
                st.success(f"Estimated OOIP (No Influx): {N_intercept * 1e6:,.2f} STB")
            
            ### Drive Mechanism Analysis
            st.header("Drive Mechanism Analysis")
            E_o_drive = np.array([((Bo_data[i] - B_oi) + (r_s_input - Rs_data[i]) * Bg_data[i]) 
                                 if pressure_data[i] < p_b else (Bo_data[i] - B_oi) 
                                 for i in range(n_points)])
            E_g_drive = B_oi * ((Bg_data / B_gi) - 1)
            E_fw_drive = (1 + m) * B_oi * c_e * (res_p - pressure_data)
            We_term = We_data / N_intercept if influx_present else np.zeros(n_points)
            total_expansion = E_o_drive + m * E_g_drive + E_fw_drive + We_term
            oil_gas_pct = np.nan_to_num((E_o_drive / total_expansion) * 100)
            gas_cap_pct = np.nan_to_num((m * E_g_drive / total_expansion) * 100)
            rock_water_pct = np.nan_to_num((E_fw_drive / total_expansion) * 100)
            water_influx_pct = np.nan_to_num((We_term / total_expansion) * 100)
            plot_pressure = pressure_data[1:]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.stackplot(plot_pressure, oil_gas_pct[1:], gas_cap_pct[1:], rock_water_pct[1:], water_influx_pct[1:],
                         labels=['Oil & Gas Expansion', 'Gas Cap Expansion', 'Rock & Water Expansion', 'Water Influx'],
                         colors=['#B22222', '#264653', '#333333', '#2A9D8F'], linewidth=2)
            ax.set_xlabel('Pressure (psi)')
            ax.set_ylabel('Drive Mechanism Contribution (%)')
            ax.set_title('Reservoir Drive Mechanism Analysis')
            ax.legend()
            ax.grid(True)
            ax.invert_xaxis()
            st.pyplot(fig)
            
            ### Download Results
            st.header("Download Results")
            output_df = pd.DataFrame({
                'Date': dates,
                'Pressure (psi)': pressure_data,
                'Cum Oil Production (STB)': Np_data,
                'Cum Gas Production (SCF)': Gp_data,
                'Cum Water Production (STB)': Wp_data,
                'Rs (SCF/STB)': Rs_data,
                'Bo (RB/STB)': Bo_data,
                'Bg (RB/SCF)': Bg_data,
                'Bw (RB/STB)': Bw_data,
                'Viscosity (cP)': Viscosity_data,
                'Oil Compressibility (1/psi)': c_o_data,
                'F (RB)': F_data,
                'W_e (MMBBL)': We_data if influx_present else np.zeros(n_points),
                'E_o (RB/STB)': E_o_data,
                'E_g (RB/STB)': E_g_data,
                'E_fw (RB/STB)': E_fw_data,
                'E_t (RB/STB)': E_t_data,
                'F_over_Et (STB)': F_over_Et
            })
            csv = output_df.to_csv(index=False).encode('utf-8')
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                output_df.to_excel(writer, index=False, sheet_name='Results')
            excel_data = excel_buffer.getvalue()
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="Download as CSV", data=csv, 
                                 file_name="oil_material_balance.csv", mime="text/csv")
            with col2:
                st.download_button(label="Download as Excel", data=excel_data, 
                                 file_name="oil_material_balance.xlsx", 
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload production data to begin analysis.")

# -----------------------------------------------------
# Gas Reservoir Section
# -----------------------------------------------------
elif reservoir_type == "Gas Reservoir":
    with st.sidebar:
        st.header("Reservoir Parameters")
        res_p = st.number_input("Reservoir Pressure (psi)", min_value=0.0, step=100.0)
        t = st.number_input("Temperature (F)", min_value=0, step=5)
        sg_g = st.number_input("Gas Specific Gravity", min_value=0.0, step=0.01)
        c_f = st.number_input("Rock Compressibility (1/psi)", min_value=0.0, step=1e-7, format="%.2e")
        c_w = st.number_input("Water Compressibility (1/psi)", min_value=0.0, step=1e-7, format="%.2e")
        S_wc = st.number_input("Initial Water Saturation (fraction)", min_value=0.0, max_value=1.0, step=0.01)
        
        st.header("Water Influx Parameters")
        influx_present = st.checkbox("Water Influx Present?")
        if influx_present:
            W_ei = st.number_input("Aquifer Volume (MMft^3)", min_value=0.0, step=100.0)
    
    uploaded_file = st.file_uploader("Upload Production Data (Excel/CSV)", type=["xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                history_data = pd.read_csv(uploaded_file)
            else:
                history_data = pd.read_excel(uploaded_file)
            
            required_columns = ['Date', 'Pressure', 'Cum Gas Production']
            if not all(col in history_data.columns for col in required_columns):
                st.error(f"Uploaded file must contain these columns: {', '.join(required_columns)}")
                st.stop()
            
            dates = pd.to_datetime(history_data['Date'])
            pressure_data = history_data['Pressure'].values
            Gp_data = history_data['Cum Gas Production'].values
            Wp_data = history_data.get('Cum Water Production', np.zeros(len(pressure_data))).values
            n_points = len(pressure_data)
            
            ### Fluid Properties Calculation
            Z_prod = np.array([compute_z(sg_g, p_val, t) for p_val in pressure_data])
            Bg_data = 0.02827 * Z_prod * (t + 460) / pressure_data
            Bw_data = np.array([calculate_Bw(p_val, t) for p_val in pressure_data])
            viscosity_data = np.array([calculate_gas_viscosity(p, t, sg_g, Z) 
                                    for p, Z in zip(pressure_data, Z_prod)])
            B_gi = Bg_data[0]
            
            ### Production Data Visualization
            st.header("Production Data Visualization")
            col1, col2 = st.columns(2)
            with col1:
                fig, ax1 = plt.subplots(figsize=(8, 4))
                ax1.plot(dates, Gp_data, 'b-', linewidth=2, label='Cum Gas Production (MMSCF)')
                ax1.set_xlabel('Time', fontweight='bold')
                ax1.set_ylabel('Cum Gas Production (MMSCF)', fontweight='bold')
                ax1.set_ylim(min(Gp_data) * 0.9, max(Gp_data) * 1.1)
                ax1.legend(prop={'weight':'bold'})
                ax1.grid()
                ax1_twin = ax1.twiny()
                ax1_twin.plot(pressure_data, Gp_data, 'r--', linewidth=2, alpha=0)
                ax1_twin.set_xlabel('Pressure (psi)', fontweight='bold')
                ax1_twin.invert_xaxis()
                st.pyplot(fig)
            with col2:
                fig, ax2 = plt.subplots(figsize=(8, 4))
                ax2.plot(dates, Wp_data, 'g-', linewidth=2, label='Cum Water Production (MMSTB)')
                ax2.set_xlabel('Time', fontweight='bold')
                ax2.set_ylabel('Cum Water Production (MMSTB)', fontweight='bold')
                ax2.set_ylim(min(Wp_data) * 0.9, max(Wp_data) * 1.1)
                ax2.legend(prop={'weight':'bold'})
                ax2.grid()
                ax2_twin = ax2.twiny()
                ax2_twin.plot(pressure_data, Wp_data, 'r--', linewidth=2, alpha=0)
                ax2_twin.set_xlabel('Pressure (psi)', fontweight='bold')
                ax2_twin.invert_xaxis()
                st.pyplot(fig)
            
            ### Fluid Properties Visualization
            st.header("Fluid Properties")
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            axs[0, 0].plot(pressure_data, Bg_data, 'b-o', linewidth=2, label='Bg (RB/SCF)')
            axs[0, 0].set_xlabel('Pressure (psi)', fontweight='bold')
            axs[0, 0].set_ylabel('Bg (RB/SCF)', fontweight='bold')
            axs[0, 0].set_title('Bg vs. Pressure', fontweight='bold')
            axs[0, 0].legend(prop={'weight':'bold'})
            axs[0, 0].grid()
            axs[0, 1].plot(pressure_data, Bw_data, 'g-o', linewidth=2, label='Bw (RB/STB)')
            axs[0, 1].set_xlabel('Pressure (psi)', fontweight='bold')
            axs[0, 1].set_ylabel('Bw (RB/STB)', fontweight='bold')
            axs[0, 1].set_title('Bw vs. Pressure', fontweight='bold')
            axs[0, 1].legend(prop={'weight':'bold'})
            axs[0, 1].grid()
            axs[1, 0].plot(pressure_data, viscosity_data, 'r-o', linewidth=2, label='Viscosity (cp)')
            axs[1, 0].set_xlabel('Pressure (psi)', fontweight='bold')
            axs[1, 0].set_ylabel('Viscosity (cp)', fontweight='bold')
            axs[1, 0].set_title('Gas Viscosity', fontweight='bold')
            axs[1, 0].legend(prop={'weight':'bold'})
            axs[1, 0].grid()
            axs[1, 1].plot(pressure_data, Z_prod, 'k-o', linewidth=2, label='Z-factor')
            axs[1, 1].set_xlabel('Pressure (psi)', fontweight='bold')
            axs[1, 1].set_ylabel('Z-factor', fontweight='bold')
            axs[1, 1].set_title('Z vs. Pressure', fontweight='bold')
            axs[1, 1].legend(prop={'weight':'bold'})
            axs[1, 1].grid()
            st.pyplot(fig)
            
            ### Cole Plot
            st.header("Cole Plot Analysis")
            cole_y = np.full(n_points, np.nan)
            for i in range(1, n_points):
                if Bg_data[i] != B_gi:
                    cole_y[i] = Gp_data[i] * Bg_data[i] / (Bg_data[i] - B_gi)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(Gp_data[1:], cole_y[1:], 'bo-', linewidth=2, label='Cole Plot')
            ax.set_xlabel('Cumulative Gas Production (MMSCF)', fontweight='bold')
            ax.set_ylabel(r'$\frac{G_p\,Bg}{Bg - B_{gi}}$', fontweight='bold')
            ax.set_title('Cole Plot', fontweight='bold')
            ax.legend(prop={'weight':'bold'})
            ax.grid()
            st.pyplot(fig)
            
            ### Water Influx
            if influx_present:
                c_t_total = c_w + c_f
                We_data = (W_ei / 5.615) * c_t_total * (res_p - pressure_data)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(dates, We_data, 'r--', linewidth=2, label='Water Influx (MMBBL)')
                ax.set_xlabel('Time', fontweight='bold')
                ax.set_ylabel('Water Influx (MMBBL)', fontweight='bold')
                ax.set_title('Water Influx Over Time', fontweight='bold')
                ax.legend(prop={'weight':'bold'})
                ax.grid()
                st.pyplot(fig)
            else:
                We_data = np.zeros(n_points)
            
            ### Material Balance Analysis
            st.header("Material Balance Analysis")
            F_data = Gp_data * Bg_data + Wp_data * Bw_data
            C = (c_f + c_w * S_wc) / (1 - S_wc)
            E_t_data = (Bg_data - B_gi) + B_gi * C * (res_p - pressure_data)
            numerator = np.sum(E_t_data * F_data)
            denom = np.sum(E_t_data**2)
            G_intercept = numerator / denom if denom != 0 else np.nan
            if not influx_present:
                ratio = F_data / E_t_data
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(Gp_data, ratio, 'bo-', linewidth=2, 
                        label=f'F/E_t vs. Cum Gas\nOGIP = {G_intercept * 1e3:.2f} BSCF')
                ax.set_xlabel('Cumulative Gas Production (MMSCF)', fontweight='bold')
                ax.set_ylabel('F / E_t (RB)', fontweight='bold')
                ax.set_title('Material Balance Plot (No Influx)', fontweight='bold')
                ax.legend(prop={'weight':'bold'})
                ax.grid()
                st.pyplot(fig)
                st.success(f"Estimated OGIP (No Influx): {G_intercept * 1e3:.2f} BSCF")
            else:
                ratio = (F_data - (We_data / (G_intercept * 1e3))) / E_t_data
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(Gp_data, ratio, 'bo-', linewidth=2, 
                        label=f'(F - W_e)/E_t vs. Cum Gas\nOGIP = {G_intercept:.2f} BSCF')
                ax.set_xlabel('Cumulative Gas Production (MMSCF)', fontweight='bold')
                ax.set_ylabel('(F - W_e) / E_t (RB)', fontweight='bold')
                ax.set_title('Material Balance Plot (With Influx)', fontweight='bold')
                ax.legend(prop={'weight':'bold'})
                ax.grid()
                st.pyplot(fig)
                st.success(f"Estimated OGIP (With Influx): {G_intercept:.2f} BSCF")
            
            ### Modified p/Z Plot
            st.header("Modified p/Z Plot")
            G_initial = Gp_data[0]
            x_modified = Gp_data - G_initial
            y_modified = pressure_data / Z_prod
            X = x_modified.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y_modified)
            slope_val = model.coef_[0]
            intercept_val = model.intercept_
            OGIP_estimated = intercept_val / -slope_val
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x_modified, y_modified, 'bo', linewidth=2, label='Data Points')
            x_fit = np.linspace(min(x_modified), max(x_modified), 100)
            y_fit = model.predict(x_fit.reshape(-1, 1))
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Fit: y = {slope_val:.4f}x + {intercept_val:.1f}\nOGIP = {OGIP_estimated:.2f} BSCF')
            ax.set_xlabel('Cumulative Gas Production minus Initial (SCF)', fontweight='bold')
            ax.set_ylabel('p / Z (psi)', fontweight='bold')
            ax.set_title('Modified p/Z Plot', fontweight='bold')
            ax.legend(prop={'weight':'bold'})
            ax.grid()
            st.pyplot(fig)
            
            ### Drive Mechanism Analysis
            st.header("Drive Mechanism Analysis")
            E_g_drive = Bg_data - B_gi
            C_val = (c_f + c_w * S_wc) / (1 - S_wc)
            E_fw_drive = B_gi * C_val * (res_p - pressure_data)
            We_term_drive = We_data / (G_intercept * 1e3) if G_intercept != 0 else np.zeros_like(We_data)
            total_expansion_drive = E_g_drive + E_fw_drive + We_term_drive
            gas_exp_pct = np.nan_to_num((E_g_drive / total_expansion_drive) * 100)
            rock_water_pct = np.nan_to_num((E_fw_drive / total_expansion_drive) * 100)
            water_influx_pct = np.nan_to_num((We_term_drive / total_expansion_drive) * 100)
            plot_pressure = pressure_data[1:]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.stackplot(plot_pressure, gas_exp_pct[1:], rock_water_pct[1:], water_influx_pct[1:],
                         labels=['Gas Expansion', 'Rock & Fluid Expansion', 'Water Influx'],
                         colors=['red', 'blue', 'green'], linewidth=2)
            ax.set_xlabel('Pressure (psi)', fontweight='bold')
            ax.set_ylabel('Drive Mechanism Contribution (%)', fontweight='bold')
            ax.set_title('Drive Mechanism Analysis', fontweight='bold')
            ax.legend(prop={'weight':'bold'})
            ax.grid(True)
            ax.invert_xaxis()
            st.pyplot(fig)
            
            ### Download Results
            st.header("Download Results")
            output_df = pd.DataFrame({
                'Date': dates,
                'Pressure (psi)': pressure_data,
                'Cum Gas Production (SCF)': Gp_data,
                'Cum Water Production (STB)': Wp_data,
                'Bg (RB/SCF)': Bg_data,
                'Bw (RB/STB)': Bw_data,
                'Viscosity (cp)': viscosity_data,
                'Z-factor': Z_prod,
                'F (RB)': F_data,
                'We (MMBBL)': We_data,
                'E_t (RB)': E_t_data,
                'Gas Expansion (RB)': E_g_drive,
                'Rock & Fluid Expansion (RB)': E_fw_drive,
                'Total Expansion (RB)': total_expansion_drive
            })
            csv = output_df.to_csv(index=False).encode('utf-8')
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                output_df.to_excel(writer, index=False, sheet_name='Results')
            excel_data = excel_buffer.getvalue()
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="Download as CSV", data=csv, 
                                 file_name="gas_material_balance.csv", mime="text/csv")
            with col2:
                st.download_button(label="Download as Excel", data=excel_data, 
                                 file_name="gas_material_balance.xlsx", 
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload production data to begin analysis.")