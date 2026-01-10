import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SCC Probability Visualization", layout="wide")

st.title("🔬 SCC Probability Estimation - CHAKSU MATHURA SECTION")
st.markdown("Upload `scc_IV_dataset.xlsx` to visualize normalized Stress Corrosion Probability Score vs Stationing.")

# File uploader
uploaded_file = st.file_uploader("Choose Excel file", type=["xlsx"], help="Upload scc_IV_dataset.xlsx")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)
    
    df_scc_II = load_data(uploaded_file)
    st.success(f"Loaded dataset: {df_scc_II.shape[0]} rows, {df_scc_II.shape[1]} columns")
    
    st.subheader("Original Dataset Preview")
    st.dataframe(df_scc_II.head(), use_container_width=True)
    
    # Process data
    if st.button("🧮 Compute Normalized SCC Probability Score", type="primary"):
        with st.spinner("Processing features..."):
            # Normalize Distance
            scaler_distance = MinMaxScaler()
            df_scc_II['Normalized_Distance_from_Pump(KM)'] = scaler_distance.fit_transform(df_scc_II[['Distance from Pump(KM)']])
            
            # Normalize OFF PSP
            scaler_off_psp = MinMaxScaler()
            df_scc_II['Normalized_OFF_PSP_VE_V'] = scaler_off_psp.fit_transform(df_scc_II[['OFF PSP (VE V)']])
            df_scc_II['Inverse_Normalized_OFF_PSP_VE_V'] = 1 - df_scc_II['Normalized_OFF_PSP_VE_V']
            
            # Weights
            feature_weights_normalized = {
                'conductivity': 0.186,
                'Hoop stress% of SMYS': 0.08,
                'Normalized_Distance_from_Pump(KM)': 0.165,
                'Inverse_Normalized_OFF_PSP_VE_V': 0.142
            }
            
            # Calculate score
            df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'] = (
                df_scc_II['conductivity'] * feature_weights_normalized['conductivity'] +
                df_scc_II['Hoop stress% of SMYS'] * feature_weights_normalized['Hoop stress% of SMYS'] +
                df_scc_II['Normalized_Distance_from_Pump(KM)'] * feature_weights_normalized['Normalized_Distance_from_Pump(KM)'] +
                df_scc_II['Inverse_Normalized_OFF_PSP_VE_V'] * feature_weights_normalized['Inverse_Normalized_OFF_PSP_VE_V']
            )
            
            st.success("✅ Score calculated!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Processed Dataset Preview")
            st.dataframe(df_scc_II[['Stationing (m)', 'Stress_Corrosion_Probability_Score_Normalized_V2', 
                                  'Normalized_Distance_from_Pump(KM)', 'Inverse_Normalized_OFF_PSP_VE_V']].head(), use_container_width=True)
        
        with col2:
            st.subheader("📈 Score Statistics")
            st.metric("Mean SCC Score", f"{df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'].mean():.4f}")
            high_risk_threshold_normalized = df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'].quantile(0.95)
            st.metric("95th Percentile (High Risk)", f"{high_risk_threshold_normalized:.4f}")
            st.metric("High Risk Segments (>95th)", f"{(df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'] > high_risk_threshold_normalized).sum()}")
        
        # Visualization
        st.subheader("📍 Normalized SCC Probability Score vs. Stationing (m)")
        
        fig = px.scatter(df_scc_II, x='Stationing (m)', y='Stress_Corrosion_Probability_Score_Normalized_V2',
                         title="Stress Corrosion Probability Score vs. Stationing",
                         labels={'Stress_Corrosion_Probability_Score_Normalized_V2': 'Normalized SCC Probability Score'},
                         hover_data=['conductivity', 'Hoop stress% of SMYS'])
        
        high_risk_threshold_normalized = df_scc_II['Stress_Corrosion_Probability_Score_Normalized_V2'].quantile(0.95)
        fig.add_hline(y=high_risk_threshold_normalized, line_dash="dash", line_color="red",
                      annotation_text=f'High Risk Threshold (95th percentile: {high_risk_threshold_normalized:.4f})')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download processed data
        csv = df_scc_II.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Processed Dataset (CSV)", csv, "scc_processed.csv", "text/csv")

else:
    st.info("👆 Upload your `scc_IV_dataset.xlsx` file to get started.")

# Sidebar for GitHub Codespaces instructions
with st.sidebar:
    st.header("🚀 GitHub Codespaces Setup")
    st.markdown("""
    1. Create `requirements.txt`:
    ```
    streamlit
    pandas
    scikit-learn
    plotly
    openpyxl
    ```
    
    2. Save this as `app.py`
    
    3. Run: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`
    
    4. Click the URL in terminal (ports tab) [web:7][web:10]
    """)
