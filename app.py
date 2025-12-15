import streamlit as st
import pandas as pd
import numpy as np

st.title("Pipeline SCC Probability Estimator")

uploaded = st.file_uploader("Upload pd.xlsx (pipeline data)", type=["xlsx"])

if uploaded is not None:
    df = pd.read_excel(uploaded)

    # thresholds
    psp_thresh = -0.85
    soil_res_thresh = 5000
    hoop_stress_thresh = 0.5
    pipe_age_thresh = 20
    remaining_thk_thresh = 0.8

    # derive binary features
    risk_features = pd.DataFrame(index=df.index)
    risk_features["X1_psp"] = (df["OFF PSP (VE V)"] < psp_thresh).astype(int)
    risk_features["X2_soil_res"] = (df["Soil Resistivity (Ω-cm)"] < soil_res_thresh).astype(int)
    risk_features["X3_hoop"] = (df["Hoop stress% of SMYS"] > hoop_stress_thresh).astype(int)
    risk_features["X4_age"] = (df["Pipe Age "].fillna(0) > pipe_age_thresh).astype(int)
    max_thk = df["Remaining Thickness(mm)"].max()
    risk_features["X5_thk"] = (
        df["Remaining Thickness(mm)"] / max_thk < remaining_thk_thresh
    ).astype(int)

    betas = np.array([1, 0.6, 1, 0.5, 1])
    beta_0 = -1

    lin_score = (risk_features * betas).sum(axis=1) + beta_0
    scc_prob = 1 / (1 + np.exp(-lin_score))

    df_result = df.copy()
    df_result["SCC_Prob"] = scc_prob

    st.subheader("Sample of SCC probabilities")
    st.dataframe(df_result[["Stationing (m)", "SCC_Prob"]].head(10))

    st.subheader("SCC probability along the pipeline")
    chart_data = df_result[["Stationing (m)", "SCC_Prob"]].set_index("Stationing (m)")
    st.line_chart(chart_data)

    # CSV download
    out_csv = df_result[["Stationing (m)", "SCC_Prob"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full SCC probabilities as CSV",
        data=out_csv,
        file_name="Pipeline_SCC_Probabilities.csv",
        mime="text/csv",
    )
