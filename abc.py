import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Excel Data Analyzer", layout="wide")

st.title("📊 Excel Data Forecasting Readiness Checker")

st.write("""
Upload an Excel file. The app will:
- Detect numerical columns
- Create histograms
- Calculate Mean and Median
- Check if Mean and Median are close
- Suggest whether the data may be suitable for forecasting
""")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"]
)

if uploaded_file is not None:

    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) == 0:
            st.warning("No numerical columns found.")
        else:

            st.subheader("Analysis Results")

            results = []

            for col in numeric_cols:

                data = df[col].dropna()

                mean_val = data.mean()
                median_val = data.median()

                if mean_val != 0:
                    diff_percent = abs(mean_val - median_val) / abs(mean_val) * 100
                else:
                    diff_percent = 0

                forecast_ready = (
                    "✅ Suitable for Forecasting"
                    if diff_percent <= 10
                    else "⚠️ Check Distribution Before Forecasting"
                )

                results.append({
                    "Column": col,
                    "Mean": round(mean_val, 2),
                    "Median": round(median_val, 2),
                    "Difference %": round(diff_percent, 2),
                    "Assessment": forecast_ready
                })

                st.markdown(f"## {col}")

                col1, col2 = st.columns([2, 1])

                with col1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(data, bins=15)
                    ax.set_title(f"Histogram - {col}")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

                with col2:
                    st.metric("Mean", round(mean_val, 2))
                    st.metric("Median", round(median_val, 2))
                    st.write(f"Difference %: {round(diff_percent,2)}")
                    st.write(forecast_ready)

            st.subheader("Summary Table")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False)

            st.download_button(
                "Download Summary",
                csv,
                "forecast_analysis.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error reading file: {e}")
