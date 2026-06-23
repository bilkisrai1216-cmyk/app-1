
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Indian Stock ARIMA Forecast",
    layout="wide"
)

st.title("📈 Indian Stock Forecasting using ARIMA")

st.write("""
Upload an Excel file containing:
- One Date column
- One or more stock price columns

The app will:
- Use the last 5 years of data
- Train an ARIMA model
- Forecast until June 2027
- Display graphs and forecast tables
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

        # Detect date column
        date_col = None

        for col in df.columns:
            try:
                converted = pd.to_datetime(df[col])
                date_col = col
                break
            except:
                pass

        if date_col is None:
            st.error("No valid date column found.")
            st.stop()

        df[date_col] = pd.to_datetime(df[date_col])

        df = df.sort_values(date_col)

        df.set_index(date_col, inplace=True)

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) == 0:
            st.error("No numerical stock columns found.")
            st.stop()

        forecast_tables = []

        for stock in numeric_cols:

            st.markdown("---")
            st.header(f"📊 {stock}")

            stock_data = df[stock].dropna()

            if len(stock_data) < 30:
                st.warning(f"Not enough data for {stock}")
                continue

            # Use last 5 years only
            last_date = stock_data.index.max()
            start_date = last_date - pd.DateOffset(years=5)

            stock_data = stock_data[
                stock_data.index >= start_date
            ]

            try:

                model = ARIMA(
                    stock_data,
                    order=(5, 1, 0)
                )

                model_fit = model.fit()

                target_date = pd.Timestamp("2027-06-30")

                months_to_forecast = (
                    (target_date.year - last_date.year) * 12
                    + (target_date.month - last_date.month)
                )

                if months_to_forecast <= 0:
                    months_to_forecast = 12

                forecast = model_fit.forecast(
                    steps=months_to_forecast
                )

                forecast_dates = pd.date_range(
                    start=last_date,
                    periods=months_to_forecast + 1,
                    freq="M"
                )[1:]

                forecast_df = pd.DataFrame({
                    "Date": forecast_dates,
                    "Forecast": forecast.values
                })

                forecast_df["Stock"] = stock

                forecast_tables.append(forecast_df)

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data.values,
                        mode="lines",
                        name="Historical"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecast.values,
                        mode="lines",
                        name="Forecast"
                    )
                )

                fig.update_layout(
                    title=f"{stock} Forecast till June 2027",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader("Forecast Values")

                st.dataframe(
                    forecast_df[
                        ["Date", "Forecast"]
                    ]
                )

            except Exception as e:
                st.error(
                    f"Error forecasting {stock}: {e}"
                )

        if len(forecast_tables) > 0:

            final_forecast = pd.concat(
                forecast_tables,
                ignore_index=True
            )

            st.markdown("---")
            st.header("📥 Download Forecast Results")

            csv = final_forecast.to_csv(
                index=False
            )

            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name="stock_forecast.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
```
