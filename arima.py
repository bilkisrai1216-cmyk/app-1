import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARIMA Stock Forecast", layout="wide")

st.title("📈 Indian Stock ARIMA Forecast")

uploaded_file = st.file_uploader(
"Upload Excel File",
type=["xlsx", "xls"]
)

if uploaded_file is not None:

```
try:
    df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    date_col = st.selectbox(
        "Select Date Column",
        df.columns
    )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_cols) == 0:
        st.error("No numeric stock columns found.")
        st.stop()

    selected_stock = st.selectbox(
        "Select Stock",
        numeric_cols
    )

    stock_data = df[selected_stock].dropna()

    last_date = stock_data.index.max()
    start_date = last_date - pd.DateOffset(years=5)

    stock_data = stock_data[
        stock_data.index >= start_date
    ]

    if len(stock_data) < 30:
        st.error("Not enough data.")
        st.stop()

    model = ARIMA(
        stock_data,
        order=(5, 1, 0)
    )

    model_fit = model.fit()

    target_date = pd.Timestamp(
        "2027-06-30"
    )

    months_to_forecast = (
        (target_date.year - last_date.year) * 12
        + (target_date.month - last_date.month)
    )

    if months_to_forecast < 1:
        months_to_forecast = 12

    forecast = model_fit.forecast(
        steps=months_to_forecast
    )

    forecast_dates = pd.date_range(
        start=last_date,
        periods=months_to_forecast + 1,
        freq="M"
    )[1:]

    forecast_df = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Forecast": np.array(forecast)
        }
    )

    st.subheader("Forecast Values")
    st.dataframe(forecast_df)

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
            y=np.array(forecast),
            mode="lines",
            name="Forecast"
        )
    )

    fig.update_layout(
        title=f"{selected_stock} Forecast to June 2027",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    csv = forecast_df.to_csv(
        index=False
    )

    st.download_button(
        "Download Forecast CSV",
        csv,
        "forecast.csv",
        "text/csv"
    )

except Exception as e:
    st.error(str(e))
```
