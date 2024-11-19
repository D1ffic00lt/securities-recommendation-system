import random

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.api import get_portfolio
from utils.portfolio_analysis import calculate_summary_statistics

st.title("Balanced investment portfolio")

investment_sum = st.number_input(
    "Enter the amount of investment (in roubles):", min_value=100, step=100
)

if st.button("Generate portfolio"):
    if investment_sum > 0:
        portfolio = get_portfolio(random.randint(0, 100), investment_sum)

        if portfolio:
            df = pd.DataFrame(portfolio)

            st.subheader("Generated portfolio")
            st.write(df.sort_values(by="final_rating", ascending=False))

            summary_stats = calculate_summary_statistics(df)
            st.subheader("Portfolio summary statistics")
            st.write(summary_stats)

            st.subheader("Sectoral distribution")
            sector_distribution = df.groupby("sector")["price"].sum().reset_index()
            fig = px.pie(
                sector_distribution,
                names="sector",
                values="price",
                title="Sectoral distribution",
            )
            st.plotly_chart(fig)

            st.subheader("Types distribution")
            types_distribution = df.groupby("type")["price"].sum().reset_index()
            fig = px.pie(
                types_distribution,
                names="type",
                values="price",
                title="Types distribution",
            )
            st.plotly_chart(fig)
        else:
            st.error("Failed to retrieve portfolio data. Try again.")
    else:
        st.warning("Please fill in all fields.")
