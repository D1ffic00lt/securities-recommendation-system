import streamlit as st
import pandas as pd
import plotly.express as px

from utils.api import get_portfolio
from utils.portfolio_analysis import calculate_summary_statistics

st.title("Сбалансированный инвестиционный портфель")

investment_sum = st.number_input(
    "Введите сумму инвестиций (в рублях):", min_value=1, step=1
)
user_id = st.text_input("Введите ваш ID пользователя:")

if st.button("Сгенерировать портфель"):
    if investment_sum > 0 and user_id:
        portfolio = get_portfolio(user_id, investment_sum)

        if portfolio:
            df = pd.DataFrame(portfolio)
            df["final_rating"] = df.price_rating + df.company_rating

            st.subheader("Сгенерированный Портфель")
            st.write(df.sort_values(by="final_rating", ascending=False))

            summary_stats = calculate_summary_statistics(df)
            st.subheader("Сводная статистика портфеля")
            st.write(summary_stats)

            st.subheader("Распределение по секторам")
            sector_distribution = df.groupby("sector")["price"].sum().reset_index()
            fig = px.pie(
                sector_distribution,
                names="sector",
                values="price",
                title="Распределение по секторам",
            )
            st.plotly_chart(fig)

            st.subheader("Рейтинги по стоимости активов")
            fig = px.histogram(df.price_rating, title="Рейтинги по стоимости активов")
            st.plotly_chart(fig)

            st.subheader("Рейтинги по данным компании")
            fig = px.histogram(df.company_rating, title="Рейтинги по данным компании")
            st.plotly_chart(fig)

            st.subheader("Общий рейтинг")
            fig = px.histogram(df.final_rating, title="Общий рейтинг")
            st.plotly_chart(fig)
        else:
            st.error("Не удалось получить данные портфеля. Попробуйте еще раз.")
    else:
        st.warning("Пожалуйста, заполните все поля.")
