from .api import get_price

def calculate_summary_statistics(portfolio_df):
    total_value = get_price(portfolio_df.figi.tolist(), how="old")

    # sector_distribution = portfolio_df.groupby("sector").count().figi

    # actual_value = get_price(portfolio_df.figi.tolist())
    summary = {
        "Общая стоимость портфеля (в рублях)": total_value
    }

    return summary
