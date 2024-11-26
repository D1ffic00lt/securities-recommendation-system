from .api import get_price

def calculate_summary_statistics(portfolio_df):
    total_value = get_price(portfolio_df.figi.tolist(), how="old")
    summary = {
        "Total portfolio value (in roubles)": total_value,
    }

    return summary
