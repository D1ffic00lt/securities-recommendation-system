def calculate_summary_statistics(portfolio_df):
    # FIXME
    total_value = portfolio_df['price'].sum()

    sector_distribution = portfolio_df.groupby('sector').value_counts()

    summary = {
        'Общая стоимость портфеля (в рублях)': total_value,
        'Распределение по секторам': sector_distribution.to_dict()
    }

    return summary