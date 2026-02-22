from pathlib import Path
import pandas as pd


def main():

    # Load raw data
    df = pd.read_csv("data/raw/transactions_monthly.csv")

    # Aggregate per customer
    agg = (
        df.groupby("customer_id")
        .agg(
            age=("age", "first"),
            income=("income", "first"),
            mean_balance=("balance", "mean"),
            std_balance=("balance", "std"),
            mean_card_spend=("card_spend", "mean"),
            mean_utilization=("utilization", "mean"),
            mean_pix=("pix_count", "mean"),
            late_payment_rate=("late_payment", "mean"),
            adopted_ever=("event_investment", "first"),
            time_to_investment=("time_to_investment", "first"),
        )
        .reset_index()
    )

    # Fill NaN from std (single month edge case, though unlikely)
    agg["std_balance"] = agg["std_balance"].fillna(0)

    # Save processed dataset
    out_path = Path("data/processed/customer_features.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)

    print(f"Saved: {out_path} | rows={len(agg)}")


if __name__ == "__main__":
    main()