from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter


def main() -> None:
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    # Load features (1 row per customer)
    # Prefer the clustered file if it exists (includes cluster/cluster_name)
    p1 = Path("data/processed/customer_features_with_cluster_named.csv")
    p2 = Path("data/processed/customer_features_with_cluster.csv")
    p3 = Path("data/processed/customer_features.csv")

    if p1.exists():
        df = pd.read_csv(p1)
    elif p2.exists():
        df = pd.read_csv(p2)
    elif p3.exists():
        df = pd.read_csv(p3)
    else:
        raise FileNotFoundError("Run build_features / cluster_profiles first. No feature table found.")

    # Survival target
    # time_to_investment: month of adoption (1..n_months) or censoring time (n_months)
    # adopted_ever/event_investment: 1 if adopted else 0
    if "time_to_investment" not in df.columns:
        raise ValueError("Missing 'time_to_investment' in features.")
    if "adopted_ever" not in df.columns:
        raise ValueError("Missing 'adopted_ever' in features (event indicator).")

    duration_col = "time_to_investment"
    event_col = "adopted_ever"

    # Choose covariates (use what exists)
    candidate_cols = [
        "age",
        "income",
        "m12_mean_balance",
        "m12_std_balance",
        "m12_mean_card_spend",
        "m12_mean_utilization",
        "m12_mean_pix",
        "m12_late_payment_rate",
        "m3_mean_balance",
        "m3_std_balance",
        "m3_mean_card_spend",
        "m3_mean_utilization",
        "m3_mean_pix",
        "m3_late_payment_rate",
        "cluster",  # optional numeric cluster
    ]
    covariates = [c for c in candidate_cols if c in df.columns]

    use = df[["customer_id", duration_col, event_col] + covariates].copy()

    # Clean types
    use[duration_col] = pd.to_numeric(use[duration_col], errors="coerce")
    use[event_col] = pd.to_numeric(use[event_col], errors="coerce").fillna(0).astype(int)

    for c in covariates:
        use[c] = pd.to_numeric(use[c], errors="coerce")

    use = use.dropna(subset=[duration_col]).copy()
    use = use.fillna(use.median(numeric_only=True))

    # Fit Cox model
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(use.drop(columns=["customer_id"]), duration_col=duration_col, event_col=event_col)

    # Predict survival curves per customer and extract P(adopt <= t) at 3 horizons
    horizons = [3, 6, 9]  # months
    surv = cph.predict_survival_function(use.drop(columns=["customer_id"]), times=horizons)  # (len(horizons), n)

    # Convert survival S(t) to adoption probability by t: 1 - S(t)
    probs = (1.0 - surv.T).copy()
    probs.columns = [f"p_adopt_{h}m" for h in horizons]
    probs.insert(0, "customer_id", use["customer_id"].astype(int).to_numpy())

    out_probs = Path("reports/tables/survival_probabilities.csv")
    probs.to_csv(out_probs, index=False)

    # Expected time approximation:
    # We approximate expected time by integrating survival curve on a grid of months 1..maxT
    max_t = int(np.nanmax(use[duration_col]))
    grid = list(range(1, max_t + 1))
    sf = cph.predict_survival_function(use.drop(columns=["customer_id"]), times=grid)  # (T, n)
    exp_time = sf.sum(axis=0).to_numpy()  # E[T] approx = sum_{t} S(t)

    exp_df = pd.DataFrame(
        {
            "customer_id": use["customer_id"].astype(int).to_numpy(),
            "expected_time_months": exp_time.round(3),
        }
    )
    out_exp = Path("reports/tables/survival_expected_time.csv")
    exp_df.to_csv(out_exp, index=False)

    # Save model summary (coefficients)
    summary = cph.summary.reset_index().rename(columns={"index": "feature"})
    out_summary = Path("reports/tables/survival_cox_summary.csv")
    summary.to_csv(out_summary, index=False)

    # Save markdown report
    out_md = Path("reports/survival_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Survival Model Report (CoxPH)\n\n")
        f.write(f"- duration: {duration_col}\n")
        f.write(f"- event: {event_col}\n")
        f.write(f"- covariates: {', '.join(covariates)}\n\n")
        f.write("## Coefficients (top)\n\n")
        f.write(summary.sort_values("p", ascending=True).head(15).to_markdown(index=False))
        f.write("\n\n")
        f.write(f"Saved tables:\n- {out_probs}\n- {out_exp}\n- {out_summary}\n")

    # Optionally save model (pickle)
    try:
        import joblib

        joblib.dump(cph, Path("models/survival_cox.pkl"))
        print("Saved: models/survival_cox.pkl")
    except Exception:
        pass

    print(f"Saved: {out_probs}")
    print(f"Saved: {out_exp}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()