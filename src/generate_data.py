from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    n_customers: int = 1000
    n_months: int = 12
    seed: int = 42
    out_path: str = "data/raw/transactions_monthly.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    # --- customer-level attributes (fixed over time) ---
    customer_id = np.arange(1, cfg.n_customers + 1)

    age = rng.integers(22, 70, size=cfg.n_customers)

    # Income: lognormal, with mild correlation with age
    income_base = rng.lognormal(mean=9.0, sigma=0.45, size=cfg.n_customers)  # ~8k-15k BRL-ish scale
    income = income_base * (0.85 + (age - 22) / (70 - 22) * 0.35)
    income = np.clip(income, 1500, 50000)

    # Risk propensity: latent trait (higher => more risk)
    risk_latent = rng.normal(0, 1, size=cfg.n_customers)

    customers = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "income": income.round(2),
            "risk_latent": risk_latent,
        }
    )

    # --- monthly panel generation ---
    months = np.arange(1, cfg.n_months + 1)
    panel = customers.assign(key=1).merge(pd.DataFrame({"month": months, "key": 1}), on="key").drop(columns=["key"])

    # Seasonality (simple)
    season = np.sin(2 * np.pi * (panel["month"].to_numpy() - 1) / 12)

    # Balance: depends on income, risk, and noise (positive)
    bal_mu = (
        0.9 * np.log(panel["income"].to_numpy())
        - 0.25 * panel["risk_latent"].to_numpy()
        + 0.2 * season
    )
    balance = rng.lognormal(mean=bal_mu, sigma=0.35)
    balance = np.clip(balance, 0, 500000)

    # Card spend: depends on income, balance, and seasonality
    card_spend = (
        0.06 * panel["income"].to_numpy()
        + 0.02 * np.sqrt(balance)
        + 120 * (season + 1.0)
        + rng.normal(0, 250, size=len(panel))
    )
    card_spend = np.clip(card_spend, 0, None)

    # PIX count: more digital if younger; add noise
    pix_lambda = (
        10
        + (60 - panel["age"].to_numpy()) * 0.25
        + 2.5 * (season + 1.0)
        + rng.normal(0, 1.0, size=len(panel))
    )
    pix_lambda = np.clip(pix_lambda, 1, 40)
    pix_count = rng.poisson(lam=pix_lambda)

    # Credit limit and utilization
    credit_limit = 0.35 * panel["income"].to_numpy() + rng.normal(0, 800, size=len(panel))
    credit_limit = np.clip(credit_limit, 1000, 40000)

    util = (
        0.25
        + 0.18 * sigmoid(panel["risk_latent"].to_numpy())
        + 0.08 * (card_spend / (credit_limit + 1e-6))
        + rng.normal(0, 0.06, size=len(panel))
    )
    utilization = np.clip(util, 0.01, 0.98)

    # Late payment probability (binary), related to risk + high utilization
    late_prob = sigmoid(
        -2.2
        + 1.2 * panel["risk_latent"].to_numpy()
        + 2.0 * (utilization - 0.5)
        + rng.normal(0, 0.2, size=len(panel))
    )
    late_payment = (rng.random(len(panel)) < late_prob).astype(int)

    # Products currently used (simple simulated signals)
    uses_card = (card_spend > 200).astype(int)
    uses_credit = ((utilization > 0.45) | (late_payment == 1)).astype(int)

    # --- Investment adoption event over time (for survival + propensity) ---
    # We simulate a "hazard" that increases for higher income, higher balance, and lower risk
    # and also increases over months (marketing exposure / maturation).
    month_effect = (panel["month"].to_numpy() - 1) / (cfg.n_months - 1)

    invest_hazard = sigmoid(
        -3.0
        + 0.55 * np.log(panel["income"].to_numpy())
        + 0.25 * np.log(balance + 1.0)
        - 0.75 * panel["risk_latent"].to_numpy()
        + 0.6 * month_effect
        + rng.normal(0, 0.15, size=len(panel))
    )
    # Convert hazard to per-month adoption probability (keep small)
    invest_prob = np.clip(invest_hazard * 0.04, 0.001, 0.15)
    # For each customer, determine first month of adoption
    panel["adopt_investment"] = 0
    first_adopt_month = np.full(cfg.n_customers, fill_value=0, dtype=int)

    # We'll loop per customer to enforce "first adoption month" only once
    for i, cid in enumerate(customer_id):
        rows = panel.index[panel["customer_id"] == cid].to_numpy()
        adopted = False
        for idx in rows:
            if adopted:
                continue
            if rng.random() < invest_prob[idx]:
                panel.at[idx, "adopt_investment"] = 1
                first_adopt_month[i] = int(panel.at[idx, "month"])
                adopted = True

    # Duration + censoring at n_months
    # duration = month of adoption (1..n_months); if never adopted => duration=n_months and event=0
    duration = np.where(first_adopt_month > 0, first_adopt_month, cfg.n_months)
    event_observed = (first_adopt_month > 0).astype(int)

    # Merge survival targets back to panel (repeated per month; fine for feature building later)
    surv = pd.DataFrame(
        {
            "customer_id": customer_id,
            "time_to_investment": duration,
            "event_investment": event_observed,
            "first_adopt_month": first_adopt_month,
        }
    )
    panel = panel.merge(surv, on="customer_id", how="left")

    # Assemble final dataset
    panel["balance"] = balance.round(2)
    panel["card_spend"] = card_spend.round(2)
    panel["pix_count"] = pix_count.astype(int)
    panel["credit_limit"] = credit_limit.round(2)
    panel["utilization"] = utilization.round(4)
    panel["late_payment"] = late_payment.astype(int)
    panel["uses_card"] = uses_card.astype(int)
    panel["uses_credit"] = uses_credit.astype(int)

    # Basic checks
    assert panel["customer_id"].nunique() == cfg.n_customers
    assert panel["month"].nunique() == cfg.n_months

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.drop(columns=["risk_latent"]).to_csv(out_path, index=False)

    adopted_rate = surv["event_investment"].mean()
    print(f"Saved: {out_path} | rows={len(panel):,} | customers={cfg.n_customers:,} | adoption_rate={adopted_rate:.2%}")


if __name__ == "__main__":
    main()