from __future__ import annotations

from pathlib import Path
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# (opcional) se você já usa joblib no projeto, mantenha.
# caso não tenha instalado, o script ainda funciona (só não salva o modelo).
try:
    import joblib
except ImportError:
    joblib = None


def ensure_dirs() -> None:
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)


def load_or_train_kmeans(k: int = 4, seed: int = 42):
    """
    Para Markov funcionar, precisamos de cluster por mês.
    Como seu KMeans atual foi treinado em features agregadas por cliente,
    aqui usamos uma aproximação compatível (boa para demo/portfólio).

    Se existirem modelos salvos em models/, carrega.
    Se não, treina a partir de data/processed/customer_features.csv.
    """
    scaler_path = Path("models/scaler.pkl")
    km_path = Path("models/kmeans.pkl")

    if joblib is not None and scaler_path.exists() and km_path.exists():
        scaler = joblib.load(scaler_path)
        km = joblib.load(km_path)
        return scaler, km

    feats_path = Path("data/processed/customer_features.csv")
    if not feats_path.exists():
        raise FileNotFoundError("Rode primeiro: python -m src.build_features")

    df = pd.read_csv(feats_path)

    train_cols = [
    "age",
    "income",
    "m12_mean_balance",
    "m12_std_balance",
    "m12_mean_card_spend",
    "m12_mean_utilization",
    "m12_mean_pix",
    "m12_late_payment_rate",
]

    X = df[train_cols].copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    km.fit(Xs)

    if joblib is not None:
        joblib.dump(scaler, scaler_path)
        joblib.dump(km, km_path)

    return scaler, km


def build_monthly_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    1 linha por (cliente, mês), usando métricas do próprio mês.
    """
    cols = [
        "customer_id",
        "month",
        "age",
        "income",
        "balance",
        "card_spend",
        "utilization",
        "pix_count",
        "late_payment",
    ]
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Colunas faltando no raw: {missing}")

    df = raw[cols].copy()
    df["customer_id"] = df["customer_id"].astype(int)
    df["month"] = df["month"].astype(int)

    # garante numéricos
    for c in ["income", "balance", "card_spend", "utilization"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["pix_count", "late_payment", "age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["customer_id", "month"]).sort_values(["customer_id", "month"]).reset_index(drop=True)
    return df


def assign_clusters_monthly(df_month: pd.DataFrame, scaler: StandardScaler, km: KMeans) -> pd.DataFrame:
    """
    Aproximação para aplicar o KMeans (treinado em agregados) no dado mensal:
      mean_balance      <- balance
      std_balance       <- 0 (não existe no mensal puro; você pode melhorar depois)
      mean_card_spend   <- card_spend
      mean_utilization  <- utilization
      mean_pix          <- pix_count
      late_payment_rate <- late_payment (0/1)
    """
    X = pd.DataFrame(
    {
        "age": df_month["age"].astype(float),
        "income": df_month["income"].astype(float),
        "m12_mean_balance": df_month["balance"].astype(float),
        "m12_std_balance": np.zeros(len(df_month), dtype=float),
        "m12_mean_card_spend": df_month["card_spend"].astype(float),
        "m12_mean_utilization": df_month["utilization"].astype(float),
        "m12_mean_pix": df_month["pix_count"].astype(float),
        "m12_late_payment_rate": df_month["late_payment"].astype(float),
    }
)

    Xs = scaler.transform(X)
    out = df_month.copy()
    out["cluster"] = km.predict(Xs).astype(int)
    return out


def compute_markov(df_clusters: pd.DataFrame):
    """
    Matriz de transição:
      P(cluster_{t+1} | cluster_t)
    """
    df = df_clusters.sort_values(["customer_id", "month"]).copy()
    df["cluster_next"] = df.groupby("customer_id")["cluster"].shift(-1)
    df["month_next"] = df.groupby("customer_id")["month"].shift(-1)

    trans = df[(df["cluster_next"].notna()) & (df["month_next"] == df["month"] + 1)].copy()
    trans["cluster_next"] = trans["cluster_next"].astype(int)

    counts = pd.crosstab(trans["cluster"], trans["cluster_next"]).sort_index(axis=0).sort_index(axis=1)

    # garantir estados completos 0..k-1
    k = int(max(df["cluster"].max(), trans["cluster_next"].max()))
    all_states = list(range(k + 1))
    counts = counts.reindex(index=all_states, columns=all_states, fill_value=0)

    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    counts.index.name = "state"
    counts.columns.name = "next_state"
    probs.index.name = "state"
    probs.columns.name = "next_state"

    return counts, probs


def main() -> None:
    ensure_dirs()

    raw_path = Path("data/raw/transactions_monthly.csv")
    if not raw_path.exists():
        raise FileNotFoundError("Arquivo não encontrado: data/raw/transactions_monthly.csv")

    raw = pd.read_csv(raw_path)

    # 1) tabela mensal
    df_month = build_monthly_table(raw)

    # 2) modelo de cluster
    scaler, km = load_or_train_kmeans(k=4, seed=42)

    # 3) cluster por mês
    df_clusters = assign_clusters_monthly(df_month, scaler, km)
    df_clusters.to_csv("data/processed/customer_monthly_with_cluster.csv", index=False)

    # 4) Markov
    counts, probs = compute_markov(df_clusters)
    print("COUNTS TYPE:", type(counts), "SHAPE:", getattr(counts, "shape", None))
    print("COUNTS HEAD:\n", counts.head() if hasattr(counts, "head") else counts)
    

    out_dir = ROOT / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_path = out_dir / "markov_transition_counts.csv"
    probs_path  = out_dir / "markov_transition_matrix.csv"

    counts.to_csv(counts_path)
    probs.to_csv(probs_path)

    print("CWD:", Path.cwd().resolve())
    print("Counts saved to:", counts_path.resolve(), "| exists:", counts_path.exists(), "| shape:", getattr(counts, "shape", None))
    print("Probs  saved to:", probs_path.resolve(),  "| exists:", probs_path.exists(),  "| shape:", getattr(probs, "shape", None))

if __name__ == "__main__":
    main()