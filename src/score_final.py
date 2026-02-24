from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mx - mn, 0.0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def find_risk_cluster_id() -> int | None:
    """
    Identifica cluster de risco (maior late_payment_rate) a partir do cluster_profile_cards.csv.
    Se não existir, retorna None.
    """
    p = Path("reports/tables/cluster_profile_cards.csv")
    if not p.exists():
        return None
    cards = pd.read_csv(p)

    # tenta detectar o nome da coluna de atraso
    for col in ["late_payment_rate", "m12_late_payment_rate", "m3_late_payment_rate"]:
        if col in cards.columns:
            risk_col = col
            break
    else:
        return None

    if "cluster" not in cards.columns:
        return None

    risk_cluster = int(cards.sort_values(risk_col, ascending=False).iloc[0]["cluster"])
    return risk_cluster


def load_markov_matrix() -> pd.DataFrame:
    p = Path("reports/tables/markov_transition_matrix.csv")
    if not p.exists():
        raise FileNotFoundError("Missing reports/tables/markov_transition_matrix.csv (run markov_transitions).")
    M = pd.read_csv(p, index_col=0)
    # garantir colunas/índices como strings para facilitar
    M.index = M.index.map(str)
    M.columns = M.columns.map(str)
    return M


def main() -> None:
    Path("reports/tables").mkdir(parents=True, exist_ok=True)

    # --- Base de clientes (para cluster atual) ---
    base_path = Path("data/processed/customer_features_with_cluster_named.csv")
    if not base_path.exists():
        base_path = Path("data/processed/customer_features_with_cluster.csv")
    if not base_path.exists():
        raise FileNotFoundError(
            "Missing clustered features. Run:\n"
            "- python -m src.build_features\n"
            "- python -m src.cluster_profiles"
        )

    base = pd.read_csv(base_path)
    if "customer_id" not in base.columns or "cluster" not in base.columns:
        raise ValueError("Base must contain customer_id and cluster.")

    base = base[["customer_id", "cluster"]].copy()
    base["customer_id"] = base["customer_id"].astype(int)

    # --- Propensão ---
    prop_path = Path("reports/tables/propensity_scores.csv")
    if not prop_path.exists():
        raise FileNotFoundError("Missing reports/tables/propensity_scores.csv (run propensity_model).")
    prop = pd.read_csv(prop_path)
    if "customer_id" not in prop.columns:
        raise ValueError("propensity_scores.csv must have customer_id.")
    # tenta achar a coluna de propensity
    prop_col = None
    for c in ["propensity_investment", "propensity", "score", "p_investment"]:
        if c in prop.columns:
            prop_col = c
            break
    if prop_col is None:
        # pega a primeira coluna que não seja customer_id
        prop_col = [c for c in prop.columns if c != "customer_id"][0]
    prop = prop[["customer_id", prop_col]].rename(columns={prop_col: "propensity"})
    prop["customer_id"] = prop["customer_id"].astype(int)

    # --- Survival: probabilidades por horizonte + tempo esperado ---
    surv_prob_path = Path("reports/tables/survival_probabilities.csv")
    surv_time_path = Path("reports/tables/survival_expected_time.csv")
    if not surv_prob_path.exists() or not surv_time_path.exists():
        raise FileNotFoundError("Missing survival outputs. Run: python -m src.survival_model")

    surv_prob = pd.read_csv(surv_prob_path)
    surv_time = pd.read_csv(surv_time_path)

    # pegar horizonte principal (3m) se existir
    p3_col = None
    for c in ["p_adopt_3m", "p_adopt_30d", "p_30d"]:
        if c in surv_prob.columns:
            p3_col = c
            break
    if p3_col is None:
        # fallback: primeira coluna de prob além de customer_id
        pcols = [c for c in surv_prob.columns if c != "customer_id"]
        p3_col = pcols[0] if pcols else None

    surv_prob = surv_prob[["customer_id", p3_col]].rename(columns={p3_col: "p_adopt_3m"})
    surv_time = surv_time[["customer_id", "expected_time_months"]]

    surv_prob["customer_id"] = surv_prob["customer_id"].astype(int)
    surv_time["customer_id"] = surv_time["customer_id"].astype(int)

    # --- Markov: risco de migrar para “cluster de risco” no próximo mês ---
    M = load_markov_matrix()

    # caso A: matriz com estados nomeados (ex: High/Low/Medium)
    # caso B: matriz com estados numéricos (0,1,2,3)
    risk_cluster_id = find_risk_cluster_id()
    risk_state_name = "Low"  # se a matriz usar nomes

    def transition_to_risk(current_cluster: int) -> float:
        row_key = str(current_cluster)
        # se for matriz numérica e tiver essa linha
        if row_key in M.index and risk_cluster_id is not None and str(risk_cluster_id) in M.columns:
            return float(M.loc[row_key, str(risk_cluster_id)])
        # se matriz nomeada
        if risk_state_name in M.columns:
            # aqui precisamos mapear cluster -> estado textual; se não existir, não dá
            # então retorna NaN e depois a gente zera
            return np.nan
        return np.nan

    base["p_next_risk"] = base["cluster"].apply(transition_to_risk)

    # Se matriz era nomeada (High/Low/Medium), tenta usar o estado atual também nomeado (se existir na base)
    if base["p_next_risk"].isna().all() and risk_state_name in M.columns:
        # tenta usar uma coluna cluster_name se existir em algum arquivo (named)
        full = pd.read_csv(base_path)
        if "cluster_name" in full.columns:
            tmp = full[["customer_id", "cluster_name"]].copy()
            tmp["customer_id"] = tmp["customer_id"].astype(int)

            base = base.merge(tmp, on="customer_id", how="left")

            def transition_named(cluster_name: str) -> float:
                if pd.isna(cluster_name):
                    return np.nan
                s = str(cluster_name)
                if s in M.index and risk_state_name in M.columns:
                    return float(M.loc[s, risk_state_name])
                return np.nan

            base["p_next_risk"] = base["cluster_name"].apply(transition_named)

    # limpa NaNs: se não conseguiu, seta 0
    base["p_next_risk"] = base["p_next_risk"].fillna(0.0)

    # --- Merge final ---
    df = base.merge(prop, on="customer_id", how="left")
    df = df.merge(surv_prob, on="customer_id", how="left")
    df = df.merge(surv_time, on="customer_id", how="left")

    df["propensity"] = df["propensity"].fillna(0.0)
    df["p_adopt_3m"] = df["p_adopt_3m"].fillna(0.0)
    df["expected_time_months"] = df["expected_time_months"].fillna(df["expected_time_months"].median())

    # --- Construção do score único ---
    # 1) propensão (quanto maior, melhor)
    s_prop = minmax(df["propensity"])

    # 2) urgência: maior prob em 3 meses e menor tempo esperado
    s_p3 = minmax(df["p_adopt_3m"])
    s_time = 1.0 - minmax(df["expected_time_months"])  # menor tempo => maior score
    s_urgency = 0.6 * s_p3 + 0.4 * s_time

    # 3) risco de migração (quanto maior, maior prioridade de ação preventiva)
    s_risk = minmax(df["p_next_risk"])

    # pesos (ajuste livre)
    w_prop, w_urg, w_risk = 0.50, 0.30, 0.20
    df["final_score"] = (w_prop * s_prop + w_urg * s_urgency + w_risk * s_risk).round(6)

    # prioridade (faixas)
    df["priority"] = pd.cut(
        df["final_score"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"],
    )

    out_path = Path("reports/tables/final_scores.csv")
    df.sort_values("final_score", ascending=False).to_csv(out_path, index=False)

    print(f"Saved: {out_path} | rows={len(df):,}")
    print(f"Risk cluster id (if numeric Markov): {risk_cluster_id}")


if __name__ == "__main__":
    main()
