from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    # Load features (1 row per customer)
    df = pd.read_csv("data/processed/customer_features.csv")

    # Choose columns for clustering (exclude IDs and targets)
    feature_cols = [
        "age",
        "income",
        "mean_balance",
        "std_balance",
        "mean_card_spend",
        "mean_utilization",
        "mean_pix",
        "late_payment_rate",
    ]

    X = df[feature_cols].copy()

    # Scale features (critical for distance-based clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    k = 4
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    clusters = km.fit_predict(X_scaled)

    # Add cluster to dataset
    df_out = df.copy()
    df_out["cluster"] = clusters

    # Cluster quality (silhouette)
    sil = silhouette_score(X_scaled, clusters)

    # Summarize clusters (means)
    summary = (
        df_out.groupby("cluster")[feature_cols]
        .mean()
        .round(3)
        .reset_index()
        .sort_values("cluster")
    )
    
    # --- Profile naming + descriptions (simple heuristics) ---
    counts = df_out["cluster"].value_counts().sort_index()
    summary2 = summary.merge(counts.rename("n_customers"), left_on="cluster", right_index=True)

    # Rank clusters to define human labels
    risk_rank = summary2["late_payment_rate"].rank(method="dense", ascending=False)
    income_rank = summary2["income"].rank(method="dense", ascending=False)
    pix_rank = summary2["mean_pix"].rank(method="dense", ascending=False)
    util_rank = summary2["mean_utilization"].rank(method="dense", ascending=False)

    cluster_name_map = {}
    cluster_desc_map = {}

    for _, row in summary2.iterrows():
        c = int(row["cluster"])

        # Identify characteristics
        high_risk = risk_rank[row.name] == 1
        high_income = income_rank[row.name] == 1
        high_digital = pix_rank[row.name] == 1
        high_credit = util_rank[row.name] == 1

        # Name (prioritize risk, then income, then digital)
        if high_risk and high_digital:
            name = "Digital Crédito Intensivo"
            desc = "Alto uso digital (PIX) e maior risco (atrasos). Perfil sensível a gestão de crédito e prevenção."
        elif high_income:
            name = "Alta Renda Estável"
            desc = "Maior renda e baixo risco. Perfil com alto potencial para investimento e cross-sell premium."
        elif high_digital:
            name = "Digital Estável"
            desc = "Uso digital alto com baixa inadimplência. Bom candidato a expansão de portfólio (investimentos/seguros)."
        else:
            name = "Conservador Tradicional"
            desc = "Uso digital moderado e baixo risco. Perfil mais tradicional, boa resposta a ofertas simples e educação financeira."

        cluster_name_map[c] = name
        cluster_desc_map[c] = desc

    # Add to per-customer table
    df_out["cluster_name"] = df_out["cluster"].map(cluster_name_map)

    # Build profile cards table (one row per cluster)
    cards = summary2.copy()
    cards["cluster_name"] = cards["cluster"].map(cluster_name_map)
    cards["profile_description"] = cards["cluster"].map(cluster_desc_map)

    # Select/rename columns for readability
    cards_out = cards[
        ["cluster", "cluster_name", "n_customers",
         "age", "income", "mean_balance", "std_balance",
         "mean_card_spend", "mean_utilization", "mean_pix", "late_payment_rate",
         "profile_description"]
    ].copy()    

    # Save outputs
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)

    df_out.to_csv("data/processed/customer_features_with_cluster.csv", index=False)
    summary.to_csv("reports/tables/cluster_summary.csv", index=False)
    
    cards_out.to_csv("reports/tables/cluster_profile_cards.csv", index=False)
    df_out.to_csv("data/processed/customer_features_with_cluster_named.csv", index=False)

    # Human-readable Markdown cards
    with open("reports/cluster_profile_cards.md", "w", encoding="utf-8") as f:
        f.write("# Perfis (Clusters) — Resumo\n\n")
        f.write(f"- k = {k}\n")
        f.write(f"- Silhouette = {sil:.3f}\n\n")
        for _, r in cards_out.sort_values("cluster").iterrows():
            f.write(f"## Cluster {int(r['cluster'])} — {r['cluster_name']}\n")
            f.write(f"- Clientes: {int(r['n_customers'])}\n")
            f.write(f"- Idade média: {r['age']:.1f}\n")
            f.write(f"- Renda média: {r['income']:.0f}\n")
            f.write(f"- Utilização média: {r['mean_utilization']:.3f}\n")
            f.write(f"- PIX médio: {r['mean_pix']:.1f}\n")
            f.write(f"- Taxa de atraso: {r['late_payment_rate']:.3f}\n")
            f.write(f"- Descrição: {r['profile_description']}\n\n")
    
    # Also save a human-readable markdown report
    Path("reports").mkdir(parents=True, exist_ok=True)

    with open("reports/cluster_report.md", "w", encoding="utf-8") as f:
        f.write("# Cluster Report (K-Means)\n\n")
        f.write(f"- k = {k}\n")
        f.write(f"- Silhouette = {sil:.3f}\n\n")
        f.write("## Cluster Summary (means)\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    print(f"Saved: data/processed/customer_features_with_cluster.csv | rows={len(df_out)}")
    print("Saved: reports/tables/cluster_summary.csv")
    print(f"Silhouette (k={k}): {sil:.3f}")


if __name__ == "__main__":
    main()