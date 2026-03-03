import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Markov Dashboard", layout="wide")
st.title("Markov Transitions Dashboard")

BASE = os.path.join("reports", "tables")
MATRIX_PATH = os.path.join(BASE, "markov_transition_matrix.csv")
COUNTS_PATH = os.path.join(BASE, "markov_transition_counts.csv")

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_read(path: str, label: str):
    if not os.path.exists(path):
        st.warning(f"Arquivo não encontrado: {path}")
        return None
    try:
        return load_csv(path)
    except Exception as e:
        st.error(f"Falha ao ler {label}: {path}")
        st.code(str(e))
        return None

matrix_df = safe_read(MATRIX_PATH, "matriz")
counts_df = safe_read(COUNTS_PATH, "contagens")

c1, c2 = st.columns([1.2, 1])

with c1:
    st.subheader("Matriz de transição")
    if matrix_df is None:
        st.stop()

    st.dataframe(matrix_df, width="stretch")

    # Tentativa opcional de heatmap, sem quebrar
    st.markdown("**Visual (heatmap)**")
    try:
        st.dataframe(matrix_df.style.background_gradient(axis=None), width="stretch")
    except Exception:
        st.info("Heatmap não disponível para este formato de dados (ok).")

with c2:
    st.subheader("Contagens de transições")
    if counts_df is None:
        st.info("Arquivo de contagens não carregado (ok).")
    else:
        st.dataframe(counts_df, width="stretch")

    st.divider()
    st.subheader("Insights (opcional)")
    st.caption("Se o CSV estiver no formato esperado (matriz quadrada), calculamos a diagonal. Caso contrário, não mostramos.")

    try:
        df = matrix_df.copy()
        # tenta usar 'state' como índice; senão usa primeira coluna
        if "state" in df.columns:
            M = df.set_index("state")
        else:
            M = df.set_index(df.columns[0])

        diag = {}
        for s in M.index:
            if s in M.columns:
                diag[s] = float(M.loc[s, s])

        if diag:
            st.table(pd.DataFrame({"stay_prob": diag}).sort_values("stay_prob", ascending=False))
        else:
            st.info("Não foi possível identificar diagonal automaticamente (formato diferente).")
    except Exception:
        st.info("Insights não disponíveis para este formato (ok).")