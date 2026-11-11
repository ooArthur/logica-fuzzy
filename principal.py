"""Ponto de entrada do app.  Rode com:  streamlit run principal.py"""
import streamlit as st

st.set_page_config(page_title="Lógica Fuzzy · Desempenho de Alunos", page_icon="🎓", layout="wide")

paginas = [
    st.Page("pagina1_base.py", title="1 · A base de dados", icon="📊"),
    st.Page("pagina2_conjuntos.py", title="2 · Conjuntos fuzzy", icon="📈"),
    st.Page("pagina3_regras.py", title="3 · Regras e simulador", icon="🎛️"),
    st.Page("pagina4_avaliacao.py", title="4 · Resultados", icon="✅"),
]
st.navigation(paginas).run()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Base: *Student Performance* (Cortez & Silva, 2008) · UCI / Kaggle — português, 649 alunos."
)
