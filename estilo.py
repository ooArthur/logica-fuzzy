"""Estilo visual e funções de apoio compartilhadas pelas páginas do app."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

TINTA = "#1F2937"
CINZA = "#9CA3AF"
GRADE = "#E5E7EB"
AZUL = "#2563EB"
VERDE = "#059669"
LARANJA = "#D97706"
VERMELHO = "#DC2626"
ROXO = "#7C3AED"
# uma cor por conjunto fuzzy, na ordem natural de "pior" para "melhor"
CORES_CONJUNTOS = [VERMELHO, LARANJA, AZUL, VERDE, ROXO]

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.edgecolor": CINZA,
    "axes.labelcolor": TINTA,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.titlecolor": TINTA,
    "xtick.color": TINTA,
    "ytick.color": TINTA,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def como_ler(texto: str) -> None:
    """Legenda curta abaixo de cada gráfico explicando como interpretá-lo."""
    st.caption(f"**Como ler:** {texto}")


def configurar_pagina(titulo: str) -> None:
    st.title(titulo)


def glossario() -> None:
    with st.expander("📖 Glossário rápido (clique para abrir)"):
        st.markdown(
            """
- **Lógica fuzzy (nebulosa):** em vez de dizer só "sim" ou "não", diz **o quanto** algo é verdade (de 0 a 1).
  Ex.: 3 faltas são "poucas faltas" com grau 0,9, e "faltas médias" com grau 0,1.
- **Conjunto fuzzy:** uma categoria com fronteira suave, como *poucas faltas*, *estuda muito*, *nota boa*.
- **Função de pertinência:** a curva que diz o grau (0 a 1) de cada valor em cada conjunto.
- **Fuzzificação:** transformar o número medido (ex.: 6 faltas) em graus de pertinência.
- **Regra fuzzy:** frase do tipo *SE estuda muito E nunca reprovou ENTÃO nota é boa*.
  O **E** vale o menor dos graus (o "elo mais fraco").
- **Inferência de Mamdani:** cada regra "corta" o conjunto de saída na altura do seu grau; juntam-se todos os cortes.
- **Defuzzificação (centroide):** o "centro de gravidade" da área resultante vira **um número**: a nota prevista.
- **MAE:** erro médio absoluto, em pontos de nota (quanto menor, melhor).
- **AUC:** de 0,5 (chute) a 1 (perfeito); mede se o modelo coloca os alunos em risco acima dos demais.
"""
        )
