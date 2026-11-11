"""Página 1 — A base de dados e a ideia da lógica fuzzy."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import estilo as e
from fuzzy_core import carregar_dados

e.configurar_pagina("📊 Lógica Fuzzy aplicada ao desempenho de alunos")


@st.cache_data
def dados():
    return carregar_dados()


df = dados()
n = len(df)

st.markdown(
    """
**Objetivo:** prever a **nota final (G3, de 0 a 20)** de alunos do ensino médio português usando **lógica fuzzy**,
com regras que um professor entenderia e poderia conferir, sem "caixa-preta".

**Por que fuzzy?** Hábitos de estudo não têm fronteiras nítidas. Quem estuda "entre 2 e 5 horas" não é
*pouco* nem *muito*, é um pouco dos dois. A lógica fuzzy trabalha com esses **graus** (0 a 1) e permite
escrever regras em linguagem comum, como *"SE estuda muito E nunca reprovou ENTÃO a nota tende a ser boa"*.
"""
)
e.glossario()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Alunos", n)
c2.metric("Nota final média", f"{df.G3.mean():.1f}")
c3.metric("Reprovados (G3 < 10)", f"{df.reprovado.sum()} ({df.reprovado.mean():.0%})")
c4.metric("Valores ausentes", int(df.isna().sum().sum()))

st.subheader("Como as notas se distribuem")
fig, ax = plt.subplots(figsize=(8, 3.4))
cont = df.G3.value_counts().sort_index()
cores = [e.VERMELHO if k < 10 else e.AZUL for k in cont.index]
ax.bar(cont.index, cont.values, color=cores, width=0.8)
ax.axvline(9.5, color=e.TINTA, ls="--", lw=1)
ax.text(9.3, cont.max() * 0.95, "nota de corte (10)", ha="right", va="top", fontsize=9, color=e.TINTA)
ax.set_xlabel("Nota final (G3)")
ax.set_ylabel("Número de alunos")
ax.set_xticks(range(0, 21, 2))
fig.tight_layout()
st.pyplot(fig)
e.como_ler(
    "cada barra é uma nota; as vermelhas são reprovações (abaixo de 10). A maioria fica entre 10 e 14, "
    f"e há {int((df.G3 == 0).sum())} alunos com nota zero (provavelmente desistências), uma cauda difícil de prever."
)

st.subheader("O que muda a nota? (antes de qualquer modelo)")
fig, axs = plt.subplots(2, 2, figsize=(9, 6))


def barras(ax, serie, rotulos, titulo, xlabel):
    g = df.groupby(serie).G3.agg(["mean", "count"])
    cores = [e.AZUL] * len(g)
    ax.bar(range(len(g)), g["mean"], color=cores)
    for i, (m, c) in enumerate(zip(g["mean"], g["count"])):
        ax.text(i, m + 0.2, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold", color=e.TINTA)
        ax.text(i, 0.4, f"n={c}", ha="center", fontsize=8, color="white")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(rotulos)
    ax.set_ylim(0, 16)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Nota final média")


barras(axs[0, 0], "studytime", ["<2h", "2–5h", "5–10h", ">10h"], "Tempo de estudo", "horas por semana")
barras(axs[0, 1], "failures", ["0", "1", "2", "3"], "Reprovações anteriores", "reprovações")
barras(axs[1, 0], "higher", ["não", "sim"], "Quer cursar ensino superior?", "")
barras(axs[1, 1], "Walc", ["1", "2", "3", "4", "5"], "Álcool no fim de semana", "1 = muito baixo … 5 = muito alto")
fig.tight_layout()
st.pyplot(fig)
e.como_ler(
    "cada barra é a nota média de um grupo (o *n* mostra quantos alunos há nele). "
    "Quem já reprovou e quem não pretende fazer faculdade tem nota bem menor; quem estuda mais e bebe menos tende a ir melhor."
)

g_f = df.groupby("failures").G3.mean()
g_h = df.groupby("higher").G3.mean()
g_s = df.groupby("studytime").G3.mean()
corr = df[["G1", "G2", "G3"]].corr()
st.subheader("Principais achados")
st.markdown(
    f"""
- **Reprovações pesam muito:** quem nunca reprovou tem média **{g_f[0]:.1f}**; quem reprovou uma vez, **{g_f[1]:.1f}**.
- **Ambição importa:** quem quer ensino superior tem média **{g_h['yes']:.1f}**, contra **{g_h['no']:.1f}** de quem não quer.
- **Estudo ajuda, com limite:** a média sobe de **{g_s[1]:.1f}** (<2h) para **{g_s[3]:.1f}** (5–10h) e estabiliza depois.
- **As notas parciais dominam:** a correlação de G2 com a nota final é **{corr.loc['G2','G3']:.2f}** (G1: {corr.loc['G1','G3']:.2f}).
  Por isso montamos **dois sistemas**: o **A** usa só o perfil do aluno (serve como alerta precoce, antes de qualquer prova)
  e o **B** acrescenta a nota do 2º período.
- **Faltas quase não explicam a nota** nesta base (correlação de {df[['absences','G3']].corr().iloc[0,1]:.2f}), por isso ficaram fora das regras.
"""
)

with st.expander("Ver uma amostra dos dados"):
    st.dataframe(
        df[["school", "sex", "age", "studytime", "failures", "higher", "Walc", "absences", "G1", "G2", "G3"]].head(15),
        width="stretch",
    )
