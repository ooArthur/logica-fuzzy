"""Página 2 — Conjuntos fuzzy e fuzzificação."""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import estilo as e
from fuzzy_core import SISTEMAS

e.configurar_pagina("📈 Conjuntos fuzzy: do número ao grau")

st.markdown(
    """
Aqui cada variável vira **conjuntos fuzzy**. Em vez de "estuda pouco: sim/não", cada valor recebe
um **grau de pertinência** de 0 a 1 em cada conjunto. As curvas abaixo (**funções de pertinência**) são
triângulos e trapézios, desenhados a partir do que os dados mostram (por exemplo, onde a nota média muda de faixa).
**Mova o controle** para ver a fuzzificação de um valor.
"""
)

escolha = st.radio(
    "Sistema",
    list(SISTEMAS),
    format_func=lambda k: SISTEMAS[k]().nome,
    horizontal=True,
)
sis = SISTEMAS[escolha]()

todas = {**sis.entradas, "G3": sis.saida}
for nome, var in todas.items():
    st.markdown("---")
    lo, hi = var.universo
    eh_saida = nome == "G3"
    st.subheader(("🎯 Saída: " if eh_saida else "") + var.rotulo)
    if var.unidade:
        st.caption(f"Escala: {var.unidade}")
    passo = 1 if nome in ("studytime", "failures", "Walc") else (1 if nome == "higher_n" else 0.5)
    inteiro = nome in ("studytime", "failures", "Walc", "higher_n")
    padrao = (lo + hi) / 2
    if inteiro:
        padrao = int(round(padrao))
    if nome == "G3":
        valor = st.slider("Valor para fuzzificar", float(lo), float(hi), 12.0, 0.5, key=f"s_{escolha}_{nome}")
    elif inteiro:
        valor = st.slider("Valor para fuzzificar", int(lo), int(hi), int(padrao), 1, key=f"s_{escolha}_{nome}")
    else:
        valor = st.slider("Valor para fuzzificar", float(lo), float(hi), float(padrao), float(passo), key=f"s_{escolha}_{nome}")

    x = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    graus = {}
    for i, c in enumerate(var.conjuntos):
        cor = e.CORES_CONJUNTOS[i % len(e.CORES_CONJUNTOS)]
        mu = var.pertinencia(x, c)
        ax.plot(x, mu, color=cor, lw=2.2, label=c)
        ax.fill_between(x, mu, alpha=0.12, color=cor)
        g = float(var.pertinencia([valor], c)[0])
        graus[c] = g
        if g > 0:
            ax.plot([valor], [g], "o", color=cor, ms=8, zorder=5)
            ax.plot([lo, valor], [g, g], ls=":", color=cor, lw=1)
    ax.axvline(valor, color=e.TINTA, ls="--", lw=1)
    ax.set_ylim(-0.03, 1.1)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(var.rotulo)
    ax.set_ylabel("Grau de pertinência")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(var.conjuntos), fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    ativos = [f"**{c}** = {g:.2f}" for c, g in graus.items() if g > 0]
    st.markdown(f"Valor **{valor:g}** pertence a: " + (" · ".join(ativos) if ativos else "nenhum conjunto"))
    e.como_ler(
        "cada curva é um conjunto; a altura mostra o grau (1 = pertence totalmente, 0 = não pertence). "
        "A linha tracejada é o valor escolhido e os pontos mostram os graus que ele recebe. "
        "Onde as curvas se cruzam, o valor é um pouco de cada conjunto, e é aí que a lógica fuzzy se diferencia da lógica clássica."
    )
