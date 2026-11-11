"""Página 3 — Regras fuzzy e simulador interativo."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import estilo as e
from fuzzy_core import SISTEMAS, U_NOTA

e.configurar_pagina("🎛️ Regras fuzzy e simulador")

st.markdown(
    """
Aqui está o "cérebro" do sistema: as **regras SE–ENTÃO**. Ajuste o perfil de um aluno nos controles e veja,
passo a passo, **quais regras disparam**, como as respostas são **combinadas** e qual **nota** sai no final.
"""
)

escolha = st.radio("Sistema", list(SISTEMAS), format_func=lambda k: SISTEMAS[k]().nome, horizontal=True)
sis = SISTEMAS[escolha]()

st.sidebar.header("Perfil do aluno")
entradas = {}
for nome, var in sis.entradas.items():
    lo, hi = var.universo
    if nome == "higher_n":
        entradas[nome] = 1 if st.sidebar.selectbox(var.rotulo, ["sim", "não"], key=f"p_{escolha}_{nome}") == "sim" else 0
    elif nome in ("studytime", "failures", "Walc"):
        padrao = {"studytime": 2, "failures": 0, "Walc": 2}[nome]
        entradas[nome] = st.sidebar.slider(var.rotulo, int(lo), int(hi), padrao, 1, help=var.unidade, key=f"p_{escolha}_{nome}")
    else:
        entradas[nome] = st.sidebar.slider(var.rotulo, float(lo), float(hi), 11.0, 0.5, help=var.unidade, key=f"p_{escolha}_{nome}")

res = sis.prever_um(**entradas)
forcas, agg, nota = res["forcas"], res["agregado"], res["nota"]

c1, c2, c3 = st.columns(3)
c1.metric("Nota prevista (G3)", f"{nota:.1f}")
c2.metric("Situação", "Aprovado" if nota >= 10 else "Em risco de reprovação")
c3.metric("Regras ativas", f"{int((forcas > 0.01).sum())} de {len(sis.regras)}")

st.subheader("1 · Quais regras dispararam?")
linhas_html = []
for r, f in zip(sis.regras, forcas):
    cor = e.AZUL if f > 0.01 else e.GRADE
    opac = "1" if f > 0.01 else "0.55"
    linhas_html.append(
        f"<tr style='opacity:{opac};border-bottom:1px solid #F3F4F6'>"
        f"<td style='padding:6px 8px'><b>{r.descricao()}</b><br><span style='color:#6B7280;font-size:0.85em'>{r.texto}</span></td>"
        f"<td style='padding:6px 8px;width:150px'>"
        f"<div style='background:{e.GRADE};border-radius:4px;height:12px'><div style='background:{cor};width:{f*100:.0f}%;height:12px;border-radius:4px'></div></div>"
        f"<div style='text-align:right;font-size:0.85em'>{f:.2f}</div></td></tr>"
    )
st.markdown(
    "<table style='width:100%;border-collapse:collapse'>"
    "<tr style='border-bottom:2px solid #E5E7EB;text-align:left'><th style='padding:6px 8px'>Regra</th><th style='padding:6px 8px'>Força (0 a 1)</th></tr>"
    + "".join(linhas_html)
    + "</table>",
    unsafe_allow_html=True,
)
e.como_ler(
    "a **força** de cada regra é o menor grau entre as condições (o **E** vale o elo mais fraco). "
    "Força 0 = a regra não se aplica a este aluno; quanto mais perto de 1, mais ela pesa na decisão."
)

st.subheader("2 · Como as regras viram uma nota")
nomes_saida = list(sis.saida.conjuntos)
cores = {c: e.CORES_CONJUNTOS[i % len(e.CORES_CONJUNTOS)] for i, c in enumerate(nomes_saida)}
fig, ax = plt.subplots(figsize=(9, 3.6))
for c in nomes_saida:
    ax.plot(U_NOTA, sis.saida.pertinencia(U_NOTA, c), color=cores[c], lw=1.2, ls="--", alpha=0.7)
ax.fill_between(U_NOTA, agg, color=e.AZUL, alpha=0.35, label="área resultante (regras combinadas)")
ax.plot(U_NOTA, agg, color=e.AZUL, lw=2)
if agg.sum() > 0:
    ax.axvline(nota, color=e.TINTA, lw=2)
    ax.text(nota + 0.2, max(agg.max(), 0.2) * 1.02, f"centroide = {nota:.1f}", fontsize=10, fontweight="bold", color=e.TINTA)
ax.axvline(9.5, color=e.CINZA, ls=":", lw=1)
ax.set_xlim(0, 20)
ax.set_ylim(0, 1.15)
ax.set_xlabel("Nota final (G3)")
ax.set_ylabel("Grau")
for c in nomes_saida:
    cx = U_NOTA[np.argmax(sis.saida.pertinencia(U_NOTA, c))]
    ax.text(cx, 1.07, c, color=cores[c], ha="center", fontsize=9, fontweight="bold")
fig.tight_layout()
st.pyplot(fig)
e.como_ler(
    "as linhas tracejadas são os conjuntos de saída (de risco a ótima). Cada regra ativa 'corta' o seu conjunto na altura da sua força; "
    "a área azul é a união desses cortes. A **linha preta** é o **centroide** (centro de gravidade da área): é a nota prevista. "
    "A linha pontilhada cinza marca o corte de aprovação (10)."
)

with st.expander("Como o cálculo é feito (em 4 passos)"):
    st.markdown(
        """
1. **Fuzzificação:** cada entrada vira graus de pertinência (página 2).
2. **Avaliação das regras:** em cada regra, o **E** usa o **mínimo** dos graus → *força* da regra.
3. **Agregação:** o conjunto de saída de cada regra é cortado na força da regra; todos os cortes são unidos pelo **máximo**.
4. **Defuzzificação:** o **centroide** da área resultante dá a nota.

O código está em `fuzzy_core.py` (motor próprio, vetorizado) e é conferido contra o `scikit-fuzzy` em `tests_fuzzy.py`.
"""
    )
