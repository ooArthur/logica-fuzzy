"""Página 4 — Resultados: o sistema fuzzy contra a base real e contra modelos tradicionais."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, cross_val_predict

import estilo as e
from fuzzy_core import SISTEMAS, carregar_dados, metricas

e.configurar_pagina("✅ Resultados: o fuzzy acerta?")

st.markdown(
    """
Comparamos os dois sistemas fuzzy com **três referências**: um chute (sempre prever a média), uma **regressão linear** e uma
**floresta aleatória** (as duas com validação cruzada de 5 partes, para ser justo). As regras fuzzy foram escritas à mão a partir
do que a página 1 mostrou; **não há treino automático**.
"""
)

FEATURES = {"A": ["studytime", "failures", "Walc", "higher_n"], "B": ["G2", "failures", "studytime"]}


@st.cache_data(show_spinner="Calculando previsões...")
def calcular():
    df = carregar_dados()
    cv = KFold(5, shuffle=True, random_state=42)
    saida = {}
    for k, f in SISTEMAS.items():
        sis = f()
        prev = {"Fuzzy": sis.prever(df), "Chute (média)": np.full(len(df), df.G3.mean())}
        prev["Regressão linear"] = cross_val_predict(LinearRegression(), df[FEATURES[k]], df.G3, cv=cv)
        prev["Floresta aleatória"] = cross_val_predict(
            RandomForestRegressor(300, min_samples_leaf=5, random_state=0), df[FEATURES[k]], df.G3, cv=cv
        )
        saida[k] = prev
    return df, saida


df, prev = calcular()

escolha = st.radio("Sistema", list(SISTEMAS), format_func=lambda k: SISTEMAS[k]().nome, horizontal=True)
p = prev[escolha]

linhas = []
for nome, y in p.items():
    m = metricas(df.G3, y)
    auc = roc_auc_score(df.reprovado, -y) if nome != "Chute (média)" else 0.5
    linhas.append({"Modelo": nome, "Erro médio (MAE)": m["MAE"], "R²": m["R²"], "AUC (risco de reprovar)": auc})
tab = pd.DataFrame(linhas)

m_f = tab[tab.Modelo == "Fuzzy"].iloc[0]
m_l = tab[tab.Modelo == "Regressão linear"].iloc[0]
a, b, c = st.columns(3)
a.metric("Erro médio do fuzzy", f"{m_f['Erro médio (MAE)']:.2f} pts", f"{m_f['Erro médio (MAE)'] - m_l['Erro médio (MAE)']:+.2f} vs regressão", delta_color="inverse")
b.metric("AUC do fuzzy", f"{m_f['AUC (risco de reprovar)']:.2f}", f"{m_f['AUC (risco de reprovar)'] - m_l['AUC (risco de reprovar)']:+.2f} vs regressão")
c.metric("R² do fuzzy", f"{m_f['R²']:.2f}")

st.dataframe(
    tab.style.format({"Erro médio (MAE)": "{:.2f}", "R²": "{:.2f}", "AUC (risco de reprovar)": "{:.2f}"}),
    hide_index=True,
    width="stretch",
)
e.como_ler(
    "**MAE** = quantos pontos de nota o modelo erra em média (menor é melhor). **AUC** mede se o modelo põe os alunos que "
    "realmente reprovaram acima dos demais na lista de risco (0,5 = chute, 1 = perfeito). **R²** = fração da variação das notas explicada."
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Real × previsto")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    jit = np.random.default_rng(0).uniform(-0.25, 0.25, len(df))
    ax.scatter(df.G3 + jit, p["Fuzzy"], s=12, alpha=0.35, color=e.AZUL, edgecolor="none")
    ax.plot([0, 20], [0, 20], color=e.TINTA, lw=1.2, ls="--")
    ax.axhline(9.5, color=e.CINZA, ls=":", lw=1)
    ax.axvline(9.5, color=e.CINZA, ls=":", lw=1)
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)
    ax.set_xlabel("Nota real (G3)")
    ax.set_ylabel("Nota prevista pelo fuzzy")
    fig.tight_layout()
    st.pyplot(fig)
    e.como_ler("cada ponto é um aluno; quanto mais perto da linha tracejada, melhor a previsão. As linhas pontilhadas marcam o corte 10.")
with col2:
    st.subheader("Curva ROC (detectar quem reprova)")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for nome, cor in [("Fuzzy", e.AZUL), ("Regressão linear", e.LARANJA), ("Floresta aleatória", e.VERDE)]:
        fpr, tpr, _ = roc_curve(df.reprovado, -p[nome])
        ax.plot(fpr, tpr, color=cor, lw=2.2 if nome == "Fuzzy" else 1.5, label=f"{nome} (AUC {roc_auc_score(df.reprovado, -p[nome]):.2f})")
    ax.plot([0, 1], [0, 1], color=e.CINZA, ls="--", lw=1)
    ax.set_xlabel("Falsos alarmes (aprovados marcados em risco)")
    ax.set_ylabel("Reprovados detectados")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    e.como_ler("quanto mais a curva sobe rápido para o canto superior esquerdo, melhor o modelo em separar quem reprova de quem passa.")

# Alerta precoce: se o professor acompanhar só os N% de maior risco, quantos reprovados pega?
st.subheader("Alerta precoce: acompanhando só quem o fuzzy aponta")
risco = p["Fuzzy"].argsort(kind="stable")  # menor nota prevista = maior risco
ordem = df.reprovado.to_numpy()[risco]
fracoes = [0.1, 0.2, 0.3, 0.4]
linhas = []
for fr in fracoes:
    k = int(round(len(df) * fr))
    pegos = int(ordem[:k].sum())
    linhas.append({"Alunos acompanhados": f"{fr:.0%} ({k})", "Reprovados capturados": f"{pegos} de {int(df.reprovado.sum())}", "Fração dos reprovados": pegos / df.reprovado.sum()})
st.dataframe(
    pd.DataFrame(linhas).style.format({"Fração dos reprovados": "{:.0%}"}),
    hide_index=True,
    width="stretch",
)
e.como_ler("se a escola acompanhar de perto apenas os alunos com a menor nota prevista, esta tabela mostra quanto dos reprovados reais ela alcança.")

st.subheader("Conclusões")
if escolha == "A":
    st.markdown(
        f"""
- Sem nenhuma nota parcial, **prever o valor exato da nota é difícil para qualquer modelo** (o erro de todos fica perto de 2 pontos, pouco abaixo do chute).
- Mas como **alerta de risco** o fuzzy funciona: AUC **{m_f['AUC (risco de reprovar)']:.2f}**, praticamente igual à regressão linear ({m_l['AUC (risco de reprovar)']:.2f}),
  e com a vantagem de ser **totalmente explicável**: cada alerta vem com as regras que o causaram.
- Atenção: o "acerto aprovado/reprovado" com corte fixo em 10 fica abaixo do chute, porque só 15% reprovam. O valor do fuzzy está em **ordenar o risco** (AUC e tabela de alerta precoce).
- O erro médio do fuzzy é um pouco maior que o da regressão, provavelmente porque sua saída é "agrupada" em poucos conjuntos: ele ordena bem o risco, mas não calibra a nota fina.
"""
    )
else:
    st.markdown(
        f"""
- Com a nota do 2º período, o fuzzy passa a acertar a nota final com erro médio de cerca de **{m_f['Erro médio (MAE)']:.1f} ponto**
  e AUC de **{m_f['AUC (risco de reprovar)']:.2f}**.
- A regressão linear ainda é um pouco mais precisa ({m_l['Erro médio (MAE)']:.2f} de erro), pois a nota final é quase uma reta da nota do 2º período.
  O fuzzy fica perto e entrega o raciocínio por trás de cada previsão.
- As regras de correção (reprovação prévia, muito estudo) ajustam a previsão para cima ou para baixo, como faria um professor experiente.
"""
    )
st.info(
    "**Limitação:** os conjuntos e regras foram definidos observando esta própria base; os números acima são, por isso, "
    "um pouco otimistas para dados novos. Numa aplicação real, validaríamos com outra turma."
)
