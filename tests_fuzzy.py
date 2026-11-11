"""Testes do sistema fuzzy. Rode com:  python tests_fuzzy.py

1. Confere o motor NumPy (fuzzy_core) contra o motor de referência do scikit-fuzzy
   (skfuzzy.control), linha a linha, nos dois sistemas.
2. Confere propriedades básicas: pertinências entre 0 e 1, nota sempre dentro de 0–20,
   monotonicidade (mais G2 -> nota maior) e que nenhum aluno fica sem regra ativa.
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from fuzzy_core import SISTEMAS, carregar_dados, U_NOTA


def montar_skfuzzy(sistema):
    """Constrói o mesmo sistema com a API de controle do scikit-fuzzy."""
    ant = {}
    for n, v in sistema.entradas.items():
        lo, hi = v.universo
        # linspace garante que o último ponto do universo seja exatamente `hi`
        # (com arange, erro de ponto flutuante zera os conjuntos que terminam na borda).
        a = ctrl.Antecedent(np.linspace(lo, hi, int(round((hi - lo) / 0.01)) + 1), n)
        for c, (tipo, p) in v.conjuntos.items():
            a[c] = fuzz.trimf(a.universe, p) if tipo == "tri" else fuzz.trapmf(a.universe, p)
        ant[n] = a
    out = ctrl.Consequent(U_NOTA, "G3", defuzzify_method="centroid")
    for c, (tipo, p) in sistema.saida.conjuntos.items():
        out[c] = fuzz.trimf(U_NOTA, p) if tipo == "tri" else fuzz.trapmf(U_NOTA, p)
    regras = []
    for r in sistema.regras:
        cond = None
        for var, conj in r.se.items():
            t = ant[var][conj]
            cond = t if cond is None else cond & t
        regras.append(ctrl.Rule(cond, out[r.entao]))
    return ctrl.ControlSystem(regras)


def testar_contra_skfuzzy(chave, n=40):
    sis = SISTEMAS[chave]()
    df = carregar_dados().sample(n, random_state=7)
    nossa = sis.prever(df)
    cs = montar_skfuzzy(sis)
    ref = []
    for _, linha in df.iterrows():
        sim = ctrl.ControlSystemSimulation(cs)
        for nome in sis.entradas:
            sim.input[nome] = float(linha[nome])
        sim.compute()
        ref.append(sim.output["G3"])
    dif = np.max(np.abs(nossa - np.array(ref)))
    assert dif < 0.05, f"Sistema {chave}: diferença máxima {dif:.4f} contra skfuzzy"
    return dif


def testar_propriedades(chave):
    sis = SISTEMAS[chave]()
    df = carregar_dados()
    for n, v in sis.entradas.items():
        for c in v.conjuntos:
            mu = v.pertinencia(df[n].to_numpy(), c)
            assert mu.min() >= 0 and mu.max() <= 1, f"pertinência fora de [0,1] em {n}/{c}"
    f = sis.forcas(df)
    assert (f.sum(axis=1) > 0).all(), "há aluno sem nenhuma regra ativa"
    p = sis.prever(df)
    assert p.min() >= 0 and p.max() <= 20, "nota fora de 0–20"
    return float(p.min()), float(p.max())


def testar_monotonia_b():
    sis = SISTEMAS["B"]()
    notas = [sis.prever_um(G2=g, failures=0, studytime=2)["nota"] for g in range(0, 21)]
    assert all(b >= a - 1e-9 for a, b in zip(notas, notas[1:])), "nota deveria subir com G2"


if __name__ == "__main__":
    for k in SISTEMAS:
        print(f"Sistema {k}: diferença máx. contra skfuzzy = {testar_contra_skfuzzy(k):.4f}")
        lo, hi = testar_propriedades(k)
        print(f"Sistema {k}: propriedades OK (previsões entre {lo:.1f} e {hi:.1f})")
    testar_monotonia_b()
    print("Monotonia do sistema B OK")
    print("TUDO OK")
