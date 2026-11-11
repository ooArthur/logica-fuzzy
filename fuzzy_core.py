"""Núcleo do sistema fuzzy (Mamdani) para prever a nota final (G3) de alunos.

Tudo aqui é independente do Streamlit: as páginas só importam estas funções.
O motor é vetorizado com NumPy (roda os 649 alunos de uma vez) e é conferido,
nos testes, contra o motor do scikit-fuzzy (ver tests_fuzzy.py).

Passo a passo de um sistema Mamdani:
  1. Fuzzificação  -> cada entrada vira graus de pertinência (0 a 1) em cada conjunto.
  2. Regras        -> SE ... E ... ENTÃO ...  (E = mínimo dos graus).
  3. Agregação     -> a saída de cada regra é "cortada" no grau da regra (mínimo)
                      e todas são unidas com o máximo.
  4. Defuzzificação-> o centroide da área resultante vira um número (a nota).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import skfuzzy as fuzz

DADOS = Path(__file__).parent / "data" / "student-por.csv"

# Nomes curtos das variáveis, usados nas frases das regras.
CURTO = {
    "studytime": "estudo", "failures": "reprovações", "Walc": "álcool no fds",
    "higher_n": "ensino superior", "G2": "nota G2", "G3": "nota final",
}

# Universo da nota final (0 a 20), com passo fino para o centroide ficar suave.
U_NOTA = np.arange(0, 20.01, 0.05)


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def carregar_dados() -> pd.DataFrame:
    """Lê o student-por.csv (separador vírgula; G1 e G2 vêm entre aspas).

    Cria duas colunas auxiliares: `higher_n` (higher yes/no -> 1/0) e
    `reprovado` (1 se G3 < 10, nota de corte do sistema de 0 a 20).
    """
    df = pd.read_csv(DADOS)
    df["higher_n"] = (df["higher"] == "yes").astype(int)
    df["reprovado"] = (df["G3"] < 10).astype(int)
    return df


# ----------------------------------------------------------------------------
# Variáveis e conjuntos fuzzy
# ----------------------------------------------------------------------------
@dataclass
class Variavel:
    nome: str                      # coluna na base (ex.: "studytime")
    rotulo: str                    # nome para mostrar (ex.: "Tempo de estudo")
    universo: tuple[float, float]  # (mínimo, máximo)
    # conjuntos: nome -> ("tri"|"trap", [parâmetros])
    conjuntos: dict[str, tuple[str, list[float]]] = field(default_factory=dict)
    unidade: str = ""

    def x(self, passo: float | None = None) -> np.ndarray:
        lo, hi = self.universo
        passo = passo or (hi - lo) / 400
        return np.arange(lo, hi + passo / 2, passo)

    def pertinencia(self, valores, conjunto: str) -> np.ndarray:
        tipo, p = self.conjuntos[conjunto]
        v = np.asarray(valores, dtype=float)
        return fuzz.trimf(v, p) if tipo == "tri" else fuzz.trapmf(v, p)

    def fuzzificar(self, valores) -> dict[str, np.ndarray]:
        return {c: self.pertinencia(valores, c) for c in self.conjuntos}


@dataclass
class Regra:
    se: dict[str, str]   # {variável: conjunto}, ligadas por E
    entao: str           # conjunto da saída
    texto: str = ""      # explicação em português

    def __str__(self) -> str:
        cond = " E ".join(f"{v} é {c}" for v, c in self.se.items())
        return f"SE {cond} ENTÃO nota é {self.entao}"

    def descricao(self) -> str:
        """Mesma regra, com nomes curtos e legíveis (para mostrar na tela)."""
        cond = " E ".join(f"{CURTO.get(v, v)} é {c}" for v, c in self.se.items())
        return f"SE {cond} ENTÃO nota é {self.entao}"


@dataclass
class SistemaFuzzy:
    nome: str
    entradas: dict[str, Variavel]
    saida: Variavel
    regras: list[Regra]

    # -- passo 1 e 2: força (grau de ativação) de cada regra, aluno a aluno ----
    def forcas(self, df: pd.DataFrame) -> np.ndarray:
        graus = {n: v.fuzzificar(df[n].to_numpy()) for n, v in self.entradas.items()}
        f = np.empty((len(df), len(self.regras)))
        for j, r in enumerate(self.regras):
            g = np.ones(len(df))
            for var, conj in r.se.items():
                g = np.minimum(g, graus[var][conj])   # E = mínimo
            f[:, j] = g
        return f

    # -- passos 3 e 4: agregação (máximo dos cortes) e centroide ---------------
    def agregar(self, forcas: np.ndarray) -> np.ndarray:
        mf_saida = {c: self.saida.pertinencia(U_NOTA, c) for c in self.saida.conjuntos}
        agg = np.zeros((forcas.shape[0], len(U_NOTA)))
        for j, r in enumerate(self.regras):
            corte = np.minimum(forcas[:, [j]], mf_saida[r.entao][None, :])
            agg = np.maximum(agg, corte)
        return agg

    def prever(self, df: pd.DataFrame, valor_sem_regra: float | None = None) -> np.ndarray:
        agg = self.agregar(self.forcas(df))
        area = agg.sum(axis=1)
        centro = (agg * U_NOTA[None, :]).sum(axis=1) / np.where(area == 0, 1, area)
        base = valor_sem_regra if valor_sem_regra is not None else float(U_NOTA.mean())
        return np.where(area == 0, base, centro)

    def prever_um(self, **entradas: float) -> dict:
        """Versão para o simulador: devolve tudo que a tela precisa desenhar."""
        df = pd.DataFrame([entradas])
        f = self.forcas(df)[0]
        agg = self.agregar(f[None, :])[0]
        area = agg.sum()
        nota = float((agg * U_NOTA).sum() / area) if area > 0 else float("nan")
        return {"nota": nota, "forcas": f, "agregado": agg}


# ----------------------------------------------------------------------------
# Saída comum aos dois sistemas: nota final de 0 a 20
# Partição padrão: 4 triângulos espaçados em ~3 pontos, centrados onde as notas
# reais costumam ficar (risco ~8, regular ~11,5, boa ~14,5, ótima ~17,5).
# ----------------------------------------------------------------------------
def _saida() -> Variavel:
    return Variavel(
        "G3", "Nota final (G3)", (0, 20),
        {
            "risco":   ("tri",  [5, 8, 11]),
            "regular": ("tri",  [9, 11.5, 14]),
            "boa":     ("tri",  [12, 14.5, 17]),
            "ótima":   ("trap", [15, 17.5, 20, 20]),
        },
        "pontos",
    )


def _particao5() -> dict[str, tuple[str, list[float]]]:
    """5 conjuntos centrados em 6, 9, 12, 15 e 18 (usada no sistema B, onde G2
    já é muito informativa e uma partição mais fina deixa a saída mais precisa)."""
    return {
        "crítica": ("trap", [0, 0, 6, 9]),
        "risco":   ("tri",  [6, 9, 12]),
        "regular": ("tri",  [9, 12, 15]),
        "boa":     ("tri",  [12, 15, 18]),
        "ótima":   ("trap", [15, 18, 20, 20]),
    }


# ----------------------------------------------------------------------------
# Sistema A — perfil do aluno (SEM notas parciais): hábitos e histórico
# Serve como "alerta precoce": dá para usar antes de qualquer prova.
# ----------------------------------------------------------------------------
def sistema_perfil() -> SistemaFuzzy:
    ent = {
        "studytime": Variavel("studytime", "Tempo de estudo semanal", (1, 4), {
            "pouco": ("trap", [1, 1, 1.5, 2.3]),
            "médio": ("tri",  [1.5, 2.3, 3.2]),
            "muito": ("trap", [2.5, 3.3, 4, 4]),
        }, "1 = <2h · 2 = 2–5h · 3 = 5–10h · 4 = >10h"),
        "failures": Variavel("failures", "Reprovações anteriores", (0, 3), {
            "nenhuma": ("trap", [0, 0, 0.2, 1]),
            "algumas": ("trap", [0.3, 1, 3, 3]),
        }, "quantidade"),
        "Walc": Variavel("Walc", "Álcool no fim de semana", (1, 5), {
            "baixo": ("trap", [1, 1, 1.5, 3]),
            "alto":  ("trap", [2.5, 4, 5, 5]),
        }, "1 = muito baixo … 5 = muito alto"),
        "higher_n": Variavel("higher_n", "Quer cursar ensino superior", (0, 1), {
            "não": ("trap", [0, 0, 0.1, 0.6]),
            "sim": ("trap", [0.4, 0.9, 1, 1]),
        }, "0 = não · 1 = sim"),
    }
    R = Regra
    regras = [
        R({"failures": "algumas"}, "risco",
          "Quem já reprovou tem nota média ~8,6, contra ~12,5 de quem nunca reprovou."),
        R({"higher_n": "não"}, "risco",
          "Sem intenção de ensino superior, a nota média cai para ~8,8 (contra ~12,3)."),
        R({"failures": "nenhuma", "higher_n": "sim", "studytime": "pouco"}, "regular",
          "Histórico limpo e ambição, mas pouco estudo: nota mediana."),
        R({"failures": "nenhuma", "higher_n": "sim", "studytime": "médio"}, "regular",
          "Perfil intermediário em estudo, com histórico limpo e ambição."),
        R({"failures": "nenhuma", "higher_n": "sim", "studytime": "muito"}, "boa",
          "Muito estudo, sem reprovação e com ambição: nota boa."),
        R({"failures": "nenhuma", "higher_n": "sim", "studytime": "médio", "Walc": "baixo"}, "boa",
          "Rotina equilibrada (estudo médio e pouco álcool) leva a nota boa."),
        R({"failures": "nenhuma", "higher_n": "sim", "studytime": "muito", "Walc": "baixo"}, "ótima",
          "Combinação ideal: estudo alto, pouco álcool, sem reprovação, com ambição."),
        R({"Walc": "alto", "studytime": "pouco"}, "risco",
          "Muito álcool no fim de semana com pouco estudo: risco."),
        R({"Walc": "alto", "failures": "nenhuma", "higher_n": "sim"}, "regular",
          "Álcool alto segura a nota em regular, mesmo com histórico limpo."),
    ]
    return SistemaFuzzy("A · Perfil do aluno (sem notas parciais)", ent, _saida(), regras)


# ----------------------------------------------------------------------------
# Sistema B — com a nota do 2º período (G2): o que o professor já sabe antes do fim
# ----------------------------------------------------------------------------
def sistema_com_g2() -> SistemaFuzzy:
    p5 = _particao5()
    entrada_g2 = {
        "muito baixa": p5["crítica"], "baixa": p5["risco"], "média": p5["regular"],
        "boa": p5["boa"], "alta": p5["ótima"],
    }
    ent = {
        "G2": Variavel("G2", "Nota do 2º período (G2)", (0, 20), entrada_g2, "pontos"),
        "failures": Variavel("failures", "Reprovações anteriores", (0, 3), {
            "nenhuma": ("trap", [0, 0, 0.2, 1]),
            "algumas": ("trap", [0.3, 1, 3, 3]),
        }, "quantidade"),
        "studytime": Variavel("studytime", "Tempo de estudo semanal", (1, 4), {
            "pouco": ("trap", [1, 1, 1.5, 2.3]),
            "muito": ("trap", [2.5, 3.3, 4, 4]),
        }, "1 = <2h · 2 = 2–5h · 3 = 5–10h · 4 = >10h"),
    }
    saida = Variavel("G3", "Nota final (G3)", (0, 20), p5, "pontos")
    R = Regra
    regras = [
        R({"G2": "muito baixa"}, "crítica", "Nota muito baixa no 2º período costuma terminar crítica."),
        R({"G2": "baixa"}, "risco", "Nota baixa no 2º período tende a continuar baixa."),
        R({"G2": "média"}, "regular", "Nota média se mantém média."),
        R({"G2": "boa"}, "boa", "Nota boa se mantém boa."),
        R({"G2": "alta"}, "ótima", "Nota alta se mantém ótima."),
        R({"G2": "média", "failures": "algumas"}, "risco",
          "Nota média com reprovação prévia tende a cair para a faixa de risco."),
        R({"G2": "boa", "failures": "algumas"}, "regular",
          "Nota boa com reprovação prévia tende a recuar um degrau."),
        R({"G2": "baixa", "studytime": "muito"}, "regular",
          "Nota baixa, mas com muito estudo: tende a recuperar."),
        R({"G2": "média", "studytime": "muito"}, "boa",
          "Nota média com muito estudo: tende a subir."),
    ]
    return SistemaFuzzy("B · Com a nota do 2º período (G2)", ent, saida, regras)


SISTEMAS = {"A": sistema_perfil, "B": sistema_com_g2}


# ----------------------------------------------------------------------------
# Avaliação
# ----------------------------------------------------------------------------
def metricas(y_real, y_prev, corte: float = 10.0) -> dict:
    y_real = np.asarray(y_real, float)
    y_prev = np.asarray(y_prev, float)
    erro = y_prev - y_real
    aprov_real = y_real >= corte
    aprov_prev = y_prev >= corte
    return {
        "MAE": float(np.mean(np.abs(erro))),
        "RMSE": float(np.sqrt(np.mean(erro ** 2))),
        "R²": float(1 - np.sum(erro ** 2) / np.sum((y_real - y_real.mean()) ** 2)),
        "Acerto aprovado/reprovado": float(np.mean(aprov_real == aprov_prev)),
    }
