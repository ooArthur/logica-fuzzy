# Lógica Fuzzy aplicada ao desempenho de alunos

App em **Streamlit** que usa **lógica fuzzy (inferência de Mamdani)** na base *Student Performance*
(Cortez & Silva, 2008 · UCI / Kaggle, disciplina de português, 649 alunos) para:

- prever a **nota final (G3, 0–20)**;
- apontar **alunos em risco de reprovação** (G3 < 10), com regras que qualquer professor consegue ler.

Dois sistemas são construídos:

| Sistema | Entradas | Uso |
|---|---|---|
| **A · Perfil do aluno** | tempo de estudo, reprovações anteriores, álcool no fim de semana, quer ensino superior | alerta precoce, antes de qualquer prova |
| **B · Com a nota do 2º período** | G2, reprovações anteriores, tempo de estudo | previsão da nota final perto do fim do ano |

## Resultados (validação cruzada de 5 partes para os modelos de comparação)

| Sistema | Modelo | Erro médio (MAE) | AUC (risco de reprovar) |
|---|---|---|---|
| A | **Fuzzy** | 2,35 | **0,80** |
| A | Regressão linear | 2,08 | 0,80 |
| A | Floresta aleatória | 2,08 | 0,79 |
| B | **Fuzzy** | 0,98 | **0,96** |
| B | Regressão linear | 0,81 | 0,96 |
| B | Floresta aleatória | 0,85 | 0,96 |

O fuzzy empata em AUC com os modelos tradicionais e é **totalmente explicável** (cada previsão mostra as regras que a causaram).
O erro médio é um pouco maior porque a saída é agrupada em poucos conjuntos. As regras foram escritas observando esta mesma base,
então os números são um pouco otimistas para dados novos.

## Como rodar

Requer Python 3.10+.

```bash
git clone https://github.com/ooArthur/logica-fuzzy.git
cd logica-fuzzy

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run principal.py         # abre em http://localhost:8501
```

Testes (conferem o motor contra o `scikit-fuzzy` e propriedades básicas):

```bash
python tests_fuzzy.py
```

## Estrutura

```
principal.py            ponto de entrada (menu com as 4 páginas)
pagina1_base.py         a base de dados e a ideia da lógica fuzzy
pagina2_conjuntos.py    funções de pertinência e fuzzificação interativa
pagina3_regras.py       regras, simulador e defuzzificação (centroide)
pagina4_avaliacao.py    comparação com chute, regressão linear e floresta aleatória
fuzzy_core.py           núcleo: variáveis, conjuntos, regras, motor de Mamdani, sistemas A e B
estilo.py               cores, tema dos gráficos e glossário
tests_fuzzy.py          testes do motor
data/student-por.csv    base (separador vírgula; G1 e G2 vêm entre aspas)
relatorio/              relatório em PDF e scripts que o geram
```

## Como o sistema de Mamdani funciona

1. **Fuzzificação:** cada entrada vira graus de pertinência (0 a 1) em conjuntos triangulares/trapezoidais.
2. **Regras:** `SE estuda muito E nunca reprovou ENTÃO nota é boa`; o **E** usa o mínimo dos graus.
3. **Agregação:** a saída de cada regra é cortada na força da regra e todas são unidas pelo máximo.
4. **Defuzzificação:** o **centroide** da área resultante é a nota prevista.

## Regerar o relatório em PDF

```bash
pip install playwright pygments pillow && playwright install chromium
streamlit run principal.py --server.port 8599 &
python relatorio/capturar.py      # prints das páginas
python relatorio/gerar_pdf.py     # gera relatorio/Atividade_Logica_Fuzzy.pdf
```

## Dados e créditos

P. Cortez e A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* FUBUTEC 2008.
Base: https://archive.ics.uci.edu/dataset/320/student+performance
