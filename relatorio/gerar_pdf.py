"""Gera o relatório em PDF (capa, introdução, explicação do código, código-fonte
comentado, prints do app e resultados).

    python relatorio/capturar.py     # 1) tira os prints (app rodando em :8599)
    python relatorio/gerar_pdf.py    # 2) monta o PDF
"""
import base64
import html
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from playwright.sync_api import sync_playwright
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from fuzzy_core import SISTEMAS, carregar_dados, metricas  # noqa: E402

CAP = RAIZ / "relatorio" / "capturas"
SAIDA = RAIZ / "relatorio" / "Atividade_Logica_Fuzzy.pdf"
AUTORES = "Arthur Almeida e Nicolas Soares"
DISCIPLINA = "Teoria do Aprendizado Estatístico · Professor Rômulo"
DATA = "11 de novembro de 2026"


# ----------------------------------------------------------------------------
# números do relatório (calculados na hora, nada digitado à mão)
# ----------------------------------------------------------------------------
def numeros():
    df = carregar_dados()
    cv = KFold(5, shuffle=True, random_state=42)
    feats = {"A": ["studytime", "failures", "Walc", "higher_n"], "B": ["G2", "failures", "studytime"]}
    out = {}
    for k, f in SISTEMAS.items():
        sis = f()
        prev = {
            "Fuzzy": sis.prever(df),
            "Chute (média)": np.full(len(df), df.G3.mean()),
            "Regressão linear": cross_val_predict(LinearRegression(), df[feats[k]], df.G3, cv=cv),
            "Floresta aleatória": cross_val_predict(
                RandomForestRegressor(300, min_samples_leaf=5, random_state=0), df[feats[k]], df.G3, cv=cv
            ),
        }
        linhas = []
        for nome, y in prev.items():
            m = metricas(df.G3, y)
            auc = 0.5 if nome.startswith("Chute") else roc_auc_score(df.reprovado, -y)
            linhas.append((nome, m["MAE"], m["R²"], auc, m["Acerto aprovado/reprovado"]))
        ordem = np.argsort(prev["Fuzzy"], kind="stable")
        alvo = df.reprovado.to_numpy()[ordem]
        alerta = {fr: int(alvo[: int(round(len(df) * fr))].sum()) for fr in (0.1, 0.2, 0.3)}
        out[k] = {"nome": sis.nome, "linhas": linhas, "alerta": alerta, "regras": sis.regras}
    out["df"] = df
    return out


# ----------------------------------------------------------------------------
# helpers de HTML
# ----------------------------------------------------------------------------
def img64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def cortar(caminho: Path, alvo=1250, minimo=500):
    """Divide um print alto em pedaços, cortando só em faixas brancas (sem partir gráficos)."""
    im = Image.open(caminho).convert("RGB")
    a = np.array(im.convert("L"))
    vazia = (a < 250).sum(axis=1) <= 2
    pedacos, ini = [], 0
    while im.height - ini > alvo * 1.15:
        corte = None
        for y in range(ini + alvo, ini + minimo, -1):          # procura para trás uma linha em branco
            if vazia[y : y + 12].all():
                corte = y + 6
                break
        corte = corte or ini + alvo
        pedacos.append(im.crop((0, ini, im.width, corte)))
        ini = corte
    pedacos.append(im.crop((0, ini, im.width, im.height)))
    return pedacos


def figura(img, legenda):
    return f'<figure><img src="{img64(img)}"><figcaption>{legenda}</figcaption></figure>'


def codigo(arquivo: str, titulo: str, descricao: str) -> str:
    texto = (RAIZ / arquivo).read_text(encoding="utf-8")
    fmt = HtmlFormatter(style="friendly", noclasses=True, nowrap=False)
    corpo = highlight(texto, PythonLexer(), fmt)
    return (
        f'<h3 class="code-h">{html.escape(titulo)} <code>{arquivo}</code></h3>'
        f"<p>{descricao}</p>{corpo}"
    )


def tabela_modelos(linhas):
    cab = "<tr><th>Modelo</th><th>Erro médio (MAE)</th><th>R²</th><th>AUC (risco)</th><th>Acerto aprov./reprov.</th></tr>"
    corpo = ""
    for nome, mae, r2, auc, ac in linhas:
        destaque = ' class="dest"' if nome == "Fuzzy" else ""
        corpo += f"<tr{destaque}><td>{nome}</td><td>{mae:.2f}</td><td>{r2:.2f}</td><td>{auc:.2f}</td><td>{ac:.0%}</td></tr>"
    return f"<table>{cab}{corpo}</table>"


def tabela_regras(regras):
    corpo = "".join(f"<tr><td>{i}</td><td>{html.escape(r.descricao())}</td><td>{html.escape(r.texto)}</td></tr>" for i, r in enumerate(regras, 1))
    return f"<table class='regras'><tr><th>#</th><th>Regra</th><th>Por quê</th></tr>{corpo}</table>"


# ----------------------------------------------------------------------------
# documento
# ----------------------------------------------------------------------------
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif; font-size: 10.2pt; line-height: 1.5; color: #1F2937; }
h1 { font-size: 20pt; color: #1E3A8A; margin: 0 0 6pt; }
h2 { font-size: 14.5pt; color: #1E3A8A; border-bottom: 2px solid #2563EB; padding-bottom: 3pt; margin: 18pt 0 8pt; break-after: avoid; }
h3 { font-size: 11.5pt; color: #1F2937; margin: 12pt 0 4pt; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 8pt 0 3pt; break-after: avoid; }
p { margin: 4pt 0 6pt; } li { margin: 2pt 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; background: #F3F4F6; padding: 0 3pt; border-radius: 3px; }
.capa { height: 255mm; display: flex; flex-direction: column; justify-content: center; align-items: flex-start; padding: 0 6mm; break-after: page;
        border-left: 10px solid #2563EB; }
.capa .disc { font-size: 11pt; color: #6B7280; letter-spacing: .5px; text-transform: uppercase; margin-bottom: 14pt; }
.capa h1 { font-size: 30pt; line-height: 1.15; margin-bottom: 10pt; }
.capa .sub { font-size: 13pt; color: #374151; margin-bottom: 40pt; max-width: 135mm; }
.capa .autores { font-size: 12pt; font-weight: bold; } .capa .data { color: #6B7280; margin-top: 4pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; break-inside: avoid; }
th { background: #1E3A8A; color: #fff; text-align: left; padding: 4pt 6pt; }
td { padding: 4pt 6pt; border-bottom: 1px solid #E5E7EB; vertical-align: top; }
tr:nth-child(even) td { background: #F5F7FA; } tr.dest td { background: #DBEAFE; font-weight: bold; }
table.regras td:first-child { width: 14pt; }
figure { margin: 6pt 0 10pt; break-inside: avoid; text-align: center; }
figure img { width: 100%; border: 1px solid #E5E7EB; }
figcaption { font-size: 8.5pt; color: #6B7280; margin-top: 3pt; }
.nota { background: #EFF6FF; border-left: 4px solid #2563EB; padding: 6pt 10pt; margin: 8pt 0; break-inside: avoid; }
.passos { counter-reset: p; list-style: none; padding: 0; }
.passos li { counter-increment: p; margin: 5pt 0; padding-left: 22pt; position: relative; }
.passos li::before { content: counter(p); position: absolute; left: 0; top: 0; width: 15pt; height: 15pt; border-radius: 50%;
                     background: #2563EB; color: #fff; font-weight: bold; text-align: center; font-size: 8.5pt; line-height: 15pt; }
.code-h { margin-top: 14pt; } .code-h code { font-size: 8.5pt; }
div.highlight { border: 1px solid #E5E7EB; border-radius: 4px; background: #FAFAFA !important; padding: 6pt 8pt; margin: 4pt 0 10pt; }
div.highlight pre { margin: 0; font-family: 'DejaVu Sans Mono', monospace; font-size: 6.9pt !important; line-height: 1.32 !important;
                    white-space: pre-wrap; word-break: break-word; background: transparent !important; }
.pb { break-after: page; }
.dois { display: flex; gap: 10pt; } .dois > div { flex: 1; }
"""


def montar(n):
    df = n["df"]
    A, B = n["A"], n["B"]
    fa = {l[0]: l for l in A["linhas"]}
    fb = {l[0]: l for l in B["linhas"]}
    reprov = int(df.reprovado.sum())

    h = [f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"]

    # ---------------- Capa
    h.append(
        f"""<div class="capa">
<div class="disc">{DISCIPLINA}</div>
<h1>Lógica Fuzzy aplicada ao desempenho de alunos</h1>
<div class="sub">Sistema de inferência de Mamdani para prever a nota final e apontar alunos em risco de reprovação,
com app interativo em Streamlit.</div>
<div class="autores">{AUTORES}</div>
<div class="data">{DATA}</div>
</div>"""
    )

    # ---------------- 1 Introdução
    h.append(
        f"""<h2>1. Introdução</h2>
<p><b>Objetivo.</b> Aplicar a <b>lógica fuzzy (nebulosa)</b> à base <i>Student Performance</i> (Cortez &amp; Silva, 2008, UCI/Kaggle),
com {len(df)} alunos de língua portuguesa de duas escolas de Portugal, para <b>prever a nota final (G3, de 0 a 20)</b> e
<b>identificar alunos em risco de reprovação</b> (G3 &lt; 10), de forma que qualquer professor consiga entender o raciocínio.</p>
<p><b>O que é lógica fuzzy.</b> Na lógica clássica um valor está dentro ou fora de uma categoria. Na fuzzy, ele tem um <b>grau de pertinência</b> de 0 a 1.
Quem estuda "entre 2 e 5 horas por semana" não é simplesmente <i>pouco</i> ou <i>muito</i>: é 37% <i>pouco</i> e 63% <i>médio</i>, por exemplo.
Com isso conseguimos escrever <b>regras em linguagem comum</b> (<i>SE estuda muito E nunca reprovou ENTÃO a nota tende a ser boa</i>) e obter uma
resposta numérica suave, sem saltos bruscos entre categorias.</p>
<p><b>A base.</b> São {len(df.columns) - 2} colunas originais (hábitos, família, escola e as notas G1, G2 e G3), sem valores ausentes.
{reprov} alunos ({reprov / len(df):.0%}) reprovaram (G3 &lt; 10) e a nota média é {df.G3.mean():.1f}. O que mais separa as notas:</p>
<ul>
<li><b>Reprovações anteriores:</b> média {df[df.failures == 0].G3.mean():.1f} para quem nunca reprovou contra {df[df.failures == 1].G3.mean():.1f} para quem reprovou uma vez.</li>
<li><b>Ambição:</b> quem quer cursar ensino superior tem média {df[df.higher == 'yes'].G3.mean():.1f}, contra {df[df.higher == 'no'].G3.mean():.1f} de quem não quer.</li>
<li><b>Tempo de estudo e álcool no fim de semana:</b> mais estudo e menos álcool andam com notas maiores.</li>
<li><b>Notas parciais:</b> G2 tem correlação de {df[['G2', 'G3']].corr().iloc[0, 1]:.2f} com G3; sozinha, ela quase resolve o problema.</li>
</ul>
<div class="nota"><b>Dois sistemas.</b> Para mostrar os dois cenários, construímos o <b>Sistema A (perfil do aluno, sem notas parciais)</b>,
um alerta precoce que funciona antes de qualquer prova, e o <b>Sistema B (com a nota do 2º período, G2)</b>, que representa o que o professor já sabe perto do fim do ano.</div>"""
    )

    # ---------------- 2 Explicação do código
    h.append(
        """<h2>2. Explicação do código</h2>
<p>O projeto tem um <b>núcleo independente</b> (<code>fuzzy_core.py</code>), onde está toda a lógica fuzzy, e um <b>app Streamlit</b> com uma página por etapa.
As páginas só chamam o núcleo; assim, o motor pode ser testado sem abrir a interface.</p>
<table>
<tr><th>Arquivo</th><th>O que faz</th></tr>
<tr><td><code>fuzzy_core.py</code></td><td>Define variáveis, conjuntos fuzzy, regras e o motor de Mamdani; contém os sistemas A e B e as métricas de avaliação.</td></tr>
<tr><td><code>principal.py</code></td><td>Ponto de entrada do app; monta o menu com as 4 páginas.</td></tr>
<tr><td><code>pagina1_base.py</code></td><td>Apresenta a base, explica a ideia e mostra os gráficos que justificam as regras.</td></tr>
<tr><td><code>pagina2_conjuntos.py</code></td><td>Mostra as funções de pertinência de cada variável e a fuzzificação de um valor escolhido.</td></tr>
<tr><td><code>pagina3_regras.py</code></td><td>Simulador: ajusta o perfil de um aluno, mostra regras disparadas, área combinada e nota (centroide).</td></tr>
<tr><td><code>pagina4_avaliacao.py</code></td><td>Compara o fuzzy com chute, regressão linear e floresta aleatória (MAE, R², AUC, curva ROC, alerta precoce).</td></tr>
<tr><td><code>tests_fuzzy.py</code></td><td>Confere o motor contra o <code>scikit-fuzzy</code> e valida propriedades básicas.</td></tr>
<tr><td><code>estilo.py</code></td><td>Cores, tema dos gráficos e glossário compartilhados.</td></tr>
</table>
<h3>Como o sistema de Mamdani funciona (4 passos)</h3>
<ol class="passos">
<li><b>Fuzzificação.</b> Cada entrada vira graus de pertinência por meio de triângulos e trapézios (<code>Variavel.fuzzificar</code>). Ex.: tempo de estudo 2 → <i>pouco</i> 0,37 e <i>médio</i> 0,63.</li>
<li><b>Avaliação das regras.</b> Em cada regra <i>SE … E … ENTÃO …</i>, o <b>E</b> usa o <b>mínimo</b> dos graus; o resultado é a <b>força</b> da regra (<code>SistemaFuzzy.forcas</code>).</li>
<li><b>Agregação.</b> O conjunto de saída de cada regra é <b>cortado</b> na altura da sua força e todos os cortes são unidos pelo <b>máximo</b> (<code>SistemaFuzzy.agregar</code>).</li>
<li><b>Defuzzificação.</b> O <b>centroide</b> (centro de gravidade) da área resultante é a nota prevista (<code>SistemaFuzzy.prever</code>).</li>
</ol>
<h3>Decisões de projeto</h3>
<ul>
<li><b>Motor próprio e vetorizado.</b> Em vez de simular aluno por aluno, o NumPy calcula os 649 de uma vez; o <code>scikit-fuzzy</code> fornece as funções de pertinência e serve de referência nos testes.</li>
<li><b>Variáveis escolhidas pela análise dos dados</b> (página 1): reprovações, ambição de cursar ensino superior, tempo de estudo e álcool no fim de semana para o sistema A; G2, reprovações e estudo para o B. As <b>faltas ficaram de fora</b>, pois quase não explicam a nota nesta base.</li>
<li><b>Saída em conjuntos centrados onde as notas reais ficam.</b> No sistema A são 4 conjuntos (risco ≈ 8, regular ≈ 11,5, boa ≈ 14,5, ótima ≈ 17,5); no B, 5 conjuntos (de 6 a 18), porque G2 é muito informativa e justifica mais resolução.</li>
<li><b>Regras escritas à mão, sem treino automático</b>, cada uma com uma justificativa em português (tabelas abaixo).</li>
<li><b>Testes.</b> O motor difere do <code>scikit-fuzzy</code> em no máximo 0,014 ponto de nota; as pertinências ficam entre 0 e 1, as notas entre 0 e 20, nenhum aluno fica sem regra ativa, e a nota do sistema B cresce com G2.</li>
</ul>
<h4>Regras do Sistema A (perfil do aluno)</h4>"""
        + tabela_regras(A["regras"])
        + "<h4>Regras do Sistema B (com nota G2)</h4>"
        + tabela_regras(B["regras"])
    )

    # ---------------- 3 Código-fonte
    h.append('<div class="pb"></div><h2>3. Código-fonte comentado</h2>')
    h.append(codigo("fuzzy_core.py", "Núcleo fuzzy", "Variáveis, conjuntos, regras, motor de Mamdani, sistemas A e B e métricas."))
    h.append(codigo("principal.py", "Entrada do app", "Registra as quatro páginas na navegação do Streamlit."))
    h.append(codigo("estilo.py", "Estilo e glossário", "Paleta, tema dos gráficos e glossário usado nas páginas."))
    h.append(codigo("pagina1_base.py", "Página 1 · A base de dados", "Distribuição das notas e gráficos de médias por grupo."))
    h.append(codigo("pagina2_conjuntos.py", "Página 2 · Conjuntos fuzzy", "Funções de pertinência e fuzzificação interativa."))
    h.append(codigo("pagina3_regras.py", "Página 3 · Regras e simulador", "Regras disparadas, agregação e centroide de um aluno."))
    h.append(codigo("pagina4_avaliacao.py", "Página 4 · Resultados", "Comparação com chute, regressão linear e floresta aleatória."))
    h.append(codigo("tests_fuzzy.py", "Testes", "Conferência do motor contra o scikit-fuzzy e propriedades básicas."))

    # ---------------- 4 Prints
    h.append('<div class="pb"></div><h2>4. Aplicação rodando (prints)</h2>')
    h.append("<p>Capturas do app Streamlit executando <code>streamlit run principal.py</code>.</p>")
    for nome, titulo in [
        ("p1_base", "Página 1 · A base de dados"),
        ("p2_conjuntos", "Página 2 · Conjuntos fuzzy"),
    ]:
        h.append(f"<h3>{titulo}</h3>")
        for i, parte in enumerate(cortar(CAP / f"{nome}.png"), 1):
            h.append(figura(parte, f"{titulo} (parte {i})"))
    h.append("<h3>Página 3 · Regras e simulador</h3>")
    h.append(figura(Image.open(CAP / "p3_sistemaA.png"), "Sistema A: perfil com estudo 2, sem reprovações, álcool 2, quer ensino superior. As barras mostram a força de cada regra."))
    h.append(figura(Image.open(CAP / "p3_sistemaB.png"), "Sistema B: com G2 = 11, duas regras disparam (nota baixa e nota média) e o centroide cai entre elas."))
    h.append("<h3>Página 4 · Resultados</h3>")
    for i, parte in enumerate(cortar(CAP / "p4_avaliacao.png"), 1):
        h.append(figura(parte, f"Página 4 · Resultados (parte {i})"))

    # ---------------- 5 Resultados
    a_f, a_l = fa["Fuzzy"], fa["Regressão linear"]
    b_f, b_l = fb["Fuzzy"], fb["Regressão linear"]
    h.append(
        f"""<h2>5. Resultados e conclusões</h2>
<p>Comparamos cada sistema com um chute (prever sempre a média), uma regressão linear e uma floresta aleatória (as duas com validação cruzada de 5 partes, sobre as mesmas variáveis).
<b>AUC</b> mede a capacidade de colocar os alunos que reprovaram no topo da lista de risco (0,5 = chute; 1 = perfeito).</p>
<h4>Sistema A · perfil do aluno (sem notas parciais)</h4>{tabela_modelos(A['linhas'])}
<h4>Sistema B · com a nota do 2º período</h4>{tabela_modelos(B['linhas'])}
<h3>Leitura dos resultados</h3>
<ul>
<li><b>Sem nota parcial, prever o valor exato é difícil para qualquer modelo:</b> o erro de todos fica perto de 2 pontos, pouco abaixo do chute ({fa['Chute (média)'][1]:.2f}).
Como <b>alerta de risco</b>, porém, o fuzzy se sai bem: AUC de <b>{a_f[3]:.2f}</b>, igual à regressão linear ({a_l[3]:.2f}), e cada alerta vem com as regras que o causaram.
Note que o "acerto aprovado/reprovado" do sistema A ({a_f[4]:.0%}) fica abaixo do chute ({fa['Chute (média)'][4]:.0%}): como só {reprov / len(df):.0%} dos alunos reprovam, prever "aprovado" para todos já acerta muito; o valor do fuzzy está em <b>ordenar o risco</b> (AUC), não em fixar um corte único.</li>
<li><b>Alerta precoce (sistema A):</b> acompanhando só os 20% de alunos com menor nota prevista, a escola alcança <b>{A['alerta'][0.2]} dos {reprov}</b> reprovados;
com 30%, <b>{A['alerta'][0.3]}</b>.</li>
<li><b>Com G2 (sistema B)</b> o fuzzy erra em média <b>{b_f[1]:.2f} ponto</b>, com AUC de <b>{b_f[3]:.2f}</b> e {b_f[4]:.0%} de acerto aprovado/reprovado. A regressão linear é um pouco mais precisa ({b_l[1]:.2f} de erro),
pois a nota final é quase uma reta da nota do 2º período; o fuzzy fica perto e entrega o raciocínio.</li>
<li><b>Vantagem do fuzzy:</b> transparência. Qualquer previsão pode ser conferida regra a regra (página 3) e as regras podem ser corrigidas por um professor sem saber programar.</li>
</ul>
<div class="nota"><b>Limitações.</b> Os conjuntos e regras foram definidos observando esta própria base, então os números são um pouco otimistas para dados novos.
O fuzzy ordena bem o risco, mas calibra pior a nota fina que uma regressão (saída agrupada em poucos conjuntos). A base inclui {int((df.G3 == 0).sum())} alunos com nota zero (provável desistência), que nenhum modelo prevê bem.
Como próximos passos: validar com outra turma e ajustar os conjuntos automaticamente (por exemplo, ANFIS).</div>
<h3>Como reproduzir</h3>
<p><code>pip install -r requirements.txt</code> · <code>streamlit run principal.py</code> · <code>python tests_fuzzy.py</code>.
Repositório: <code>github.com/ooArthur/logica-fuzzy</code>. Dados: <i>Student Performance</i>, UCI Machine Learning Repository (P. Cortez e A. Silva, 2008).</p>
</body></html>"""
    )
    return "".join(h)


def main():
    n = numeros()
    doc = montar(n)
    (RAIZ / "relatorio" / "_relatorio.html").write_text(doc, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.set_content(doc, wait_until="load")
        pg.pdf(
            path=str(SAIDA), format="A4", print_background=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=(
                "<div style=\"font-size:8px;color:#6B7280;width:100%;padding:0 16mm;display:flex;justify-content:space-between\">"
                f"<span>Lógica Fuzzy · {AUTORES}</span><span>página <span class='pageNumber'></span> de <span class='totalPages'></span></span></div>"
            ),
            margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
        )
        b.close()
    print("PDF:", SAIDA)


if __name__ == "__main__":
    main()
