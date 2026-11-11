"""Tira as capturas das páginas do app (precisa do app rodando em localhost:8599).

    streamlit run principal.py --server.port 8599 &
    python relatorio/capturar.py
Requer:  pip install playwright && playwright install chromium
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://localhost:8599"
SAIDA = Path(__file__).parent / "capturas"
SAIDA.mkdir(exist_ok=True)

# (arquivo, caminho da página no app, opções extras)
PAGINAS = [
    ("p1_base", "", None),
    ("p2_conjuntos", "pagina2_conjuntos", None),
    ("p3_regras", "pagina3_regras", None),
    ("p4_avaliacao", "pagina4_avaliacao", None),
]


def recortar(caminho: Path) -> None:
    """Tira o botão 'Deploy' do topo e o branco que sobra no fim da página."""
    im = Image.open(caminho)
    a = np.array(im.convert("L"))
    linhas = np.where((a < 250).sum(axis=1) > 3)[0]
    im.crop((0, 70, im.width, int(linhas.max()) + 30)).save(caminho)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1300, "height": 6500})
        for nome, caminho, _ in PAGINAS:
            pg.goto(f"{URL}/{caminho}")
            pg.wait_for_selector("h1", timeout=60000)
            pg.wait_for_timeout(6000)
            alvo = pg.locator('[data-testid="stMain"]')
            alvo.screenshot(path=str(SAIDA / f"{nome}.png"))
            recortar(SAIDA / f"{nome}.png")
            print("ok", nome)
        # Página 3 inteira (com a barra lateral de controles), nos dois sistemas
        pg.set_viewport_size({"width": 1400, "height": 1850})
        pg.goto(f"{URL}/pagina3_regras")
        pg.wait_for_selector("h1", timeout=60000)
        pg.wait_for_timeout(5000)
        pg.screenshot(path=str(SAIDA / "p3_sistemaA.png"))
        pg.get_by_text("B · Com a nota do 2º período (G2)").click()
        pg.wait_for_timeout(5000)
        pg.screenshot(path=str(SAIDA / "p3_sistemaB.png"))
        print("ok p3 A/B")
        b.close()


if __name__ == "__main__":
    sys.exit(main())
