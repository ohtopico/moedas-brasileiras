"""Coleta fotos de moedas com grau de conservação a partir das buscas do LeilõesBR.

Os títulos dos lotes costumam trazer o grau (FC, SOB, MBC, BC...), o ano e o valor
de face, por exemplo "Moeda de 1 Real (JK) ano 2002 no estado MBC." O script busca
cada termo, lê os títulos, baixa a foto de cada lote de moeda única e salva:

    saida/imagens/<GRAU>/<leilao>_<lote>.jpg
    saida/metadados.csv   (todos os lotes vistos, inclusive os descartados)

A busca só mostra leilões em andamento. Rodar de novo outro dia acrescenta lotes
novos sem baixar de novo os que já estão no CSV.

Uso (Colab ou PC):
    python coletar_leiloesbr.py --saida /content/drive/MyDrive/moedas/leiloesbr
    python coletar_leiloesbr.py --escopo brasil --intermediario descartar
"""
import argparse
import csv
import random
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

BASE = "https://www.leiloesbr.com.br"
BUSCA = BASE + "/busca_andamento.asp?pesquisa={q}&op=2&v=126&tp=|&b=0&pag={pag}"
IMG = "https://d1o6h00a1h5k7q.cloudfront.net/imagens/{tam}/{leilao}/{lote}.jpg"

# O site ignora termos de 2 letras ("FC", "BC"), então os graus vão por extenso.
CONSULTAS_PADRAO = [
    "moeda real flor de cunho", "moeda real FDC", "moeda real soberba",
    "moeda real MBC", "moeda real bem conservada", "moeda real muito bem conservada",
    "moeda centavos flor de cunho", "moeda centavos FDC", "moeda centavos soberba",
    "moeda centavos MBC", "moeda centavos bem conservada",
    "moeda 1 real", "moeda 50 centavos", "moeda 25 centavos",
    "moeda 10 centavos", "moeda 5 centavos", "moeda 1 centavo",
    "moeda real NGC", "moeda real PCGS", "moeda real circulada",
]

# Escala brasileira, do melhor para o pior.
ESCALA = ["FC", "SOB", "MBC", "BC", "R", "UTGEG"]
SINONIMOS = {
    "FDC": "FC", "FC": "FC", "FLOR DE CUNHO": "FC",
    "SOBERBA": "SOB", "SOB": "SOB", "S": "SOB",
    "MBC": "MBC", "BC": "BC",
    "MUITO BEM CONSERVADA": "MBC", "BEM CONSERVADA": "BC",
    "REGULAR": "R", "UTGEG": "UTGEG", "UTG": "UTGEG",
}
_TOKEN = (r"(?:FLOR DE CUNHO|FDC|FC|SOBERBA|SOB|MUITO BEM CONSERVADA|MBC|"
          r"BEM CONSERVADA|BC|UTGEG|UTG|REGULAR)")
# Um grau, opcionalmente seguido de "/ outro grau" ("MBC/S", "Mbc / Soberba").
RE_GRAU = re.compile(
    rf"(?<![A-Z0-9]){_TOKEN}(?:\s*/\s*(?:{_TOKEN}|S))*(?![A-Z0-9])"
)
# Notas de certificadoras (NGC, PCGS) na escala Sheldon, convertidas para a brasileira.
RE_SHELDON = re.compile(r"(?<![A-Z])(MS|PF|PR|SP|AU|XF|EF|VF|VG|F|G)\s*-?\s*(\d{1,2})(?!\d)")
RE_ANO = re.compile(r"(?<!\d)(1[5-9]\d\d|20[0-4]\d)(?!\d)")
RE_VALOR = re.compile(
    r"(?<![\w$])(\d{1,5}(?:[.,]\d{1,3})?|[IVXLC]+)\s*"
    r"(centavos?|cents?|reais|real|r[eé]is|cruzeiros?(?: reais| novos)?|"
    r"cruzados?(?: novos)?|NCr\$|Cr\$|Cz\$|CR\$)",
    re.IGNORECASE,
)
RE_PRECO = re.compile(r"R\$\s*([\d.]+,\d{2})")
RE_LOTE = re.compile(
    r"\blote\b|\bmoedas\b|\btrio\b|\bpar\b|cole[cç][aã]o|[aá]lbum|c[eé]dula|"
    r"medalha|\bset\b|\bkit\b|s[eé]rie completa|garimpo|\bkg\b",
    re.IGNORECASE,
)
# Países e moedas estrangeiras em qualquer parte do título.
PAISES_FORA = re.compile(
    r"\b(portugal|[aá]ustria|alemanha|fran[cç]a|espanha|it[aá]lia|inglaterra|"
    r"estados unidos|eua|argentina|uruguai|paraguai|chile|m[eé]xico|camar[oõ]es|"
    r"jap[aã]o|china|r[uú]ssia|angola|mo[cç]ambique|canad[aá]|peru|bol[ií]via|"
    r"col[oô]mbia|venezuela|su[ií][cç]a|b[eé]lgica|holanda|gr[eé]cia|gana|ghana|"
    r"indon[eé]sia|ucr[aâ]nia|[ií]ndia|turquia|egito|[aá]frica|austr[aá]lia|"
    r"nova zel[aâ]ndia|cuba|equador|israel|cor[eé]ia|tail[aâ]ndia|filipinas|"
    r"vaticano|m[oô]naco|pol[oô]nia|hungria|su[eé]cia|noruega|dinamarca|finl[aâ]ndia|"
    r"irlanda|esc[oó]cia|reino unido|gr[aã][- ]bretanha|tch[eé]c\w*|iugosl\w*|uni[aã]o sovi\w*|"
    r"dollars?|d[oó]lar(es)?|euros?|yuan|pesos?|cedis?|r[uú]pias?|hryvnias?|"
    r"kreuzer|francos?|libras?|marcos?|liras?|ienes?|yen|rublos?|escudos?|"
    r"shillings?|pence|centimos?|c[eé]ntimos|one cent|cents?)\b",
    re.IGNORECASE,
)
FAMILIA_REAL = re.compile(r"\b(centavos?|reais|real)\b", re.IGNORECASE)
ANTES_DO_REAL = re.compile(r"r[eé]is|cruzeiro|cruzado|\bcr\$|\bncr\$|\bcz\$|"
                           r"\burv\b|vint[eé]ns?\b|patacas?\b|tost[aã]o|\bdobras?\b",
                           re.IGNORECASE)
BRASIL = re.compile(r"\bbra[sz]il", re.IGNORECASE)

CAMPOS = ["leilao_id", "lote_id", "titulo", "grau_bruto", "grau", "ano",
          "valor_face", "preco_rs", "status", "arquivo", "url_lote",
          "url_imagem", "consulta"]


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def extrair_grau(titulo, intermediario="descartar"):
    """Retorna (grau_bruto, grau). grau é None se não houver ou se for descartado."""
    t = sem_acento(titulo).upper()
    m = RE_GRAU.search(t)
    if not m:
        return extrair_sheldon(t)
    bruto = re.sub(r"\s*/\s*", "/", m.group(0).strip())
    graus = []
    for parte in bruto.split("/"):
        g = SINONIMOS.get(parte)
        if g and g not in graus:
            graus.append(g)
    if not graus:
        return bruto, None
    if len(graus) == 1:
        return bruto, graus[0]
    graus.sort(key=ESCALA.index)
    if intermediario == "inferior":
        return bruto, graus[-1]
    if intermediario == "superior":
        return bruto, graus[0]
    if intermediario == "manter":
        return bruto, "_".join(graus)
    return bruto, None


def extrair_sheldon(t):
    """MS 68 -> FC, AU 55 -> SOB, VF 30 -> MBC... Só com número, para evitar falsos positivos."""
    certificada = re.search(r"NGC|PCGS|ANACS|CERTIFICAD|SLAB", t)
    for m in RE_SHELDON.finditer(t):
        pref, n = m.group(1), int(m.group(2))
        if not 1 <= n <= 70:
            continue
        if len(pref) == 1 and not certificada:  # "6,92 g 28 mm" não é G28
            continue
        if pref in ("MS", "PF", "PR", "SP") and n >= 60:
            g = "FC"
        elif pref in ("AU", "XF", "EF") or (pref == "MS" and n >= 40):
            g = "SOB"
        elif pref == "VF":
            g = "MBC"
        elif pref == "F":
            g = "BC"
        elif pref in ("VG", "G"):
            g = "R"
        else:
            continue
        return f"{pref}{n}", g
    return "", None


def extrair_ano(titulo):
    m = re.search(r"\bano\s+(\d{4})", titulo, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Sem "ano", pula números seguidos de unidade ("2000 Réis 1867" -> 1867).
    for m in RE_ANO.finditer(titulo):
        if not RE_VALOR.match(titulo, m.start()):
            return int(m.group(1))
    return None


def extrair_valor(titulo):
    m = RE_VALOR.search(titulo)
    return f"{m.group(1)} {m.group(2).lower()}" if m else ""


def classificar_titulo(titulo, escopo="real"):
    """Retorna o motivo de descarte, ou '' se o lote serve.

    escopo: "real" (família do Real, 1994+), "brasil" (qualquer moeda brasileira)
    ou "todos" (só tira lotes com várias peças).
    """
    if RE_LOTE.search(titulo):
        return "varias_pecas"
    if escopo == "todos":
        return ""
    if PAISES_FORA.search(titulo):
        return "estrangeira"
    eh_real = bool(FAMILIA_REAL.search(titulo)) and not ANTES_DO_REAL.search(titulo)
    if escopo == "brasil":
        return "" if BRASIL.search(titulo) or eh_real or ANTES_DO_REAL.search(titulo) \
            else "pais_incerto"
    ano = extrair_ano(titulo)
    if not eh_real or (ano and ano < 1994):
        return "fora_do_real"
    return ""


def parse_busca(html):
    """Extrai os lotes de uma página de busca: lista de dicts sem duplicatas."""
    soup = BeautifulSoup(html, "html.parser")
    lotes = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "abre_catalogo.asp" not in href:
            continue
        partes = href.split("|")
        if len(partes) < 4 or not partes[-1].strip().isdigit():
            continue
        leilao, lote = partes[-2].strip(), partes[-1].strip()
        titulo = (a.get("title") or a.get_text(" ", strip=True)).strip()
        atual = lotes.get(lote)
        # O mesmo lote aparece em vários links (foto, título); fica o título mais longo.
        if atual and len(atual["titulo"]) >= len(titulo):
            continue
        preco = ""
        bloco = a
        for _ in range(6):
            bloco = bloco.parent
            if bloco is None:
                break
            m = RE_PRECO.search(bloco.get_text(" ", strip=True))
            if m:
                preco = m.group(1).replace(".", "").replace(",", ".")
                break
        url = href if href.startswith("http") else BASE + "/" + href.lstrip("/")
        lotes[lote] = {"leilao_id": leilao, "lote_id": lote, "titulo": titulo,
                       "preco_rs": preco or (atual or {}).get("preco_rs", ""),
                       "url_lote": url}
    return [l for l in lotes.values() if l["titulo"]]


def baixar(sessao, url, tentativas=3):
    for i in range(tentativas):
        try:
            r = sessao.get(url, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** (i + 1))
    return None


def baixar_imagem(sessao, leilao, lote, destino):
    for tam in ("img_g", "img_m", "img_p"):  # img_g é a grande, da página do lote
        url = IMG.format(tam=tam, leilao=leilao, lote=lote)
        r = baixar(sessao, url)
        if r is not None and r.headers.get("content-type", "").startswith("image"):
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(r.content)
            return url
    return ""


def atualizar_imagens(sessao, saida, csv_path, atraso):
    with open(csv_path, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    alvo = [l for l in linhas if l["arquivo"] and "/img_g/" not in l["url_imagem"]]
    print(f"{len(alvo)} fotos para baixar de novo")
    for i, l in enumerate(alvo, 1):
        url = baixar_imagem(sessao, l["leilao_id"], l["lote_id"], saida / l["arquivo"])
        if url:
            l["url_imagem"] = url
        if i % 50 == 0:
            print(f"  {i}/{len(alvo)}")
        time.sleep(atraso)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(linhas)
    print("Pronto. Metadados em", csv_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default="leiloesbr_moedas")
    ap.add_argument("--consultas", nargs="*", default=CONSULTAS_PADRAO)
    ap.add_argument("--max-paginas", type=int, default=10)
    ap.add_argument("--escopo", default="real", choices=["real", "brasil", "todos"],
                    help="real: só família do Real (padrão); brasil: qualquer moeda "
                         "brasileira; todos: sem filtro de país")
    ap.add_argument("--intermediario", default="inferior",
                    choices=["descartar", "inferior", "superior", "manter"],
                    help="o que fazer com graus como MBC/SOB")
    ap.add_argument("--atraso", type=float, default=1.5,
                    help="segundos entre requisições")
    ap.add_argument("--sem-imagens", action="store_true",
                    help="só gera o CSV, para conferir antes de baixar")
    ap.add_argument("--atualizar-imagens", action="store_true",
                    help="baixa de novo, na versão grande, as fotos que já estão no CSV")
    args = ap.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    csv_path = saida / "metadados.csv"
    vistos = set()
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            vistos = {row["lote_id"] for row in csv.DictReader(f)}
    novo = not csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CAMPOS)
    if novo:
        w.writeheader()

    sessao = requests.Session()
    sessao.headers["User-Agent"] = (
        "Mozilla/5.0 (pesquisa pessoal; coleta lenta de fotos de moedas)")
    contagem = {}

    if args.atualizar_imagens:
        atualizar_imagens(sessao, saida, csv_path, args.atraso)
        return

    for consulta in args.consultas:
        ids_anteriores = None
        for pag in range(1, args.max_paginas + 1):
            r = baixar(sessao, BUSCA.format(q=quote_plus(consulta), pag=pag))
            time.sleep(args.atraso + random.random())
            if r is None:
                print(f"[{consulta}] página {pag}: falhou")
                break
            r.encoding = r.apparent_encoding or "latin-1"
            lotes = parse_busca(r.text)
            novos = [l for l in lotes if l["lote_id"] not in vistos]
            print(f"[{consulta}] página {pag}: {len(lotes)} lotes, {len(novos)} novos")
            ids = {l["lote_id"] for l in lotes}
            if not lotes or ids == ids_anteriores:  # acabaram as páginas
                break
            ids_anteriores = ids
            for l in novos:
                vistos.add(l["lote_id"])
                titulo = l["titulo"]
                grau_bruto, grau = extrair_grau(titulo, args.intermediario)
                status = classificar_titulo(titulo, args.escopo)
                if not status and not grau:
                    status = "sem_grau" if not grau_bruto else "grau_intermediario"
                arquivo, url_img = "", ""
                if not status and not args.sem_imagens:
                    destino = saida / "imagens" / grau / f"{l['leilao_id']}_{l['lote_id']}.jpg"
                    url_img = baixar_imagem(sessao, l["leilao_id"], l["lote_id"], destino)
                    time.sleep(args.atraso)
                    if url_img:
                        arquivo = str(destino.relative_to(saida))
                    else:
                        status = "erro_imagem"
                if not status:
                    status = "ok"
                    contagem[grau] = contagem.get(grau, 0) + 1
                w.writerow({**l, "grau_bruto": grau_bruto, "grau": grau or "",
                            "ano": extrair_ano(titulo) or "",
                            "valor_face": extrair_valor(titulo), "status": status,
                            "arquivo": arquivo, "url_imagem": url_img,
                            "consulta": consulta})
            f.flush()
    f.close()
    print("\nNovos lotes aproveitados por grau:", contagem or "nenhum")
    print("Metadados em", csv_path)


if __name__ == "__main__":
    main()
