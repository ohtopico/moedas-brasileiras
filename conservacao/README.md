# Fase 2: estado de conservação (FC, SOB, MBC, BC…)

Continuação do projeto: montar um dataset de fotos de moedas rotuladas com o
estado de conservação, o ano, o valor de face e o preço de leilão, e treinar um
classificador de conservação. As fotos vêm das buscas públicas do LeilõesBR,
cujos títulos já trazem o grau dado pelo leiloeiro (os rótulos, portanto, têm o
ruído de avaliações feitas por vendedores). As fotos baixadas são dos
leiloeiros: uso pessoal, sem redistribuir — por isso este repositório guarda só
o código, não as imagens.

## Como rodar no Colab

O jeito mais fácil é abrir `coleta_moedas_colab.ipynb` no Colab (Arquivo > Fazer upload de notebook) e rodar as células em ordem. O notebook já contém o script. Para rodar o script direto:

```python
from google.colab import drive
drive.mount('/content/drive')
!pip -q install requests beautifulsoup4
# copie coletar_leiloesbr.py para o Drive ou faça upload na sessão

# 1) Só o CSV, para conferir se os títulos estão sendo lidos certo
!python coletar_leiloesbr.py --saida /content/drive/MyDrive/moedas/leiloesbr --sem-imagens --max-paginas 1

# 2) Coleta completa (apague o metadados.csv do passo 1 antes)
!python coletar_leiloesbr.py --saida /content/drive/MyDrive/moedas/leiloesbr
```

## Saída

- `imagens/<GRAU>/<leilao>_<lote>.jpg`: pronto para `ImageFolder`.
- `metadados.csv`: todos os lotes vistos, com `status` (`ok`, `sem_grau`,
  `grau_intermediario`, `varias_pecas`, `estrangeira`, `fora_do_real`, `erro_imagem`).

## Opções úteis

| Opção | O que faz |
|---|---|
| `--escopo real` | só família do Real, 1994+ (padrão); `brasil` = qualquer moeda brasileira; `todos` = sem filtro |
| `--intermediario inferior` | "MBC/SOB" vira MBC (padrão); `descartar` joga fora |
| `--intermediario manter` | cria classes como `SOB_MBC` |
| `--consultas "moeda 1 real FC" ...` | troca os termos de busca |
| `--atraso 3` | mais devagar com o site |

## Treino

Depois da coleta, abra `treino_conservacao_colab.ipynb` no Colab (com GPU) e rode as
células. Ele treina uma MobileNetV3-Large para classificar o grau de conservação,
salva o modelo em `DADOS/treino/` e mostra a matriz de confusão e os erros.
Comece com `AGRUPAR = True` (FC / SOB / MBC / BC ou pior; só junta as classes raras do fim da escala).

## Cuidados

- A busca só mostra leilões em andamento. Rode de novo a cada uma ou duas semanas:
  os lotes que já estão no CSV não são baixados de novo.
- A foto principal pode mostrar só um lado, os dois lados juntos ou a moeda num
  slab. Vale revisar as pastas e apagar o que não serve antes de treinar.
- O grau vem do leiloeiro, então os rótulos têm ruído. Juntar graus vizinhos
  (FC+SOB, MBC, BC ou pior) costuma ajudar no começo.
- As fotos são dos leiloeiros: use só para o projeto pessoal, sem redistribuir.
