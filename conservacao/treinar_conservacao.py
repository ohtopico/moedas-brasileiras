"""Treina um classificador de estado de conservação de moedas (FC, SOB, MBC, BC...).

Espera a estrutura que o coletar_leiloesbr.py gera:

    DADOS/imagens/<GRAU>/<leilao>_<lote>.jpg

Faz fine-tuning de uma MobileNetV3-Large pré-treinada no ImageNet, com split
estratificado treino/validação, balanceamento por reamostragem e, ao final,
relatório com matriz de confusão e acurácia "a um grau de distância" (errar
FC por SOB é quase acerto; FC por BC não é).

Uso:
    python treinar_conservacao.py --dados /content/drive/MyDrive/moedas/leiloesbr
    python treinar_conservacao.py --dados ... --agrupar  # FC / SOB / MBC / BC-ou-pior
"""
import argparse
import copy
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

# Ordem da escala, do melhor para o pior. Classes fora dela vão para o fim.
ESCALA = ["FC", "SOB", "MBC", "BC", "R", "UTGEG"]
# FC e SOB ficam separados: para colecionador, é a distinção que mais muda o preço.
GRUPOS = {"FC": "FC", "SOB": "SOB", "MBC": "MBC",
          "BC": "BC_OU_PIOR", "R": "BC_OU_PIOR", "UTGEG": "BC_OU_PIOR"}
ESCALA_GRUPOS = ["FC", "SOB", "MBC", "BC_OU_PIOR"]


def semear(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Moedas(Dataset):
    def __init__(self, itens, classes, treino, tamanho=224):
        self.itens = itens
        self.idx = {c: i for i, c in enumerate(classes)}
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if treino:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(tamanho, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                # Moeda não tem "lado certo" na foto: rotação ajuda bastante.
                transforms.RandomRotation(180, fill=255),
                transforms.ColorJitter(0.25, 0.25, 0.15),
                transforms.ToTensor(), norm,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(tamanho * 1.15)),
                transforms.CenterCrop(tamanho),
                transforms.ToTensor(), norm,
            ])

    def __len__(self):
        return len(self.itens)

    def __getitem__(self, i):
        caminho, classe = self.itens[i]
        img = Image.open(caminho).convert("RGB")
        return self.tf(img), self.idx[classe]


def listar_itens(pasta_imagens, agrupar, min_por_classe):
    itens = []
    for p in sorted(pasta_imagens.rglob("*")):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            classe = p.parent.name
            if agrupar:
                classe = GRUPOS.get(classe, "BC_OU_PIOR")
            itens.append((p, classe))
    contagem = Counter(c for _, c in itens)
    descartadas = {c for c, n in contagem.items() if n < min_por_classe}
    if descartadas:
        print(f"Classes com menos de {min_por_classe} fotos, fora do treino: "
              f"{sorted(descartadas)}")
        itens = [x for x in itens if x[1] not in descartadas]
    escala = ESCALA_GRUPOS if agrupar else ESCALA
    classes = sorted({c for _, c in itens},
                     key=lambda c: escala.index(c) if c in escala else 99)
    return itens, classes


def dividir(itens, frac_val, seed):
    """Split estratificado por classe."""
    rng = random.Random(seed)
    por_classe = {}
    for item in itens:
        por_classe.setdefault(item[1], []).append(item)
    treino, val = [], []
    for grupo in por_classe.values():
        rng.shuffle(grupo)
        n_val = max(1, round(len(grupo) * frac_val))
        val += grupo[:n_val]
        treino += grupo[n_val:]
    rng.shuffle(treino)
    return treino, val


def criar_modelo(n_classes):
    modelo = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    modelo.classifier[3] = nn.Linear(modelo.classifier[3].in_features, n_classes)
    return modelo


def avaliar(modelo, loader, device, n_classes):
    modelo.eval()
    conf = np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            pred = modelo(x.to(device)).argmax(1).cpu().numpy()
            for yi, pi in zip(y.numpy(), pred):
                conf[yi, pi] += 1
    acc = conf.trace() / conf.sum()
    # Acurácia balanceada: média do recall de cada classe.
    recalls = [conf[i, i] / conf[i].sum() for i in range(n_classes) if conf[i].sum()]
    bal = float(np.mean(recalls))
    # Acurácia a um grau: acerta ou erra para a classe vizinha na escala.
    viz = sum(conf[i, j] for i in range(n_classes) for j in range(n_classes)
              if abs(i - j) <= 1)
    return acc, bal, viz / conf.sum(), conf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True,
                    help="pasta com imagens/<GRAU>/*.jpg (a saída do coletor)")
    ap.add_argument("--saida", default=None,
                    help="onde salvar modelo e relatório (padrão: DADOS/treino)")
    ap.add_argument("--agrupar", action="store_true",
                    help="junta BC/R/UTGeG em BC_OU_PIOR (classes: FC / SOB / MBC / BC ou pior)")
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--epocas-congelado", type=int, default=3,
                    help="épocas iniciais treinando só a cabeça")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tamanho", type=int, default=224)
    ap.add_argument("--frac-val", type=float, default=0.2)
    ap.add_argument("--min-por-classe", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    semear(args.seed)
    dados = Path(args.dados)
    pasta_imagens = dados / "imagens" if (dados / "imagens").is_dir() else dados
    saida = Path(args.saida) if args.saida else dados / "treino"
    saida.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    itens, classes = listar_itens(pasta_imagens, args.agrupar, args.min_por_classe)
    if len(classes) < 2:
        raise SystemExit("Menos de 2 classes com fotos suficientes; colete mais.")
    treino, val = dividir(itens, args.frac_val, args.seed)
    contagem = Counter(c for _, c in treino)
    print(f"Device: {device} | classes: {classes}")
    print(f"Treino: {len(treino)} {dict(contagem)} | validação: {len(val)}")

    ds_treino = Moedas(treino, classes, treino=True, tamanho=args.tamanho)
    ds_val = Moedas(val, classes, treino=False, tamanho=args.tamanho)
    # Reamostragem: classes raras (BC...) aparecem com mais frequência no batch.
    pesos = torch.tensor([1.0 / contagem[c] for _, c in treino])
    sampler = WeightedRandomSampler(pesos, num_samples=len(treino))
    dl_treino = DataLoader(ds_treino, batch_size=args.batch, sampler=sampler,
                           num_workers=args.workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    modelo = criar_modelo(len(classes)).to(device)
    criterio = nn.CrossEntropyLoss(label_smoothing=0.1)
    melhor = {"bal": 0.0, "estado": None, "epoca": 0}

    for epoca in range(1, args.epocas + 1):
        congelado = epoca <= args.epocas_congelado
        for p in modelo.features.parameters():
            p.requires_grad = not congelado
        params = [p for p in modelo.parameters() if p.requires_grad]
        if epoca in (1, args.epocas_congelado + 1):
            otim = torch.optim.AdamW(params, lr=args.lr if congelado else args.lr / 3,
                                     weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                otim, T_max=max(1, args.epocas - epoca + 1))
        modelo.train()
        perda_soma, t0 = 0.0, time.time()
        for x, y in dl_treino:
            x, y = x.to(device), y.to(device)
            otim.zero_grad()
            perda = criterio(modelo(x), y)
            perda.backward()
            otim.step()
            perda_soma += perda.item() * len(y)
        sched.step()
        acc, bal, viz, _ = avaliar(modelo, dl_val, device, len(classes))
        print(f"época {epoca:2d}{' (congelado)' if congelado else ''}: "
              f"perda {perda_soma / len(treino):.3f} | val acc {acc:.3f} | "
              f"balanceada {bal:.3f} | ±1 grau {viz:.3f} | {time.time() - t0:.0f}s")
        if bal > melhor["bal"]:
            melhor = {"bal": bal, "estado": copy.deepcopy(modelo.state_dict()),
                      "epoca": epoca}

    modelo.load_state_dict(melhor["estado"])
    acc, bal, viz, conf = avaliar(modelo, dl_val, device, len(classes))
    print(f"\nMelhor época: {melhor['epoca']} | val acc {acc:.3f} | "
          f"balanceada {bal:.3f} | ±1 grau {viz:.3f}")
    print("Matriz de confusão (linha = verdadeiro, coluna = previsto):")
    larg = max(len(c) for c in classes)
    print(" " * (larg + 1) + " ".join(f"{c:>{larg}}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>{larg}} " + " ".join(f"{n:>{larg}}" for n in conf[i]))

    torch.save({"state_dict": modelo.state_dict(), "classes": classes,
                "tamanho": args.tamanho, "agrupar": args.agrupar},
               saida / "modelo_conservacao.pt")
    with open(saida / "relatorio.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "val_acc": acc, "val_balanceada": bal,
                   "val_um_grau": viz, "confusao": conf.tolist(),
                   "melhor_epoca": melhor["epoca"], "args": vars(args)},
                  f, ensure_ascii=False, indent=2, default=str)
    with open(saida / "predicoes_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "verdadeiro", "previsto"])
        modelo.eval()
        with torch.no_grad():
            for (caminho, classe), (x, _) in zip(val, ds_val):
                pred = modelo(x.unsqueeze(0).to(device)).argmax(1).item()
                w.writerow([caminho, classe, classes[pred]])
    print(f"\nModelo e relatório salvos em {saida}")


if __name__ == "__main__":
    main()
