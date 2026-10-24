# Classificação de moedas brasileiras com MobileNetV2

Meu primeiro dataset próprio: fotos de moedas brasileiras de circulação comum, usadas para treinar um modelo que identifica o tipo de moeda a partir de uma foto.

## Dados

- **186 fotos**, cerca de 31 por classe, guardadas no bucket `moedasbrasileiras` do Google Cloud Storage (uma pasta por classe dentro de `classes/`).
- **6 classes:** 1, 5, 10, 25 e 50 centavos e 1 real.
- Divisão de 80% para treino (148 fotos) e 20% para validação (38 fotos).

## Modelo

- *Transfer learning* com a **MobileNetV2** pré-treinada no ImageNet, congelada, mais uma camada final com uma saída por moeda.
- Fotos redimensionadas para 224×224 e normalizadas.
- Aumento de dados só no treino: giros de 90° e espelhamento aleatórios.
- Otimizador Adam (taxa de aprendizado 0,0001), 250 épocas, lotes de 32 fotos.

## Resultados

> **Atenção:** os números abaixo são da execução anterior, em que a lista de fotos era embaralhada de novo a cada leitura e parte das fotos de validação também aparecia no treino. A divisão foi corrigida (embaralhamento único, com semente fixa) e o notebook precisa ser executado de novo para atualizar estes resultados.

- Acurácia de treino em torno de **92%** nas últimas épocas e acurácia de validação em torno de **53%**: o modelo decora o treino e generaliza pouco, o esperado com tão poucas fotos.
- Na matriz de confusão, **20 das 38 fotos** de validação foram classificadas corretamente. A moeda de 1 real foi a mais fácil (5 de 5), e as de 1, 5 e 10 centavos, com cores e tamanhos parecidos, foram as mais confundidas entre si.

A apresentação do projeto está em [`Apresentação.pdf`](Apresentação.pdf).

## Como executar

O notebook foi feito para o **Google Colab** com acesso ao bucket.

1. Abra o `notebook.ipynb` no Colab e autentique sua conta Google com acesso ao bucket (ou troque `PASTA_DAS_IMAGENS` por uma pasta local com uma subpasta por classe).
2. Execute as células em ordem. A primeira instala as dependências que não vêm no Colab (lista completa em `requirements.txt`).
3. Os parâmetros do experimento (tamanho da imagem, lote, épocas, taxa de aprendizado, classes) ficam reunidos na seção **2. Parâmetros**.

## Trabalhos futuros

- Aumentar a base de fotos e usar parada antecipada (*early stopping*) no lugar de um número fixo de épocas.
- Prever o valor da moeda a partir do ano e do estado de conservação.

## Autora

**Júlia Roberta Quoos Alves**: engenheira de dados e mestranda em Ciência da Computação (UFSM).
[LinkedIn](https://www.linkedin.com/in/ohtopico) · [Site](http://ohtopico.com.br)
