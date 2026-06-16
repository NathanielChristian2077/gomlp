# Relatório: MLP manual densa

## Objetivo

Esta etapa implementa manualmente uma MLP para classificação binária de imagens de gatos e cachorros. A implementação foi feita em Go para tornar explícitos os passos principais de uma rede neural: pré-processamento, forward, ativações, loss, backpropagation, atualização de pesos, checkpoints e avaliação.

A proposta desta MLP não é superar uma CNN. O objetivo é criar uma baseline manual, auditável e matematicamente compreensível, servindo como ponto de comparação para as etapas posteriores do trabalho.

## Dataset

O dataset usado possui 500 imagens balanceadas:

| Split | Total | Cat | Dog |
|---|---:|---:|---:|
| Train | 300 | 150 | 150 |
| Validation | 100 | 50 | 50 |
| Test | 100 | 50 | 50 |

Estrutura esperada:

```text
dataset/
  train/cat
  train/dog
  validation/cat
  validation/dog
  test/cat
  test/dog
```

Rótulos:

```text
cat = 0.0
dog = 1.0
```

## Pré-processamento

Cada imagem é decodificada, redimensionada para 64x64, convertida para escala de cinza, normalizada para o intervalo [0, 1] e vetorizada em 4096 entradas.

A escala de cinza usa:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

A vetorização simplifica a implementação, mas remove a estrutura espacial local da imagem. Essa limitação é central para interpretar os resultados da MLP, já que a rede totalmente conectada não possui filtros locais nem compartilhamento de pesos.

## Arquitetura

A baseline densa oficial inicial é:

```text
4096 -> 128 -> 1
```

Configuração oficial inicial:

```text
Hidden: 128
Ativação oculta: ReLU
Saída: sigmoid
Loss: Binary Cross Entropy
Batch size: 16
Learning rate: 0.001
Épocas: 100
Seeds: 1, 2, 3, 4, 5, 42
```

A implementação atual também permite arquiteturas profundas, como:

```text
4096 -> 64 -> 1
4096 -> 256 -> 64 -> 1
4096 -> 512 -> 128 -> 32 -> 1
```

No código, isso é representado por `HiddenSizes`, um slice de inteiros. Por exemplo:

```text
HiddenSizes = [256, 64]
```

representa:

```text
4096 -> 256 -> 64 -> 1
```

## Forward

Cada camada densa calcula:

```text
z_j = b_j + soma_i(W_j,i * x_i)
```

Em uma arquitetura com várias camadas ocultas, a saída ativada de cada camada vira a entrada da próxima:

```text
entrada -> Dense -> ReLU -> Dense -> ReLU -> ... -> Dense -> Sigmoid
```

## Ativações e loss

A camada oculta usa ReLU:

```text
ReLU(x) = max(0, x)
```

A saída usa sigmoid:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

A loss é Binary Cross Entropy:

```text
BCE = -(y * log(yHat) + (1-y) * log(1-yHat))
```

A implementação limita `yHat` antes do log para evitar `log(0)`.

## Backpropagation

Com sigmoid na saída e Binary Cross Entropy, o delta da saída é simplificado para:

```text
delta_saida = yHat - y
```

Os gradientes são acumulados por mini-batch e aplicados por gradient descent:

```text
W = W - lr * (dW / batch_size)
b = b - lr * (dB / batch_size)
```

Para múltiplas camadas ocultas, o erro é propagado de trás para frente, usando os pesos da camada seguinte e a derivada da ReLU da camada atual.

## Treino e avaliação

A cada época, as amostras de treino são embaralhadas, divididas em mini-batches e usadas para atualizar os pesos. Ao fim da época, o modelo é avaliado em treino e validação.

O melhor checkpoint é escolhido por validação:

1. Maior acurácia de validação.
2. Em empate, menor loss de validação.

O teste final é feito usando esse melhor checkpoint, não necessariamente o modelo da última época.

## Métricas

Foram implementadas:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Matriz de confusão.

A classificação usa threshold 0.5:

```text
yHat >= 0.5 -> dog
yHat < 0.5 -> cat
```

A matriz de confusão é impressa como:

```text
TN FP
FN TP
```

## Como executar

Treino individual com uma camada oculta:

```bash
go run ./cmd/train --dataset ./dataset --epochs 100 --hidden 128 --batch 16 --lr 0.001 --run-dir runs/manual_h128
```

Treino individual com múltiplas camadas ocultas:

```bash
go run ./cmd/train --dataset ./dataset --epochs 100 --hidden 256x64 --batch 16 --lr 0.001 --run-dir runs/manual_h256x64
```

Sweep de experimentos:

```bash
go run ./cmd/sweep \
  --dataset ./dataset \
  --epochs 100 \
  --hidden '128;256x64;512x128' \
  --batch '16,32' \
  --lr '0.001,0.0003' \
  --seeds '1,2,3,4,5,42' \
  --workers 1 \
  --runs runs/dense_sweep_v1
```

Cada execução salva `config.json`, `summary.json`, `metrics.csv`, `confusion_matrix.csv`, `test_predictions.csv` e checkpoints.

## Resultados observados da baseline inicial

| Seed | Melhor época | Val acc | Test acc | F1 |
|---:|---:|---:|---:|---:|
| 1 | 20 | 0.61 | 0.50 | 0.55 |
| 2 | 1 | 0.58 | 0.45 | 0.52 |
| 3 | 1 | 0.52 | 0.53 | 0.48 |
| 4 | 51 | 0.59 | 0.52 | 0.47 |
| 5 | 90 | 0.56 | 0.54 | 0.38 |
| 42 | 12 | 0.60 | 0.44 | 0.22 |

Médias aproximadas:

```text
Val accuracy média: 57,7%
Test accuracy média: 49,7%
F1 médio: 43,7%
```

## Bateria densa posterior

Após a baseline inicial, foram executados sweeps adicionais variando arquitetura, batch size, learning rate e seed. A configuração densa rasa `4096 -> 64 -> 1`, com `lr = 0.003` e `batch = 16`, apresentou a melhor acurácia média em teste entre as baterias finais, aproximadamente 55,5%.

A arquitetura `4096 -> 256 -> 64 -> 1`, com os mesmos hiperparâmetros, apresentou F1 médio superior em algumas combinações, mas com custo computacional maior.

## Discussão

A MLP mostrou que o pipeline está funcional: o modelo aprende problemas sintéticos simples e reduz loss no treino real. Porém, no dataset de imagens, a acurácia de teste ficou próxima do acaso.

Esse resultado é coerente com a limitação da MLP sobre imagens vetorizadas. Ao transformar a imagem em um vetor de 4096 valores, relações espaciais importantes são perdidas. A rede não possui filtros locais, compartilhamento de pesos nem viés espacial, ao contrário de uma CNN.

A infraestrutura com múltiplas camadas, checkpoints e sweeps permite testar essa limitação com mais rigor, sem misturar resultados soltos ou perder rastreabilidade das configurações.

## Conclusão

A baseline densa está pronta para apresentação como implementação manual e independente. Ela demonstra de forma auditável os principais componentes de uma rede neural totalmente conectada e fornece uma base coerente para comparação com CNN e transfer learning nas etapas seguintes.