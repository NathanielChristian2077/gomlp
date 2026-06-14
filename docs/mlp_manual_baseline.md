# Relatório parcial: MLP manual densa

## Objetivo

Esta etapa implementa manualmente uma MLP para classificação binária de imagens de gatos e cachorros. A implementação foi feita em Go para tornar explícitos os passos principais de uma rede neural: pré-processamento, forward, ativações, loss, backpropagation, atualização de pesos e avaliação.

A proposta desta MLP não é superar uma CNN. O objetivo é criar uma baseline manual, auditável e matematicamente compreensível para posterior comparação.

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

A vetorização simplifica a implementação, mas remove a estrutura espacial local da imagem. Essa limitação é central para interpretar os resultados da MLP.

## Arquitetura baseline

```text
4096 -> 128 -> 1
```

Configuração oficial:

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

## Forward

A camada densa calcula:

```text
z_o = b_o + soma_i(x_i * W_i,o)
```

Os pesos são armazenados em layout input-major:

```text
Weights[i*Out + o]
```

Esse layout é simples para auditar e será útil na futura DSA, pois cada entrada ativa aponta para um bloco contíguo de pesos.

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

## Execução da baseline

Script oficial:

```bash
./scripts/run_dense_baseline_h128_b16_lr001.sh ./dataset
```

Resultados:

```text
results/baseline_dense_h128_b16_lr001/
```

O arquivo `summary.csv` consolida os resultados por seed.

## Resultados observados

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

## Discussão

A MLP mostrou que o pipeline está funcional: o modelo aprende problemas sintéticos simples e reduz loss no treino real. Porém, no dataset de imagens, a acurácia de teste ficou próxima do acaso.

Esse resultado é coerente com a limitação da MLP sobre imagens vetorizadas. Ao transformar a imagem em um vetor de 4096 valores, relações espaciais importantes são perdidas. A rede não possui filtros locais, compartilhamento de pesos nem viés espacial, ao contrário de uma CNN.

Também houve alta instabilidade entre seeds. Isso indica que o conjunto de validação é pequeno e ruidoso, e que a MLP encontra padrões pouco robustos.

## Conclusão parcial

A baseline densa está pronta para documentação e comparação. Ela é funcional, auditável e reproduzível, mas apresenta baixa generalização.

A próxima etapa será implementar DSA exact sparse sobre essa baseline, comparando tempo de execução, sparsity, loss e acurácia sem alterar a função matemática da rede quando apenas ativações exatamente zero forem ignoradas.
