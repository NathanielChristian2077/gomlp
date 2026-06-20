# Baseline: MLP manual densa

## Objetivo

A baseline implementa uma MLP manual em Go para classificação binária de imagens de gatos e cachorros. O propósito é auditabilidade: deixar explícitos pré-processamento, forward, ativações, loss, backpropagation, atualização, validação, checkpoints e avaliação.

Ela não pretende superar uma CNN. Ao vetorizar imagens, a MLP perde relações espaciais locais e serve justamente para demonstrar essa limitação.

## Dataset e pré-processamento

| Split | Total | Cat | Dog |
|---|---:|---:|---:|
| Train | 300 | 150 | 150 |
| Validation | 100 | 50 | 50 |
| Test | 100 | 50 | 50 |

Cada imagem é decodificada, redimensionada para `64x64`, convertida para escala de cinza, normalizada para `[0,1]` e vetorizada em 4096 valores.

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
cat = 0
dog = 1
```

## Modelo

A baseline inicial é:

```text
4096 -> 128 -> 1
```

A arquitetura aceita uma ou mais camadas ocultas, por exemplo:

```text
4096 -> 64 -> 1
4096 -> 256 -> 64 -> 1
4096 -> 512 -> 128 -> 32 -> 1
```

Camadas ocultas usam ReLU, a saída usa sigmoid e a loss é Binary Cross-Entropy.

```text
z_j = b_j + soma_i(W_i,j * x_i)
ReLU(x) = max(0, x)
sigmoid(x) = 1 / (1 + exp(-x))
BCE = -(y * log(yHat) + (1-y) * log(1-yHat))
```

Com sigmoid e BCE, o delta de saída é:

```text
delta_saida = yHat - y
```

Os gradientes são acumulados por mini-batch e a atualização SGD é:

```text
W = W - lr * (dW / batch_size)
b = b - lr * (dB / batch_size)
```

## Seleção e avaliação

O melhor checkpoint é escolhido por validação:

1. maior accuracy de validação;
2. em empate, menor loss de validação.

O teste é avaliado somente com esse melhor checkpoint. Métricas registradas: accuracy, precision, recall, F1 e matriz de confusão.

## Execução

```bash
go test ./...

go run ./cmd/train \
  --dataset ./dataset \
  --epochs 100 \
  --hidden 128 \
  --batch 16 \
  --lr 0.001 \
  --seed 42 \
  --run-dir runs/manual_h128
```

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

## Resultados observados

Na baseline inicial `4096 -> 128 -> 1`, as médias observadas foram aproximadamente:

```text
Val accuracy:  57,7%
Test accuracy: 49,7%
F1:            43,7%
```

Em baterias posteriores, `4096 -> 64 -> 1` com `lr=0.003` e `batch=16` apresentou melhor accuracy média de teste, enquanto `4096 -> 256 -> 64 -> 1` mostrou F1 superior em parte das configurações, com custo computacional maior.

## Conclusão

A MLP valida o pipeline completo e fornece uma baseline manual coerente. O desempenho próximo ao acaso no teste é compatível com a perda de estrutura espacial causada pela vetorização, o que justifica a comparação posterior com CNN e transfer learning.
