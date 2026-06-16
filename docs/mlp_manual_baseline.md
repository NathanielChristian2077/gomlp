# Relatório: MLP manual densa e extensão DSA

## Objetivo

Esta etapa implementa manualmente uma MLP para classificação binária de imagens de gatos e cachorros. A implementação foi feita em Go para tornar explícitos os passos principais de uma rede neural: pré-processamento, forward, ativações, loss, backpropagation, atualização de pesos e avaliação.

Além da baseline densa, este branch inclui uma extensão experimental de Dynamic Sparse Activation, DSA. A extensão foi implementada para avaliar se ativações nulas ou muito pequenas induzidas pela ReLU podem ser ignoradas na propagação entre camadas, preservando ou aproximando o comportamento da MLP densa.

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

## Arquitetura

A implementação aceita uma ou mais camadas ocultas. A baseline densa inicial foi:

```text
4096 -> 128 -> 1
```

As baterias posteriores testaram arquiteturas como:

```text
4096 -> 64 -> 1
4096 -> 256 -> 64 -> 1
4096 -> 128 -> 256 -> 128 -> 1
4096 -> 256 -> 256 -> 128 -> 1
4096 -> 512 -> 512 -> 128 -> 1
```

No código, isso é representado por `HiddenSizes`, um slice de inteiros. Por exemplo:

```text
HiddenSizes = [256, 64]
```

representa:

```text
4096 -> 256 -> 64 -> 1
```

## Forward denso

Cada camada densa calcula:

```text
z_o = b_o + soma_i(x_i * W_i,o)
```

Os pesos são armazenados em layout input-major:

```text
Weights[i*Out + o]
```

Esse layout facilita a propagação esparsa porque cada entrada ativa acessa um bloco contíguo de pesos para todos os neurônios da camada seguinte.

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

## Dynamic Sparse Activation

A DSA transforma a saída ReLU de uma camada oculta em uma representação compacta:

```text
ActiveVector:
  Size    -> tamanho original da camada
  Indices -> índices dos neurônios ativos
  Values  -> ativações dos neurônios ativos
```

A camada seguinte é calculada usando apenas as entradas ativas:

```text
z_o = b_o + soma_{j ativo}(a_j * W_j,o)
```

Na DSA exact, `threshold = 0`. Apenas ativações exatamente nulas são removidas. Essa versão preserva matematicamente a função da MLP densa.

Na DSA threshold, `threshold > 0`. Ativações positivas pequenas também são removidas. Essa versão é aproximada e pode alterar loss, predições e métricas.

## Comandos principais

Treino individual:

```bash
go run ./cmd/train --dataset ./dataset --epochs 100 --hidden 256x64 --batch 16 --lr 0.001 --run-dir runs/manual_h256x64
```

Sweep de experimentos densos:

```bash
go run ./cmd/sweep \
  --dataset ./dataset \
  --epochs 100 \
  --hidden '64;256x64;128x256x128' \
  --batch '16,32' \
  --lr '0.003,0.001,0.0003' \
  --seeds '1,2,3,4,5,42' \
  --workers 1 \
  --runs runs/dense_sweep
```

Comparação dense vs DSA:

```bash
go run ./cmd/compare \
  --dataset ./dataset \
  --hidden 256x64 \
  --epochs 200 \
  --batch 16 \
  --lr 0.003 \
  --seed 42 \
  --thresholds "0,0.05,0.1,0.25,0.5" \
  --runs runs/dsa_compare \
  --name compare_h256x64_lr003_bs16_seed42
```

Benchmark de forward:

```bash
go run ./cmd/bench \
  --dataset ./dataset \
  --hidden 256x64 \
  --epochs 200 \
  --batch 16 \
  --lr 0.003 \
  --seed 42 \
  --thresholds "0,0.05,0.1,0.25,0.5" \
  --repeat 500 \
  --warmup 50 \
  --gomaxprocs 1 \
  --runs runs/dsa_bench \
  --name bench_h256x64_lr003_bs16_seed42
```

## Resultados da baseline densa inicial

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

## Resultados da DSA exact

Resultados no split de teste, com `lr = 0.003`, `batch = 16`, `epochs = 200`, `seed = 42` e benchmark com `repeat = 500`, `warmup = 50`, `gomaxprocs = 1`.

| Arquitetura | Acc | F1 | Sparsity exact | Ops salvas | Dense ns/forward | Sparse exact ns/forward | Ganho real |
|---|---:|---:|---:|---:|---:|---:|---:|
| `64` | 0.56 | 0.6563 | 53.36% | 0.013% | 251182 | 232240 | 7.54% |
| `256x64` | 0.59 | 0.6435 | 44.86% | 0.733% | 950254 | 868090 | 8.65% |
| `128x256x128` | 0.60 | 0.6552 | 47.78% | 5.34% | 526198 | 471552 | 10.38% |
| `256x256x128` | 0.53 | 0.6240 | 50.21% | 4.35% | 1000383 | 901491 | 9.89% |
| `512x512x128` | 0.55 | 0.2373 | 47.39% | 6.45% | 2095225 | 1838661 | 12.25% |

Em todas as arquiteturas, a DSA exact manteve `mismatch = 0` e `max_abs_diff_from_dense = 0`. Isso confirma que a propagação esparsa exata preserva a função da MLP densa.

## Resultados de thresholds

A arquitetura `256x64` apresentou o comportamento mais claro para thresholds:

| Threshold | Loss | Acc | F1 | Sparsity | Mismatch |
|---:|---:|---:|---:|---:|---:|
| `0` exact | 0.7291 | 0.59 | 0.6435 | 44.86% | 0 |
| `0.05` | 0.7301 | 0.59 | 0.6435 | 47.44% | 0 |
| `0.10` | 0.7355 | 0.61 | 0.6549 | 50.02% | 4 |
| `0.25` | 0.7523 | 0.58 | 0.6250 | 57.76% | 3 |
| `0.50` | 0.7823 | 0.58 | 0.6250 | 70.05% | 5 |

O threshold `0.05` foi o ponto conservador mais estável nessa arquitetura, pois aumentou a esparsidade sem alterar as predições. O threshold `0.10` foi o resultado experimental mais interessante, pois melhorou acurácia e F1, mas alterou quatro predições e aumentou a loss.

## Discussão

A MLP mostrou que o pipeline está funcional, mas continua limitada para classificação de imagens vetorizadas. Essa limitação é esperada porque a rede não possui filtros locais, compartilhamento de pesos nem viés espacial, ao contrário de uma CNN.

A DSA exact mostrou que uma fração relevante das ativações ocultas pode ser removida sem alterar a saída. O ganho real de tempo foi maior nas arquiteturas com mais custo em camadas internas, especialmente `128x256x128` e `512x512x128`.

Os thresholds positivos criam uma segunda linha experimental: poda dinâmica aproximada em inferência. Essa abordagem pode aumentar esparsidade e, em alguns casos, melhorar métricas discretas, mas não preserva necessariamente a função da rede densa.

## Conclusão

A baseline densa está funcional e documentada. A extensão DSA exact preserva a saída da MLP densa e produz ganhos reais de tempo nos benchmarks realizados. A arquitetura `128x256x128` apresentou o melhor equilíbrio entre qualidade, custo e aceleração, enquanto `512x512x128` serviu como melhor stress test computacional. Thresholds moderados, especialmente `0.05` e `0.10`, merecem investigação posterior em baterias com múltiplas seeds.