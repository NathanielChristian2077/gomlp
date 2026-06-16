# MLP Manual em Go com Dynamic Sparse Activation

Este branch contém a implementação manual da MLP densa em Go e uma extensão experimental de Dynamic Sparse Activation, DSA. A ideia é manter a baseline densa auditável e, sobre ela, avaliar uma forma de propagação esparsa induzida pela ReLU.

A proposta deste branch é complementar a apresentação principal do trabalho. Ele mantém a MLP manual como base, mas inclui resultados e comandos para comparar a propagação densa com uma propagação esparsa exata e com thresholds aproximados.

## Estrutura atual

```text
cmd/train/      Executa um treino individual
cmd/sweep/      Executa uma grade de experimentos densos
cmd/compare/    Compara dense, sparse exact e sparse threshold em métricas de classificação
cmd/bench/      Mede tempo puro de forward dense e sparse com warmup e repetição
experiment/     Runner, configs, checkpoints e persistência de resultados
data/           Loader do dataset e pré-processamento das imagens
metrics/        Métricas, matriz de confusão e logger CSV
nn/             MLP, camadas densas, ativações, DSA, loss e treino
docs/           Relatórios e documentação técnica
runs/           Saídas organizadas de experimentos novos
```

## Dataset esperado

```text
dataset/
  train/
    cat/
    dog/
  validation/
    cat/
    dog/
  test/
    cat/
    dog/
```

A configuração usada nos experimentos possui 500 imagens balanceadas:

```text
train:      300 imagens, 150 cat e 150 dog
validation: 100 imagens, 50 cat e 50 dog
test:       100 imagens, 50 cat e 50 dog
```

Rótulos:

```text
cat = 0.0
dog = 1.0
```

## Pré-processamento

Cada imagem é processada assim:

1. Decodificação JPEG ou PNG.
2. Redimensionamento para 64x64 pixels.
3. Conversão para escala de cinza.
4. Normalização dos pixels para o intervalo [0, 1].
5. Vetorização para `[]float64` de tamanho 4096.

A escala de cinza usa:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

## Arquitetura da MLP

A implementação aceita uma ou mais camadas ocultas. Exemplos:

```text
4096 -> 64 -> 1
4096 -> 256 -> 64 -> 1
4096 -> 128 -> 256 -> 128 -> 1
4096 -> 256 -> 256 -> 128 -> 1
4096 -> 512 -> 512 -> 128 -> 1
```

A camada oculta usa ReLU e a saída usa sigmoid. A loss é Binary Cross Entropy. O otimizador é gradient descent manual com mini-batch.

Os pesos das camadas densas usam layout input-major:

```text
Weights[i*Out + o]
```

Esse layout favorece a DSA porque, quando uma ativação de entrada está ativa, todos os pesos associados à sua contribuição na próxima camada ficam em um bloco contíguo.

## Matemática implementada

Camada densa:

```text
z_o = b_o + soma_i(x_i * W_i,o)
```

ReLU:

```text
ReLU(x) = max(0, x)
```

Sigmoid:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

Binary Cross Entropy:

```text
BCE = -(y * log(yHat) + (1-y) * log(1-yHat))
```

Com sigmoid na saída e BCE, o delta da saída é:

```text
delta_saida = yHat - y
```

## Dynamic Sparse Activation

A DSA explora a esparsidade induzida pela ReLU. Depois de cada camada oculta, ativações nulas ou abaixo de um threshold são compactadas em um `ActiveVector`, que guarda:

```text
Size: tamanho original da camada
Indices: posições originais dos neurônios ativos
Values: ativações correspondentes
```

Na DSA exata, `threshold = 0`. Apenas ativações exatamente nulas são removidas, preservando a mesma função da MLP densa.

```text
z_o = b_o + soma_{j ativo}(a_j * W_j,o)
```

Na DSA com threshold, `threshold > 0`. Ativações positivas pequenas também são removidas. Essa versão é aproximada e pode alterar loss, predições e métricas.

## Como rodar

Treino individual:

```bash
go run ./cmd/train \
  --dataset ./dataset \
  --epochs 100 \
  --hidden 256x64 \
  --batch 16 \
  --lr 0.001 \
  --run-dir runs/manual_h256x64_b16_lr001_seed42 \
  --seed 42
```

Sweep denso:

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

Comparação de métricas dense vs DSA:

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

## Resultados resumidos da DSA exact

Resultados no split de teste, com `lr = 0.003`, `batch = 16`, `epochs = 200`, `seed = 42` e benchmark com `repeat = 500`, `warmup = 50`, `gomaxprocs = 1`.

| Arquitetura | Acc | F1 | Sparsity exact | Ops salvas | Dense ns/forward | Sparse exact ns/forward | Ganho real |
|---|---:|---:|---:|---:|---:|---:|---:|
| `64` | 0.56 | 0.6563 | 53.36% | 0.013% | 251182 | 232240 | 7.54% |
| `256x64` | 0.59 | 0.6435 | 44.86% | 0.733% | 950254 | 868090 | 8.65% |
| `128x256x128` | 0.60 | 0.6552 | 47.78% | 5.34% | 526198 | 471552 | 10.38% |
| `256x256x128` | 0.53 | 0.6240 | 50.21% | 4.35% | 1000383 | 901491 | 9.89% |
| `512x512x128` | 0.55 | 0.2373 | 47.39% | 6.45% | 2095225 | 1838661 | 12.25% |

Em todas as arquiteturas, a DSA exact manteve `mismatch = 0` e `max_abs_diff_from_dense = 0`, preservando loss, acurácia, precisão, recall e F1 da MLP densa.

## Leitura dos resultados

A arquitetura `128x256x128` apresentou o melhor equilíbrio entre qualidade, custo e ganho real de tempo. A arquitetura `512x512x128` apresentou maior ganho computacional bruto, mas pior F1, sendo mais útil como stress test de desempenho do que como classificador final.

O threshold `0.05` foi o ponto conservador mais estável nos experimentos, aumentando a esparsidade com pouca ou nenhuma alteração nas predições em algumas arquiteturas. O threshold `0.1` apresentou ganho experimental de acurácia e F1 na arquitetura `256x64`, mas alterou predições e piorou a loss, portanto deve ser tratado como DSA aproximada.

## Documentação detalhada

A documentação principal da extensão está em:

```text
docs/dsa_extension.md
```
