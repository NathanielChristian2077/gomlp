# Busca de arquitetura com PyTorch e GPU

Este branch adiciona uma busca em PyTorch para usar GPU na procura de arquiteturas MLP mais estáveis. A busca usa apenas treino e validação. O split de teste não é avaliado pelo script de busca ampla.

## Comando recomendado

```bash
python scripts/gpu_search_pytorch.py \
  --dataset ./dataset \
  --runs runs/gpu_search_pytorch_v1 \
  --device auto \
  --strategy halving \
  --workers 4
```

O modo `auto` usa GPU se disponível, caso contrário usa CPU.

## Estratégia halving

É o modo recomendado. Ele faz uma triagem progressiva:

```text
screen_s1: todas as arquiteturas, seed 1, até 20 épocas, mantém 1024 grupos
screen_s2: top 1024, seeds 1..5, até 60 épocas, mantém 256 grupos
screen_s3: top 256, seeds 1..14, até 150 épocas, mantém 64 grupos
final_s4:  top 64, seeds 1..42, até 500 épocas
```

Um grupo é a combinação:

```text
arquitetura + learning rate + batch size
```

## Estratégia exhaustive

Roda a grade completa:

```text
258 arquiteturas * 6 learning rates * 4 batch sizes * 42 seeds = 260064 execuções
```

```bash
python scripts/gpu_search_pytorch.py \
  --dataset ./dataset \
  --runs runs/gpu_search_pytorch_exhaustive \
  --device cuda \
  --strategy exhaustive
```

## Grade fixa

Arquiteturas de 1 até 3 camadas ocultas usando:

```text
16, 32, 64, 128, 256, 512
```

Learning rates:

```text
0, 0.0001, 0.0003, 0.001, 0.003, 0.01
```

Batch sizes:

```text
0, 16, 32, 64
```

`batch = 0` significa full-batch.

Seeds:

```text
1 até 42
```

## Critério de parada

Cada execução tem trava máxima de 500 épocas, mas pode parar antes por:

```text
validation_patience
low_learning
divergent_or_non_finite
max_epochs
```

## Saídas

O arquivo principal é:

```text
<runs>/summary.csv
```

Também é salvo:

```text
<runs>/selected_groups.json
```

No estágio final, o script salva artefatos por execução:

```text
config.json
metrics.csv
best.pt
```

## Como escolher candidatos

Depois da busca, agrupe o `summary.csv` por:

```text
hidden + learning_rate + batch_size
```

Priorize média alta de validação, desvio padrão baixo entre seeds, F1 coerente com accuracy e gap treino-validação moderado.

## Resultado da busca ampla

A busca por halving definiu um espaço total planejado de 260064 combinações arquitetura/hiperparâmetros/seeds e executou 17584 treinamentos efetivos na etapa exploratória inicial. Isso permitiu reduzir a busca para candidatos robustos sem avaliar exaustivamente toda a grade.

A configuração robusta inicial mais forte com a cabeça `sigmoid1` foi:

```text
hidden=32x64x512
learning_rate=0.01
batch_size=16
seed canônica=24
```

Essa configuração serviu como ponto de partida para a comparação posterior entre cabeças de saída.

## Comparação de cabeças de saída

Depois da busca inicial, as três arquiteturas mais interessantes foram retestadas com duas formulações válidas de saída:

| Cabeça | Saída | Loss | Predição |
|---|---:|---|---|
| `sigmoid1` | 1 logit | Binary Cross-Entropy | `sigmoid(logit) >= 0.5` |
| `softmax2` | 2 logits, cat/dog | Cross-Entropy | `argmax(logits)` |

`softmax1` não é usado porque softmax com um único logit é degenerado. `sigmoid2` também não foi usado como candidato final porque gato/cachorro é um problema de classes mutuamente exclusivas, não multi-label.

O ranking congelado dos seis candidatos foi:

| Rank | Arquitetura | Cabeça | LR | Batch | Seed | Parâmetros | Score | Acc média | Acc std | Acc min/max | F1 médio | Gap abs médio |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | `32x64x512` | `softmax2` | 0.01 | 16 | 7 | 167522 | 0.895363 | 0.589048 | 0.023177 | 0.54 / 0.64 | 0.574793 | 0.093968 |
| 2 | `32x64x512` | `sigmoid1` | 0.01 | 16 | 24 | 167009 | 0.881118 | 0.584524 | 0.024417 | 0.53 / 0.64 | 0.569265 | 0.115317 |
| 3 | `64x32x512` | `sigmoid1` | 0.003 | 16 | 25 | 281697 | 0.872363 | 0.578095 | 0.023728 | 0.52 / 0.62 | 0.575089 | 0.133651 |
| 4 | `128x32x512` | `sigmoid1` | 0.003 | 32 | 9 | 545953 | 0.871975 | 0.576667 | 0.024365 | 0.53 / 0.63 | 0.570688 | 0.123413 |
| 5 | `64x32x512` | `softmax2` | 0.003 | 16 | 11 | 282210 | 0.868193 | 0.578571 | 0.031965 | 0.51 / 0.69 | 0.580320 | 0.142222 |
| 6 | `128x32x512` | `softmax2` | 0.003 | 32 | 12 | 546466 | 0.857485 | 0.574048 | 0.026100 | 0.51 / 0.63 | 0.556492 | 0.131032 |

A configuração final da busca em PyTorch foi escolhida como:

```text
32x64x512 + softmax2
lr=0.01
batch=16
seed=7
```

Essa escolha se justifica pelo maior score robusto, maior acurácia média, menor gap médio e menor desvio em relação à versão `sigmoid1` da mesma arquitetura.

## Validação em Go com DSA

Os seis candidatos foram portados para a implementação manual em Go usando o script:

```bash
DATASET=./dataset bash scripts/run_top6_go_dsa.sh
```

O preflight `go test ./...` passou e os seis candidatos foram executados com:

```text
validation compare
test compare
test benchmark
```

A DSA exata (`threshold=0`) preservou as predições densas no split de teste para todos os seis candidatos, com `mismatch_count_from_dense = 0`. Isso indica que a propagação esparsa dinâmica exata manteve equivalência funcional com o forward denso no critério de classificação.

### Métricas densas no teste, Go

| Rank | Arquitetura | Cabeça | Acc teste | Precision | Recall | F1 | Observação |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `32x64x512` | `softmax2` | 0.51 | 0.6000 | 0.0600 | 0.1091 | Vencedor robusto em PyTorch, mas fraco em F1 no Go |
| 2 | `32x64x512` | `sigmoid1` | 0.50 | 0.5000 | 0.3800 | 0.4318 | Mesmo tamanho da vencedora, mas comportamento de decisão diferente |
| 3 | `64x32x512` | `sigmoid1` | 0.50 | 0.5000 | 0.1800 | 0.2647 | Baixo recall no teste em Go |
| 4 | `128x32x512` | `sigmoid1` | 0.58 | 0.5909 | 0.5200 | 0.5532 | Melhor acurácia no teste em Go |
| 5 | `64x32x512` | `softmax2` | 0.54 | 0.5278 | 0.7600 | 0.6230 | Melhor F1 no teste em Go |
| 6 | `128x32x512` | `softmax2` | 0.44 | 0.4571 | 0.6400 | 0.5333 | Recall alto, mas baixa acurácia |

Esse resultado mostra que a ordenação obtida em PyTorch não se transfere perfeitamente para o Go. A implementação manual tem diferenças de inicialização, ordem de treino, RNG e detalhes numéricos suficientes para alterar o comportamento final dos checkpoints. Por isso, a busca GPU é usada como seleção de candidatos, enquanto a implementação Go é usada como validação manual e estudo da DSA.

### Benchmark DSA exata, teste, Go

| Rank | Arquitetura | Cabeça | Dense ns/forward | Sparse exact ns/forward | Speedup real | Tempo salvo | Sparsity média | Mismatch |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `32x64x512` | `softmax2` | 147657.18 | 108273.07 | 1.36x | 26.7% | 52.35% | 0 |
| 2 | `32x64x512` | `sigmoid1` | 156653.10 | 111706.34 | 1.40x | 28.7% | 51.67% | 0 |
| 3 | `64x32x512` | `sigmoid1` | 266491.69 | 209170.35 | 1.27x | 21.5% | 51.80% | 0 |
| 4 | `128x32x512` | `sigmoid1` | 480934.48 | 380039.45 | 1.27x | 21.0% | 50.47% | 0 |
| 5 | `64x32x512` | `softmax2` | 260981.56 | 206902.93 | 1.26x | 20.7% | 53.46% | 0 |
| 6 | `128x32x512` | `softmax2` | 485016.81 | 440797.53 | 1.10x | 9.1% | 52.32% | 0 |

A DSA exata apresentou aceleração real entre 1.10x e 1.40x nos seis candidatos, preservando a classe prevista em todos os casos. A família `32x64x512` foi a mais favorecida no benchmark, tanto com `softmax2` quanto com `sigmoid1`.

Thresholds positivos foram mantidos como estudo exploratório. Eles podem aumentar a esparsidade e, em alguns casos, melhorar o tempo real, mas já não possuem garantia geral de equivalência com o modelo denso. Portanto, o resultado principal deve ser reportado com `threshold=0`, e os thresholds positivos devem ser discutidos como trade-off entre compressão dinâmica, velocidade e risco de mudança de predição.

## Conclusão operacional

A busca GPU selecionou `32x64x512 + softmax2` como configuração robusta principal. A validação em Go mostrou que diferentes interações entre arquitetura e cabeça de saída produzem melhores métricas dependendo do critério:

```text
Melhor candidato robusto em PyTorch: 32x64x512 + softmax2
Melhor acurácia no teste em Go:      128x32x512 + sigmoid1
Melhor F1 no teste em Go:            64x32x512 + softmax2
Maior aceleração DSA:                32x64x512, especialmente sigmoid1/softmax2
```

Isso reforça que a contribuição não está apenas em escolher um único modelo final, mas em estudar a interação entre arquitetura, cabeça de saída e propagação esparsa dinâmica.
