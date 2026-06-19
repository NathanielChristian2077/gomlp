# Seis candidatos finais para estudo

A comparação de cabeças de saída foi congelada com seis candidatos: três arquiteturas e duas formulações de saída válidas para classificação binária.

## Formulações

| Cabeça | Saída | Loss | Predição |
|---|---:|---|---|
| `sigmoid1` | 1 logit | Binary Cross-Entropy | `sigmoid(logit) >= 0.5` |
| `softmax2` | 2 logits, cat/dog | Cross-Entropy | `argmax(logits)` |

`softmax1` não é usado porque softmax com um único logit é degenerado. `sigmoid2` também não entra no conjunto final porque as classes são mutuamente exclusivas, não multi-label.

## Ranking congelado

| Rank | Arquitetura | Cabeça | LR | Batch | Seed canônica | Parâmetros | Score | Acc média | Acc std | Acc min/max | F1 médio | Gap abs médio | Melhor acc | Melhor F1 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | `32x64x512` | `softmax2` | 0.01 | 16 | 7 | 167522 | 0.895363 | 0.589048 | 0.023177 | 0.54 / 0.64 | 0.574793 | 0.093968 | 0.64 | 0.600000 |
| 2 | `32x64x512` | `sigmoid1` | 0.01 | 16 | 24 | 167009 | 0.881118 | 0.584524 | 0.024417 | 0.53 / 0.64 | 0.569265 | 0.115317 | 0.64 | 0.660377 |
| 3 | `64x32x512` | `sigmoid1` | 0.003 | 16 | 25 | 281697 | 0.872363 | 0.578095 | 0.023728 | 0.52 / 0.62 | 0.575089 | 0.133651 | 0.62 | 0.641509 |
| 4 | `128x32x512` | `sigmoid1` | 0.003 | 32 | 9 | 545953 | 0.871975 | 0.576667 | 0.024365 | 0.53 / 0.63 | 0.570688 | 0.123413 | 0.63 | 0.704000 |
| 5 | `64x32x512` | `softmax2` | 0.003 | 16 | 11 | 282210 | 0.868193 | 0.578571 | 0.031965 | 0.51 / 0.69 | 0.580320 | 0.142222 | 0.69 | 0.715596 |
| 6 | `128x32x512` | `softmax2` | 0.003 | 32 | 12 | 546466 | 0.857485 | 0.574048 | 0.026100 | 0.51 / 0.63 | 0.556492 | 0.131032 | 0.63 | 0.660550 |

## Decisão principal

O modelo principal para continuidade é:

```text
32x64x512 + softmax2
lr=0.01
batch=16
seed=7
```

Ele é mantido como candidato principal porque obteve o melhor score robusto, maior acurácia média e menor gap médio entre os seis candidatos.

## Por que manter os seis

Os seis candidatos preservam interações interessantes entre arquitetura e cabeça de saída:

- `32x64x512` favoreceu `softmax2` de forma robusta.
- `64x32x512` teve melhor F1 médio e maior pico individual com `softmax2`, mas com variância maior; `sigmoid1` foi mais estável no score.
- `128x32x512` permaneceu competitivo com `sigmoid1`, enquanto `softmax2` piorou o score agregado.

Isso permite discutir não apenas a melhor configuração final, mas também como a formulação de saída altera estabilidade, F1, gap e comportamento por seed.

## Rodar no Go com DSA

```bash
DATASET=./dataset bash scripts/run_top6_go_dsa.sh
```

Parâmetros úteis:

```bash
DATASET=./dataset \
RUNS_ROOT=runs/go_top6_output_heads_dsa \
EPOCHS=500 \
THRESHOLDS=0,0.025,0.05,0.075,0.1 \
BENCH_REPEAT=300 \
bash scripts/run_top6_go_dsa.sh
```

Cada candidato gera:

```text
compare_validation.csv
compare_test.csv
bench_test.csv
```

A DSA exata (`threshold=0`) deve preservar a decisão densa. Thresholds positivos são aproximações e devem ser analisados como trade-off entre esparsidade, speedup estimado/real e degradação de métricas.
