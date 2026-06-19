# Comparação de cabeças de saída

Este fluxo compara as duas formulações válidas para classificação binária gato/cachorro nos três melhores candidatos MLP encontrados na busca GPU.

## Formulações testadas

| Modo | Saída | Loss | Predição | Uso |
|---|---:|---|---|---|
| `sigmoid1` | 1 logit | `BCEWithLogitsLoss` | `sigmoid(logit) >= 0.5` | Formulação binária atual |
| `softmax2` | 2 logits | `CrossEntropyLoss` | `argmax(logits)` | Formulação multiclasse gato/cachorro |

A opção `softmax1` não existe porque softmax com um único neurônio é degenerado: a probabilidade sempre seria 1. Para duas classes mutuamente exclusivas, softmax precisa de dois logits.

Também não testamos `sigmoid2` por padrão, porque dois sigmoides independentes são mais adequados para multi-label. Aqui as classes são mutuamente exclusivas: ou gato, ou cachorro.

## Candidatos padrão

```text
32x64x512,   lr=0.01,  batch=16
64x32x512,   lr=0.003, batch=16
128x32x512,  lr=0.003, batch=32
```

## Rodar

```bash
python scripts/gpu_output_head_tournament.py \
  --dataset ./dataset \
  --runs runs/output_head_tournament_v2 \
  --device auto \
  --workers 12 \
  --seeds 1-42 \
  --max-epochs 500
```

Saídas:

```text
runs/output_head_tournament_v2/summary.csv
runs/output_head_tournament_v2/ranking.csv
runs/output_head_tournament_v2/best_head.csv
```

## Interpretação

A comparação deve ser feita por arquitetura e por modo de saída, usando média de validação, F1 médio, desvio entre seeds, pior seed e gap treino-validação. O teste deve continuar intocado até a formulação de saída ser escolhida.

Se `softmax2` vencer de forma robusta, a próxima etapa é portar a cabeça de saída para o Go: duas saídas, cross-entropy com logits e predição por `argmax`. Se `sigmoid1` vencer ou empatar, a implementação atual continua defensável para classificação binária.
