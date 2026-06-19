# Comparação de cabeças de saída

Este fluxo compara formulações de saída para os três melhores candidatos MLP encontrados na busca GPU.

## Formulações testadas

| Modo | Saída | Loss | Uso |
|---|---:|---|---|
| `sigmoid1` | 1 logit | `BCEWithLogitsLoss` | Formulação binária atual |
| `softmax2` | 2 logits | `CrossEntropyLoss` | Formulação multiclasse gato/cachorro |
| `sigmoid2` | 2 logits independentes | BCE one-hot | Diagnóstico; não é a opção preferida para classes mutuamente exclusivas |

A opção `softmax1` não existe no script porque softmax com um único neurônio é degenerado: a probabilidade sempre seria 1. Para duas classes mutuamente exclusivas, softmax precisa de dois logits.

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
  --runs runs/output_head_tournament_v1 \
  --device auto \
  --workers 12 \
  --seeds 1-42 \
  --max-epochs 500
```

Saídas:

```text
runs/output_head_tournament_v1/summary.csv
runs/output_head_tournament_v1/ranking.csv
runs/output_head_tournament_v1/best_head.csv
```

## Interpretação

A comparação deve ser feita por grupo e por modo de saída, usando média de validação, F1 médio, desvio entre seeds, pior seed e gap treino-validação. O teste deve continuar intocado até a formulação de saída ser escolhida.

Se `softmax2` vencer de forma robusta, a próxima etapa é portar a cabeça de saída para o Go: duas saídas, cross-entropy com logits e predição por `argmax`. Se `sigmoid1` vencer ou empatar, a implementação atual continua defensável para classificação binária.
