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

## Resultado final da comparação

| Rank | Arquitetura | LR | Batch | Cabeça | Parâmetros | Score | Acc média | Acc std | Acc min/max | F1 médio | Gap abs médio | Melhor seed | Melhor acc | Melhor F1 |
|---:|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | `32x64x512` | 0.01 | 16 | `softmax2` | 167522 | 0.895363 | 0.589048 | 0.023177 | 0.54 / 0.64 | 0.574793 | 0.093968 | 7 | 0.64 | 0.600000 |
| 2 | `32x64x512` | 0.01 | 16 | `sigmoid1` | 167009 | 0.881118 | 0.584524 | 0.024417 | 0.53 / 0.64 | 0.569265 | 0.115317 | 24 | 0.64 | 0.660377 |
| 3 | `64x32x512` | 0.003 | 16 | `sigmoid1` | 281697 | 0.872363 | 0.578095 | 0.023728 | 0.52 / 0.62 | 0.575089 | 0.133651 | 25 | 0.62 | 0.641509 |
| 4 | `128x32x512` | 0.003 | 32 | `sigmoid1` | 545953 | 0.871975 | 0.576667 | 0.024365 | 0.53 / 0.63 | 0.570688 | 0.123413 | 9 | 0.63 | 0.704000 |
| 5 | `64x32x512` | 0.003 | 16 | `softmax2` | 282210 | 0.868193 | 0.578571 | 0.031965 | 0.51 / 0.69 | 0.580320 | 0.142222 | 11 | 0.69 | 0.715596 |
| 6 | `128x32x512` | 0.003 | 32 | `softmax2` | 546466 | 0.857485 | 0.574048 | 0.026100 | 0.51 / 0.63 | 0.556492 | 0.131032 | 12 | 0.63 | 0.660550 |

## Decisão

A configuração final escolhida para continuidade é:

```text
arquitetura: 32x64x512
cabeça: softmax2
saída: 2 logits
loss: CrossEntropyLoss
predição: argmax(logits)
learning rate: 0.01
batch size: 16
seed canônica: 7
```

A escolha é justificada porque `32x64x512 + softmax2` obteve o maior score robusto, maior acurácia média, menor gap médio que sua versão `sigmoid1` e manteve desvio baixo entre seeds. Embora a melhor seed `sigmoid1` tenha F1 individual maior, a decisão final prioriza robustez agregada entre 42 seeds.

## Interpretação

A comparação deve ser feita por arquitetura e por modo de saída, usando média de validação, F1 médio, desvio entre seeds, pior seed e gap treino-validação. O teste deve continuar intocado até a formulação de saída ser escolhida.

Como `softmax2` venceu de forma robusta no candidato principal, a próxima etapa é portar a cabeça de saída para o Go: duas saídas, cross-entropy com logits e predição por `argmax`.
