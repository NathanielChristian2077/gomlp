# Busca de arquitetura com PyTorch e GPU

Este branch adiciona uma busca em PyTorch para usar GPU na procura de arquiteturas MLP mais estáveis. A busca usa apenas treino e validação. O split de teste não é avaliado por este script.

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
