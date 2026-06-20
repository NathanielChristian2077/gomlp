# Dynamic Sparse Activation

## Escopo

Esta branch contém exclusivamente a extensão DSA sobre a baseline de `main`. Ela não versiona checkpoints, logs, CSVs de execução ou datasets. Esses artefatos são gerados localmente em `runs/` e permanecem ignorados pelo Git.

## Ideia

Após uma ReLU, ativações nulas não contribuem para a camada seguinte. A DSA representa apenas ativações positivas em um `ActiveVector` com índices e valores, e usa essa representação para calcular a próxima camada.

```text
z_o = b_o + soma_{j ativo}(a_j * W_j,o)
```

## Modos

- `threshold=0`: DSA exact. Remove apenas ativações exatamente nulas e deve preservar a decisão do forward denso.
- `threshold>0`: DSA aproximada. Remove ativações pequenas adicionais; mede trade-off entre esparsidade, velocidade e alteração de predição.

## Comandos

Comparação de equivalência e métricas:

```bash
go run ./cmd/compare \
  --dataset ./dataset \
  --hidden 256x64 \
  --epochs 200 \
  --batch 16 \
  --lr 0.003 \
  --seed 42 \
  --thresholds '0,0.05,0.1' \
  --runs runs/dsa_compare
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
  --thresholds '0,0.05,0.1' \
  --repeat 500 \
  --warmup 50 \
  --gomaxprocs 1 \
  --runs runs/dsa_bench
```

## Critérios

Para DSA exact, registrar pelo menos:

```text
mismatch_count_from_dense = 0
max_abs_diff_from_dense ≈ 0
```

Thresholds positivos são experimentais e nunca devem ser usados para afirmar equivalência exata.
