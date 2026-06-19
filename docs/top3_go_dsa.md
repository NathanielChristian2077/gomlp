# Top-3 MLP no Go e DSA

Este fluxo congela os três melhores candidatos individuais encontrados no torneio em PyTorch/GPU e os testa na implementação manual em Go, incluindo comparação densa vs DSA e benchmark de inferência.

## Candidatos congelados

| Rank | Arquitetura | LR | Batch | Seed | Score GPU | Val acc média GPU | Val F1 médio GPU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `32x64x512` | 0.01 | 16 | 24 | 0.881118 | 0.584524 | 0.569265 |
| 2 | `64x32x512` | 0.003 | 16 | 25 | 0.872363 | 0.578095 | 0.575089 |
| 3 | `128x32x512` | 0.003 | 32 | 9 | 0.871975 | 0.576667 | 0.570688 |

A escolha é pelo ranking robusto de validação, não pelo melhor pico isolado de uma seed.

## Rodar tudo

```bash
DATASET=./dataset bash scripts/run_top3_go_dsa.sh
```

Parâmetros úteis:

```bash
DATASET=./dataset \
RUNS_ROOT=runs/go_top3_dsa \
EPOCHS=500 \
THRESHOLDS=0,0.025,0.05,0.075,0.1 \
BENCH_REPEAT=300 \
BENCH_WARMUP=20 \
bash scripts/run_top3_go_dsa.sh
```

O script treina/carrega os modelos pelo pipeline de experimentos do Go. A primeira comparação de cada candidato cria o checkpoint `best.gob`; as chamadas seguintes reutilizam o mesmo experimento pelo cache.

## Saídas

Arquivos principais:

```text
runs/go_top3_dsa/top3_manifest.csv
runs/go_top3_dsa/top3_compare_combined.csv
runs/go_top3_dsa/top3_bench_combined.csv
runs/go_top3_dsa/top3_dense_summary.csv
runs/go_top3_dsa/top3_dsa_summary.csv
runs/go_top3_dsa/top3_bench_summary.csv
runs/go_top3_dsa/top3_report_snippet.md
```

Por candidato, também serão gerados:

```text
runs/go_top3_dsa/<label>/compare_validation.csv
runs/go_top3_dsa/<label>/compare_test.csv
runs/go_top3_dsa/<label>/bench_test.csv
```

Os checkpoints e summaries do treino ficam em:

```text
runs/go_top3_dsa/go_runs/<run_id>_<name>/
```

## Interpretação

A linha `dense` é a MLP convencional. A linha `sparse_exact` com `threshold = 0` é a DSA exata: ela deve preservar a saída/predição do modelo denso. Para o relatório, isso deve ser verificado por:

```text
mismatch_count_from_dense = 0
max_abs_diff_from_dense próximo de 0
```

Linhas `sparse_threshold` usam threshold maior que zero. Elas são aproximações: podem aumentar esparsidade e operações economizadas, mas podem alterar predições. No relatório, devem ser discutidas como variação experimental separada, não como equivalência exata.

## Ordem recomendada para o relatório

1. Descrever que a busca GPU foi usada apenas para seleção de candidatos com base em validação.
2. Congelar o top-3 antes de olhar o teste.
3. Reexecutar os três candidatos na implementação manual em Go.
4. Avaliar dense vs DSA no split de validação e no teste.
5. Usar DSA exata como evidência de equivalência funcional.
6. Usar thresholds maiores como estudo de trade-off entre esparsidade e degradação de métricas.
7. Usar benchmark para separar economia teórica de operações e speedup real medido.
