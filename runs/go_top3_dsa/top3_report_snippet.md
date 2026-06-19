# Resultados Top-3 MLP Go + DSA

## Candidatos congelados

| Rank | Arquitetura | LR | Batch | Seed | Score GPU | Val acc média GPU | Val F1 médio GPU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `32x64x512` | 0.01 | 16 | 24 | 0.881118 | 0.584524 | 0.569265 |
| 2 | `64x32x512` | 0.003 | 16 | 25 | 0.872363 | 0.578095 | 0.575089 |
| 3 | `128x32x512` | 0.003 | 32 | 9 | 0.871975 | 0.576667 | 0.570688 |

## Métricas densas no Go

| Rank | Arquitetura | Split | Acc | F1 | Loss | Matriz TN/FP/FN/TP |
|---:|---|---|---:|---:|---:|---|
| 1 | `32x64x512` | test | 0.5000 | 0.4318 | 0.7825 | 31/19/31/19 |
| 1 | `32x64x512` | validation | 0.5900 | 0.5393 | 0.7865 | 35/15/26/24 |
| 2 | `64x32x512` | test | 0.5000 | 0.2647 | 0.7342 | 41/9/41/9 |
| 2 | `64x32x512` | validation | 0.5400 | 0.4103 | 0.7433 | 38/12/34/16 |
| 3 | `128x32x512` | test | 0.5800 | 0.5532 | 0.6766 | 32/18/24/26 |
| 3 | `128x32x512` | validation | 0.6400 | 0.6538 | 0.6692 | 30/20/16/34 |

## DSA exacta, threshold = 0

| Rank | Split | Acc delta | F1 delta | Sparsity | Speedup estimado | Ops salvas | Mismatch | Max diff |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | test | 0.0000 | 0.0000 | 0.5167 | 1.1096 | 0.0987 | 0 | 0.0000 |
| 1 | validation | 0.0000 | 0.0000 | 0.5187 | 1.1115 | 0.1003 | 0 | 0.0000 |
| 2 | test | 0.0000 | 0.0000 | 0.5180 | 1.0399 | 0.0383 | 0 | 0.0000 |
| 2 | validation | 0.0000 | 0.0000 | 0.5208 | 1.0400 | 0.0385 | 0 | 0.0000 |
| 3 | test | 0.0000 | 0.0000 | 0.5047 | 1.0195 | 0.0191 | 0 | 0.0000 |
| 3 | validation | 0.0000 | 0.0000 | 0.5067 | 1.0196 | 0.0192 | 0 | 0.0000 |

## Melhor threshold por candidato no teste

| Rank | Threshold | Acc | F1 | Sparsity | Mismatch | Speedup estimado |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.5000 | 0.4318 | 0.5167 | 0 | 1.1096 |
| 2 | 0.0500 | 0.5100 | 0.2899 | 0.5439 | 1 | 1.0413 |
| 3 | 0.0000 | 0.5800 | 0.5532 | 0.5047 | 0 | 1.0195 |

## Benchmark de inferência, teste

| Rank | Modo | Threshold | ns/forward | Speedup real vs dense | Sparsity |
|---:|---|---:|---:|---:|---:|
| 1 | dense | 0.0000 | 148063.0204 | 1.0000 | 0.0000 |
| 1 | sparse_exact | 0.0000 | 109821.8914 | 1.3482 | 0.5167 |
| 2 | dense | 0.0000 | 267045.6347 | 1.0000 | 0.0000 |
| 2 | sparse_exact | 0.0000 | 208653.2213 | 1.2799 | 0.5180 |
| 3 | dense | 0.0000 | 485599.7842 | 1.0000 | 0.0000 |
| 3 | sparse_exact | 0.0000 | 382816.7589 | 1.2685 | 0.5047 |

## Observações para o relatório

A DSA exata usa threshold zero e, portanto, deve preservar as decisões do modelo denso. A evidência esperada é `mismatch_count_from_dense = 0` e `max_abs_diff_from_dense` próximo de zero. Thresholds maiores que zero passam a ser aproximações: podem aumentar a esparsidade, mas precisam ser discutidos separadamente porque podem alterar predições.
