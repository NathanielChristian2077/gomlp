# Extensão experimental: Dynamic Sparse Activation

## Objetivo

Esta documentação descreve a extensão Dynamic Sparse Activation, DSA, implementada sobre a MLP manual em Go. A proposta é avaliar se a esparsidade natural produzida pela ReLU pode ser explorada durante a inferência para reduzir operações sem alterar, ou alterando de forma controlada, o comportamento da MLP densa.

A extensão não substitui a apresentação principal da baseline densa. Ela funciona como uma investigação adicional: primeiro validando equivalência matemática e depois medindo tempo, esparsidade e impacto de thresholds aproximados.

## Ideia central

Em uma MLP com ReLU, parte das ativações ocultas se torna exatamente zero:

```text
ReLU(x) = max(0, x)
```

Quando uma ativação é zero, sua contribuição para a camada seguinte também é zero. Na propagação densa, a camada seguinte ainda percorre essa entrada e realiza multiplicações e somas que não alteram o resultado.

A DSA transforma a saída ReLU em uma representação compacta chamada `ActiveVector`:

```text
ActiveVector:
  Size    -> tamanho original da camada
  Indices -> índices originais dos neurônios ativos
  Values  -> valores das ativações ativas
```

A próxima camada então calcula apenas as contribuições dos neurônios ativos:

```text
z_o = b_o + soma_{j ativo}(a_j * W_j,o)
```

## DSA exact

A DSA exact usa `threshold = 0`. Ela remove apenas ativações exatamente nulas. Por isso, é matematicamente equivalente à MLP densa:

```text
soma_i(a_i * W_i,o) = soma_{i ativo}(a_i * W_i,o), quando a_i = 0 para os inativos
```

O objetivo da DSA exact é preservar a saída da rede densa e medir se essa equivalência pode ser explorada computacionalmente.

## DSA threshold

A DSA threshold usa `threshold > 0`. Nesse caso, ativações positivas pequenas também são removidas:

```text
a_i é mantido somente se a_i > threshold
```

Essa versão não é matematicamente equivalente. Ela altera a função da rede e deve ser analisada separadamente, como uma forma de poda dinâmica aproximada em inferência.

## Implementação

A implementação foi organizada em três níveis:

1. `Forward` mantém a propagação densa tradicional.
2. `ForwardSparseWithStats` executa DSA e coleta métricas de esparsidade e operações.
3. `ForwardSparseFast` executa DSA com buffers reutilizáveis para benchmark puro de inferência.

O benchmark usa `SparseForwardWorkspace` para evitar alocações repetidas de `ActiveVector` durante o loop de inferência.

## Comandos usados

Comparação de métricas:

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

## Métricas registradas

O `cmd/compare` registra:

```text
loss
accuracy
precision
recall
f1
matriz de confusão
active_total
activation_slots_total
avg_active_ratio
avg_sparsity
avg_active_by_layer
max_abs_diff_from_dense
mismatch_count_from_dense
```

O `cmd/bench` registra:

```text
ns_per_forward
forwards_per_second
dense_ops_per_pass
sparse_ops_per_pass
estimated_speedup
ops_saved_ratio
avg_sparsity
checksum
```

## Resultados da DSA exact

Resultados no split de teste, com `lr = 0.003`, `batch = 16`, `epochs = 200`, `seed = 42`, `repeat = 500`, `warmup = 50`, `gomaxprocs = 1`.

| Arquitetura | Best epoch | Acc | F1 | Sparsity exact | Ops salvas | Dense ns/forward | Sparse exact ns/forward | Ganho real |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `64` | 161 | 0.56 | 0.6563 | 53.36% | 0.013% | 251182 | 232240 | 7.54% |
| `256x64` | 88 | 0.59 | 0.6435 | 44.86% | 0.733% | 950254 | 868090 | 8.65% |
| `128x256x128` | 53 | 0.60 | 0.6552 | 47.78% | 5.34% | 526198 | 471552 | 10.38% |
| `256x256x128` | 2 | 0.53 | 0.6240 | 50.21% | 4.35% | 1000383 | 901491 | 9.89% |
| `512x512x128` | 41 | 0.55 | 0.2373 | 47.39% | 6.45% | 2095225 | 1838661 | 12.25% |

Em todas as arquiteturas, a DSA exact teve:

```text
mismatch_count_from_dense = 0
max_abs_diff_from_dense = 0
```

Isso valida a equivalência exata entre o forward denso e o forward esparso exact.

## Interpretação por arquitetura

### 64

A arquitetura `4096 -> 64 -> 1` apresentou alta esparsidade, mas economia estimada quase nula. Isso ocorre porque há apenas uma camada oculta, então a DSA só reduz trabalho de forma relevante na camada de saída, que é pequena.

Mesmo assim, houve ganho real de tempo no benchmark. Esse ganho deve ser interpretado com cautela, pois provavelmente mistura efeito da DSA com diferenças de implementação entre o caminho denso e o caminho sparse fast.

### 256x64

A arquitetura `4096 -> 256 -> 64 -> 1` foi o melhor caso para observar thresholds. A DSA exact preservou as métricas e removeu 44,86% das ativações ocultas.

O threshold `0.05` manteve as mesmas predições e aumentou a esparsidade. O threshold `0.10` melhorou accuracy e F1, mas alterou quatro predições e aumentou a loss.

### 128x256x128

A arquitetura `4096 -> 128 -> 256 -> 128 -> 1` apresentou o melhor equilíbrio entre qualidade e desempenho. Ela alcançou 0.60 de accuracy, 0.6552 de F1, 47,78% de esparsidade e 10,38% de ganho real de tempo com DSA exact.

Essa arquitetura é a candidata mais equilibrada para discussão de DSA no contexto do trabalho.

### 256x256x128

A arquitetura `4096 -> 256 -> 256 -> 128 -> 1` teve boa esparsidade e ganho real de tempo, mas qualidade inferior. O best epoch foi 2, sugerindo que a configuração pode ter começado a degradar cedo no treino.

### 512x512x128

A arquitetura `4096 -> 512 -> 512 -> 128 -> 1` apresentou o maior ganho bruto de tempo com DSA exact, 12,25%. Porém, o F1 foi baixo, com recall muito reduzido. Por isso, ela é mais útil como stress test computacional do que como melhor classificador.

## Resultados de thresholds na arquitetura 256x64

| Threshold | Loss | Acc | Precision | Recall | F1 | Sparsity | Mismatch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` exact | 0.7291 | 0.59 | 0.5692 | 0.74 | 0.6435 | 44.86% | 0 |
| `0.05` | 0.7301 | 0.59 | 0.5692 | 0.74 | 0.6435 | 47.44% | 0 |
| `0.10` | 0.7355 | 0.61 | 0.5873 | 0.74 | 0.6549 | 50.02% | 4 |
| `0.25` | 0.7523 | 0.58 | 0.5645 | 0.70 | 0.6250 | 57.76% | 3 |
| `0.50` | 0.7823 | 0.58 | 0.5645 | 0.70 | 0.6250 | 70.05% | 5 |

## Conclusões

1. A DSA exact preservou integralmente a função da MLP densa em todas as arquiteturas testadas.
2. A esparsidade induzida pela ReLU foi significativa, ficando entre 44% e 53% nos modelos avaliados.
3. O ganho real de tempo aumentou nas arquiteturas com mais custo em camadas internas.
4. A arquitetura `128x256x128` apresentou o melhor equilíbrio entre qualidade e desempenho.
5. Thresholds positivos devem ser tratados como aproximação, não como equivalência.
6. O threshold `0.05` é conservador; o threshold `0.10` é experimentalmente interessante na `256x64`, mas altera predições.

## Limitações

Os ganhos de tempo devem ser interpretados em relação à implementação manual densa deste projeto. Implementações industriais baseadas em BLAS, GEMM, SIMD ou GPU podem apresentar desempenho absoluto superior para o caminho denso, especialmente quando processam batches como multiplicações matriz-matriz.

Assim, a DSA neste branch demonstra uma propriedade técnica real da MLP manual, mas não deve ser apresentada como superior a implementações profissionais otimizadas sem benchmarks adicionais.

## Próximos passos sugeridos

1. Comparar a MLP manual com uma implementação densa batched.
2. Criar uma baseline externa usando Gonum, BLAS ou PyTorch para referência de desempenho.
3. Avaliar DSA em múltiplas seeds e splits.
4. Testar DSA em arquiteturas onde camadas internas representem maior fração do custo.
5. Investigar uma implementação HPC em C/C++ e CUDA.
