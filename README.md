# GoMLP

Implementação manual, auditável e experimental de uma MLP para classificação binária de imagens de gatos e cachorros.

A baseline não tenta competir com CNNs. Ela existe para expor, sem abstrações de framework, o pipeline de leitura de dados, forward, loss, backpropagation, atualização de parâmetros, validação, checkpoints e métricas.

## Branches

| Branch | Papel | Regra |
|---|---|---|
| `main` | baseline densa estável | recebe apenas funcionalidades consolidadas e documentação fundacional |
| `proto-DSAP` | experimento Dynamic Sparse Activation | deriva de `main`; contém DSA, comparação e benchmark, sem artefatos gerados |
| `GOptimize` | trilha de otimização | deriva de `main`; contém kernels, precisão reduzida e otimizadores, sem scripts ou resultados de busca GPU |
| `gpu-search-pytorch` | investigação de candidatos com GPU | histórico experimental, separado da baseline de produção |

## Estrutura

```text
cmd/              executáveis de linha de comando
data/             leitura, normalização e organização do dataset
experiment/       configurações, runner, checkpoint e persistência
metrics/          métricas, matriz de confusão e logger
nn/               arquitetura da MLP, camadas e treino
internal/tensor/  primitivas matemáticas manuais de matriz e operações elementares
internal/         demais detalhes privados de implementação e kernels
scripts/          automações reprodutíveis de pesquisa
docs/             documentação organizada por escopo
```

`internal/tensor` é mantido como base matemática manual do projeto. Mesmo quando a MLP usa loops especializados para a operação matriz-vetor, essas primitivas preservam a implementação explícita de produto matricial, transposição, operações elemento a elemento e aplicação de funções.

Os testes unitários ficam próximos do pacote que testam, conforme a convenção de Go. Isso permite validar invariantes internos sem transformar a API pública em um vazamento de detalhes de implementação. Testes de integração, quando necessários, pertencem a `test/integration`.

## Dataset esperado

```text
dataset/
  train/
    cat/
    dog/
  validation/
    cat/
    dog/
  test/
    cat/
    dog/
```

O dataset fornecido para o projeto é parte da distribuição de trabalho e permanece versionável. A configuração usada possui 500 imagens balanceadas: 300 para treino, 100 para validação e 100 para teste. As imagens são convertidas para escala de cinza, redimensionadas para 64x64 e vetorizadas em 4096 entradas.

## Comandos principais

Teste sintético:

```bash
go test ./...
go run ./cmd/train
```

Treino individual:

```bash
go run ./cmd/train \
  --dataset ./dataset \
  --epochs 100 \
  --hidden 128 \
  --batch 16 \
  --lr 0.001 \
  --seed 42 \
  --run-dir runs/manual_h128_b16_lr001_seed42
```

Sweep pequeno:

```bash
go run ./cmd/sweep \
  --dataset ./dataset \
  --epochs 100 \
  --hidden '128;256x64;512x128' \
  --batch '16,32' \
  --lr '0.001,0.0003' \
  --seeds '1,2,3,4,5,42' \
  --workers 1 \
  --runs runs/dense_sweep_v1
```

## Higiene de resultados

`runs/` contém artefatos transitórios de execução e permanece ignorado pelo Git. Dataset, operações matemáticas manuais e resultados curados não são descartados automaticamente: eles fazem parte da rastreabilidade acadêmica do projeto.

Consulte [docs/README.md](docs/README.md) para o índice de documentação.
