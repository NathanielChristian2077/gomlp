# Organização do repositório

## Responsabilidades por diretório

- `cmd/`: executáveis. Um comando coordena dependências; regras matemáticas e lógica de modelo não devem nascer aqui.
- `data/`: leitura do dataset e pré-processamento.
- `experiment/`: ciclo de execução, configuração, checkpoints e persistência.
- `metrics/`: métricas e serialização associada a métricas.
- `nn/`: modelo, camadas, forward, backpropagation e contratos de treino.
- `internal/`: detalhes privados, kernels específicos de arquitetura e utilitários que não são API do modelo.
- `docs/`: documentação curada.
- `scripts/`: automações reproduzíveis que combinam comandos existentes.

## Testes

Em Go, testes unitários permanecem no mesmo diretório do pacote para testar detalhes internos sem ampliar a API pública. Arquivos devem usar o sufixo `_test.go` e focar uma responsabilidade por arquivo. Testes integrados de comandos ou fluxos completos pertencem a `test/integration` quando forem introduzidos.

## Artefatos gerados

Não versionar datasets locais, checkpoints, logs, CSVs de execução nem diretórios `runs/` e `results/`. O código deve gerar esses artefatos de forma reprodutível; a documentação deve guardar apenas tabelas e conclusões curadas.

## Política de branches

Cada branch experimental deve nascer da ponta atual de `main`. Ela pode trazer somente o código necessário ao seu experimento e não deve carregar scripts, resultados ou documentação de experimentos não relacionados. Antes de promover código para `main`, remova artefatos gerados e revalide `go test ./...`.
