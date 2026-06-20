# Índice de documentação

A documentação é organizada por escopo, não por ordem acidental de criação.

```text
docs/
  foundation/    baseline estável e fundamentos do projeto
  research/      relatórios de extensões experimentais
  optimization/  decisões e contratos da trilha GOptimize
  development/   convenções de repositório e reprodução
```

## Convenções

- Documentação fundacional acompanha `main`.
- Documentação de DSA acompanha `proto-DSAP`.
- Documentação de kernels, precisão e otimizadores acompanha `GOptimize`.
- Resultados brutos, checkpoints, CSVs extensos e logs não são documentação de código e não devem ser versionados em `runs/`.
- Resumos curados devem registrar configuração, split, seed, métrica e comando de reprodução.
