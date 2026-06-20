# Adam

A implementação Adam vive em `optimizer/`, fora de `nn/`. A MLP fornece apenas o contrato `nn.BatchOptimizer`; estratégias de atualização não pertencem à arquitetura do modelo.

A primeira avaliação deve comparar SGD e Adam mantendo arquitetura, seed, batch, split e regra de early stopping constantes. Para Adam, iniciar a busca de learning rate em `1e-4`, `3e-4`, `1e-3` e `3e-3`. Não reutilizar automaticamente taxas de SGD.

O estado de Adam deve persistir durante toda a execução. Checkpoints para retomada de treino futuramente devem serializar momentos e contador de passos.
