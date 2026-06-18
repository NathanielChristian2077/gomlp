# Super sweep de arquiteturas MLP

## Objetivo

Este branch adiciona um experimento exaustivo para procurar configurações de MLP mais estáveis, usando somente treino e validação. O conjunto de teste é carregado pelo loader padrão do projeto, mas não é avaliado pelo comando.

A intenção é evitar escolher uma arquitetura apenas por sorte de seed ou por memorização. Por isso, o comando testa muitas combinações de arquitetura, learning rate, batch size e seed, parando automaticamente quando o aprendizado deixa de avançar.

## Comando

```bash
go run ./cmd/super-sweep \
  --dataset ./dataset \
  --workers 4 \
  --runs runs/super_sweep_v1
```

Os únicos parâmetros operacionais previstos são:

```text
--workers  número de experimentos simultâneos
--runs     diretório raiz de saída
--dataset  caminho do dataset, mantido como opção prática, não como hiperparâmetro
```

## Grade fixa

Camadas ocultas permitidas:

```text
16, 32, 64, 128, 256, 512
```

Profundidade:

```text
1, 2 ou 3 camadas ocultas
```

Total de arquiteturas:

```text
6 + 6^2 + 6^3 = 258
```

Learning rates:

```text
0, 0.0001, 0.0003, 0.001, 0.003, 0.01
```

Batch sizes:

```text
0, 16, 32, 64
```

No batch size, `0` significa full-batch, ou seja, o batch efetivo vira o tamanho completo do split de treino.

Seeds:

```text
1 até 42
```

Total de configurações:

```text
258 arquiteturas * 6 learning rates * 4 batch sizes * 42 seeds = 260064 execuções
```

Sim, é absurdo. Essa é a parte em que o computador começa a reconsiderar suas escolhas de vida.

## Critério de parada

Não há um número de épocas escolhido como alvo experimental. Existe apenas uma trava de segurança:

```text
max_epochs = 500
```

Cada execução pode parar antes por:

```text
low_learning              treino e validação quase não melhoram na janela recente
validation_patience       validação não melhora por tempo suficiente
divergent_or_non_finite   loss não finita ou alta demais
max_epochs                atingiu a trava de segurança
```

Constantes usadas:

```text
min_epochs = 30
patience = 35
low_learning_window = 15
min_validation_delta = 1e-4
low_learning_delta = 1e-4
divergent_loss_limit = 10.0
```

## Saídas

Cada execução cria uma pasta própria com:

```text
config.json
summary.json
metrics.csv
checkpoints/best.gob
```

O arquivo geral fica em:

```text
<runs>/summary.csv
```

Ele inclui:

```text
arquitetura
profundidade
número de parâmetros
seed
learning rate
batch size
batch efetivo
épocas executadas
motivo da parada
melhor época
métricas de treino na melhor validação
métricas de validação
lacuna treino-validação
diretório da execução
erro, se houver
```

## Interpretação esperada

A seleção do modelo deve priorizar validação consistente, não apenas maior acurácia isolada.

Critérios recomendados:

1. Alta média de validação por arquitetura e hiperparâmetros.
2. Baixa variância entre seeds.
3. Gap moderado entre treino e validação.
4. F1 de validação coerente com accuracy.
5. Ausência de parada precoce por divergência.
6. Número de parâmetros compatível com o ganho observado.

A análise posterior deve agrupar `summary.csv` por arquitetura, learning rate e batch size, usando média, desvio padrão, melhor caso e pior caso entre seeds.
