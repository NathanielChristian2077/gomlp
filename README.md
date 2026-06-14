# MLP Manual em Go para classificação de gatos e cachorros

Este repositório contém a implementação manual de uma MLP para classificação binária de imagens de gatos e cachorros. A implementação foi feita em Go para deixar explícitos os passos matemáticos da rede neural: pré-processamento, propagação direta, funções de ativação, função de perda, retropropagação, atualização de pesos e avaliação.

A proposta desta etapa do trabalho não é competir com uma CNN, mas construir uma base auditável para entender o funcionamento interno de uma rede neural totalmente conectada. As imagens são reduzidas para 64x64 em escala de cinza e depois vetorizadas em 4096 entradas. Essa escolha simplifica a implementação, mas também remove parte importante da estrutura espacial da imagem, o que limita a generalização do modelo.

## Estrutura atual

```text
cmd/train/      Executável principal de treino e avaliação
data/           Loader do dataset e pré-processamento das imagens
metrics/        Métricas, matriz de confusão e logger CSV
nn/             Implementação da MLP, camadas densas, ativações, loss e treino
scripts/        Scripts para repetir experimentos de baseline
results/        Saídas experimentais em CSV e logs
```

## Dataset esperado

O loader espera a seguinte estrutura de diretórios:

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

A versão usada na baseline possui 500 imagens no total, balanceadas entre 250 gatos e 250 cachorros:

```text
train:      300 imagens, 150 cat e 150 dog
validation: 100 imagens, 50 cat e 50 dog
test:       100 imagens, 50 cat e 50 dog
```

Os rótulos adotados são:

```text
cat = 0.0
dog = 1.0
```

## Pré-processamento

Cada imagem é processada da seguinte forma:

1. Decodificação da imagem JPEG ou PNG.
2. Redimensionamento para 64x64 pixels.
3. Conversão para escala de cinza usando combinação ponderada dos canais RGB.
4. Normalização dos pixels para o intervalo [0, 1].
5. Vetorização em um slice `[]float64` de tamanho 4096.

O redimensionamento atual usa amostragem por vizinho mais próximo. Isso foi escolhido por simplicidade e auditabilidade nesta primeira fase. Uma interpolação bilinear manual pode ser estudada depois, mas ainda não é necessária para fechar a baseline densa.

## Arquitetura da MLP baseline

A baseline densa oficial atual é:

```text
Entrada: 4096 valores
Camada oculta: 128 neurônios
Ativação oculta: ReLU
Saída: 1 neurônio
Ativação de saída: Sigmoid
Loss: Binary Cross Entropy
Otimizador: Gradient Descent manual com mini-batch
Batch size: 16
Learning rate: 0.001
Épocas: 100
Seeds avaliadas: 1, 2, 3, 4, 5 e 42
```

A camada densa usa pesos em layout `W[input][output]`, armazenados linearmente como:

```text
Weights[i*Out + o]
```

Essa organização favorece clareza e também prepara o código para a futura versão esparsa dinâmica, pois cada entrada ativa aponta para um bloco contíguo de pesos da próxima camada.

## Matemática implementada

A propagação direta de uma camada densa calcula:

```text
z_o = b_o + soma_i(x_i * W_i,o)
```

A camada oculta aplica ReLU:

```text
ReLU(x) = max(0, x)
```

A saída aplica sigmoid:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

Para classificação binária, a loss usada é Binary Cross Entropy:

```text
BCE = -(y * log(yHat) + (1-y) * log(1-yHat))
```

Na retropropagação, a combinação de sigmoid na saída com BCE permite a simplificação:

```text
delta_saida = yHat - y
```

A camada oculta recebe o erro ponderado pelos pesos da saída e multiplica pela derivada da ReLU.

## Como rodar

Teste sintético OR:

```bash
go run ./cmd/train
```

Treino no dataset real:

```bash
go run ./cmd/train --dataset ./dataset --epochs 100 --hidden 128 --batch 16 --lr 0.001 --log-every 10
```

Rodada oficial da baseline densa:

```bash
chmod +x scripts/run_dense_baseline_h128_b16_lr001.sh
./scripts/run_dense_baseline_h128_b16_lr001.sh ./dataset
```

O script cria a pasta:

```text
results/baseline_dense_h128_b16_lr001/
```

E salva:

```text
seed_1.csv, seed_1.log, ..., seed_42.csv, seed_42.log
summary.csv
```

## Seleção de checkpoint

Durante o treino, a cada época o modelo é avaliado no conjunto de validação. O código clona o estado da MLP sempre que a validação melhora. Ao final, o conjunto de teste é avaliado usando o melhor checkpoint de validação, e não necessariamente o modelo da última época.

O critério de escolha é:

1. Maior acurácia de validação.
2. Em caso de empate, menor loss de validação.

Isso reduz o risco de reportar uma época posterior já afetada por overfitting.

## Resultado da baseline densa

A execução oficial com 6 seeds mostrou forte instabilidade e desempenho médio próximo do acaso no teste. A tabela consolidada está em:

```text
results/baseline_dense_h128_b16_lr001/summary.csv
```

Resumo observado:

```text
Acurácia média de validação: aproximadamente 57,7%
Acurácia média de teste: aproximadamente 49,7%
F1 médio: aproximadamente 43,7%
Melhor test accuracy: 54%
Pior test accuracy: 44%
```

A interpretação é que a MLP aprende parte do conjunto de treino, mas não encontra uma representação robusta para generalizar bem em imagens vetorizadas. Isso é esperado: ao transformar a imagem em vetor, a rede perde relações espaciais locais importantes como bordas, texturas e formas. Essa limitação será importante para comparar a MLP com a CNN nas etapas seguintes do trabalho.

## Próximos passos planejados

1. Congelar e documentar a baseline densa.
2. Escrever a seção de análise experimental da MLP.
3. Implementar a versão DSA exact sparse, preservando a mesma função da rede densa.
4. Comparar MLP densa e DSA em tempo, loss, acurácia e sparsity.
5. Só depois estudar goroutines, paralelismo e possíveis experimentos CUDA.

A prioridade atual é manter a MLP correta, mensurável e auditável antes de adicionar otimizações.
