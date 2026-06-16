# MLP Manual em Go para classificação de gatos e cachorros

Este repositório contém uma implementação manual de uma MLP para classificação binária de imagens de gatos e cachorros. A implementação foi feita em Go para deixar explícitos os principais passos matemáticos da rede neural: pré-processamento, propagação direta, funções de ativação, função de perda, retropropagação, atualização de pesos e avaliação.

A proposta desta etapa não é competir com uma CNN. O objetivo é construir uma baseline densa auditável para entender o funcionamento interno de uma rede neural totalmente conectada aplicada a imagens vetorizadas.

## Estrutura atual

```text
cmd/train/      Executa um treino individual
cmd/sweep/      Executa uma grade de experimentos
experiment/     Runner, configs, checkpoints e persistência de resultados
data/           Loader do dataset e pré-processamento das imagens
metrics/        Métricas, matriz de confusão e logger CSV
nn/             MLP, camadas densas, ativações, loss e treino
scripts/        Scripts auxiliares para reproduzir baselines
results/        Resultados históricos pequenos
runs/           Saídas organizadas de experimentos novos
```

## Dataset esperado

O loader espera a estrutura:

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

A baseline atual usa 500 imagens balanceadas:

```text
train:      300 imagens, 150 cat e 150 dog
validation: 100 imagens, 50 cat e 50 dog
test:       100 imagens, 50 cat e 50 dog
```

Rótulos:

```text
cat = 0.0
dog = 1.0
```

## Pré-processamento

Cada imagem é processada assim:

1. Decodificação JPEG ou PNG.
2. Redimensionamento para 64x64 pixels.
3. Conversão para escala de cinza.
4. Normalização dos pixels para o intervalo [0, 1].
5. Vetorização para `[]float64` de tamanho 4096.

A escala de cinza usa:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

A vetorização simplifica a implementação, mas remove relações espaciais locais importantes. Essa limitação é parte central da análise da MLP e justifica a comparação posterior com arquiteturas convolucionais.

## Arquitetura da MLP

A implementação aceita uma ou mais camadas ocultas. A baseline oficial original é:

```text
4096 -> 128 -> 1
```

Também é possível executar arquiteturas como:

```text
4096 -> 64 -> 1
4096 -> 256 -> 64 -> 1
4096 -> 512 -> 128 -> 32 -> 1
```

A camada oculta usa ReLU e a saída usa sigmoid. A loss é Binary Cross Entropy. O otimizador atual é gradient descent manual com mini-batch.

## Matemática implementada

Camada densa:

```text
z_j = b_j + soma_i(W_j,i * x_i)
```

ReLU:

```text
ReLU(x) = max(0, x)
```

Sigmoid:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

Binary Cross Entropy:

```text
BCE = -(y * log(yHat) + (1-y) * log(1-yHat))
```

Com sigmoid na saída e BCE, o delta da saída é:

```text
delta_saida = yHat - y
```

Os gradientes são acumulados por mini-batch e aplicados por gradient descent:

```text
W = W - lr * (dW / batch_size)
b = b - lr * (dB / batch_size)
```

## Como rodar

Teste sintético OR:

```bash
go run ./cmd/train
```

Treino individual com a baseline original:

```bash
go run ./cmd/train \
  --dataset ./dataset \
  --epochs 100 \
  --hidden 128 \
  --batch 16 \
  --lr 0.001 \
  --run-dir runs/manual_h128_b16_lr001_seed42 \
  --seed 42 \
  --log-every 10
```

Treino individual com mais de uma camada oculta:

```bash
go run ./cmd/train \
  --dataset ./dataset \
  --epochs 100 \
  --hidden 256x64 \
  --batch 16 \
  --lr 0.001 \
  --run-dir runs/manual_h256x64_b16_lr001_seed42 \
  --seed 42
```

Formatos aceitos em `--hidden`:

```text
128
256x64
512-128
1024:256:64
```

## Sweep de experimentos

O comando `cmd/sweep` executa várias combinações de arquitetura, batch size, learning rate e seed.

Exemplo pequeno:

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

Cada execução gera uma pasta própria dentro de `runs/`, contendo:

```text
config.json
summary.json
metrics.csv
confusion_matrix.csv
test_predictions.csv
checkpoints/best.gob
checkpoints/latest.gob
```

Se uma execução já tiver `summary.json` completo, o runner reutiliza o resultado e marca a execução como `cached`.

## Seleção de checkpoint

Durante o treino, a cada época o modelo é avaliado no conjunto de validação. O código clona o estado da MLP sempre que a validação melhora.

Critério de escolha:

1. Maior acurácia de validação.
2. Em empate, menor loss de validação.

O conjunto de teste é avaliado usando o melhor checkpoint de validação, não necessariamente a última época.

## Resultado da baseline densa original

A configuração original documentada foi:

```text
hidden = 128
batch = 16
lr = 0.001
epochs = 100
seeds = 1, 2, 3, 4, 5, 42
```

Resumo observado:

```text
Acurácia média de validação: aproximadamente 57,7%
Acurácia média de teste: aproximadamente 49,7%
F1 médio: aproximadamente 43,7%
Melhor test accuracy: 54%
Pior test accuracy: 44%
```

A interpretação é que a MLP aprende parte do conjunto de treino, mas não encontra uma representação robusta para generalizar bem em imagens vetorizadas.

## Resultado da bateria densa otimizada

Após a infraestrutura de sweeps, foram testadas arquiteturas e hiperparâmetros adicionais. Duas configurações se destacaram:

```text
4096 -> 64 -> 1
lr = 0.003
batch = 16

4096 -> 256 -> 64 -> 1
lr = 0.003
batch = 16
```

A configuração `4096 -> 64 -> 1` apresentou a melhor acurácia média em teste entre os experimentos densos finais, com aproximadamente 55,5%. A configuração `4096 -> 256 -> 64 -> 1` apresentou F1 médio superior em algumas baterias, mas com custo computacional maior.

## Discussão

A MLP manual valida o pipeline completo de classificação: leitura do dataset, pré-processamento, forward, backpropagation, atualização por mini-batch, checkpoints, avaliação e registro de métricas.

Os resultados também evidenciam a limitação da MLP densa em imagens vetorizadas. Ao transformar cada imagem em um vetor de 4096 valores, a rede perde relações espaciais locais que são essenciais para visão computacional. Por isso, os resultados ficam próximos do acaso em várias configurações, mesmo quando o treino reduz a loss.

## Conclusão

A baseline densa está pronta como implementação manual, auditável e independente. Ela cumpre o papel de etapa inicial do projeto: demonstrar o funcionamento interno de uma rede neural totalmente conectada e fornecer uma base justa para comparação posterior com CNN e transfer learning.