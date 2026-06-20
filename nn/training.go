package nn

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/NathanielChristian2077/gomlp/metrics"
)

// DefaultClassificationThreshold converte a saída sigmoid em classe binária.
const DefaultClassificationThreshold = 0.5

// Sample representa uma amostra supervisionada da MLP.
// X é o vetor de entrada e Y é o rótulo: 0 para gato, 1 para cachorro.
type Sample struct {
	X []float64
	Y float64
}

// EpochResult concentra as métricas produzidas por treino ou avaliação.
type EpochResult struct {
	Loss      float64
	Accuracy  float64
	Confusion metrics.ConfusionMatrix
	Duration  time.Duration
}

// BatchOptimizer descreve um otimizador com estado que atualiza uma MLP após um batch.
// Implementações concretas vivem fora do pacote nn para não misturar arquitetura e estratégia de atualização.
type BatchOptimizer interface {
	Model() *MLP
	Step(batchSize int)
}

// TrainEpoch executa uma época em full-batch com SGD.
func TrainEpoch(model *MLP, samples []Sample, lr float64) (EpochResult, error) {
	return TrainEpochMiniBatch(model, samples, lr, len(samples), nil)
}

// TrainEpochMiniBatch executa treino SGD por mini-batch com shuffle opcional.
func TrainEpochMiniBatch(model *MLP, samples []Sample, lr float64, batchSize int, rng *rand.Rand) (EpochResult, error) {
	return trainEpochMiniBatch(model, samples, batchSize, rng, func(actualBatchSize int) {
		model.ApplyGrad(lr, actualBatchSize)
	})
}

// TrainEpochMiniBatchWithOptimizer executa treino com um otimizador explícito.
// O otimizador deve ser criado uma vez e reutilizado entre batches e épocas.
func TrainEpochMiniBatchWithOptimizer(optimizer BatchOptimizer, samples []Sample, batchSize int, rng *rand.Rand) (EpochResult, error) {
	if optimizer == nil {
		return EpochResult{}, fmt.Errorf("nil optimizer")
	}
	return trainEpochMiniBatch(optimizer.Model(), samples, batchSize, rng, optimizer.Step)
}

func trainEpochMiniBatch(model *MLP, samples []Sample, batchSize int, rng *rand.Rand, step func(batchSize int)) (EpochResult, error) {
	if err := validateTrainingInput(model, samples); err != nil {
		return EpochResult{}, err
	}
	if batchSize <= 0 {
		return EpochResult{}, fmt.Errorf("invalid batch size: %d", batchSize)
	}
	if step == nil {
		return EpochResult{}, fmt.Errorf("nil optimizer step")
	}

	startedAt := time.Now()
	work := samples
	if rng != nil {
		work = append([]Sample(nil), samples...)
		rng.Shuffle(len(work), func(i, j int) { work[i], work[j] = work[j], work[i] })
	}

	for start := 0; start < len(work); start += batchSize {
		end := start + batchSize
		if end > len(work) {
			end = len(work)
		}

		batch := work[start:end]
		model.ZeroGrad()
		for _, sample := range batch {
			yHat := model.Forward(sample.X)
			model.Backward(sample.X, yHat, sample.Y)
		}
		step(len(batch))
	}

	result, err := Evaluate(model, samples)
	if err != nil {
		return EpochResult{}, err
	}
	result.Duration = time.Since(startedAt)
	return result, nil
}

// Evaluate calcula métricas sem alterar os pesos da rede.
func Evaluate(model *MLP, samples []Sample) (EpochResult, error) {
	if err := validateTrainingInput(model, samples); err != nil {
		return EpochResult{}, err
	}

	startedAt := time.Now()
	loss := 0.0
	confusion := metrics.NewConfusionMatrix()
	for _, sample := range samples {
		yHat := model.Forward(sample.X)
		loss += BinaryCrossEntropy(yHat, sample.Y)
		confusion.Add(yHat, sample.Y, DefaultClassificationThreshold)
	}

	return EpochResult{
		Loss:      loss / float64(len(samples)),
		Accuracy:  confusion.Accuracy(),
		Confusion: confusion,
		Duration:  time.Since(startedAt),
	}, nil
}

func validateTrainingInput(model *MLP, samples []Sample) error {
	if model == nil {
		return fmt.Errorf("nil model")
	}
	if len(model.Hidden) == 0 {
		return fmt.Errorf("model has no hidden layers")
	}
	if len(samples) == 0 {
		return fmt.Errorf("empty sample set")
	}

	expectedInputSize := model.InputSize()
	for i, sample := range samples {
		if len(sample.X) != expectedInputSize {
			return fmt.Errorf("sample %d has invalid input length: expected %d, got %d", i, expectedInputSize, len(sample.X))
		}
		if sample.Y != 0 && sample.Y != 1 {
			return fmt.Errorf("sample %d has invalid label: expected 0 or 1, got %f", i, sample.Y)
		}
	}
	return nil
}
