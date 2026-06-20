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

// TrainEpoch executa uma época full-batch usando o otimizador SGD padrão.
func TrainEpoch(model *MLP, samples []Sample, learningRate float64) (EpochResult, error) {
	return TrainEpochMiniBatch(model, samples, learningRate, len(samples), nil)
}

// TrainEpochMiniBatch executa treino SGD por mini-batch com shuffle opcional.
// A função preserva a API da baseline e delega a atualização para SGDOptimizer.
func TrainEpochMiniBatch(model *MLP, samples []Sample, learningRate float64, batchSize int, rng *rand.Rand) (EpochResult, error) {
	if err := validateTrainingInput(model, samples); err != nil {
		return EpochResult{}, err
	}
	if learningRate <= 0 {
		return EpochResult{}, fmt.Errorf("invalid SGD learning rate: %g", learningRate)
	}
	return TrainEpochMiniBatchWithOptimizer(NewSGDOptimizer(model, learningRate), samples, batchSize, rng)
}

// TrainEpochMiniBatchWithOptimizer executa treino com um otimizador explícito.
// Otimizadores com estado devem ser criados uma vez e reutilizados entre épocas.
func TrainEpochMiniBatchWithOptimizer(optimizer Optimizer, samples []Sample, batchSize int, rng *rand.Rand) (EpochResult, error) {
	if optimizer == nil {
		return EpochResult{}, fmt.Errorf("nil optimizer")
	}

	model := optimizer.Model()
	if err := validateTrainingInput(model, samples); err != nil {
		return EpochResult{}, err
	}
	if batchSize <= 0 {
		return EpochResult{}, fmt.Errorf("invalid batch size: %d", batchSize)
	}

	startedAt := time.Now()
	work := samples
	if rng != nil {
		work = append([]Sample(nil), samples...)
		rng.Shuffle(len(work), func(i, j int) {
			work[i], work[j] = work[j], work[i]
		})
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
		optimizer.Step(len(batch))
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
		loss += model.Loss(yHat, sample.Y)
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
