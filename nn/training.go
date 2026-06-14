package nn

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/NathanielChristian2077/gomlp/metrics"
)

const DefaultClassificationThreshold = 0.5

type Sample struct {
	X []float64
	Y float64
}

type EpochResult struct {
	Loss      float64
	Accuracy  float64
	Confusion metrics.ConfusionMatrix
	Duration  time.Duration
}

func TrainEpoch(model *MLP, samples []Sample, lr float64) (EpochResult, error) {
	return TrainEpochMiniBatch(model, samples, lr, len(samples), nil)
}

func TrainEpochMiniBatch(model *MLP, samples []Sample, lr float64, batchSize int, rng *rand.Rand) (EpochResult, error) {
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

		model.ApplyGrad(lr, len(batch))
	}

	result, err := Evaluate(model, samples)
	if err != nil {
		return EpochResult{}, err
	}

	result.Duration = time.Since(startedAt)
	return result, nil
}

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
	if len(samples) == 0 {
		return fmt.Errorf("empty sample set")
	}

	for i, sample := range samples {
		if len(sample.X) != model.Hidden.In {
			return fmt.Errorf("sample %d has invalid input length: expected %d, got %d", i, model.Hidden.In, len(sample.X))
		}
		if sample.Y != 0 && sample.Y != 1 {
			return fmt.Errorf("sample %d has invalid label: expected 0 or 1, got %f", i, sample.Y)
		}
	}

	return nil
}
