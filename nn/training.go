package nn

import (
	"fmt"
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
	if err := validateTrainingInput(model, samples); err != nil {
		return EpochResult{}, err
	}

	startedAt := time.Now()
	model.ZeroGrad()

	loss := 0.0
	confusion := metrics.NewConfusionMatrix()

	for _, sample := range samples {
		yHat := model.Forward(sample.X)
		loss += BinaryCrossEntropy(yHat, sample.Y)
		confusion.Add(yHat, sample.Y, DefaultClassificationThreshold)
		model.Backward(sample.X, yHat, sample.Y)
	}

	model.ApplyGrad(lr, len(samples))

	return EpochResult{
		Loss:      loss / float64(len(samples)),
		Accuracy:  confusion.Accuracy(),
		Confusion: confusion,
		Duration:  time.Since(startedAt),
	}, nil
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
