package nn

import (
	"math"
	"testing"
)

func TestAdamOptimizerFirstStepMatchesManualUpdate(t *testing.T) {
	model := NewMLPWithHiddenSizes(2, []int{2}, 1)

	hidden := &model.Hidden[0]
	copy(hidden.Weights, []float64{1.0, -2.0, 3.0, -4.0})
	copy(hidden.Biases, []float64{0.5, -0.5})
	copy(hidden.GradW, []float64{0.4, -0.8, 1.2, -1.6})
	copy(hidden.GradB, []float64{0.2, -0.6})

	output := &model.Output
	copy(output.Weights, []float64{0.25, -0.75})
	copy(output.Biases, []float64{0.1})
	copy(output.GradW, []float64{0.3, -0.9})
	copy(output.GradB, []float64{-0.5})

	beforeHiddenWeights := cloneFloat64Slice(hidden.Weights)
	beforeHiddenBiases := cloneFloat64Slice(hidden.Biases)
	beforeOutputWeights := cloneFloat64Slice(output.Weights)
	beforeOutputBiases := cloneFloat64Slice(output.Biases)

	config := AdamConfig{
		LearningRate: 0.01,
		Beta1:        0.8,
		Beta2:        0.9,
		Epsilon:      1e-8,
	}
	optimizer := NewAdamOptimizerWithConfig(model, config)
	optimizer.Step(2)

	if got := optimizer.StepCount(); got != 1 {
		t.Fatalf("expected Adam step count 1, got %d", got)
	}

	assertAdamFirstStep(t, hidden.Weights, beforeHiddenWeights, []float64{0.4, -0.8, 1.2, -1.6}, config, 2)
	assertAdamFirstStep(t, hidden.Biases, beforeHiddenBiases, []float64{0.2, -0.6}, config, 2)
	assertAdamFirstStep(t, output.Weights, beforeOutputWeights, []float64{0.3, -0.9}, config, 2)
	assertAdamFirstStep(t, output.Biases, beforeOutputBiases, []float64{-0.5}, config, 2)
}

func TestAdamConfigUsesDefaultsForZeroMoments(t *testing.T) {
	model := NewMLP(2, 3, 1)
	optimizer := NewAdamOptimizerWithConfig(model, AdamConfig{LearningRate: 0.001})
	config := optimizer.Config()

	if config.Beta1 != DefaultAdamBeta1 {
		t.Fatalf("expected default beta1 %.3f, got %.3f", DefaultAdamBeta1, config.Beta1)
	}
	if config.Beta2 != DefaultAdamBeta2 {
		t.Fatalf("expected default beta2 %.3f, got %.3f", DefaultAdamBeta2, config.Beta2)
	}
	if config.Epsilon != DefaultAdamEpsilon {
		t.Fatalf("expected default epsilon %g, got %g", DefaultAdamEpsilon, config.Epsilon)
	}
}

func TestTrainEpochMiniBatchWithAdamReducesLoss(t *testing.T) {
	samples := []Sample{
		{X: []float64{0, 0}, Y: 0},
		{X: []float64{0, 1}, Y: 1},
		{X: []float64{1, 0}, Y: 1},
		{X: []float64{1, 1}, Y: 1},
	}

	model := NewMLP(2, 4, 42)
	optimizer := NewAdamOptimizer(model, 0.03)
	initialLoss := averageSampleLoss(model, samples)

	for epoch := 0; epoch < 800; epoch++ {
		if _, err := TrainEpochMiniBatchWithOptimizer(optimizer, samples, len(samples), nil); err != nil {
			t.Fatalf("Adam training failed: %v", err)
		}
	}

	finalLoss := averageSampleLoss(model, samples)
	if finalLoss >= initialLoss {
		t.Fatalf("expected Adam loss to decrease: initial=%f final=%f", initialLoss, finalLoss)
	}
	if got := optimizer.StepCount(); got != 800 {
		t.Fatalf("expected 800 Adam steps, got %d", got)
	}
}

func assertAdamFirstStep(t *testing.T, actual, before, gradients []float64, config AdamConfig, batchSize int) {
	t.Helper()

	if len(actual) != len(before) || len(actual) != len(gradients) {
		t.Fatalf("invalid test slices: actual=%d before=%d gradients=%d", len(actual), len(before), len(gradients))
	}

	for i := range actual {
		expected := expectedAdamFirstStep(before[i], gradients[i], config, batchSize)
		if diff := math.Abs(actual[i] - expected); diff > 1e-12 {
			t.Fatalf("index %d: expected %.12f, got %.12f, diff %.12f", i, expected, actual[i], diff)
		}
	}
}

func expectedAdamFirstStep(before, gradient float64, config AdamConfig, batchSize int) float64 {
	g := gradient / float64(batchSize)
	return before - config.LearningRate*g/(math.Abs(g)+config.Epsilon)
}

func averageSampleLoss(model *MLP, samples []Sample) float64 {
	total := 0.0
	for _, sample := range samples {
		model.Forward(sample.X)
		total += model.LossFromLastForward(sample.Y)
	}
	return total / float64(len(samples))
}
