package nn

import "testing"

type sample struct {
	x []float64
	y float64
}

func averageLoss(model *MLP, samples []sample) float64 {
	total := 0.0

	for _, s := range samples {
		yHat := model.Forward(s.x)
		total += BinaryCrossEntropy(yHat, s.y)
	}

	return total / float64(len(samples))
}

func TestMLPTrainingOnORSyntheticDataset(t *testing.T) {
	samples := []sample{
		{x: []float64{0, 0}, y: 0},
		{x: []float64{0, 1}, y: 1},
		{x: []float64{1, 0}, y: 1},
		{x: []float64{1, 1}, y: 1},
	}

	model := NewMLP(2, 4, 42)

	initialLoss := averageLoss(model, samples)

	for epoch := 0; epoch < 3000; epoch++ {
		model.ZeroGrad()

		for _, s := range samples {
			yHat := model.Forward(s.x)
			model.Backward(s.x, yHat, s.y)
		}

		model.ApplyGrad(0.1, len(samples))
	}

	finalLoss := averageLoss(model, samples)

	if finalLoss >= initialLoss {
		t.Fatalf("expected loss to decrease: initial=%f final=%f", initialLoss, finalLoss)
	}

	if finalLoss > 0.35 {
		t.Fatalf("expected final loss <= 0.35, got %f", finalLoss)
	}
}

func TestNewMLPWithHiddenSizesBuildsMultipleHiddenLayers(t *testing.T) {
	model := NewMLPWithHiddenSizes(3, []int{5, 4, 2}, 7)

	if got := model.InputSize(); got != 3 {
		t.Fatalf("expected input size 3, got %d", got)
	}

	expected := []int{5, 4, 2}
	actual := model.HiddenSizes()
	if len(actual) != len(expected) {
		t.Fatalf("expected %d hidden layers, got %d", len(expected), len(actual))
	}

	for i := range expected {
		if actual[i] != expected[i] {
			t.Fatalf("hidden layer %d: expected %d, got %d", i, expected[i], actual[i])
		}
	}

	model.Forward([]float64{1, 0, 1})
}

func TestMultilayerMLPTrainingReducesLoss(t *testing.T) {
	samples := []sample{
		{x: []float64{0, 0}, y: 0},
		{x: []float64{0, 1}, y: 1},
		{x: []float64{1, 0}, y: 1},
		{x: []float64{1, 1}, y: 1},
	}

	model := NewMLPWithHiddenSizes(2, []int{8, 4}, 42)
	initialLoss := averageLoss(model, samples)

	for epoch := 0; epoch < 2000; epoch++ {
		model.ZeroGrad()
		for _, s := range samples {
			yHat := model.Forward(s.x)
			model.Backward(s.x, yHat, s.y)
		}
		model.ApplyGrad(0.1, len(samples))
	}

	finalLoss := averageLoss(model, samples)
	if finalLoss >= initialLoss {
		t.Fatalf("expected multilayer loss to decrease: initial=%f final=%f", initialLoss, finalLoss)
	}
}
