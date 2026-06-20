package nn

import (
	"math"
	"testing"
)

func TestSGDOptimizerAppliesAverageBatchGradient(t *testing.T) {
	model := NewMLP(2, 1, 7)
	layer := &model.Hidden[0]
	layer.Weights[0] = 2
	layer.Biases[0] = -1
	layer.GradW[0] = 0.6
	layer.GradB[0] = -0.4

	optimizer := NewSGDOptimizer(model, 0.1)
	optimizer.Step(2)

	if diff := math.Abs(layer.Weights[0] - 1.97); diff > 1e-12 {
		t.Fatalf("weight: got %.15f want 1.97", layer.Weights[0])
	}
	if diff := math.Abs(layer.Biases[0] - (-0.98)); diff > 1e-12 {
		t.Fatalf("bias: got %.15f want -0.98", layer.Biases[0])
	}
}

func TestSGDOptimizerPanicsOnInvalidBatchSize(t *testing.T) {
	optimizer := NewSGDOptimizer(NewMLP(2, 1, 7), 0.1)
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	optimizer.Step(0)
}

func TestTrainEpochMiniBatchWithOptimizerReducesLoss(t *testing.T) {
	samples := []Sample{
		{X: []float64{0, 0}, Y: 0},
		{X: []float64{0, 1}, Y: 1},
		{X: []float64{1, 0}, Y: 1},
		{X: []float64{1, 1}, Y: 1},
	}

	model := NewMLP(2, 4, 42)
	optimizer := NewSGDOptimizer(model, 0.1)
	before, err := Evaluate(model, samples)
	if err != nil {
		t.Fatal(err)
	}

	for epoch := 0; epoch < 800; epoch++ {
		if _, err := TrainEpochMiniBatchWithOptimizer(optimizer, samples, len(samples), nil); err != nil {
			t.Fatal(err)
		}
	}

	after, err := Evaluate(model, samples)
	if err != nil {
		t.Fatal(err)
	}
	if after.Loss >= before.Loss {
		t.Fatalf("expected loss to decrease: before=%f after=%f", before.Loss, after.Loss)
	}
}
