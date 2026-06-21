package nn

import (
	"math"
	"testing"
)

func TestSGDOptimizerAppliesAverageGradient(t *testing.T) {
	model := NewMLP(2, 1, 7)
	layer := &model.Hidden[0]
	layer.Weights[0] = 2
	layer.GradW[0] = 0.6
	layer.Biases[0] = -1
	layer.GradB[0] = -0.4

	NewSGDOptimizer(model, 0.1).Step(2)
	if diff := math.Abs(layer.Weights[0] - 1.97); diff > 1e-12 {
		t.Fatalf("weight: got %.15f want 1.97", layer.Weights[0])
	}
	if diff := math.Abs(layer.Biases[0] - (-0.98)); diff > 1e-12 {
		t.Fatalf("bias: got %.15f want -0.98", layer.Biases[0])
	}
}
