package optimizer

import (
	"math"
	"testing"

	"github.com/NathanielChristian2077/gomlp/nn"
)

func TestAdamFirstStepMatchesBiasCorrectedUpdate(t *testing.T) {
	model := nn.NewMLP(2, 2, 1)
	layer := &model.Hidden[0]
	layer.Weights[0] = 1
	layer.GradW[0] = 0.4
	layer.Biases[0] = 0.5
	layer.GradB[0] = -0.2

	adam := NewAdamWithConfig(model, AdamConfig{LearningRate: 0.01, Beta1: 0.8, Beta2: 0.9, Epsilon: 1e-8})
	adam.Step(2)

	gWeight := 0.4 / 2.0
	wantWeight := 1.0 - 0.01*gWeight/(math.Abs(gWeight)+1e-8)
	if diff := math.Abs(layer.Weights[0] - wantWeight); diff > 1e-12 { t.Fatalf("weight: got %.15f want %.15f", layer.Weights[0], wantWeight) }

	gBias := -0.2 / 2.0
	wantBias := 0.5 - 0.01*gBias/(math.Abs(gBias)+1e-8)
	if diff := math.Abs(layer.Biases[0] - wantBias); diff > 1e-12 { t.Fatalf("bias: got %.15f want %.15f", layer.Biases[0], wantBias) }
}

func TestAdamTrainsORSyntheticDataset(t *testing.T) {
	samples := []nn.Sample{{X: []float64{0, 0}, Y: 0}, {X: []float64{0, 1}, Y: 1}, {X: []float64{1, 0}, Y: 1}, {X: []float64{1, 1}, Y: 1}}
	model := nn.NewMLP(2, 4, 42)
	adam := NewAdam(model, 0.03)
	before, err := nn.Evaluate(model, samples)
	if err != nil { t.Fatal(err) }
	for epoch := 0; epoch < 800; epoch++ {
		if _, err := nn.TrainEpochMiniBatchWithOptimizer(adam, samples, len(samples), nil); err != nil { t.Fatal(err) }
	}
	after, err := nn.Evaluate(model, samples)
	if err != nil { t.Fatal(err) }
	if after.Loss >= before.Loss { t.Fatalf("expected Adam loss to decrease: before=%f after=%f", before.Loss, after.Loss) }
}
