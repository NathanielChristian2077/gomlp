package nn

import (
	"math"
	"math/rand"
	"testing"
)

func TestDenseLayerForwardManualWeights(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	layer := NewDenseLayer(2, 3, rng)

	copy(layer.Weights, []float64{
		1, 2, 3,
		4, 5, 6,
	})

	copy(layer.Biases, []float64{
		0.5, -1, 2,
	})

	input := []float64{2, 3}
	output := make([]float64, 3)

	layer.Forward(input, output)

	expected := []float64{
		14.5,
		18,
		26,
	}

	for i := range expected {
		if math.Abs(output[i]-expected[i]) > 1e-9 {
			t.Fatalf("output[%d]: expected %f, got %f", i, expected[i], output[i])
		}
	}
}

func TestDenseLayerForwardPanicsOnInvalidInputShape(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	layer := NewDenseLayer(2, 1, rng)

	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic")
		}
	}()

	layer.Forward([]float64{1}, make([]float64, 1))
}

func TestDenseLayerAccumulateGradPanicsOnInvalidDeltaShape(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	layer := NewDenseLayer(2, 1, rng)

	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic")
		}
	}()

	layer.AccumulateGrad([]float64{1, 2}, []float64{})
}

func TestDenseLayerApplyGradPanicsOnInvalidBatchSize(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	layer := NewDenseLayer(2, 1, rng)

	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic")
		}
	}()

	layer.ApplyGrad(0.1, 0)
}
