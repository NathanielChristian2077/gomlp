package nn

import (
	"math"
	"testing"
)

func TestReLUToActivePreservesPositiveIndices(t *testing.T) {
	z := []float64{-2.0, 0.0, 1.5, -0.0, 3.25, -7.0, 0.01}

	active := ReLUToActive(z)

	if active.Size != len(z) {
		t.Fatalf("expected active vector size %d, got %d", len(z), active.Size)
	}

	expectedIdx := []int{2, 4, 6}
	expectedValues := []float64{1.5, 3.25, 0.01}

	assertIntSlicesEqual(t, active.Idx, expectedIdx)
	assertFloatSlicesClose(t, active.Values, expectedValues, 0)
}

func TestForwardSparseMatchesDenseForwardAfterReLU(t *testing.T) {
	layer := DenseLayer{
		In:  5,
		Out: 3,
		Weights: []float64{
			0.10, -0.20, 0.30,
			0.40, 0.50, -0.60,
			-0.70, 0.80, 0.90,
			1.00, -1.10, 1.20,
			-1.30, 1.40, -1.50,
		},
		Biases: []float64{0.25, -0.50, 0.75},
	}

	preActivation := []float64{-2.0, 1.5, 0.0, 3.0, 0.25}
	denseInput := make([]float64, len(preActivation))
	for i, z := range preActivation {
		denseInput[i] = ReLU(z)
	}

	activeInput := ReLUToActive(preActivation)
	denseOutput := make([]float64, layer.Out)
	sparseOutput := make([]float64, layer.Out)

	layer.Forward(denseInput, denseOutput)
	layer.ForwardSparse(activeInput, sparseOutput)

	assertFloatSlicesClose(t, sparseOutput, denseOutput, 1e-12)
}

func TestForwardSparseWithNoActiveInputsReturnsBiases(t *testing.T) {
	layer := DenseLayer{
		In:  4,
		Out: 2,
		Weights: []float64{
			1.0, 2.0,
			3.0, 4.0,
			5.0, 6.0,
			7.0, 8.0,
		},
		Biases: []float64{-0.75, 1.25},
	}

	activeInput := ReLUToActive([]float64{-3.0, 0.0, -1.0, -0.5})
	output := []float64{999.0, -999.0}

	layer.ForwardSparse(activeInput, output)

	assertFloatSlicesClose(t, output, layer.Biases, 0)
}

func TestForwardSparseOverwritesPreviousOutputValues(t *testing.T) {
	layer := DenseLayer{
		In:      2,
		Out:     2,
		Weights: []float64{2.0, -1.0, 0.5, 3.0},
		Biases:  []float64{0.1, -0.2},
	}

	activeInput := ActiveVector{
		Size:   2,
		Idx:    []int{0, 1},
		Values: []float64{4.0, 2.0},
	}

	firstOutput := make([]float64, layer.Out)
	secondOutput := []float64{100.0, 100.0}

	layer.ForwardSparse(activeInput, firstOutput)
	layer.ForwardSparse(activeInput, secondOutput)

	assertFloatSlicesClose(t, secondOutput, firstOutput, 0)
}

func assertIntSlicesEqual(t *testing.T, actual, expected []int) {
	t.Helper()

	if len(actual) != len(expected) {
		t.Fatalf("expected %d integers, got %d: actual=%v expected=%v", len(expected), len(actual), actual, expected)
	}

	for i := range expected {
		if actual[i] != expected[i] {
			t.Fatalf("index %d: expected %d, got %d: actual=%v expected=%v", i, expected[i], actual[i], actual, expected)
		}
	}
}

func assertFloatSlicesClose(t *testing.T, actual, expected []float64, tolerance float64) {
	t.Helper()

	if len(actual) != len(expected) {
		t.Fatalf("expected %d floats, got %d: actual=%v expected=%v", len(expected), len(actual), actual, expected)
	}

	for i := range expected {
		diff := math.Abs(actual[i] - expected[i])
		if diff > tolerance {
			t.Fatalf("index %d: expected %.12f, got %.12f, diff %.12f > tolerance %.12f", i, expected[i], actual[i], diff, tolerance)
		}
	}
}
