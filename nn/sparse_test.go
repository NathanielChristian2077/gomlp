package nn

import (
	"math"
	"testing"
)

func TestReLUToActivePreservesPositiveIndices(t *testing.T) {
	z := []float64{-2.0, 0.0, 1.5, 0.0, 3.25, -7.0, 0.01}

	active := ReLUToActive(z, 0)

	if active.Size != len(z) {
		t.Fatalf("expected active vector size %d, got %d", len(z), active.Size)
	}

	expectedIndices := []int{2, 4, 6}
	expectedValues := []float64{1.5, 3.25, 0.01}

	assertIntSlicesEqual(t, active.Indices, expectedIndices)
	assertFloatSlicesClose(t, active.Values, expectedValues, 0)
}

func TestReLUToActiveAppliesThreshold(t *testing.T) {
	z := []float64{-1.0, 0.0, 0.10, 0.50, 1.25}

	active := ReLUToActive(z, 0.25)

	assertIntSlicesEqual(t, active.Indices, []int{3, 4})
	assertFloatSlicesClose(t, active.Values, []float64{0.50, 1.25}, 0)
}

func TestReLUToActivePanicsForNegativeThreshold(t *testing.T) {
	assertPanics(t, func() {
		ReLUToActive([]float64{1.0}, -0.1)
	})
}

func TestActiveVectorWritesDenseRepresentation(t *testing.T) {
	active := ActiveVector{
		Size:    5,
		Indices: []int{1, 3},
		Values:  []float64{2.5, 4.5},
	}

	out := []float64{9, 9, 9, 9, 9}
	active.WriteDense(out)

	assertFloatSlicesClose(t, out, []float64{0, 2.5, 0, 4.5, 0}, 0)
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

	activeInput := ReLUToActive(preActivation, 0)
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

	activeInput := ReLUToActive([]float64{-3.0, 0.0, -1.0, -0.5}, 0)
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
		Size:    2,
		Indices: []int{0, 1},
		Values:  []float64{4.0, 2.0},
	}

	firstOutput := make([]float64, layer.Out)
	secondOutput := []float64{100.0, 100.0}

	layer.ForwardSparse(activeInput, firstOutput)
	layer.ForwardSparse(activeInput, secondOutput)

	assertFloatSlicesClose(t, secondOutput, firstOutput, 0)
}

func TestForwardSparsePanicsForInvalidActiveVector(t *testing.T) {
	layer := DenseLayer{In: 2, Out: 1, Weights: []float64{1, 2}, Biases: []float64{0}}

	assertPanics(t, func() {
		layer.ForwardSparse(ActiveVector{Size: 3, Indices: []int{0}, Values: []float64{1}}, []float64{0})
	})

	assertPanics(t, func() {
		layer.ForwardSparse(ActiveVector{Size: 2, Indices: []int{0, 1}, Values: []float64{1}}, []float64{0})
	})

	assertPanics(t, func() {
		layer.ForwardSparse(ActiveVector{Size: 2, Indices: []int{2}, Values: []float64{1}}, []float64{0})
	})
}

func TestAccumulateGradSparseMatchesDenseGradientAfterReLU(t *testing.T) {
	denseLayer := DenseLayer{
		In:      4,
		Out:     2,
		Weights: []float64{0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9},
		Biases:  []float64{0.1, -0.2},
		GradW:   make([]float64, 8),
		GradB:   make([]float64, 2),
	}
	sparseLayer := denseLayer.Clone()

	preActivation := []float64{-1.0, 2.0, 0.0, 3.0}
	activeInput := ReLUToActive(preActivation, 0)
	denseInput := activeInput.ToDense()
	deltaOut := []float64{0.4, -0.6}

	denseLayer.AccumulateGrad(denseInput, deltaOut)
	sparseLayer.AccumulateGradSparse(activeInput, deltaOut)

	assertFloatSlicesClose(t, sparseLayer.GradW, denseLayer.GradW, 1e-12)
	assertFloatSlicesClose(t, sparseLayer.GradB, denseLayer.GradB, 1e-12)
}

func TestMLPForwardSparseExactMatchesDenseForward(t *testing.T) {
	model := NewMLPWithHiddenSizes(4, []int{5, 3}, 11)
	x := []float64{0.1, -0.2, 0.7, 1.0}

	denseYHat := model.Forward(x)
	sparseYHat := model.ForwardSparseExact(x)

	if diff := math.Abs(sparseYHat - denseYHat); diff > 1e-12 {
		t.Fatalf("expected sparse exact forward to match dense forward: dense=%.12f sparse=%.12f diff=%.12f", denseYHat, sparseYHat, diff)
	}
}

func TestMLPForwardSparseStatsCaptureSparsityAndOps(t *testing.T) {
	model := NewMLPWithHiddenSizes(4, []int{5, 3}, 11)
	x := []float64{0.1, -0.2, 0.7, 1.0}

	_, stats := model.ForwardSparseWithStats(x, 0)

	if len(stats.Hidden) != 2 {
		t.Fatalf("expected stats for 2 hidden layers, got %d", len(stats.Hidden))
	}

	if stats.DenseOps != 38 {
		t.Fatalf("expected 38 dense ops, got %d", stats.DenseOps)
	}

	if stats.SparseOps <= 0 || stats.SparseOps > stats.DenseOps {
		t.Fatalf("expected sparse ops in range [1, %d], got %d", stats.DenseOps, stats.SparseOps)
	}

	if stats.EstimatedSpeedup() < 1 {
		t.Fatalf("expected estimated speedup >= 1, got %.6f", stats.EstimatedSpeedup())
	}

	for _, layerStats := range stats.Hidden {
		if layerStats.Active < 0 || layerStats.Active > layerStats.Size {
			t.Fatalf("invalid active count for layer %d: active=%d size=%d", layerStats.LayerIndex, layerStats.Active, layerStats.Size)
		}
		if layerStats.ActiveRatio < 0 || layerStats.ActiveRatio > 1 {
			t.Fatalf("invalid active ratio for layer %d: %.6f", layerStats.LayerIndex, layerStats.ActiveRatio)
		}
		if layerStats.Sparsity < 0 || layerStats.Sparsity > 1 {
			t.Fatalf("invalid sparsity for layer %d: %.6f", layerStats.LayerIndex, layerStats.Sparsity)
		}
	}
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

func assertPanics(t *testing.T, fn func()) {
	t.Helper()

	defer func() {
		if recovered := recover(); recovered == nil {
			t.Fatalf("expected panic")
		}
	}()

	fn()
}
