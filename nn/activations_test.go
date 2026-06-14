package nn

import (
	"math"
	"testing"
)

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestReLU(t *testing.T) {
	if ReLU(-2) != 0 {
		t.Fatalf("expected ReLU(-2) = 0")
	}
	if ReLU(0) != 0 {
		t.Fatalf("expected ReLU(0) = 0")
	}
	if ReLU(3.5) != 3.5 {
		t.Fatalf("expected ReLU(3.5) = 3.5")
	}
}

func TestReLUDerivativeFromActivation(t *testing.T) {
	if ReLUDerivativeFromActivation(0) != 0 {
		t.Fatalf("expected derivative at inactive activation to be 0")
	}
	if ReLUDerivativeFromActivation(2) != 1 {
		t.Fatalf("expected derivative at active activation to be 1")
	}
}

func TestSigmoid(t *testing.T) {
	if !almostEqual(Sigmoid(0), 0.5) {
		t.Fatalf("expected sigmoid(0) = 0.5, got %f", Sigmoid(0))
	}

	if Sigmoid(1000) <= 0.999 {
		t.Fatalf("expected sigmoid(1000) near 1")
	}

	if Sigmoid(-1000) >= 0.001 {
		t.Fatalf("expected sigmoid(-1000) near 0")
	}
}

func TestBinaryCrossEntropy(t *testing.T) {
	got := BinaryCrossEntropy(0.8, 1)
	want := -math.Log(0.8)

	if !almostEqual(got, want) {
		t.Fatalf("expected %f, got %f", want, got)
	}

	got = BinaryCrossEntropy(0.2, 0)
	want = -math.Log(0.8)

	if !almostEqual(got, want) {
		t.Fatalf("expected %f, got %f", want, got)
	}
}
