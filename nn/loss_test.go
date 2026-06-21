package nn

import (
	"math"
	"testing"
)

func TestSigmoidBinaryCrossEntropyFromLogits(t *testing.T) {
	loss := DefaultSigmoidBinaryCrossEntropy()
	logits := []float64{0}
	if got, want := loss.LossFromLogits(logits, 1), math.Log(2); math.Abs(got-want) > 1e-12 {
		t.Fatalf("loss: got %.15f want %.15f", got, want)
	}
	delta := make([]float64, 1)
	loss.FillOutputDelta(logits, 1, delta)
	if got, want := delta[0], -0.5; math.Abs(got-want) > 1e-12 {
		t.Fatalf("delta: got %.15f want %.15f", got, want)
	}
}

func TestSoftmaxCrossEntropyFromLogits(t *testing.T) {
	loss := SoftmaxCrossEntropy{}
	logits := []float64{0, 0}
	if got, want := loss.LossFromLogits(logits, 1), math.Log(2); math.Abs(got-want) > 1e-12 {
		t.Fatalf("loss: got %.15f want %.15f", got, want)
	}
	delta := make([]float64, 2)
	loss.FillOutputDelta(logits, 1, delta)
	if math.Abs(delta[0]-0.5) > 1e-12 || math.Abs(delta[1]+0.5) > 1e-12 {
		t.Fatalf("delta: got %v want [0.5 -0.5]", delta)
	}
}
