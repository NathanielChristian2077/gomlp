package nn

import (
	"math"
	"testing"
)

func TestSigmoidBinaryCrossEntropyValue(t *testing.T) {
	loss := DefaultSigmoidBinaryCrossEntropy()
	got := loss.Value(0.8, 1)
	want := -math.Log(0.8)
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("positive target: got %.15f want %.15f", got, want)
	}

	got = loss.Value(0.2, 0)
	want = -math.Log(0.8)
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("negative target: got %.15f want %.15f", got, want)
	}
}

func TestSigmoidBinaryCrossEntropyOutputDelta(t *testing.T) {
	loss := DefaultLoss()
	if got := loss.OutputDelta(0.8, 1); math.Abs(got+0.2) > 1e-12 {
		t.Fatalf("positive target delta: got %.15f want -0.2", got)
	}
	if got := loss.OutputDelta(0.2, 0); math.Abs(got-0.2) > 1e-12 {
		t.Fatalf("negative target delta: got %.15f want 0.2", got)
	}
}

func TestSigmoidBinaryCrossEntropyClampsExtremePredictions(t *testing.T) {
	loss := DefaultSigmoidBinaryCrossEntropy()
	for _, prediction := range []float64{0, 1} {
		value := loss.Value(prediction, 1)
		if math.IsInf(value, 0) || math.IsNaN(value) {
			t.Fatalf("prediction %f produced non-finite loss %f", prediction, value)
		}
	}
}
