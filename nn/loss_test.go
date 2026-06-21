package nn

import (
	"math"
	"testing"
)

func TestDefaultLossValueAndDelta(t *testing.T) {
	loss := DefaultLoss()
	if got, want := loss.Value(0.8, 1), -math.Log(0.8); math.Abs(got-want) > 1e-12 {
		t.Fatalf("loss: got %.15f want %.15f", got, want)
	}
	if got := loss.OutputDelta(0.8, 1); math.Abs(got+0.2) > 1e-12 {
		t.Fatalf("delta: got %.15f want -0.2", got)
	}
}
