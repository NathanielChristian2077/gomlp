package tensor

import "testing"

func TestDot(t *testing.T) {
	a := New(2, 3)
	b := New(3, 2)
	copy(a.Data, []float64{1, 2, 3, 4, 5, 6})
	copy(b.Data, []float64{7, 8, 9, 10, 11, 12})

	got := Dot(a, b)
	want := []float64{58, 64, 139, 154}
	assertDataEqual(t, got.Data, want)
}

func TestElementwiseOperationsDoNotMutateInput(t *testing.T) {
	a := New(1, 3)
	b := New(1, 3)
	copy(a.Data, []float64{1, 2, 3})
	copy(b.Data, []float64{4, 5, 6})

	assertDataEqual(t, Add(a, b).Data, []float64{5, 7, 9})
	assertDataEqual(t, Sub(a, b).Data, []float64{-3, -3, -3})
	assertDataEqual(t, MulScalar(a, 2).Data, []float64{2, 4, 6})
	assertDataEqual(t, Apply(a, func(v float64) float64 { return v * v }).Data, []float64{1, 4, 9})
	assertDataEqual(t, a.Data, []float64{1, 2, 3})
}

func TestTranspose(t *testing.T) {
	a := New(2, 3)
	copy(a.Data, []float64{1, 2, 3, 4, 5, 6})

	got := Transpose(a)
	if got.Rows != 3 || got.Cols != 2 {
		t.Fatalf("shape: got %dx%d want 3x2", got.Rows, got.Cols)
	}
	assertDataEqual(t, got.Data, []float64{1, 4, 2, 5, 3, 6})
}

func assertDataEqual(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}
