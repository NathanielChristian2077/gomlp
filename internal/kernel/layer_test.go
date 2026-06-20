package kernel

import "testing"

func TestDenseForwardF32x8MatchesScalar(t *testing.T) {
	const (
		in  = 17
		out = 13 // 8 SIMD + 5 de cauda
	)

	input, weights, bias := denseForwardFixture(in, out)
	want := make([]float32, out)
	got := make([]float32, out)

	denseForwardScalarF32(input, weights, bias, want, in, out)
	denseForwardF32x8(input, weights, bias, got, in, out)

	assertFloat32SlicesClose(t, got, want, 1e-4)
}

func TestDenseForwardF32DispatcherMatchesScalar(t *testing.T) {
	tests := []struct {
		name string
		in   int
		out  int
	}{
		{name: "short output uses scalar path", in: 7, out: 3},
		{name: "exact vector width", in: 9, out: 8},
		{name: "vector width with scalar tail", in: 17, out: 13},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, weights, bias := denseForwardFixture(tt.in, tt.out)
			want := make([]float32, tt.out)
			got := make([]float32, tt.out)

			denseForwardScalarF32(input, weights, bias, want, tt.in, tt.out)
			DenseForwardF32(input, weights, bias, got, tt.in, tt.out)

			assertFloat32SlicesClose(t, got, want, 1e-4)
		})
	}
}

func TestDenseForwardF32PanicsOnInvalidShape(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered == nil {
			t.Fatalf("expected panic")
		}
	}()

	DenseForwardF32(
		[]float32{1, 2},
		[]float32{1, 2, 3},
		[]float32{0, 0},
		make([]float32, 2),
		2,
		2,
	)
}

func denseForwardFixture(in, out int) ([]float32, []float32, []float32) {
	input := make([]float32, in)
	weights := make([]float32, in*out)
	bias := make([]float32, out)

	for i := 0; i < in; i++ {
		input[i] = float32(i+1) * 0.07
	}
	for i := 0; i < len(weights); i++ {
		weights[i] = float32((i%11)-5) * 0.03
	}
	for i := 0; i < out; i++ {
		bias[i] = float32(i) * 0.01
	}

	return input, weights, bias
}

func assertFloat32SlicesClose(t *testing.T, actual, expected []float32, tolerance float32) {
	t.Helper()

	if len(actual) != len(expected) {
		t.Fatalf("expected %d floats, got %d: actual=%v expected=%v", len(expected), len(actual), actual, expected)
	}

	for i := range expected {
		diff := actual[i] - expected[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tolerance {
			t.Fatalf("index %d: expected %.8f, got %.8f, diff %.8f > tolerance %.8f", i, expected[i], actual[i], diff, tolerance)
		}
	}
}
