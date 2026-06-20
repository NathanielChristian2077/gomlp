package f32

import "testing"

func TestDenseForwardMatchesScalar(t *testing.T) {
	for _, shape := range [][2]int{{7, 3}, {9, 8}, {17, 13}} {
		in, out := shape[0], shape[1]
		input := make([]float32, in)
		weights := make([]float32, in*out)
		bias := make([]float32, out)
		for i := range input { input[i] = float32(i+1) * 0.07 }
		for i := range weights { weights[i] = float32((i%11)-5) * 0.03 }
		for i := range bias { bias[i] = float32(i) * 0.01 }
		want, got := make([]float32, out), make([]float32, out)
		denseForwardScalar(input, weights, bias, want, in, out)
		DenseForward(input, weights, bias, got, in, out)
		for i := range want {
			diff := got[i] - want[i]
			if diff < 0 { diff = -diff }
			if diff > 1e-4 { t.Fatalf("shape=%dx%d index=%d got=%f want=%f", in, out, i, got[i], want[i]) }
		}
	}
}
