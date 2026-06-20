//go:build !goexperiment.simd || !amd64

package kernel

func canUseF32x8(out int) bool {
	return false
}

func denseForwardF32x8(input, weights, bias, output []float32, in, out int) {
	denseForwardScalarF32(input, weights, bias, output, in, out)
}
