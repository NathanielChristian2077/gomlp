//go:build !goexperiment.simd || !amd64

package f32

func canUseX8(out int) bool { return false }
func denseForwardX8(input, weights, bias, output []float32, in, out int) { denseForwardScalar(input, weights, bias, output, in, out) }
