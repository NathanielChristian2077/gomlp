// Package f32 contains private float32 kernels used by GOptimize experiments.
package f32

import "fmt"

const lanesX8 = 8

// DenseForward computes output = input * weights + bias with input-major weights[i*out+o].
func DenseForward(input, weights, bias, output []float32, in, out int) {
	mustMatchDenseForward(input, weights, bias, output, in, out)
	denseForward(input, weights, bias, output, in, out)
}

func denseForward(input, weights, bias, output []float32, in, out int) {
	if canUseX8(out) {
		denseForwardX8(input, weights, bias, output, in, out)
		return
	}
	denseForwardScalar(input, weights, bias, output, in, out)
}

func denseForwardScalar(input, weights, bias, output []float32, in, out int) {
	copy(output[:out], bias[:out])
	for i := 0; i < in; i++ {
		x := input[i]
		row := i * out
		for o := 0; o < out; o++ { output[o] += x * weights[row+o] }
	}
}

func mustMatchDenseForward(input, weights, bias, output []float32, in, out int) {
	if in <= 0 || out <= 0 { panic(fmt.Sprintf("invalid dense forward shape: in=%d out=%d", in, out)) }
	if len(input) < in { panic(fmt.Sprintf("invalid input length: expected at least %d got %d", in, len(input))) }
	if len(weights) < in*out { panic(fmt.Sprintf("invalid weight length: expected at least %d got %d", in*out, len(weights))) }
	if len(bias) < out || len(output) < out { panic("invalid bias or output length") }
}
