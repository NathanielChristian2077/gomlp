package kernel

import "fmt"

const lanesF32x8 = 8

// DenseForwardF32 calcula output = input*weights + bias para uma camada densa
// com pesos em layout input-major: weights[i*out+o].
func DenseForwardF32(input, weights, bias, output []float32, in, out int) {
	mustMatchDenseForwardF32(input, weights, bias, output, in, out)
	denseForwardF32(input, weights, bias, output, in, out)
}

func denseForwardF32(input, weights, bias, output []float32, in, out int) {
	if canUseF32x8(out) {
		denseForwardF32x8(input, weights, bias, output, in, out)
		return
	}

	denseForwardScalarF32(input, weights, bias, output, in, out)
}

func denseForwardScalarF32(input, weights, bias, output []float32, in, out int) {
	copy(output[:out], bias[:out])

	for i := 0; i < in; i++ {
		x := input[i]
		weightRow := i * out

		for o := 0; o < out; o++ {
			output[o] += x * weights[weightRow+o]
		}
	}
}

func mustMatchDenseForwardF32(input, weights, bias, output []float32, in, out int) {
	if in <= 0 || out <= 0 {
		panic(fmt.Sprintf("invalid dense forward shape: in=%d out=%d", in, out))
	}
	if len(input) < in {
		panic(fmt.Sprintf("invalid dense forward input length: expected at least %d, got %d", in, len(input)))
	}
	if len(weights) < in*out {
		panic(fmt.Sprintf("invalid dense forward weights length: expected at least %d, got %d", in*out, len(weights)))
	}
	if len(bias) < out {
		panic(fmt.Sprintf("invalid dense forward bias length: expected at least %d, got %d", out, len(bias)))
	}
	if len(output) < out {
		panic(fmt.Sprintf("invalid dense forward output length: expected at least %d, got %d", out, len(output)))
	}
}
