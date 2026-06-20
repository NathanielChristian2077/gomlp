//go:build goexperiment.simd && amd64

package kernel

import "simd/archsimd"

func canUseF32x8(out int) bool {
	return out >= lanesF32x8 && archsimd.X86.AVX2()
}

func denseForwardF32x8(input, weights, bias, output []float32, in, out int) {
	if !canUseF32x8(out) {
		denseForwardScalarF32(input, weights, bias, output, in, out)
		return
	}

	copy(output[:out], bias[:out])

	vectorEnd := out &^ (lanesF32x8 - 1)
	useFMA := archsimd.X86.FMA()

	for i := 0; i < in; i++ {
		x := input[i]
		xVec := archsimd.BroadcastFloat32x8(x)
		weightRow := i * out

		for o := 0; o < vectorEnd; o += lanesF32x8 {
			acc := archsimd.LoadFloat32x8Slice(output[o:])
			w := archsimd.LoadFloat32x8Slice(weights[weightRow+o:])

			if useFMA {
				acc = xVec.MulAdd(w, acc)
			} else {
				acc = acc.Add(xVec.Mul(w))
			}

			acc.StoreSlice(output[o:])
		}

		for o := vectorEnd; o < out; o++ {
			output[o] += x * weights[weightRow+o]
		}
	}
}
