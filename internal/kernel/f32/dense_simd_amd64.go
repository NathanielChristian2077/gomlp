//go:build goexperiment.simd && amd64

package f32

import "simd/archsimd"

func canUseX8(out int) bool { return out >= lanesX8 && archsimd.X86.AVX2() }

func denseForwardX8(input, weights, bias, output []float32, in, out int) {
	if !canUseX8(out) { denseForwardScalar(input, weights, bias, output, in, out); return }
	copy(output[:out], bias[:out])
	vectorEnd := out &^ (lanesX8 - 1)
	useFMA := archsimd.X86.FMA()
	for i := 0; i < in; i++ {
		xVec := archsimd.BroadcastFloat32x8(input[i])
		row := i * out
		for o := 0; o < vectorEnd; o += lanesX8 {
			acc := archsimd.LoadFloat32x8Slice(output[o:])
			w := archsimd.LoadFloat32x8Slice(weights[row+o:])
			if useFMA { acc = xVec.MulAdd(w, acc) } else { acc = acc.Add(xVec.Mul(w)) }
			acc.StoreSlice(output[o:])
		}
		for o := vectorEnd; o < out; o++ { output[o] += input[i] * weights[row+o] }
	}
}
