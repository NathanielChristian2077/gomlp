package nn

import "math"

// ReLU applies the hidden-layer activation.
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLUToActive returns only activations greater than threshold.
// threshold=0 is the exact DSA representation; positive thresholds are approximate.
func ReLUToActive(z []float64, threshold float64) ActiveVector {
	active := NewActiveVector(len(z))
	ReLUToActiveInto(z, threshold, &active)
	return active
}

// ReLUToActiveInto reuses ActiveVector buffers for sparse forward paths.
func ReLUToActiveInto(z []float64, threshold float64, out *ActiveVector) {
	if threshold < 0 {
		panic("activation threshold must be non-negative")
	}
	if out == nil {
		panic("nil active vector output")
	}

	out.Reset(len(z))
	for i, value := range z {
		if value > threshold {
			out.Indices = append(out.Indices, i)
			out.Values = append(out.Values, value)
		}
	}
}

func ReLUDerivativeFromActivation(a float64) float64 {
	if a > 0 {
		return 1
	}
	return 0
}

// Sigmoid transforms a scalar logit into a probability.
func Sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}
