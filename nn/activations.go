package nn

import "math"

func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func ReLUDerivativeFromActivation(a float64) float64 {
	if a > 0 {
		return 1
	}
	return 0
}

// Sigmoid transforms a scalar logit into a probability while avoiding exp overflow.
func Sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}
