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

func Sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}

	z := math.Exp(x)
	return z / (1 + z)
}

func BinaryCrossEntropy(yHat, y float64) float64 {
	const eps = 1e-12

	if yHat < eps {
		yHat = eps
	}
	if yHat > 1-eps {
		yHat = 1 - eps
	}

	return -(y*math.Log(yHat) + (1-y)*math.Log(1-yHat))
}
