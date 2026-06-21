package nn

import "math"

const defaultLossEpsilon = 1e-12

// Loss describes a scalar binary loss and its output-layer delta.
type Loss interface {
	Value(prediction, target float64) float64
	OutputDelta(prediction, target float64) float64
}

// SigmoidBinaryCrossEntropy is the baseline loss for one sigmoid output.
type SigmoidBinaryCrossEntropy struct {
	Epsilon float64
}

func DefaultLoss() Loss {
	return DefaultSigmoidBinaryCrossEntropy()
}

func DefaultSigmoidBinaryCrossEntropy() SigmoidBinaryCrossEntropy {
	return SigmoidBinaryCrossEntropy{Epsilon: defaultLossEpsilon}
}

func (l SigmoidBinaryCrossEntropy) Value(prediction, target float64) float64 {
	epsilon := l.epsilon()
	if prediction < epsilon {
		prediction = epsilon
	}
	if prediction > 1-epsilon {
		prediction = 1 - epsilon
	}
	return -(target*math.Log(prediction) + (1-target)*math.Log(1-prediction))
}

func (SigmoidBinaryCrossEntropy) OutputDelta(prediction, target float64) float64 {
	return prediction - target
}

func (l SigmoidBinaryCrossEntropy) epsilon() float64 {
	if l.Epsilon <= 0 {
		return defaultLossEpsilon
	}
	return l.Epsilon
}

func BinaryCrossEntropy(prediction, target float64) float64 {
	return DefaultLoss().Value(prediction, target)
}
