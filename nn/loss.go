package nn

import "math"

const defaultLossEpsilon = 1e-12

// Loss descreve uma função de perda escalar e o delta correspondente da camada de saída.
// O contrato atual assume uma saída sigmoid binária; perdas vetoriais, como softmax,
// terão um contrato próprio quando forem introduzidas.
type Loss interface {
	Value(prediction, target float64) float64
	OutputDelta(prediction, target float64) float64
}

// SigmoidBinaryCrossEntropy combina BCE com uma saída sigmoid.
// Essa combinação permite usar prediction-target como delta do logit de saída.
type SigmoidBinaryCrossEntropy struct {
	Epsilon float64
}

// DefaultSigmoidBinaryCrossEntropy retorna a loss padrão da baseline.
func DefaultSigmoidBinaryCrossEntropy() SigmoidBinaryCrossEntropy {
	return SigmoidBinaryCrossEntropy{Epsilon: defaultLossEpsilon}
}

// Value calcula Binary Cross Entropy para uma probabilidade prevista e rótulo binário.
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

// OutputDelta retorna dL/dz para sigmoid seguida de BCE.
func (SigmoidBinaryCrossEntropy) OutputDelta(prediction, target float64) float64 {
	return prediction - target
}

func (l SigmoidBinaryCrossEntropy) epsilon() float64 {
	if l.Epsilon <= 0 {
		return defaultLossEpsilon
	}
	return l.Epsilon
}

// BinaryCrossEntropy preserva a API simples da baseline e delega para loss.go.
func BinaryCrossEntropy(prediction, target float64) float64 {
	return DefaultSigmoidBinaryCrossEntropy().Value(prediction, target)
}
