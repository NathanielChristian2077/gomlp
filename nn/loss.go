package nn

import (
	"fmt"
	"math"
)

const defaultLossEpsilon = 1e-12

// OutputLoss evaluates logits and fills dL/dlogits for an output head.
type OutputLoss interface {
	LossFromLogits(logits []float64, target float64) float64
	FillOutputDelta(logits []float64, target float64, delta []float64)
}

// SigmoidBinaryCrossEntropy is the loss for one sigmoid output logit.
type SigmoidBinaryCrossEntropy struct {
	Epsilon float64
}

// SoftmaxCrossEntropy is the loss for mutually exclusive classes represented by logits.
type SoftmaxCrossEntropy struct{}

func DefaultSigmoidBinaryCrossEntropy() SigmoidBinaryCrossEntropy {
	return SigmoidBinaryCrossEntropy{Epsilon: defaultLossEpsilon}
}

func LossForOutputHead(head OutputHead) OutputLoss {
	switch normalizeModelOutputHead(head) {
	case OutputHeadSigmoid1:
		return DefaultSigmoidBinaryCrossEntropy()
	case OutputHeadSoftmax2:
		return SoftmaxCrossEntropy{}
	default:
		panic(fmt.Sprintf("unknown output head: %q", head))
	}
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

func (l SigmoidBinaryCrossEntropy) LossFromLogits(logits []float64, target float64) float64 {
	if len(logits) != 1 {
		panic(fmt.Sprintf("sigmoid1 expects 1 logit, got %d", len(logits)))
	}
	return l.Value(Sigmoid(logits[0]), target)
}

func (l SigmoidBinaryCrossEntropy) FillOutputDelta(logits []float64, target float64, delta []float64) {
	if len(logits) != 1 || len(delta) != 1 {
		panic(fmt.Sprintf("sigmoid1 expects 1 logit and 1 delta, got %d and %d", len(logits), len(delta)))
	}
	delta[0] = l.OutputDelta(Sigmoid(logits[0]), target)
}

func (l SigmoidBinaryCrossEntropy) epsilon() float64 {
	if l.Epsilon <= 0 {
		return defaultLossEpsilon
	}
	return l.Epsilon
}

func (SoftmaxCrossEntropy) LossFromLogits(logits []float64, target float64) float64 {
	class := int(target)
	if target != float64(class) || class < 0 || class >= len(logits) {
		panic(fmt.Sprintf("invalid softmax target %v for %d logits", target, len(logits)))
	}
	if len(logits) == 0 {
		panic("softmax cross-entropy requires at least one logit")
	}

	maxLogit := logits[0]
	for _, value := range logits[1:] {
		if value > maxLogit {
			maxLogit = value
		}
	}
	sumExp := 0.0
	for _, value := range logits {
		sumExp += math.Exp(value - maxLogit)
	}
	return maxLogit + math.Log(sumExp) - logits[class]
}

func (SoftmaxCrossEntropy) FillOutputDelta(logits []float64, target float64, delta []float64) {
	if len(delta) != len(logits) {
		panic(fmt.Sprintf("softmax delta length: expected %d got %d", len(logits), len(delta)))
	}
	class := int(target)
	if target != float64(class) || class < 0 || class >= len(logits) {
		panic(fmt.Sprintf("invalid softmax target %v for %d logits", target, len(logits)))
	}
	SoftmaxInto(logits, delta)
	delta[class] -= 1
}

// BinaryCrossEntropy preserves the baseline scalar helper API.
func BinaryCrossEntropy(prediction, target float64) float64 {
	return DefaultSigmoidBinaryCrossEntropy().Value(prediction, target)
}

// CrossEntropyFromLogits preserves the output-head helper API.
func CrossEntropyFromLogits(logits []float64, target int) float64 {
	return SoftmaxCrossEntropy{}.LossFromLogits(logits, float64(target))
}
