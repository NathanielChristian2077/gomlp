package nn

import (
	"fmt"
	"math"
	"strings"
)

// OutputHead defines how output logits become probabilities and classes.
type OutputHead string

const (
	OutputHeadSigmoid1 OutputHead = "sigmoid1"
	OutputHeadSoftmax2 OutputHead = "softmax2"
)

func NormalizeOutputHead(raw string) (OutputHead, error) {
	switch OutputHead(strings.ToLower(strings.TrimSpace(raw))) {
	case "", OutputHeadSigmoid1:
		return OutputHeadSigmoid1, nil
	case OutputHeadSoftmax2:
		return OutputHeadSoftmax2, nil
	case "softmax1":
		return "", fmt.Errorf("softmax1 is invalid: softmax over one logit is always 1")
	default:
		return "", fmt.Errorf("unknown output head %q: expected %q or %q", raw, OutputHeadSigmoid1, OutputHeadSoftmax2)
	}
}

func mustNormalizeOutputHead(raw string) OutputHead {
	head, err := NormalizeOutputHead(raw)
	if err != nil {
		panic(err)
	}
	return head
}

func normalizeModelOutputHead(head OutputHead) OutputHead {
	if head == "" {
		return OutputHeadSigmoid1
	}
	return head
}

func (h OutputHead) OutputSize() int {
	switch normalizeModelOutputHead(h) {
	case OutputHeadSigmoid1:
		return 1
	case OutputHeadSoftmax2:
		return 2
	default:
		panic(fmt.Sprintf("unknown output head: %q", h))
	}
}

func (h OutputHead) PositiveProbability(logits []float64) float64 {
	switch normalizeModelOutputHead(h) {
	case OutputHeadSigmoid1:
		if len(logits) != 1 {
			panic(fmt.Sprintf("sigmoid1 expects 1 logit, got %d", len(logits)))
		}
		return Sigmoid(logits[0])
	case OutputHeadSoftmax2:
		if len(logits) != 2 {
			panic(fmt.Sprintf("softmax2 expects 2 logits, got %d", len(logits)))
		}
		return SoftmaxClassProbability(logits, 1)
	default:
		panic(fmt.Sprintf("unknown output head: %q", h))
	}
}

func (h OutputHead) PredictClass(logits []float64) int {
	switch normalizeModelOutputHead(h) {
	case OutputHeadSigmoid1:
		if h.PositiveProbability(logits) >= DefaultClassificationThreshold {
			return 1
		}
		return 0
	case OutputHeadSoftmax2:
		if len(logits) != 2 {
			panic(fmt.Sprintf("softmax2 expects 2 logits, got %d", len(logits)))
		}
		if logits[1] >= logits[0] {
			return 1
		}
		return 0
	default:
		panic(fmt.Sprintf("unknown output head: %q", h))
	}
}

// LossFromLogits delegates loss mathematics to loss.go.
func (h OutputHead) LossFromLogits(logits []float64, y float64) float64 {
	return LossForOutputHead(h).LossFromLogits(logits, y)
}

// FillOutputDelta delegates dL/dlogits to loss.go.
func (h OutputHead) FillOutputDelta(logits []float64, y float64, delta []float64) {
	LossForOutputHead(h).FillOutputDelta(logits, y, delta)
}

func SoftmaxInto(logits []float64, out []float64) {
	if len(logits) == 0 {
		panic("softmax requires at least one logit")
	}
	if len(out) != len(logits) {
		panic(fmt.Sprintf("invalid softmax output length: expected %d got %d", len(logits), len(out)))
	}

	maxLogit := logits[0]
	for _, value := range logits[1:] {
		if value > maxLogit {
			maxLogit = value
		}
	}

	sum := 0.0
	for i, value := range logits {
		exp := math.Exp(value - maxLogit)
		out[i] = exp
		sum += exp
	}
	if sum == 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		panic("invalid softmax normalization")
	}
	for i := range out {
		out[i] /= sum
	}
}

func SoftmaxClassProbability(logits []float64, classIndex int) float64 {
	if classIndex < 0 || classIndex >= len(logits) {
		panic(fmt.Sprintf("invalid softmax class index %d for %d logits", classIndex, len(logits)))
	}
	maxLogit := logits[0]
	for _, value := range logits[1:] {
		if value > maxLogit {
			maxLogit = value
		}
	}
	numerator := math.Exp(logits[classIndex] - maxLogit)
	denominator := 0.0
	for _, value := range logits {
		denominator += math.Exp(value - maxLogit)
	}
	return numerator / denominator
}
