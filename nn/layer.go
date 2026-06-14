package nn

import (
	"math"
	"math/rand"
)

type DenseLayer struct {
	In  int
	Out int

	Weights []float64 // W[input][output] -> já explico...
	Biases  []float64

	GradW []float64
	GradB []float64
}

func NewDenseLayer(in, out int, rng *rand.Rand) DenseLayer {
	w := make([]float64, in*out)
	b := make([]float64, out)

	scale := math.Sqrt(2.0 / float64(in))

	for i := range w {
		w[i] = rng.NormFloat64() * scale
	}

	return DenseLayer{
		In:      in,
		Out:     out,
		Weights: w,
		Biases:  b,
		GradW:   make([]float64, in*out),
		GradB:   make([]float64, out),
	}
}

func (l *DenseLayer) Forward(input, output []float64) {
	copy(output, l.Biases)

	for i := 0; i < l.In; i++ {
		x := input[i]
		base := i * l.Out

		for o := 0; o < l.Out; o++ {
			output[o] += x * l.Weights[base+o]
		}
	}
}

func (l *DenseLayer) ZeroGrad() {
	for i := range l.GradW {
		l.GradW[i] = 0
	}
	for i := range l.GradB {
		l.GradB[i] = 0
	}
}

func (l *DenseLayer) AccumulateGrad(input []float64, deltaOut []float64) {
	for o := 0; o < l.Out; o++ {
		l.GradB[o] += deltaOut[o]
	}

	for i := 0; i < l.In; i++ {
		x := input[i]
		base := i * l.Out

		for o := 0; o < l.Out; o++ {
			l.GradW[base+o] += x * deltaOut[o]
		}
	}
}

func (l *DenseLayer) ApplyGrad(lr float64, batchSize int) {
	scale := lr / float64(batchSize)

	for i := range l.Weights {
		l.Weights[i] -= scale * l.GradW[i]
	}

	for i := range l.Biases {
		l.Biases[i] -= scale * l.GradB[i]
	}
}
