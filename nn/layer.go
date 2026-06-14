package nn

import (
	"fmt"
	"math"
	"math/rand"
)

type DenseLayer struct {
	In  int
	Out int

	Weights []float64
	Biases  []float64

	GradW []float64
	GradB []float64
}

func NewDenseLayer(in, out int, rng *rand.Rand) DenseLayer {
	if in <= 0 || out <= 0 {
		panic(fmt.Sprintf("invalid dense layer shape: in=%d, out=%d", in, out))
	}
	if rng == nil {
		panic("nil random source")
	}

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

func (l DenseLayer) Clone() DenseLayer {
	return DenseLayer{
		In:      l.In,
		Out:     l.Out,
		Weights: cloneFloat64Slice(l.Weights),
		Biases:  cloneFloat64Slice(l.Biases),
		GradW:   cloneFloat64Slice(l.GradW),
		GradB:   cloneFloat64Slice(l.GradB),
	}
}

func (l *DenseLayer) Forward(input, output []float64) {
	l.mustMatchInput(input, "forward input")
	l.mustMatchOutput(output, "forward output")

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
	l.mustMatchInput(input, "gradient input")
	l.mustMatchOutput(deltaOut, "gradient delta")

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
	if batchSize <= 0 {
		panic(fmt.Sprintf("invalid batch size: %d", batchSize))
	}

	scale := lr / float64(batchSize)

	for i := range l.Weights {
		l.Weights[i] -= scale * l.GradW[i]
	}

	for i := range l.Biases {
		l.Biases[i] -= scale * l.GradB[i]
	}
}

func (l *DenseLayer) mustMatchInput(values []float64, name string) {
	if len(values) != l.In {
		panic(fmt.Sprintf("invalid %s length: expected %d, got %d", name, l.In, len(values)))
	}
}

func (l *DenseLayer) mustMatchOutput(values []float64, name string) {
	if len(values) != l.Out {
		panic(fmt.Sprintf("invalid %s length: expected %d, got %d", name, l.Out, len(values)))
	}
}

func cloneFloat64Slice(values []float64) []float64 {
	out := make([]float64, len(values))
	copy(out, values)
	return out
}
