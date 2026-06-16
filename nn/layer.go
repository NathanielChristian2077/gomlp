package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// DenseLayer representa uma camada totalmente conectada.
// A camada armazena pesos, biases e gradientes manualmente, sem depender de bibliotecas de ML.
// O layout dos pesos é input-major: o peso W[i,o] fica em Weights[i*Out+o].
// Esse formato deixa claro como cada entrada contribui para cada neurônio de saída.
type DenseLayer struct {
	In  int
	Out int

	Weights []float64
	Biases  []float64

	GradW []float64
	GradB []float64
}

type ActiveVector struct {
	Size   int
	Idx    []int
	Values []float64
}

// NewDenseLayer cria uma camada densa com inicialização He.
// A escala sqrt(2/in) é adequada para camadas que usam ReLU,
// ajudando a manter a magnitude das ativações mais estável no início do treino.
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

// Clone cria uma cópia profunda da camada.
// Isso é usado para guardar o melhor checkpoint por validação sem compartilhar slices
// com o modelo que continua sendo treinado nas épocas seguintes.
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

// Forward calcula z = xW + b.
// input contém as ativações da camada anterior.
// output recebe os logits da camada atual.
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

func (l *DenseLayer) ForwardSparse(input ActiveVector, z []float64) {
	copy(z, l.Biases)

	for k, j := range input.Idx {
		x := input.Values[k]

		base := j * l.Out

		for o := 0; o < l.Out; o++ {
			z[o] += x * l.Weights[base+o]
		}
	}
}

// ZeroGrad zera os acumuladores de gradiente antes de processar um novo batch.
func (l *DenseLayer) ZeroGrad() {
	for i := range l.GradW {
		l.GradW[i] = 0
	}
	for i := range l.GradB {
		l.GradB[i] = 0
	}
}

// AccumulateGrad acumula os gradientes da camada para uma amostra.
// Para cada peso W[i,o], o gradiente é input[i] * deltaOut[o].
// Para cada bias b[o], o gradiente é deltaOut[o].
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

// ApplyGrad aplica gradient descent aos pesos e biases.
// Os gradientes acumulados são divididos pelo tamanho do batch para usar a média do batch.
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
