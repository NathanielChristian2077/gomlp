package nn

import (
	"fmt"
	"math/rand"
)

// MLP representa uma rede totalmente conectada manual:
// entrada -> uma ou mais camadas ocultas ReLU -> saída sigmoid.
// Hidden é um slice para permitir comparar arquiteturas com diferentes profundidades.
type MLP struct {
	Hidden []DenseLayer
	Output DenseLayer

	HiddenZ [][]float64
	HiddenA [][]float64
	OutputZ []float64

	DeltaHidden [][]float64
	DeltaOutput []float64
}

// NewMLP preserva a API da baseline original de uma camada oculta.
func NewMLP(inputSize, hiddenSize int, seed int64) *MLP {
	return NewMLPWithHiddenSizes(inputSize, []int{hiddenSize}, seed)
}

// NewMLPWithHiddenSizes cria uma MLP com uma ou mais camadas ocultas.
// O seed fixa a inicialização dos pesos para tornar os experimentos reproduzíveis.
func NewMLPWithHiddenSizes(inputSize int, hiddenSizes []int, seed int64) *MLP {
	if inputSize <= 0 {
		panic(fmt.Sprintf("invalid input size: %d", inputSize))
	}
	if len(hiddenSizes) == 0 {
		panic("MLP requires at least one hidden layer")
	}

	rng := rand.New(rand.NewSource(seed))
	hidden := make([]DenseLayer, len(hiddenSizes))
	hiddenZ := make([][]float64, len(hiddenSizes))
	hiddenA := make([][]float64, len(hiddenSizes))
	deltaHidden := make([][]float64, len(hiddenSizes))

	previousSize := inputSize
	for i, size := range hiddenSizes {
		if size <= 0 {
			panic(fmt.Sprintf("invalid hidden layer %d size: %d", i, size))
		}

		hidden[i] = NewDenseLayer(previousSize, size, rng)
		hiddenZ[i] = make([]float64, size)
		hiddenA[i] = make([]float64, size)
		deltaHidden[i] = make([]float64, size)
		previousSize = size
	}

	return &MLP{
		Hidden: hidden,
		Output: NewDenseLayer(previousSize, 1, rng),

		HiddenZ: hiddenZ,
		HiddenA: hiddenA,
		OutputZ: make([]float64, 1),

		DeltaHidden: deltaHidden,
		DeltaOutput: make([]float64, 1),
	}
}

func (m *MLP) InputSize() int {
	if m == nil || len(m.Hidden) == 0 {
		return 0
	}
	return m.Hidden[0].In
}

func (m *MLP) HiddenSizes() []int {
	if m == nil {
		return nil
	}

	sizes := make([]int, len(m.Hidden))
	for i, layer := range m.Hidden {
		sizes[i] = layer.Out
	}
	return sizes
}

// Clone copia o modelo para guardar o melhor checkpoint de validação.
func (m *MLP) Clone() *MLP {
	if m == nil {
		return nil
	}

	return &MLP{
		Hidden: cloneDenseLayerSlice(m.Hidden),
		Output: m.Output.Clone(),

		HiddenZ: cloneFloat64Matrix(m.HiddenZ),
		HiddenA: cloneFloat64Matrix(m.HiddenA),
		OutputZ: cloneFloat64Slice(m.OutputZ),

		DeltaHidden: cloneFloat64Matrix(m.DeltaHidden),
		DeltaOutput: cloneFloat64Slice(m.DeltaOutput),
	}
}

// Forward calcula a saída da rede para uma amostra.
func (m *MLP) Forward(x []float64) float64 {
	input := x

	for layerIndex := range m.Hidden {
		m.Hidden[layerIndex].Forward(input, m.HiddenZ[layerIndex])

		for i, z := range m.HiddenZ[layerIndex] {
			m.HiddenA[layerIndex][i] = ReLU(z)
		}

		input = m.HiddenA[layerIndex]
	}

	m.Output.Forward(input, m.OutputZ)

	return Sigmoid(m.OutputZ[0])
}

// Backward acumula os gradientes de uma amostra.
// Com sigmoid + Binary Cross Entropy, o delta da saída é yHat - y.
func (m *MLP) Backward(x []float64, yHat, y float64) {
	m.DeltaOutput[0] = yHat - y

	lastHidden := len(m.Hidden) - 1
	m.Output.AccumulateGrad(m.HiddenA[lastHidden], m.DeltaOutput)

	for layerIndex := lastHidden; layerIndex >= 0; layerIndex-- {
		for neuron := 0; neuron < m.Hidden[layerIndex].Out; neuron++ {
			sum := 0.0

			if layerIndex == lastHidden {
				base := neuron * m.Output.Out
				for o := 0; o < m.Output.Out; o++ {
					sum += m.Output.Weights[base+o] * m.DeltaOutput[o]
				}
			} else {
				nextLayer := m.Hidden[layerIndex+1]
				base := neuron * nextLayer.Out
				for o := 0; o < nextLayer.Out; o++ {
					sum += nextLayer.Weights[base+o] * m.DeltaHidden[layerIndex+1][o]
				}
			}

			m.DeltaHidden[layerIndex][neuron] = sum * ReLUDerivativeFromActivation(m.HiddenA[layerIndex][neuron])
		}

		var previousActivation []float64
		if layerIndex == 0 {
			previousActivation = x
		} else {
			previousActivation = m.HiddenA[layerIndex-1]
		}

		m.Hidden[layerIndex].AccumulateGrad(previousActivation, m.DeltaHidden[layerIndex])
	}
}

// ZeroGrad zera gradientes antes de processar um novo batch.
func (m *MLP) ZeroGrad() {
	for i := range m.Hidden {
		m.Hidden[i].ZeroGrad()
	}
	m.Output.ZeroGrad()
}

// ApplyGrad aplica a atualização de pesos após o batch.
func (m *MLP) ApplyGrad(lr float64, batchSize int) {
	for i := range m.Hidden {
		m.Hidden[i].ApplyGrad(lr, batchSize)
	}
	m.Output.ApplyGrad(lr, batchSize)
}

func cloneDenseLayerSlice(layers []DenseLayer) []DenseLayer {
	out := make([]DenseLayer, len(layers))
	for i, layer := range layers {
		out[i] = layer.Clone()
	}
	return out
}

func cloneFloat64Matrix(values [][]float64) [][]float64 {
	out := make([][]float64, len(values))
	for i, row := range values {
		out[i] = cloneFloat64Slice(row)
	}
	return out
}
