package nn

import "math/rand"

// MLP representa a arquitetura da baseline: entrada -> camada oculta -> saída.
// A camada oculta usa ReLU e a saída usa sigmoid para classificação binária.
type MLP struct {
	Hidden DenseLayer
	Output DenseLayer

	HiddenZ []float64
	HiddenA []float64
	OutputZ []float64

	DeltaHidden []float64
	DeltaOutput []float64
}

// NewMLP cria a rede com uma camada oculta e seed fixa para reprodutibilidade.
func NewMLP(inputSize, hiddenSize int, seed int64) *MLP {
	rng := rand.New(rand.NewSource(seed))

	return &MLP{
		Hidden: NewDenseLayer(inputSize, hiddenSize, rng),
		Output: NewDenseLayer(hiddenSize, 1, rng),

		HiddenZ: make([]float64, hiddenSize),
		HiddenA: make([]float64, hiddenSize),
		OutputZ: make([]float64, 1),

		DeltaHidden: make([]float64, hiddenSize),
		DeltaOutput: make([]float64, 1),
	}
}

// Clone copia o modelo para guardar o melhor checkpoint de validação.
func (m *MLP) Clone() *MLP {
	if m == nil {
		return nil
	}

	return &MLP{
		Hidden: m.Hidden.Clone(),
		Output: m.Output.Clone(),

		HiddenZ: cloneFloat64Slice(m.HiddenZ),
		HiddenA: cloneFloat64Slice(m.HiddenA),
		OutputZ: cloneFloat64Slice(m.OutputZ),

		DeltaHidden: cloneFloat64Slice(m.DeltaHidden),
		DeltaOutput: cloneFloat64Slice(m.DeltaOutput),
	}
}

// Forward calcula a saída da rede para uma amostra.
func (m *MLP) Forward(x []float64) float64 {
	m.Hidden.Forward(x, m.HiddenZ)

	for i, z := range m.HiddenZ {
		m.HiddenA[i] = ReLU(z)
	}

	m.Output.Forward(m.HiddenA, m.OutputZ)

	return Sigmoid(m.OutputZ[0])
}

// Backward acumula os gradientes de uma amostra.
// Com sigmoid + Binary Cross Entropy, o delta da saída é yHat - y.
func (m *MLP) Backward(x []float64, yHat, y float64) {
	m.DeltaOutput[0] = yHat - y

	m.Output.AccumulateGrad(m.HiddenA, m.DeltaOutput)

	for i := 0; i < m.Hidden.Out; i++ {
		w := m.Output.Weights[i*m.Output.Out]
		m.DeltaHidden[i] = w * m.DeltaOutput[0] * ReLUDerivativeFromActivation(m.HiddenA[i])
	}

	m.Hidden.AccumulateGrad(x, m.DeltaHidden)
}

// ZeroGrad zera gradientes antes de processar um novo batch.
func (m *MLP) ZeroGrad() {
	m.Hidden.ZeroGrad()
	m.Output.ZeroGrad()
}

// ApplyGrad aplica a atualização de pesos após o batch.
func (m *MLP) ApplyGrad(lr float64, batchSize int) {
	m.Hidden.ApplyGrad(lr, batchSize)
	m.Output.ApplyGrad(lr, batchSize)
}
