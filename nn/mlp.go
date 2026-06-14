package nn

import "math/rand"

type MLP struct {
	Hidden DenseLayer
	Output DenseLayer

	HiddenZ []float64
	HiddenA []float64
	OutputZ []float64

	DeltaHidden []float64
	DeltaOutput []float64
}

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

func (m *MLP) Forward(x []float64) float64 {
	m.Hidden.Forward(x, m.HiddenZ)

	for i, z := range m.HiddenZ {
		m.HiddenA[i] = ReLU(z)
	}

	m.Output.Forward(m.HiddenA, m.OutputZ)

	return Sigmoid(m.OutputZ[0])
}

func (m *MLP) Backward(x []float64, yHat, y float64) {
	// Sigmoid + BCE simplifica:
	// delta = yHat - y
	m.DeltaOutput[0] = yHat - y

	m.Output.AccumulateGrad(m.HiddenA, m.DeltaOutput)

	for i := 0; i < m.Hidden.Out; i++ {
		w := m.Output.Weights[i*m.Output.Out] // Out = 1
		m.DeltaHidden[i] = w * m.DeltaOutput[0] * ReLUDerivativeFromActivation(m.HiddenA[i])
	}

	m.Hidden.AccumulateGrad(x, m.DeltaHidden)
}

func (m *MLP) ZeroGrad() {
	m.Hidden.ZeroGrad()
	m.Output.ZeroGrad()
}

func (m *MLP) ApplyGrad(lr float64, batchSize int) {
	m.Hidden.ApplyGrad(lr, batchSize)
	m.Output.ApplyGrad(lr, batchSize)
}
