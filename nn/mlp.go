package nn

import (
	"fmt"
	"math/rand"
)

// MLP is a manual fully connected network with ReLU hidden layers and sigmoid output.
type MLP struct {
	Hidden []DenseLayer
	Output DenseLayer

	HiddenZ [][]float64
	HiddenA [][]float64
	OutputZ []float64

	DeltaHidden [][]float64
	DeltaOutput []float64
}

type SparseActivationStats struct {
	LayerIndex  int
	Size        int
	Active      int
	ActiveRatio float64
	Sparsity    float64
}

type SparseForwardStats struct {
	Hidden    []SparseActivationStats
	DenseOps  int
	SparseOps int
}

func (s SparseForwardStats) EstimatedSpeedup() float64 {
	if s.SparseOps == 0 {
		return 0
	}
	return float64(s.DenseOps) / float64(s.SparseOps)
}

// SparseForwardWorkspace owns reusable buffers for DSA inference and benchmarks.
type SparseForwardWorkspace struct {
	Active      []ActiveVector
	HiddenStats []SparseActivationStats
}

func NewSparseForwardWorkspace(model *MLP) *SparseForwardWorkspace {
	if model == nil || len(model.Hidden) == 0 {
		panic("cannot create sparse workspace for nil or empty model")
	}
	active := make([]ActiveVector, len(model.Hidden))
	for i, layer := range model.Hidden {
		active[i] = NewActiveVector(layer.Out)
	}
	return &SparseForwardWorkspace{
		Active: active, HiddenStats: make([]SparseActivationStats, 0, len(model.Hidden)),
	}
}

func (w *SparseForwardWorkspace) mustMatchModel(model *MLP) {
	if w == nil {
		panic("nil sparse forward workspace")
	}
	if model == nil || len(model.Hidden) == 0 {
		panic("invalid model for sparse forward workspace")
	}
	if len(w.Active) != len(model.Hidden) {
		panic(fmt.Sprintf("invalid sparse workspace hidden count: expected %d got %d", len(model.Hidden), len(w.Active)))
	}
	for i, layer := range model.Hidden {
		if cap(w.Active[i].Indices) < layer.Out || cap(w.Active[i].Values) < layer.Out {
			w.Active[i] = NewActiveVector(layer.Out)
		}
	}
}

func NewMLP(inputSize, hiddenSize int, seed int64) *MLP {
	return NewMLPWithHiddenSizes(inputSize, []int{hiddenSize}, seed)
}

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
		Hidden: hidden, Output: NewDenseLayer(previousSize, 1, rng),
		HiddenZ: hiddenZ, HiddenA: hiddenA, OutputZ: make([]float64, 1),
		DeltaHidden: deltaHidden, DeltaOutput: make([]float64, 1),
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

func (m *MLP) Clone() *MLP {
	if m == nil {
		return nil
	}
	return &MLP{
		Hidden: cloneDenseLayerSlice(m.Hidden), Output: m.Output.Clone(),
		HiddenZ: cloneFloat64Matrix(m.HiddenZ), HiddenA: cloneFloat64Matrix(m.HiddenA), OutputZ: cloneFloat64Slice(m.OutputZ),
		DeltaHidden: cloneFloat64Matrix(m.DeltaHidden), DeltaOutput: cloneFloat64Slice(m.DeltaOutput),
	}
}

func (m *MLP) Forward(x []float64) float64 {
	if len(x) != m.InputSize() {
		panic(fmt.Sprintf("invalid forward input length: expected %d, got %d", m.InputSize(), len(x)))
	}
	return m.forwardUnchecked(x)
}

func (m *MLP) ForwardFast(x []float64) float64 {
	return m.forwardUnchecked(x)
}

func (m *MLP) forwardUnchecked(x []float64) float64 {
	input := x
	for layerIndex := range m.Hidden {
		m.Hidden[layerIndex].forwardUnchecked(input, m.HiddenZ[layerIndex])
		for i, z := range m.HiddenZ[layerIndex] {
			m.HiddenA[layerIndex][i] = ReLU(z)
		}
		input = m.HiddenA[layerIndex]
	}
	m.Output.forwardUnchecked(input, m.OutputZ)
	return Sigmoid(m.OutputZ[0])
}

func (m *MLP) ForwardSparseExact(x []float64) float64 {
	workspace := NewSparseForwardWorkspace(m)
	return m.ForwardSparseFast(x, 0, workspace)
}

func (m *MLP) ForwardSparseFast(x []float64, threshold float64, workspace *SparseForwardWorkspace) float64 {
	workspace.mustMatchModel(m)
	return m.ForwardSparsePrepared(x, threshold, workspace)
}

func (m *MLP) ForwardSparsePrepared(x []float64, threshold float64, workspace *SparseForwardWorkspace) float64 {
	firstLayer := &m.Hidden[0]
	firstLayer.forwardUnchecked(x, m.HiddenZ[0])
	ReLUToActiveInto(m.HiddenZ[0], threshold, &workspace.Active[0])

	for layerIndex := 1; layerIndex < len(m.Hidden); layerIndex++ {
		layer := &m.Hidden[layerIndex]
		layer.forwardSparseUnchecked(workspace.Active[layerIndex-1], m.HiddenZ[layerIndex])
		ReLUToActiveInto(m.HiddenZ[layerIndex], threshold, &workspace.Active[layerIndex])
	}
	m.Output.forwardSparseUnchecked(workspace.Active[len(m.Hidden)-1], m.OutputZ)
	return Sigmoid(m.OutputZ[0])
}

func (m *MLP) ForwardSparseWithStats(x []float64, threshold float64) (float64, SparseForwardStats) {
	workspace := NewSparseForwardWorkspace(m)
	return m.ForwardSparseWithStatsWorkspace(x, threshold, workspace)
}

func (m *MLP) ForwardSparseWithStatsWorkspace(x []float64, threshold float64, workspace *SparseForwardWorkspace) (float64, SparseForwardStats) {
	workspace.mustMatchModel(m)
	workspace.HiddenStats = workspace.HiddenStats[:0]
	stats := SparseForwardStats{}

	firstLayer := &m.Hidden[0]
	firstLayer.forwardUnchecked(x, m.HiddenZ[0])
	stats.DenseOps += estimateDenseOps(*firstLayer)
	stats.SparseOps += estimateDenseOps(*firstLayer)
	ReLUToActiveInto(m.HiddenZ[0], threshold, &workspace.Active[0])
	workspace.HiddenStats = append(workspace.HiddenStats, sparseActivationStats(0, workspace.Active[0]))

	for layerIndex := 1; layerIndex < len(m.Hidden); layerIndex++ {
		layer := &m.Hidden[layerIndex]
		layer.forwardSparseUnchecked(workspace.Active[layerIndex-1], m.HiddenZ[layerIndex])
		stats.DenseOps += estimateDenseOps(*layer)
		stats.SparseOps += estimateSparseOps(workspace.Active[layerIndex-1], layer.Out)
		ReLUToActiveInto(m.HiddenZ[layerIndex], threshold, &workspace.Active[layerIndex])
		workspace.HiddenStats = append(workspace.HiddenStats, sparseActivationStats(layerIndex, workspace.Active[layerIndex]))
	}

	lastActive := workspace.Active[len(m.Hidden)-1]
	m.Output.forwardSparseUnchecked(lastActive, m.OutputZ)
	stats.DenseOps += estimateDenseOps(m.Output)
	stats.SparseOps += estimateSparseOps(lastActive, m.Output.Out)
	stats.Hidden = workspace.HiddenStats
	return Sigmoid(m.OutputZ[0]), stats
}

func (m *MLP) Backward(x []float64, yHat, y float64) {
	m.DeltaOutput[0] = m.OutputDelta(yHat, y)
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

		previousActivation := x
		if layerIndex > 0 {
			previousActivation = m.HiddenA[layerIndex-1]
		}
		m.Hidden[layerIndex].AccumulateGrad(previousActivation, m.DeltaHidden[layerIndex])
	}
}

func (m *MLP) ZeroGrad() {
	for i := range m.Hidden {
		m.Hidden[i].ZeroGrad()
	}
	m.Output.ZeroGrad()
}

// ApplyGrad is retained for small manual experiments; training uses Optimizer.
func (m *MLP) ApplyGrad(learningRate float64, batchSize int) {
	for i := range m.Hidden {
		m.Hidden[i].ApplyGrad(learningRate, batchSize)
	}
	m.Output.ApplyGrad(learningRate, batchSize)
}

func estimateDenseOps(layer DenseLayer) int { return layer.In * layer.Out }
func estimateSparseOps(input ActiveVector, outputSize int) int { return input.ActiveCount() * outputSize }

func sparseActivationStats(layerIndex int, active ActiveVector) SparseActivationStats {
	return SparseActivationStats{LayerIndex: layerIndex, Size: active.Size, Active: active.ActiveCount(), ActiveRatio: active.ActiveRatio(), Sparsity: active.Sparsity()}
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
