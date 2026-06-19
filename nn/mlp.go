package nn

import (
	"fmt"
	"math/rand"
)

// MLP representa uma rede totalmente conectada manual:
// entrada -> uma ou mais camadas ocultas ReLU -> cabeça de saída configurável.
// Hidden é um slice para permitir comparar arquiteturas com diferentes profundidades.
type MLP struct {
	Hidden []DenseLayer
	Output DenseLayer

	OutputHead OutputHead

	HiddenZ [][]float64
	HiddenA [][]float64
	OutputZ []float64

	DeltaHidden [][]float64
	DeltaOutput []float64
}

// SparseActivationStats resume a esparsidade observada em uma camada oculta.
type SparseActivationStats struct {
	LayerIndex  int
	Size        int
	Active      int
	ActiveRatio float64
	Sparsity    float64
}

// SparseForwardStats resume o custo estimado de um forward esparso.
// DenseOps e SparseOps contam multiplicações de camadas lineares.
// A primeira camada oculta continua densa nesta fase, pois a imagem de entrada
// ainda é tratada como vetor denso.
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

// SparseForwardWorkspace guarda buffers reutilizáveis para forward esparso.
// Ele evita alocações repetidas de ActiveVector durante benchmarks e inferência.
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
		Active:      active,
		HiddenStats: make([]SparseActivationStats, 0, len(model.Hidden)),
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

// NewMLP preserva a API da baseline original de uma camada oculta.
func NewMLP(inputSize, hiddenSize int, seed int64) *MLP {
	return NewMLPWithHiddenSizes(inputSize, []int{hiddenSize}, seed)
}

// NewMLPWithHiddenSizes cria uma MLP com saída sigmoid binária por padrão.
func NewMLPWithHiddenSizes(inputSize int, hiddenSizes []int, seed int64) *MLP {
	return NewMLPWithHiddenSizesAndHead(inputSize, hiddenSizes, seed, string(OutputHeadSigmoid1))
}

// NewMLPWithHiddenSizesAndHead cria uma MLP com uma ou mais camadas ocultas e cabeça configurável.
// O seed fixa a inicialização dos pesos para tornar os experimentos reproduzíveis.
func NewMLPWithHiddenSizesAndHead(inputSize int, hiddenSizes []int, seed int64, outputHead string) *MLP {
	if inputSize <= 0 {
		panic(fmt.Sprintf("invalid input size: %d", inputSize))
	}
	if len(hiddenSizes) == 0 {
		panic("MLP requires at least one hidden layer")
	}

	head := mustNormalizeOutputHead(outputHead)
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

	outputSize := head.OutputSize()
	return &MLP{
		Hidden:     hidden,
		Output:     NewDenseLayer(previousSize, outputSize, rng),
		OutputHead: head,

		HiddenZ: hiddenZ,
		HiddenA: hiddenA,
		OutputZ: make([]float64, outputSize),

		DeltaHidden: deltaHidden,
		DeltaOutput: make([]float64, outputSize),
	}
}

func (m *MLP) Head() OutputHead {
	if m == nil {
		return OutputHeadSigmoid1
	}
	return normalizeModelOutputHead(m.OutputHead)
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
		Hidden:     cloneDenseLayerSlice(m.Hidden),
		Output:     m.Output.Clone(),
		OutputHead: m.Head(),

		HiddenZ: cloneFloat64Matrix(m.HiddenZ),
		HiddenA: cloneFloat64Matrix(m.HiddenA),
		OutputZ: cloneFloat64Slice(m.OutputZ),

		DeltaHidden: cloneFloat64Matrix(m.DeltaHidden),
		DeltaOutput: cloneFloat64Slice(m.DeltaOutput),
	}
}

// Forward calcula a probabilidade da classe positiva, dog, para uma amostra.
func (m *MLP) Forward(x []float64) float64 {
	if len(x) != m.InputSize() {
		panic(fmt.Sprintf("invalid forward input length: expected %d, got %d", m.InputSize(), len(x)))
	}
	return m.forwardUnchecked(x)
}

// ForwardFast calcula a saída densa com validação mínima, para benchmarks controlados.
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
	return m.Head().PositiveProbability(m.OutputZ)
}

func (m *MLP) LossFromLastForward(y float64) float64 {
	return m.Head().LossFromLogits(m.OutputZ, y)
}

func (m *MLP) PredictClassFromLastForward() int {
	return m.Head().PredictClass(m.OutputZ)
}

// ForwardSparseExact calcula a saída usando propagação esparsa dinâmica exata.
// Apenas ativações ReLU exatamente nulas são removidas, preservando a saída da MLP densa.
func (m *MLP) ForwardSparseExact(x []float64) float64 {
	workspace := NewSparseForwardWorkspace(m)
	return m.ForwardSparseFast(x, 0, workspace)
}

// ForwardSparseFast calcula a saída esparsa sem produzir métricas.
// Use este método para benchmark de inferência, reutilizando workspace entre amostras.
func (m *MLP) ForwardSparseFast(x []float64, threshold float64, workspace *SparseForwardWorkspace) float64 {
	workspace.mustMatchModel(m)
	return m.ForwardSparsePrepared(x, threshold, workspace)
}

// ForwardSparsePrepared é a versão sem validações por chamada.
// O workspace deve ter sido criado para o mesmo modelo por NewSparseForwardWorkspace.
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
	return m.Head().PositiveProbability(m.OutputZ)
}

// ForwardSparseWithStats calcula a saída usando propagação esparsa dinâmica.
// threshold=0 corresponde à DSA exata. Thresholds positivos geram uma aproximação
// experimental, removendo também ativações positivas pequenas.
func (m *MLP) ForwardSparseWithStats(x []float64, threshold float64) (float64, SparseForwardStats) {
	workspace := NewSparseForwardWorkspace(m)
	return m.ForwardSparseWithStatsWorkspace(x, threshold, workspace)
}

// ForwardSparseWithStatsWorkspace é a versão instrumentada com buffers reutilizáveis.
// Os slices de stats retornados são válidos até a próxima chamada usando o mesmo workspace.
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

	return m.Head().PositiveProbability(m.OutputZ), stats
}

// Backward acumula os gradientes de uma amostra.
// Para sigmoid+BCE e softmax+cross-entropy, o delta da saída é probabilidade - alvo.
func (m *MLP) Backward(x []float64, yHat, y float64) {
	_ = yHat // mantido na assinatura para preservar a API anterior.
	m.Head().FillOutputDelta(m.OutputZ, y, m.DeltaOutput)

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

func estimateDenseOps(layer DenseLayer) int {
	return layer.In * layer.Out
}

func estimateSparseOps(input ActiveVector, outputSize int) int {
	return input.ActiveCount() * outputSize
}

func sparseActivationStats(layerIndex int, active ActiveVector) SparseActivationStats {
	return SparseActivationStats{
		LayerIndex:  layerIndex,
		Size:        active.Size,
		Active:      active.ActiveCount(),
		ActiveRatio: active.ActiveRatio(),
		Sparsity:    active.Sparsity(),
	}
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
