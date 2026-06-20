package nn

import (
	"fmt"
	"math"
)

const (
	DefaultAdamBeta1   = 0.9
	DefaultAdamBeta2   = 0.999
	DefaultAdamEpsilon = 1e-8
)

// Optimizer aplica gradientes acumulados em um modelo.
// Implementações com estado, como Adam, devem ser reutilizadas entre épocas.
type Optimizer interface {
	Model() *MLP
	Step(batchSize int)
}

// AdamConfig guarda os hiperparâmetros do Adam.
// LearningRate é obrigatório. Beta1, Beta2 e Epsilon usam defaults quando zero.
type AdamConfig struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
}

// DefaultAdamConfig retorna a configuração clássica do Adam.
func DefaultAdamConfig(learningRate float64) AdamConfig {
	return AdamConfig{
		LearningRate: learningRate,
		Beta1:        DefaultAdamBeta1,
		Beta2:        DefaultAdamBeta2,
		Epsilon:      DefaultAdamEpsilon,
	}
}

// AdamOptimizer implementa Adam manualmente para todos os pesos e biases da MLP.
type AdamOptimizer struct {
	model  *MLP
	config AdamConfig

	step       int
	beta1Power float64
	beta2Power float64

	hidden []adamLayerState
	output adamLayerState
}

type adamLayerState struct {
	WeightM []float64
	WeightV []float64
	BiasM   []float64
	BiasV   []float64
}

// NewAdamOptimizer cria Adam usando beta1=0.9, beta2=0.999 e epsilon=1e-8.
func NewAdamOptimizer(model *MLP, learningRate float64) *AdamOptimizer {
	return NewAdamOptimizerWithConfig(model, DefaultAdamConfig(learningRate))
}

// NewAdamOptimizerWithConfig cria Adam para o modelo informado.
// O otimizador mantém momentos internos e deve ser reaproveitado a cada batch.
func NewAdamOptimizerWithConfig(model *MLP, config AdamConfig) *AdamOptimizer {
	if model == nil {
		panic("cannot create Adam optimizer for nil model")
	}
	if len(model.Hidden) == 0 {
		panic("cannot create Adam optimizer for model without hidden layers")
	}

	config = normalizeAdamConfig(config)
	if err := validateAdamConfig(config); err != nil {
		panic(err.Error())
	}

	hidden := make([]adamLayerState, len(model.Hidden))
	for i := range model.Hidden {
		hidden[i] = newAdamLayerState(model.Hidden[i])
	}

	return &AdamOptimizer{
		model:      model,
		config:     config,
		beta1Power: 1,
		beta2Power: 1,
		hidden:     hidden,
		output:     newAdamLayerState(model.Output),
	}
}

func (o *AdamOptimizer) Model() *MLP {
	if o == nil {
		return nil
	}
	return o.model
}

func (o *AdamOptimizer) Config() AdamConfig {
	if o == nil {
		return AdamConfig{}
	}
	return o.config
}

func (o *AdamOptimizer) StepCount() int {
	if o == nil {
		return 0
	}
	return o.step
}

// Step aplica uma atualização Adam usando os gradientes acumulados no modelo.
// batchSize converte os gradientes acumulados em média de batch, igual ao SGD atual.
func (o *AdamOptimizer) Step(batchSize int) {
	if o == nil {
		panic("nil Adam optimizer")
	}
	if batchSize <= 0 {
		panic(fmt.Sprintf("invalid batch size: %d", batchSize))
	}
	o.mustMatchModel()

	o.step++
	o.beta1Power *= o.config.Beta1
	o.beta2Power *= o.config.Beta2

	gradientScale := 1.0 / float64(batchSize)
	weightLearningRate := o.config.LearningRate
	beta1Correction := 1.0 - o.beta1Power
	beta2Correction := 1.0 - o.beta2Power

	for i := range o.model.Hidden {
		applyAdamToLayer(
			&o.model.Hidden[i],
			&o.hidden[i],
			gradientScale,
			weightLearningRate,
			o.config.Beta1,
			o.config.Beta2,
			o.config.Epsilon,
			beta1Correction,
			beta2Correction,
		)
	}

	applyAdamToLayer(
		&o.model.Output,
		&o.output,
		gradientScale,
		weightLearningRate,
		o.config.Beta1,
		o.config.Beta2,
		o.config.Epsilon,
		beta1Correction,
		beta2Correction,
	)
}

func newAdamLayerState(layer DenseLayer) adamLayerState {
	return adamLayerState{
		WeightM: make([]float64, len(layer.Weights)),
		WeightV: make([]float64, len(layer.Weights)),
		BiasM:   make([]float64, len(layer.Biases)),
		BiasV:   make([]float64, len(layer.Biases)),
	}
}

func normalizeAdamConfig(config AdamConfig) AdamConfig {
	if config.Beta1 == 0 {
		config.Beta1 = DefaultAdamBeta1
	}
	if config.Beta2 == 0 {
		config.Beta2 = DefaultAdamBeta2
	}
	if config.Epsilon == 0 {
		config.Epsilon = DefaultAdamEpsilon
	}
	return config
}

func validateAdamConfig(config AdamConfig) error {
	if config.LearningRate <= 0 || math.IsNaN(config.LearningRate) || math.IsInf(config.LearningRate, 0) {
		return fmt.Errorf("invalid Adam learning rate: %g", config.LearningRate)
	}
	if config.Beta1 < 0 || config.Beta1 >= 1 || math.IsNaN(config.Beta1) {
		return fmt.Errorf("invalid Adam beta1: %g", config.Beta1)
	}
	if config.Beta2 < 0 || config.Beta2 >= 1 || math.IsNaN(config.Beta2) {
		return fmt.Errorf("invalid Adam beta2: %g", config.Beta2)
	}
	if config.Epsilon <= 0 || math.IsNaN(config.Epsilon) || math.IsInf(config.Epsilon, 0) {
		return fmt.Errorf("invalid Adam epsilon: %g", config.Epsilon)
	}
	return nil
}

func (o *AdamOptimizer) mustMatchModel() {
	if o.model == nil {
		panic("Adam optimizer has nil model")
	}
	if len(o.hidden) != len(o.model.Hidden) {
		panic(fmt.Sprintf("Adam state hidden layer count mismatch: expected %d, got %d", len(o.model.Hidden), len(o.hidden)))
	}
	for i := range o.model.Hidden {
		mustMatchAdamLayerState(o.model.Hidden[i], o.hidden[i], fmt.Sprintf("hidden layer %d", i))
	}
	mustMatchAdamLayerState(o.model.Output, o.output, "output layer")
}

func mustMatchAdamLayerState(layer DenseLayer, state adamLayerState, name string) {
	if len(state.WeightM) != len(layer.Weights) || len(state.WeightV) != len(layer.Weights) {
		panic(fmt.Sprintf("Adam state weight length mismatch for %s", name))
	}
	if len(state.BiasM) != len(layer.Biases) || len(state.BiasV) != len(layer.Biases) {
		panic(fmt.Sprintf("Adam state bias length mismatch for %s", name))
	}
	if len(layer.GradW) != len(layer.Weights) {
		panic(fmt.Sprintf("gradient weight length mismatch for %s", name))
	}
	if len(layer.GradB) != len(layer.Biases) {
		panic(fmt.Sprintf("gradient bias length mismatch for %s", name))
	}
}

func applyAdamToLayer(layer *DenseLayer, state *adamLayerState, gradientScale, learningRate, beta1, beta2, epsilon, beta1Correction, beta2Correction float64) {
	for i, grad := range layer.GradW {
		g := grad * gradientScale
		state.WeightM[i] = beta1*state.WeightM[i] + (1-beta1)*g
		state.WeightV[i] = beta2*state.WeightV[i] + (1-beta2)*g*g

		mHat := state.WeightM[i] / beta1Correction
		vHat := state.WeightV[i] / beta2Correction
		layer.Weights[i] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
	}

	for i, grad := range layer.GradB {
		g := grad * gradientScale
		state.BiasM[i] = beta1*state.BiasM[i] + (1-beta1)*g
		state.BiasV[i] = beta2*state.BiasV[i] + (1-beta2)*g*g

		mHat := state.BiasM[i] / beta1Correction
		vHat := state.BiasV[i] / beta2Correction
		layer.Biases[i] -= learningRate * mHat / (math.Sqrt(vHat) + epsilon)
	}
}
