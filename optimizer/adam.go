// Package optimizer contains parameter-update strategies for nn models.
package optimizer

import (
	"fmt"
	"math"

	"github.com/NathanielChristian2077/gomlp/nn"
)

const (
	DefaultAdamBeta1   = 0.9
	DefaultAdamBeta2   = 0.999
	DefaultAdamEpsilon = 1e-8
)

// AdamConfig controls Adam. LearningRate is mandatory; zero-valued moment parameters use defaults.
type AdamConfig struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
}

func DefaultAdamConfig(learningRate float64) AdamConfig {
	return AdamConfig{LearningRate: learningRate, Beta1: DefaultAdamBeta1, Beta2: DefaultAdamBeta2, Epsilon: DefaultAdamEpsilon}
}

// Adam is a stateful Adam optimizer. Reuse one instance across all batches of a training run.
type Adam struct {
	model  *nn.MLP
	config AdamConfig

	step       int
	beta1Power float64
	beta2Power float64

	hidden []layerState
	output layerState
}

type layerState struct {
	weightM []float64
	weightV []float64
	biasM   []float64
	biasV   []float64
}

func NewAdam(model *nn.MLP, learningRate float64) *Adam {
	return NewAdamWithConfig(model, DefaultAdamConfig(learningRate))
}

func NewAdamWithConfig(model *nn.MLP, config AdamConfig) *Adam {
	if model == nil || len(model.Hidden) == 0 {
		panic("cannot create Adam for nil or empty model")
	}
	config = normalize(config)
	if err := validate(config); err != nil {
		panic(err)
	}

	hidden := make([]layerState, len(model.Hidden))
	for i := range model.Hidden {
		hidden[i] = newLayerState(model.Hidden[i])
	}

	return &Adam{model: model, config: config, beta1Power: 1, beta2Power: 1, hidden: hidden, output: newLayerState(model.Output)}
}

func (o *Adam) Model() *nn.MLP { return o.model }
func (o *Adam) Config() AdamConfig { return o.config }
func (o *Adam) StepCount() int { return o.step }

// Step applies one Adam update using gradients accumulated over batchSize samples.
func (o *Adam) Step(batchSize int) {
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

	gradientScale := 1 / float64(batchSize)
	beta1Correction := 1 - o.beta1Power
	beta2Correction := 1 - o.beta2Power
	for i := range o.model.Hidden {
		applyLayer(&o.model.Hidden[i], &o.hidden[i], gradientScale, o.config, beta1Correction, beta2Correction)
	}
	applyLayer(&o.model.Output, &o.output, gradientScale, o.config, beta1Correction, beta2Correction)
}

func newLayerState(layer nn.DenseLayer) layerState {
	return layerState{weightM: make([]float64, len(layer.Weights)), weightV: make([]float64, len(layer.Weights)), biasM: make([]float64, len(layer.Biases)), biasV: make([]float64, len(layer.Biases))}
}

func normalize(config AdamConfig) AdamConfig {
	if config.Beta1 == 0 { config.Beta1 = DefaultAdamBeta1 }
	if config.Beta2 == 0 { config.Beta2 = DefaultAdamBeta2 }
	if config.Epsilon == 0 { config.Epsilon = DefaultAdamEpsilon }
	return config
}

func validate(config AdamConfig) error {
	if config.LearningRate <= 0 || math.IsNaN(config.LearningRate) || math.IsInf(config.LearningRate, 0) { return fmt.Errorf("invalid Adam learning rate: %g", config.LearningRate) }
	if config.Beta1 < 0 || config.Beta1 >= 1 || math.IsNaN(config.Beta1) { return fmt.Errorf("invalid Adam beta1: %g", config.Beta1) }
	if config.Beta2 < 0 || config.Beta2 >= 1 || math.IsNaN(config.Beta2) { return fmt.Errorf("invalid Adam beta2: %g", config.Beta2) }
	if config.Epsilon <= 0 || math.IsNaN(config.Epsilon) || math.IsInf(config.Epsilon, 0) { return fmt.Errorf("invalid Adam epsilon: %g", config.Epsilon) }
	return nil
}

func (o *Adam) mustMatchModel() {
	if o.model == nil || len(o.hidden) != len(o.model.Hidden) { panic("Adam state/model mismatch") }
	for i := range o.model.Hidden { mustMatchLayer(o.model.Hidden[i], o.hidden[i]) }
	mustMatchLayer(o.model.Output, o.output)
}

func mustMatchLayer(layer nn.DenseLayer, state layerState) {
	if len(state.weightM) != len(layer.Weights) || len(state.weightV) != len(layer.Weights) || len(state.biasM) != len(layer.Biases) || len(state.biasV) != len(layer.Biases) { panic("Adam layer-state shape mismatch") }
	if len(layer.GradW) != len(layer.Weights) || len(layer.GradB) != len(layer.Biases) { panic("Adam gradient shape mismatch") }
}

func applyLayer(layer *nn.DenseLayer, state *layerState, gradientScale float64, config AdamConfig, beta1Correction, beta2Correction float64) {
	for i, grad := range layer.GradW {
		g := grad * gradientScale
		state.weightM[i] = config.Beta1*state.weightM[i] + (1-config.Beta1)*g
		state.weightV[i] = config.Beta2*state.weightV[i] + (1-config.Beta2)*g*g
		mHat := state.weightM[i] / beta1Correction
		vHat := state.weightV[i] / beta2Correction
		layer.Weights[i] -= config.LearningRate * mHat / (math.Sqrt(vHat) + config.Epsilon)
	}
	for i, grad := range layer.GradB {
		g := grad * gradientScale
		state.biasM[i] = config.Beta1*state.biasM[i] + (1-config.Beta1)*g
		state.biasV[i] = config.Beta2*state.biasV[i] + (1-config.Beta2)*g*g
		mHat := state.biasM[i] / beta1Correction
		vHat := state.biasV[i] / beta2Correction
		layer.Biases[i] -= config.LearningRate * mHat / (math.Sqrt(vHat) + config.Epsilon)
	}
}
