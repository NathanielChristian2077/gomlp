package nn

import "fmt"

// Optimizer updates the parameters of an MLP after gradients are accumulated.
// Stateful optimizers must be reused across batches and epochs.
type Optimizer interface {
	Model() *MLP
	Step(batchSize int)
}

// SGDOptimizer implements mini-batch stochastic gradient descent.
type SGDOptimizer struct {
	model        *MLP
	learningRate float64
}

func NewSGDOptimizer(model *MLP, learningRate float64) *SGDOptimizer {
	if model == nil {
		panic("cannot create SGD optimizer for nil model")
	}
	if learningRate <= 0 {
		panic(fmt.Sprintf("invalid SGD learning rate: %g", learningRate))
	}
	return &SGDOptimizer{model: model, learningRate: learningRate}
}

func (o *SGDOptimizer) Model() *MLP {
	if o == nil {
		return nil
	}
	return o.model
}

func (o *SGDOptimizer) LearningRate() float64 {
	if o == nil {
		return 0
	}
	return o.learningRate
}

func (o *SGDOptimizer) Step(batchSize int) {
	if o == nil || o.model == nil {
		panic("nil SGD optimizer or model")
	}
	if batchSize <= 0 {
		panic(fmt.Sprintf("invalid batch size: %d", batchSize))
	}
	for i := range o.model.Hidden {
		applySGDToLayer(&o.model.Hidden[i], o.learningRate, batchSize)
	}
	applySGDToLayer(&o.model.Output, o.learningRate, batchSize)
}

func applySGDToLayer(layer *DenseLayer, learningRate float64, batchSize int) {
	scale := learningRate / float64(batchSize)
	for i := range layer.Weights {
		layer.Weights[i] -= scale * layer.GradW[i]
	}
	for i := range layer.Biases {
		layer.Biases[i] -= scale * layer.GradB[i]
	}
}
