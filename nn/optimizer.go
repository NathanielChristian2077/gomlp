package nn

import "fmt"

// Optimizer atualiza os parâmetros de uma MLP após a acumulação de gradientes.
// Otimizadores com estado devem ser reutilizados entre batches e épocas.
type Optimizer interface {
	Model() *MLP
	Step(batchSize int)
}

// SGDOptimizer implementa gradient descent estocástico com gradiente médio por batch.
type SGDOptimizer struct {
	model        *MLP
	learningRate float64
}

// NewSGDOptimizer cria o otimizador padrão da baseline.
func NewSGDOptimizer(model *MLP, learningRate float64) *SGDOptimizer {
	if model == nil {
		panic("cannot create SGD optimizer for nil model")
	}
	if learningRate <= 0 {
		panic(fmt.Sprintf("invalid SGD learning rate: %g", learningRate))
	}

	return &SGDOptimizer{
		model:        model,
		learningRate: learningRate,
	}
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

// Step aplica uma atualização SGD usando os gradientes acumulados no modelo.
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
