package nn

import "math"

// ReLU aplica a função de ativação da camada oculta.
// Valores negativos são zerados e valores positivos passam sem alteração.
// Essa ativação também é o ponto que permite observar sparsity,
// pois neurônios com saída exatamente zero podem ser ignorados na DSA exata.
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLUToActive aplica ReLU e retorna apenas as ativações maiores que threshold.
// Com threshold igual a zero, a representação esparsa é matematicamente equivalente
// ao vetor denso após ReLU, apenas removendo entradas exatamente nulas.
// Thresholds positivos são aproximados e devem ser tratados como experimento separado.
func ReLUToActive(z []float64, threshold float64) ActiveVector {
	if threshold < 0 {
		panic("activation threshold must be non-negative")
	}

	indices := make([]int, 0, len(z))
	values := make([]float64, 0, len(z))

	for i, v := range z {
		if v > threshold {
			indices = append(indices, i)
			values = append(values, v)
		}
	}

	return ActiveVector{
		Size:    len(z),
		Indices: indices,
		Values:  values,
	}
}

// ReLUDerivativeFromActivation calcula a derivada da ReLU usando a ativação já computada.
// Se a ativação final foi maior que zero, o neurônio estava ativo e a derivada é 1.
// Se a ativação foi zero, o neurônio é tratado como inativo e a derivada é 0.
func ReLUDerivativeFromActivation(a float64) float64 {
	if a > 0 {
		return 1
	}
	return 0
}

// Sigmoid transforma o logit da saída em uma probabilidade no intervalo [0, 1].
// A implementação separa valores positivos e negativos para reduzir risco de overflow
// em math.Exp quando o valor absoluto do logit é muito alto.
func Sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}

	z := math.Exp(x)
	return z / (1 + z)
}

// BinaryCrossEntropy mede o erro para classificação binária.
// y representa o rótulo real, com 0 para gato e 1 para cachorro.
// yHat representa a probabilidade prevista pela sigmoid.
// O clamp evita log(0), que geraria infinito e quebraria o treino.
func BinaryCrossEntropy(yHat, y float64) float64 {
	const eps = 1e-12

	if yHat < eps {
		yHat = eps
	}
	if yHat > 1-eps {
		yHat = 1 - eps
	}

	return -(y*math.Log(yHat) + (1-y)*math.Log(1-yHat))
}
