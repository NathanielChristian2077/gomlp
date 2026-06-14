package metrics

// ConfusionMatrix armazena os quatro casos da classificação binária.
// Neste projeto, a classe positiva é dog e a classe negativa é cat.
type ConfusionMatrix struct {
	TruePositive  int
	TrueNegative  int
	FalsePositive int
	FalseNegative int
}

func NewConfusionMatrix() ConfusionMatrix {
	return ConfusionMatrix{}
}

// Add adiciona uma previsão à matriz de confusão.
func (m *ConfusionMatrix) Add(yHat, y, threshold float64) {
	predicted := Classify(yHat, threshold)
	actual := int(y)

	switch {
	case predicted == 1 && actual == 1:
		m.TruePositive++
	case predicted == 0 && actual == 0:
		m.TrueNegative++
	case predicted == 1 && actual == 0:
		m.FalsePositive++
	case predicted == 0 && actual == 1:
		m.FalseNegative++
	}
}

func (m ConfusionMatrix) Total() int {
	return m.TruePositive + m.TrueNegative + m.FalsePositive + m.FalseNegative
}

func (m ConfusionMatrix) Correct() int {
	return m.TruePositive + m.TrueNegative
}

func (m ConfusionMatrix) Accuracy() float64 {
	return Accuracy(m.Correct(), m.Total())
}

// Precision mede, entre as previsões positivas, quantas estavam corretas.
func (m ConfusionMatrix) Precision() float64 {
	denominator := m.TruePositive + m.FalsePositive
	if denominator == 0 {
		return 0
	}
	return float64(m.TruePositive) / float64(denominator)
}

// Recall mede, entre os positivos reais, quantos foram recuperados pelo modelo.
func (m ConfusionMatrix) Recall() float64 {
	denominator := m.TruePositive + m.FalseNegative
	if denominator == 0 {
		return 0
	}
	return float64(m.TruePositive) / float64(denominator)
}

// F1 combina precision e recall em uma média harmônica.
func (m ConfusionMatrix) F1() float64 {
	precision := m.Precision()
	recall := m.Recall()
	if precision+recall == 0 {
		return 0
	}
	return 2 * precision * recall / (precision + recall)
}
