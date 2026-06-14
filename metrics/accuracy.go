package metrics

func Classify(yHat float64, threshold float64) int {
	if yHat >= threshold {
		return 1
	}
	return 0
}

func CorrectPrediction(yHat, y, threshold float64) bool {
	return Classify(yHat, threshold) == int(y)
}

func Accuracy(correct, total int) float64 {
	if total <= 0 {
		return 0
	}
	return float64(correct) / float64(total)
}

func AccuracyFromPredictions(yHats []float64, ys []float64, threshold float64) float64 {
	if len(yHats) == 0 || len(yHats) != len(ys) {
		return 0
	}

	correct := 0

	for i := range yHats {
		if CorrectPrediction(yHats[i], ys[i], threshold) {
			correct++
		}
	}

	return Accuracy(correct, len(yHats))
}
