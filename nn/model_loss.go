package nn

// Loss evaluates the baseline output loss for a sigmoid prediction.
func (m *MLP) Loss(prediction, target float64) float64 {
	return DefaultLoss().Value(prediction, target)
}

// OutputDelta returns the baseline output-layer delta for sigmoid plus BCE.
func (m *MLP) OutputDelta(prediction, target float64) float64 {
	return DefaultLoss().OutputDelta(prediction, target)
}
