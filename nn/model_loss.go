package nn

// Loss evaluates the baseline sigmoid loss for a model prediction.
func (m *MLP) Loss(prediction, target float64) float64 {
	return DefaultLoss().Value(prediction, target)
}

// OutputDelta returns the baseline sigmoid plus BCE output delta.
func (m *MLP) OutputDelta(prediction, target float64) float64 {
	return DefaultLoss().OutputDelta(prediction, target)
}
