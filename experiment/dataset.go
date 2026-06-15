package experiment

import (
	"fmt"

	"github.com/NathanielChristian2077/gomlp/data"
	"github.com/NathanielChristian2077/gomlp/nn"
)

type DatasetBundle struct {
	Train               []nn.Sample
	Validation          []nn.Sample
	Test                []nn.Sample
	InputSize           int
	DefaultLearningRate float64
}

func LoadDataset(path string) (DatasetBundle, error) {
	if path == "" {
		samples := []nn.Sample{
			{X: []float64{0, 0}, Y: 0},
			{X: []float64{0, 1}, Y: 1},
			{X: []float64{1, 0}, Y: 1},
			{X: []float64{1, 1}, Y: 1},
		}

		return DatasetBundle{
			Train:               samples,
			Validation:          samples,
			Test:                samples,
			InputSize:           2,
			DefaultLearningRate: 0.1,
		}, nil
	}

	dataset, err := data.LoadDataset(path)
	if err != nil {
		return DatasetBundle{}, err
	}

	if len(dataset.Train) == 0 {
		return DatasetBundle{}, fmt.Errorf("empty training split")
	}
	if len(dataset.Validation) == 0 {
		return DatasetBundle{}, fmt.Errorf("empty validation split")
	}
	if len(dataset.Test) == 0 {
		return DatasetBundle{}, fmt.Errorf("empty test split")
	}

	return DatasetBundle{
		Train:               dataset.Train,
		Validation:          dataset.Validation,
		Test:                dataset.Test,
		InputSize:           len(dataset.Train[0].X),
		DefaultLearningRate: 0.001,
	}, nil
}
