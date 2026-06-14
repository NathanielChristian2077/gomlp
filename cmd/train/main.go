package main

import (
	"fmt"
	"log"

	"github.com/NathanielChristian2077/gomlp/metrics"
	"github.com/NathanielChristian2077/gomlp/nn"
)

func main() {
	samples := []nn.Sample{
		{X: []float64{0, 0}, Y: 0},
		{X: []float64{0, 1}, Y: 1},
		{X: []float64{1, 0}, Y: 1},
		{X: []float64{1, 1}, Y: 1},
	}

	model := nn.NewMLP(2, 4, 42)

	logger, err := metrics.NewEpochCSVLogger("results/or_synthetic.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := logger.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	epochs := 3000
	lr := 0.1

	for epoch := 1; epoch <= epochs; epoch++ {
		trainResult, err := nn.TrainEpoch(model, samples, lr)
		if err != nil {
			log.Fatal(err)
		}

		valResult, err := nn.Evaluate(model, samples)
		if err != nil {
			log.Fatal(err)
		}

		if err := logger.WriteEpoch(
			epoch,
			trainResult.Loss,
			trainResult.Accuracy,
			valResult.Loss,
			valResult.Accuracy,
			trainResult.Duration.Milliseconds(),
		); err != nil {
			log.Fatal(err)
		}

		if epoch == 1 || epoch%500 == 0 {
			fmt.Printf(
				"epoch=%d train_loss=%.6f train_acc=%.2f val_loss=%.6f val_acc=%.2f\n",
				epoch,
				trainResult.Loss,
				trainResult.Accuracy,
				valResult.Loss,
				valResult.Accuracy,
			)
		}
	}

	fmt.Println("predictions:")

	for _, sample := range samples {
		yHat := model.Forward(sample.X)
		fmt.Printf("x=%v y=%.0f yHat=%.6f\n", sample.X, sample.Y, yHat)
	}
}
