package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/NathanielChristian2077/gomlp/data"
	"github.com/NathanielChristian2077/gomlp/metrics"
	"github.com/NathanielChristian2077/gomlp/nn"
)

func main() {
	datasetPath := flag.String("dataset", "", "dataset root with train, validation and test folders")
	epochs := flag.Int("epochs", 3000, "number of training epochs")
	hiddenSize := flag.Int("hidden", 128, "hidden layer size")
	learningRate := flag.Float64("lr", -1, "learning rate; if negative, a default is selected")
	outputPath := flag.String("out", "", "csv output path")
	flag.Parse()

	trainSamples, validationSamples, inputSize, defaultLR, defaultOutput := loadTrainingData(*datasetPath)

	lr := *learningRate
	if lr < 0 {
		lr = defaultLR
	}

	out := *outputPath
	if out == "" {
		out = defaultOutput
	}

	model := nn.NewMLP(inputSize, *hiddenSize, 42)

	logger, err := metrics.NewEpochCSVLogger(out)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := logger.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	fmt.Printf("train=%d validation=%d input=%d hidden=%d epochs=%d lr=%.6f out=%s\n", len(trainSamples), len(validationSamples), inputSize, *hiddenSize, *epochs, lr, out)

	for epoch := 1; epoch <= *epochs; epoch++ {
		trainResult, err := nn.TrainEpoch(model, trainSamples, lr)
		if err != nil {
			log.Fatal(err)
		}

		validationResult, err := nn.Evaluate(model, validationSamples)
		if err != nil {
			log.Fatal(err)
		}

		if err := logger.WriteEpoch(
			epoch,
			trainResult.Loss,
			trainResult.Accuracy,
			validationResult.Loss,
			validationResult.Accuracy,
			trainResult.Duration.Milliseconds(),
		); err != nil {
			log.Fatal(err)
		}

		if epoch == 1 || epoch%500 == 0 || epoch == *epochs {
			fmt.Printf(
				"epoch=%d train_loss=%.6f train_acc=%.2f val_loss=%.6f val_acc=%.2f\n",
				epoch,
				trainResult.Loss,
				trainResult.Accuracy,
				validationResult.Loss,
				validationResult.Accuracy,
			)
		}
	}

	fmt.Println("predictions:")

	limit := len(validationSamples)
	if limit > 10 {
		limit = 10
	}

	for i := 0; i < limit; i++ {
		sample := validationSamples[i]
		yHat := model.Forward(sample.X)
		fmt.Printf("i=%d y=%.0f yHat=%.6f\n", i, sample.Y, yHat)
	}
}

func loadTrainingData(datasetPath string) ([]nn.Sample, []nn.Sample, int, float64, string) {
	if datasetPath == "" {
		samples := []nn.Sample{
			{X: []float64{0, 0}, Y: 0},
			{X: []float64{0, 1}, Y: 1},
			{X: []float64{1, 0}, Y: 1},
			{X: []float64{1, 1}, Y: 1},
		}

		return samples, samples, 2, 0.1, "results/or_synthetic.csv"
	}

	dataset, err := data.LoadDataset(datasetPath)
	if err != nil {
		log.Fatal(err)
	}

	if len(dataset.Train) == 0 {
		log.Fatal("empty training split")
	}
	if len(dataset.Validation) == 0 {
		log.Fatal("empty validation split")
	}

	return dataset.Train, dataset.Validation, len(dataset.Train[0].X), 0.001, "results/dataset_train.csv"
}
