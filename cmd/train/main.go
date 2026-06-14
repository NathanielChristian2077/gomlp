package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"

	"github.com/NathanielChristian2077/gomlp/data"
	"github.com/NathanielChristian2077/gomlp/metrics"
	"github.com/NathanielChristian2077/gomlp/nn"
)

func main() {
	datasetPath := flag.String("dataset", "", "dataset root with train, validation and test folders")
	epochs := flag.Int("epochs", 3000, "number of training epochs")
	hiddenSize := flag.Int("hidden", 128, "hidden layer size")
	batchSize := flag.Int("batch", 0, "mini-batch size; if <= 0, full-batch training is used")
	seed := flag.Int64("seed", 42, "random seed")
	learningRate := flag.Float64("lr", -1, "learning rate; if negative, a default is selected")
	outputPath := flag.String("out", "", "csv output path")
	logEvery := flag.Int("log-every", 50, "print progress every N epochs")
	flag.Parse()

	trainSamples, validationSamples, testSamples, inputSize, defaultLR, defaultOutput := loadTrainingData(*datasetPath)

	lr := *learningRate
	if lr < 0 {
		lr = defaultLR
	}

	out := *outputPath
	if out == "" {
		out = defaultOutput
	}

	batch := *batchSize
	if batch <= 0 || batch > len(trainSamples) {
		batch = len(trainSamples)
	}

	model := nn.NewMLP(inputSize, *hiddenSize, *seed)
	rng := rand.New(rand.NewSource(*seed + 1))

	logger, err := metrics.NewEpochCSVLogger(out)
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := logger.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	fmt.Printf("train=%d validation=%d test=%d input=%d hidden=%d epochs=%d batch=%d lr=%.6f out=%s\n", len(trainSamples), len(validationSamples), len(testSamples), inputSize, *hiddenSize, *epochs, batch, lr, out)
	printClassBalance("train", trainSamples)
	printClassBalance("validation", validationSamples)
	printClassBalance("test", testSamples)

	for epoch := 1; epoch <= *epochs; epoch++ {
		trainResult, err := nn.TrainEpochMiniBatch(model, trainSamples, lr, batch, rng)
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

		if shouldLog(epoch, *epochs, *logEvery) {
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

	testResult, err := nn.Evaluate(model, testSamples)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("test_loss=%.6f test_acc=%.2f precision=%.2f recall=%.2f f1=%.2f\n", testResult.Loss, testResult.Accuracy, testResult.Confusion.Precision(), testResult.Confusion.Recall(), testResult.Confusion.F1())
	printConfusionMatrix(testResult.Confusion)

	fmt.Println("predictions:")

	limit := len(testSamples)
	if limit > 10 {
		limit = 10
	}

	for i := 0; i < limit; i++ {
		sample := testSamples[i]
		yHat := model.Forward(sample.X)
		fmt.Printf("i=%d y=%.0f yHat=%.6f\n", i, sample.Y, yHat)
	}
}

func loadTrainingData(datasetPath string) ([]nn.Sample, []nn.Sample, []nn.Sample, int, float64, string) {
	if datasetPath == "" {
		samples := []nn.Sample{
			{X: []float64{0, 0}, Y: 0},
			{X: []float64{0, 1}, Y: 1},
			{X: []float64{1, 0}, Y: 1},
			{X: []float64{1, 1}, Y: 1},
		}

		return samples, samples, samples, 2, 0.1, "results/or_synthetic.csv"
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
	if len(dataset.Test) == 0 {
		log.Fatal("empty test split")
	}

	return dataset.Train, dataset.Validation, dataset.Test, len(dataset.Train[0].X), 0.001, "results/dataset_train.csv"
}

func shouldLog(epoch, epochs, logEvery int) bool {
	if epoch == 1 || epoch == epochs {
		return true
	}
	return logEvery > 0 && epoch%logEvery == 0
}

func printClassBalance(name string, samples []nn.Sample) {
	cats := 0
	dogs := 0

	for _, sample := range samples {
		if sample.Y == data.CatLabel {
			cats++
		} else if sample.Y == data.DogLabel {
			dogs++
		}
	}

	fmt.Printf("%s: total=%d cat=%d dog=%d\n", name, len(samples), cats, dogs)
}

func printConfusionMatrix(matrix metrics.ConfusionMatrix) {
	fmt.Println("confusion_matrix:")
	fmt.Printf("TN=%d FP=%d\n", matrix.TrueNegative, matrix.FalsePositive)
	fmt.Printf("FN=%d TP=%d\n", matrix.FalseNegative, matrix.TruePositive)
}
