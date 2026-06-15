package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/NathanielChristian2077/gomlp/data"
	"github.com/NathanielChristian2077/gomlp/experiment"
	"github.com/NathanielChristian2077/gomlp/metrics"
	"github.com/NathanielChristian2077/gomlp/nn"
)

type classBalance struct {
	Total int `json:"total"`
	Cats  int `json:"cats"`
	Dogs  int `json:"dogs"`
}

type trainRunConfig struct {
	RunName      string       `json:"run_name"`
	DatasetPath  string       `json:"dataset_path"`
	Epochs       int          `json:"epochs"`
	HiddenSizes  []int        `json:"hidden_sizes"`
	BatchSize    int          `json:"batch_size"`
	Seed         int64        `json:"seed"`
	LearningRate float64      `json:"learning_rate"`
	InputSize    int          `json:"input_size"`
	OutputPath   string       `json:"output_path"`
	CreatedAt    string       `json:"created_at"`
	Train        classBalance `json:"train"`
	Validation   classBalance `json:"validation"`
	Test         classBalance `json:"test"`
}

type trainRunSummary struct {
	RunName               string                  `json:"run_name"`
	SelectedEpoch         int                     `json:"selected_epoch"`
	SelectedValLoss       float64                 `json:"selected_val_loss"`
	SelectedValAccuracy   float64                 `json:"selected_val_accuracy"`
	SelectedValPrecision  float64                 `json:"selected_val_precision"`
	SelectedValRecall     float64                 `json:"selected_val_recall"`
	SelectedValF1         float64                 `json:"selected_val_f1"`
	TestLoss              float64                 `json:"test_loss"`
	TestAccuracy          float64                 `json:"test_accuracy"`
	TestPrecision         float64                 `json:"test_precision"`
	TestRecall            float64                 `json:"test_recall"`
	TestF1                float64                 `json:"test_f1"`
	Confusion             metrics.ConfusionMatrix `json:"confusion"`
	TrainTimeMilliseconds int64                   `json:"train_time_ms"`
	CompletedAt           string                  `json:"completed_at"`
}

func main() {
	datasetPath := flag.String("dataset", "", "dataset root with train, validation and test folders")
	epochs := flag.Int("epochs", 100, "number of training epochs")
	hiddenRaw := flag.String("hidden", "128", "hidden architecture, e.g. 128, 256x64 or 512-128")
	batchSize := flag.Int("batch", 0, "mini-batch size; if <= 0, full-batch training is used")
	seed := flag.Int64("seed", 42, "random seed")
	learningRate := flag.Float64("lr", -1, "learning rate; if negative, a default is selected")
	outputPath := flag.String("out", "", "csv output path")
	runDirFlag := flag.String("run-dir", "", "directory used to save metrics, config, summary and confusion matrix")
	runName := flag.String("name", "dense_baseline", "human-readable run name used in logs")
	logEvery := flag.Int("log-every", 50, "print progress every N epochs")
	flag.Parse()

	hiddenSizes, err := experiment.ParseHiddenArchitecture(*hiddenRaw)
	if err != nil {
		log.Fatal(err)
	}

	trainSamples, validationSamples, testSamples, inputSize, defaultLR, defaultOutput := loadTrainingData(*datasetPath)

	lr := *learningRate
	if lr < 0 {
		lr = defaultLR
	}

	out := *outputPath
	if out == "" {
		if *runDirFlag != "" {
			out = filepath.Join(*runDirFlag, "metrics.csv")
		} else {
			out = defaultOutput
		}
	}

	runDir := *runDirFlag
	if runDir == "" {
		runDir = filepath.Dir(out)
	}
	if err := os.MkdirAll(runDir, 0755); err != nil {
		log.Fatal(err)
	}

	batch := *batchSize
	if batch <= 0 || batch > len(trainSamples) {
		batch = len(trainSamples)
	}

	config := trainRunConfig{
		RunName:      *runName,
		DatasetPath:  *datasetPath,
		Epochs:       *epochs,
		HiddenSizes:  append([]int(nil), hiddenSizes...),
		BatchSize:    batch,
		Seed:         *seed,
		LearningRate: lr,
		InputSize:    inputSize,
		OutputPath:   out,
		CreatedAt:    time.Now().Format(time.RFC3339),
		Train:        computeClassBalance(trainSamples),
		Validation:   computeClassBalance(validationSamples),
		Test:         computeClassBalance(testSamples),
	}
	if err := writeJSON(filepath.Join(runDir, "config.json"), config); err != nil {
		log.Fatal(err)
	}

	model := nn.NewMLPWithHiddenSizes(inputSize, hiddenSizes, *seed)
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

	fmt.Printf("run=%s run_dir=%s\n", *runName, runDir)
	fmt.Printf("train=%d validation=%d test=%d input=%d hidden=%s epochs=%d batch=%d lr=%.6f out=%s\n", len(trainSamples), len(validationSamples), len(testSamples), inputSize, experiment.HiddenSizesLabel(hiddenSizes), *epochs, batch, lr, out)
	printClassBalance("train", trainSamples)
	printClassBalance("validation", validationSamples)
	printClassBalance("test", testSamples)

	startedAt := time.Now()
	var bestModel *nn.MLP
	var bestEpoch int
	var bestValidationResult nn.EpochResult

	for epoch := 1; epoch <= *epochs; epoch++ {
		trainResult, err := nn.TrainEpochMiniBatch(model, trainSamples, lr, batch, rng)
		if err != nil {
			log.Fatal(err)
		}

		validationResult, err := nn.Evaluate(model, validationSamples)
		if err != nil {
			log.Fatal(err)
		}

		if bestModel == nil || isBetterValidation(validationResult, bestValidationResult) {
			bestModel = model.Clone()
			bestEpoch = epoch
			bestValidationResult = validationResult
		}

		if err := logger.WriteEpochDetailed(
			epoch,
			trainResult.Loss,
			trainResult.Accuracy,
			trainResult.Confusion.Precision(),
			trainResult.Confusion.Recall(),
			trainResult.Confusion.F1(),
			validationResult.Loss,
			validationResult.Accuracy,
			validationResult.Confusion.Precision(),
			validationResult.Confusion.Recall(),
			validationResult.Confusion.F1(),
			trainResult.Duration.Milliseconds(),
		); err != nil {
			log.Fatal(err)
		}

		if shouldLog(epoch, *epochs, *logEvery) {
			fmt.Printf(
				"epoch=%d train_loss=%.6f train_acc=%.2f train_f1=%.2f val_loss=%.6f val_acc=%.2f val_f1=%.2f epoch_ms=%d\n",
				epoch,
				trainResult.Loss,
				trainResult.Accuracy,
				trainResult.Confusion.F1(),
				validationResult.Loss,
				validationResult.Accuracy,
				validationResult.Confusion.F1(),
				trainResult.Duration.Milliseconds(),
			)
		}
	}

	if bestModel == nil {
		validationResult, err := nn.Evaluate(model, validationSamples)
		if err != nil {
			log.Fatal(err)
		}

		bestModel = model.Clone()
		bestEpoch = 0
		bestValidationResult = validationResult
	}

	fmt.Printf("selected_epoch=%d selected_val_loss=%.6f selected_val_acc=%.2f selected_val_f1=%.2f\n", bestEpoch, bestValidationResult.Loss, bestValidationResult.Accuracy, bestValidationResult.Confusion.F1())

	testResult, err := nn.Evaluate(bestModel, testSamples)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("test_loss=%.6f test_acc=%.2f precision=%.2f recall=%.2f f1=%.2f\n", testResult.Loss, testResult.Accuracy, testResult.Confusion.Precision(), testResult.Confusion.Recall(), testResult.Confusion.F1())
	printConfusionMatrix(testResult.Confusion)

	if err := writeConfusionMatrixCSV(filepath.Join(runDir, "confusion_matrix.csv"), testResult.Confusion); err != nil {
		log.Fatal(err)
	}
	if err := writePredictionsCSV(filepath.Join(runDir, "test_predictions.csv"), bestModel, testSamples); err != nil {
		log.Fatal(err)
	}

	summary := trainRunSummary{
		RunName:               *runName,
		SelectedEpoch:         bestEpoch,
		SelectedValLoss:       bestValidationResult.Loss,
		SelectedValAccuracy:   bestValidationResult.Accuracy,
		SelectedValPrecision:  bestValidationResult.Confusion.Precision(),
		SelectedValRecall:     bestValidationResult.Confusion.Recall(),
		SelectedValF1:         bestValidationResult.Confusion.F1(),
		TestLoss:              testResult.Loss,
		TestAccuracy:          testResult.Accuracy,
		TestPrecision:         testResult.Confusion.Precision(),
		TestRecall:            testResult.Confusion.Recall(),
		TestF1:                testResult.Confusion.F1(),
		Confusion:             testResult.Confusion,
		TrainTimeMilliseconds: time.Since(startedAt).Milliseconds(),
		CompletedAt:           time.Now().Format(time.RFC3339),
	}
	if err := writeJSON(filepath.Join(runDir, "summary.json"), summary); err != nil {
		log.Fatal(err)
	}

	fmt.Println("predictions:")

	limit := len(testSamples)
	if limit > 10 {
		limit = 10
	}

	for i := 0; i < limit; i++ {
		sample := testSamples[i]
		yHat := bestModel.Forward(sample.X)
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

func isBetterValidation(candidate, current nn.EpochResult) bool {
	if candidate.Accuracy != current.Accuracy {
		return candidate.Accuracy > current.Accuracy
	}
	return candidate.Loss < current.Loss
}

func shouldLog(epoch, epochs, logEvery int) bool {
	if epoch == 1 || epoch == epochs {
		return true
	}
	return logEvery > 0 && epoch%logEvery == 0
}

func printClassBalance(name string, samples []nn.Sample) {
	balance := computeClassBalance(samples)
	fmt.Printf("%s: total=%d cat=%d dog=%d\n", name, balance.Total, balance.Cats, balance.Dogs)
}

func computeClassBalance(samples []nn.Sample) classBalance {
	balance := classBalance{Total: len(samples)}

	for _, sample := range samples {
		if sample.Y == data.CatLabel {
			balance.Cats++
		} else if sample.Y == data.DogLabel {
			balance.Dogs++
		}
	}

	return balance
}

func printConfusionMatrix(matrix metrics.ConfusionMatrix) {
	fmt.Println("confusion_matrix:")
	fmt.Printf("TN=%d FP=%d\n", matrix.TrueNegative, matrix.FalsePositive)
	fmt.Printf("FN=%d TP=%d\n", matrix.FalseNegative, matrix.TruePositive)
}

func writeJSON(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}

func writeConfusionMatrixCSV(path string, matrix metrics.ConfusionMatrix) error {
	logger, err := metrics.NewCSVLogger(path, []string{"metric", "value"})
	if err != nil {
		return err
	}
	defer logger.Close()

	rows := [][]string{
		{"true_negative", fmt.Sprintf("%d", matrix.TrueNegative)},
		{"false_positive", fmt.Sprintf("%d", matrix.FalsePositive)},
		{"false_negative", fmt.Sprintf("%d", matrix.FalseNegative)},
		{"true_positive", fmt.Sprintf("%d", matrix.TruePositive)},
		{"accuracy", fmt.Sprintf("%.8f", matrix.Accuracy())},
		{"precision", fmt.Sprintf("%.8f", matrix.Precision())},
		{"recall", fmt.Sprintf("%.8f", matrix.Recall())},
		{"f1", fmt.Sprintf("%.8f", matrix.F1())},
	}

	for _, row := range rows {
		if err := logger.WriteRow(row...); err != nil {
			return err
		}
	}

	return nil
}

func writePredictionsCSV(path string, model *nn.MLP, samples []nn.Sample) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if err := writer.Write([]string{"index", "y", "y_hat", "predicted"}); err != nil {
		return err
	}

	for i, sample := range samples {
		yHat := model.Forward(sample.X)
		row := []string{
			fmt.Sprintf("%d", i),
			fmt.Sprintf("%.0f", sample.Y),
			fmt.Sprintf("%.8f", yHat),
			fmt.Sprintf("%d", metrics.Classify(yHat, nn.DefaultClassificationThreshold)),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return writer.Error()
}
