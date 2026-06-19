package experiment

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/NathanielChristian2077/gomlp/metrics"
	"github.com/NathanielChristian2077/gomlp/nn"
)

type RunResult struct {
	RunID                  string                  `json:"run_id"`
	Name                   string                  `json:"name"`
	RunDirectory           string                  `json:"run_directory"`
	Completed              bool                    `json:"completed"`
	LoadedFromSummary      bool                    `json:"loaded_from_summary"`
	BestEpoch              int                     `json:"best_epoch"`
	BestValidationLoss     float64                 `json:"best_validation_loss"`
	BestValidationAccuracy float64                 `json:"best_validation_accuracy"`
	OutputHead             string                  `json:"output_head"`
	TestLoss               float64                 `json:"test_loss"`
	TestAccuracy           float64                 `json:"test_accuracy"`
	TestPrecision          float64                 `json:"test_precision"`
	TestRecall             float64                 `json:"test_recall"`
	TestF1                 float64                 `json:"test_f1"`
	Confusion              metrics.ConfusionMatrix `json:"confusion"`
	TrainTimeMilliseconds  int64                   `json:"train_time_ms"`
	EpochsRun              int                     `json:"epochs_run"`
	Error                  string                  `json:"error,omitempty"`
}

func RunExperiments(configs []RunConfig, dataset DatasetBundle, maxWorkers int) []RunResult {
	if maxWorkers <= 0 {
		maxWorkers = 1
	}
	if maxWorkers > len(configs) && len(configs) > 0 {
		maxWorkers = len(configs)
	}

	jobs := make(chan RunConfig)
	results := make(chan RunResult)

	for workerID := 0; workerID < maxWorkers; workerID++ {
		go func() {
			for config := range jobs {
				results <- RunOrLoadExperiment(config, dataset)
			}
		}()
	}

	go func() {
		for _, config := range configs {
			jobs <- config
		}
		close(jobs)
	}()

	out := make([]RunResult, 0, len(configs))
	for range configs {
		out = append(out, <-results)
	}

	return out
}

func RunOrLoadExperiment(config RunConfig, dataset DatasetBundle) RunResult {
	config = config.Normalize(dataset.DefaultLearningRate)
	if err := config.Validate(); err != nil {
		return failedResult(config, "", "", err)
	}

	runID, err := config.ID()
	if err != nil {
		return failedResult(config, "", "", err)
	}

	runDir, err := config.RunDirectory()
	if err != nil {
		return failedResult(config, runID, "", err)
	}

	summaryPath := filepath.Join(runDir, "summary.json")
	if _, err := os.Stat(summaryPath); err == nil {
		var summary RunResult
		if err := readJSON(summaryPath, &summary); err == nil && summary.Completed {
			summary.LoadedFromSummary = true
			return summary
		}
	}

	if err := os.MkdirAll(filepath.Join(runDir, "checkpoints"), 0755); err != nil {
		return failedResult(config, runID, runDir, err)
	}
	if err := writeJSONAtomic(filepath.Join(runDir, "config.json"), config); err != nil {
		return failedResult(config, runID, runDir, err)
	}

	batchSize := config.BatchSize
	if batchSize <= 0 || batchSize > len(dataset.Train) {
		batchSize = len(dataset.Train)
	}

	model := nn.NewMLPWithHiddenSizesAndHead(dataset.InputSize, config.HiddenSizes, config.Seed, config.OutputHead)
	rng := rand.New(rand.NewSource(config.Seed + 1))

	logger, err := metrics.NewEpochCSVLogger(filepath.Join(runDir, "metrics.csv"))
	if err != nil {
		return failedResult(config, runID, runDir, err)
	}
	defer logger.Close()

	startedAt := time.Now()
	var bestModel *nn.MLP
	var bestEpoch int
	var bestValidationResult nn.EpochResult

	for epoch := 1; epoch <= config.Epochs; epoch++ {
		trainResult, err := nn.TrainEpochMiniBatch(model, dataset.Train, config.LearningRate, batchSize, rng)
		if err != nil {
			return failedResult(config, runID, runDir, err)
		}

		validationResult, err := nn.Evaluate(model, dataset.Validation)
		if err != nil {
			return failedResult(config, runID, runDir, err)
		}

		if bestModel == nil || isBetterValidation(validationResult, bestValidationResult) {
			bestModel = model.Clone()
			bestEpoch = epoch
			bestValidationResult = validationResult
			best := NewCheckpoint(runID, epoch, bestEpoch, bestValidationResult, bestModel)
			if err := SaveCheckpoint(filepath.Join(runDir, "checkpoints", "best.gob"), best); err != nil {
				return failedResult(config, runID, runDir, err)
			}
		}

		latest := NewCheckpoint(runID, epoch, bestEpoch, bestValidationResult, model)
		if err := SaveCheckpoint(filepath.Join(runDir, "checkpoints", "latest.gob"), latest); err != nil {
			return failedResult(config, runID, runDir, err)
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
			return failedResult(config, runID, runDir, err)
		}
	}

	if bestModel == nil {
		bestModel = model.Clone()
		bestEpoch = config.Epochs
		bestValidationResult, err = nn.Evaluate(bestModel, dataset.Validation)
		if err != nil {
			return failedResult(config, runID, runDir, err)
		}
	}

	testResult, err := nn.Evaluate(bestModel, dataset.Test)
	if err != nil {
		return failedResult(config, runID, runDir, err)
	}

	result := RunResult{
		RunID:                  runID,
		Name:                   config.Name,
		RunDirectory:           runDir,
		Completed:              true,
		BestEpoch:              bestEpoch,
		BestValidationLoss:     bestValidationResult.Loss,
		BestValidationAccuracy: bestValidationResult.Accuracy,
		OutputHead:             string(bestModel.Head()),
		TestLoss:               testResult.Loss,
		TestAccuracy:           testResult.Accuracy,
		TestPrecision:          testResult.Confusion.Precision(),
		TestRecall:             testResult.Confusion.Recall(),
		TestF1:                 testResult.Confusion.F1(),
		Confusion:              testResult.Confusion,
		TrainTimeMilliseconds:  time.Since(startedAt).Milliseconds(),
		EpochsRun:              config.Epochs,
	}

	if err := writeConfusionMatrixCSV(filepath.Join(runDir, "confusion_matrix.csv"), testResult.Confusion); err != nil {
		return failedResult(config, runID, runDir, err)
	}
	if err := writePredictionsCSV(filepath.Join(runDir, "test_predictions.csv"), bestModel, dataset.Test); err != nil {
		return failedResult(config, runID, runDir, err)
	}
	if err := writeJSONAtomic(summaryPath, result); err != nil {
		return failedResult(config, runID, runDir, err)
	}

	return result
}

func failedResult(config RunConfig, runID string, runDir string, err error) RunResult {
	result := RunResult{
		RunID:        runID,
		Name:         config.Name,
		RunDirectory: runDir,
		Completed:    false,
		OutputHead:   config.OutputHead,
	}
	if err != nil {
		result.Error = err.Error()
	}
	return result
}

func isBetterValidation(candidate, current nn.EpochResult) bool {
	if candidate.Accuracy != current.Accuracy {
		return candidate.Accuracy > current.Accuracy
	}
	return candidate.Loss < current.Loss
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

	if err := writer.Write([]string{"index", "label", "positive_probability", "predicted"}); err != nil {
		return err
	}

	for i, sample := range samples {
		yHat := model.Forward(sample.X)
		predicted := model.PredictClassFromLastForward()
		if err := writer.Write([]string{
			fmt.Sprintf("%d", i),
			fmt.Sprintf("%.0f", sample.Y),
			fmt.Sprintf("%.8f", yHat),
			fmt.Sprintf("%d", predicted),
		}); err != nil {
			return err
		}
	}

	return writer.Error()
}
