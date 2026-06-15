package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/NathanielChristian2077/gomlp/experiment"
)

func main() {
	datasetPath := flag.String("dataset", "", "dataset root with train, validation and test folders")
	epochs := flag.Int("epochs", 100, "number of epochs for each experiment")
	hiddenArchitecturesRaw := flag.String("hidden", "128;256;256x64", "hidden architectures separated by semicolon, e.g. 128;256x64;512-128")
	batchValues := flag.String("batch", "16,32", "comma-separated batch sizes; use 0 for full-batch")
	learningRateValues := flag.String("lr", "0.001,0.0003", "comma-separated learning rates")
	seedValues := flag.String("seeds", "1,2,3,4,5,42", "comma-separated seeds")
	workers := flag.Int("workers", 1, "maximum number of CPU experiments running concurrently")
	outputRoot := flag.String("runs", "runs", "root directory for experiment outputs")
	logEvery := flag.Int("log-every", 50, "reserved logging interval recorded in the config")
	flag.Parse()

	hiddenArchitectures, err := experiment.ParseHiddenArchitectures(*hiddenArchitecturesRaw)
	if err != nil {
		log.Fatal(err)
	}
	batchSizes, err := parseInts(*batchValues)
	if err != nil {
		log.Fatal(err)
	}
	learningRates, err := parseFloats(*learningRateValues)
	if err != nil {
		log.Fatal(err)
	}
	seeds, err := parseInt64s(*seedValues)
	if err != nil {
		log.Fatal(err)
	}

	dataset, err := experiment.LoadDataset(*datasetPath)
	if err != nil {
		log.Fatal(err)
	}

	configs := make([]experiment.RunConfig, 0, len(hiddenArchitectures)*len(batchSizes)*len(learningRates)*len(seeds))
	for _, hiddenSizes := range hiddenArchitectures {
		for _, batch := range batchSizes {
			for _, lr := range learningRates {
				for _, seed := range seeds {
					name := fmt.Sprintf("dense_h%s_lr%g_bs%d_seed%d", experiment.HiddenSizesLabel(hiddenSizes), lr, batch, seed)
					configs = append(configs, experiment.RunConfig{
						Name:         name,
						DatasetPath:  *datasetPath,
						Epochs:       *epochs,
						HiddenSizes:  append([]int(nil), hiddenSizes...),
						BatchSize:    batch,
						Seed:         seed,
						LearningRate: lr,
						OutputRoot:   *outputRoot,
						LogEvery:     *logEvery,
					})
				}
			}
		}
	}

	fmt.Printf("experiments=%d workers=%d train=%d validation=%d test=%d input=%d\n", len(configs), *workers, len(dataset.Train), len(dataset.Validation), len(dataset.Test), dataset.InputSize)

	results := experiment.RunExperiments(configs, dataset, *workers)
	for _, result := range results {
		if !result.Completed {
			fmt.Printf("FAILED run=%s id=%s error=%s\n", result.Name, result.RunID, result.Error)
			continue
		}
		status := "trained"
		if result.LoadedFromSummary {
			status = "cached"
		}
		fmt.Printf(
			"%s run=%s id=%s best_epoch=%d val_acc=%.4f val_loss=%.6f test_acc=%.4f test_loss=%.6f time_ms=%d dir=%s\n",
			status,
			result.Name,
			result.RunID,
			result.BestEpoch,
			result.BestValidationAccuracy,
			result.BestValidationLoss,
			result.TestAccuracy,
			result.TestLoss,
			result.TrainTimeMilliseconds,
			result.RunDirectory,
		)
	}

	if err := writeSweepSummary(filepath.Join(*outputRoot, "summary.csv"), results); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("summary=%s\n", filepath.Join(*outputRoot, "summary.csv"))
}

func writeSweepSummary(path string, results []experiment.RunResult) error {
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

	if err := writer.Write([]string{"run_id", "name", "completed", "cached", "best_epoch", "val_loss", "val_accuracy", "test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1", "train_time_ms", "run_directory", "error"}); err != nil {
		return err
	}

	for _, result := range results {
		row := []string{
			result.RunID,
			result.Name,
			strconv.FormatBool(result.Completed),
			strconv.FormatBool(result.LoadedFromSummary),
			strconv.Itoa(result.BestEpoch),
			formatFloat(result.BestValidationLoss),
			formatFloat(result.BestValidationAccuracy),
			formatFloat(result.TestLoss),
			formatFloat(result.TestAccuracy),
			formatFloat(result.TestPrecision),
			formatFloat(result.TestRecall),
			formatFloat(result.TestF1),
			strconv.FormatInt(result.TrainTimeMilliseconds, 10),
			result.RunDirectory,
			result.Error,
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return writer.Error()
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 8, 64)
}

func parseInts(raw string) ([]int, error) {
	parts := strings.Split(raw, ",")
	values := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid integer %q: %w", part, err)
		}
		values = append(values, value)
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("empty integer list")
	}
	return values, nil
}

func parseInt64s(raw string) ([]int64, error) {
	parts := strings.Split(raw, ",")
	values := make([]int64, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.ParseInt(part, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid int64 %q: %w", part, err)
		}
		values = append(values, value)
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("empty int64 list")
	}
	return values, nil
}

func parseFloats(raw string) ([]float64, error) {
	parts := strings.Split(raw, ",")
	values := make([]float64, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.ParseFloat(part, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid float %q: %w", part, err)
		}
		values = append(values, value)
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("empty float list")
	}
	return values, nil
}
