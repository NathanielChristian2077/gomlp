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
	"time"

	"github.com/NathanielChristian2077/gomlp/experiment"
	"github.com/NathanielChristian2077/gomlp/metrics"
	"github.com/NathanielChristian2077/gomlp/nn"
)

type compareResult struct {
	Mode                   string
	Threshold              float64
	Split                  string
	Samples                int
	Loss                   float64
	Accuracy               float64
	Precision              float64
	Recall                 float64
	F1                     float64
	TrueNegative           int
	FalsePositive          int
	FalseNegative          int
	TruePositive           int
	DurationMilliseconds   int64
	DenseOpsTotal          int
	SparseOpsTotal         int
	EstimatedSpeedup       float64
	ActiveTotal            int
	ActivationSlotsTotal   int
	AverageActiveCount     float64
	AverageActiveRatio     float64
	AverageSparsity        float64
	AverageActiveByLayer   string
	MaxAbsDiffFromDense    float64
	MismatchCountFromDense int
}

type samplePrediction struct {
	YHat      float64
	Predicted int
}

func main() {
	datasetPath := flag.String("dataset", "", "dataset root with train, validation and test folders; empty uses OR synthetic dataset")
	epochs := flag.Int("epochs", 100, "number of training epochs used when no cached run is available")
	hiddenRaw := flag.String("hidden", "128", "hidden architecture, e.g. 64, 256x64 or 512-128")
	batchSize := flag.Int("batch", 0, "mini-batch size; if <= 0, full-batch training is used")
	seed := flag.Int64("seed", 42, "random seed")
	learningRate := flag.Float64("lr", -1, "learning rate; if negative, the dataset default is selected")
	runsRoot := flag.String("runs", "runs", "root directory used to find or create the dense training run")
	runName := flag.String("name", "dense_compare", "human-readable run name for the dense model")
	checkpointPath := flag.String("checkpoint", "", "optional checkpoint path; if set, training/cache lookup is skipped")
	splitName := flag.String("split", "test", "split to compare: train, validation or test")
	thresholdsRaw := flag.String("thresholds", "0", "comma-separated sparse thresholds; 0 is exact DSA")
	outPath := flag.String("out", "", "CSV output path; default is <run_dir>/compare.csv or compare.csv when using --checkpoint")
	flag.Parse()

	hiddenSizes, err := experiment.ParseHiddenArchitecture(*hiddenRaw)
	if err != nil {
		log.Fatal(err)
	}

	thresholds, err := parseThresholds(*thresholdsRaw)
	if err != nil {
		log.Fatal(err)
	}

	dataset, err := experiment.LoadDataset(*datasetPath)
	if err != nil {
		log.Fatal(err)
	}

	samples, normalizedSplit, err := selectSplit(dataset, *splitName)
	if err != nil {
		log.Fatal(err)
	}

	var model *nn.MLP
	var runDir string
	var runID string
	var bestEpoch int

	if *checkpointPath != "" {
		checkpoint, err := experiment.LoadCheckpoint(*checkpointPath)
		if err != nil {
			log.Fatal(err)
		}
		model, err = experiment.RestoreModel(checkpoint)
		if err != nil {
			log.Fatal(err)
		}
		runID = checkpoint.RunID
		bestEpoch = checkpoint.BestEpoch
		runDir = filepath.Dir(filepath.Dir(*checkpointPath))
	} else {
		config := experiment.RunConfig{
			Name:         *runName,
			DatasetPath:  *datasetPath,
			Epochs:       *epochs,
			HiddenSizes:  hiddenSizes,
			BatchSize:    *batchSize,
			Seed:         *seed,
			LearningRate: *learningRate,
			OutputRoot:   *runsRoot,
		}

		result := experiment.RunOrLoadExperiment(config, dataset)
		if !result.Completed {
			log.Fatalf("dense run failed: %s", result.Error)
		}

		runID = result.RunID
		runDir = result.RunDirectory
		bestEpoch = result.BestEpoch

		checkpoint, err := experiment.LoadCheckpoint(filepath.Join(runDir, "checkpoints", "best.gob"))
		if err != nil {
			log.Fatal(err)
		}
		model, err = experiment.RestoreModel(checkpoint)
		if err != nil {
			log.Fatal(err)
		}
	}

	out := *outPath
	if out == "" {
		if runDir != "" {
			out = filepath.Join(runDir, "compare.csv")
		} else {
			out = "compare.csv"
		}
	}

	hiddenLabel := experiment.HiddenSizesLabel(model.HiddenSizes())
	denseResult, densePredictions, err := evaluateDense(model, samples, normalizedSplit)
	if err != nil {
		log.Fatal(err)
	}

	results := []compareResult{denseResult}
	for _, threshold := range thresholds {
		result, err := evaluateSparse(model, samples, normalizedSplit, threshold, densePredictions)
		if err != nil {
			log.Fatal(err)
		}
		results = append(results, result)
	}

	if err := writeCompareCSV(out, runID, *runName, hiddenLabel, bestEpoch, results); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("run_id=%s run_dir=%s best_epoch=%d split=%s samples=%d out=%s\n", runID, runDir, bestEpoch, normalizedSplit, len(samples), out)
	fmt.Printf(
		"%-18s %-10s %-10s %-10s %-10s %-10s %-10s %-12s %-12s %-14s %-12s %-10s %-10s\n",
		"mode", "threshold", "loss", "acc", "precision", "recall", "f1", "ms", "speedup", "active", "avg_active", "sparsity", "mismatch",
	)
	for _, result := range results {
		fmt.Printf(
			"%-18s %-10.6g %-10.6f %-10.4f %-10.4f %-10.4f %-10.4f %-12d %-12.4f %-14s %-12.4f %-10.4f %-10d\n",
			result.Mode,
			result.Threshold,
			result.Loss,
			result.Accuracy,
			result.Precision,
			result.Recall,
			result.F1,
			result.DurationMilliseconds,
			result.EstimatedSpeedup,
			formatActivationFraction(result.ActiveTotal, result.ActivationSlotsTotal),
			result.AverageActiveCount,
			result.AverageSparsity,
			result.MismatchCountFromDense,
		)
		fmt.Printf("  activations_by_layer: %s\n", result.AverageActiveByLayer)
	}
}

func evaluateDense(model *nn.MLP, samples []nn.Sample, split string) (compareResult, []samplePrediction, error) {
	if len(samples) == 0 {
		return compareResult{}, nil, fmt.Errorf("empty %s split", split)
	}

	startedAt := time.Now()
	loss := 0.0
	confusion := metrics.NewConfusionMatrix()
	predictions := make([]samplePrediction, len(samples))

	for i, sample := range samples {
		yHat := model.Forward(sample.X)
		predicted := metrics.Classify(yHat, nn.DefaultClassificationThreshold)
		predictions[i] = samplePrediction{YHat: yHat, Predicted: predicted}
		loss += nn.BinaryCrossEntropy(yHat, sample.Y)
		confusion.Add(yHat, sample.Y, nn.DefaultClassificationThreshold)
	}

	activeTotal, slotsTotal := fullDenseActivationTotals(model, len(samples))
	averageActiveCount := averageActiveCount(activeTotal, len(samples), len(model.Hidden))

	return compareResult{
		Mode:                   "dense",
		Threshold:              0,
		Split:                  split,
		Samples:                len(samples),
		Loss:                   loss / float64(len(samples)),
		Accuracy:               confusion.Accuracy(),
		Precision:              confusion.Precision(),
		Recall:                 confusion.Recall(),
		F1:                     confusion.F1(),
		TrueNegative:           confusion.TrueNegative,
		FalsePositive:          confusion.FalsePositive,
		FalseNegative:          confusion.FalseNegative,
		TruePositive:           confusion.TruePositive,
		DurationMilliseconds:   time.Since(startedAt).Milliseconds(),
		DenseOpsTotal:          modelDenseOps(model) * len(samples),
		SparseOpsTotal:         modelDenseOps(model) * len(samples),
		EstimatedSpeedup:       1,
		ActiveTotal:            activeTotal,
		ActivationSlotsTotal:   slotsTotal,
		AverageActiveCount:     averageActiveCount,
		AverageActiveRatio:     1,
		AverageSparsity:        0,
		AverageActiveByLayer:   denseLayerActivationSummary(model),
		MaxAbsDiffFromDense:    0,
		MismatchCountFromDense: 0,
	}, predictions, nil
}

func evaluateSparse(model *nn.MLP, samples []nn.Sample, split string, threshold float64, densePredictions []samplePrediction) (compareResult, error) {
	if len(samples) == 0 {
		return compareResult{}, fmt.Errorf("empty %s split", split)
	}
	if len(densePredictions) != len(samples) {
		return compareResult{}, fmt.Errorf("invalid dense predictions length: expected %d got %d", len(samples), len(densePredictions))
	}

	startedAt := time.Now()
	loss := 0.0
	confusion := metrics.NewConfusionMatrix()
	denseOpsTotal := 0
	sparseOpsTotal := 0
	activeTotal := 0
	activationSlotsTotal := 0
	layerActiveTotals := make([]int, len(model.Hidden))
	layerSlotTotals := make([]int, len(model.Hidden))
	maxAbsDiff := 0.0
	mismatchCount := 0

	for i, sample := range samples {
		yHat, stats := model.ForwardSparseWithStats(sample.X, threshold)
		predicted := metrics.Classify(yHat, nn.DefaultClassificationThreshold)

		loss += nn.BinaryCrossEntropy(yHat, sample.Y)
		confusion.Add(yHat, sample.Y, nn.DefaultClassificationThreshold)

		denseOpsTotal += stats.DenseOps
		sparseOpsTotal += stats.SparseOps
		for _, layerStats := range stats.Hidden {
			if layerStats.LayerIndex < 0 || layerStats.LayerIndex >= len(layerActiveTotals) {
				return compareResult{}, fmt.Errorf("invalid sparse stats layer index: %d", layerStats.LayerIndex)
			}
			activeTotal += layerStats.Active
			activationSlotsTotal += layerStats.Size
			layerActiveTotals[layerStats.LayerIndex] += layerStats.Active
			layerSlotTotals[layerStats.LayerIndex] += layerStats.Size
		}

		diff := absFloat64(yHat - densePredictions[i].YHat)
		if diff > maxAbsDiff {
			maxAbsDiff = diff
		}
		if predicted != densePredictions[i].Predicted {
			mismatchCount++
		}
	}

	averageActiveCount := averageActiveCount(activeTotal, len(samples), len(model.Hidden))
	averageActiveRatio := 0.0
	averageSparsity := 0.0
	if activationSlotsTotal > 0 {
		averageActiveRatio = float64(activeTotal) / float64(activationSlotsTotal)
		averageSparsity = 1 - averageActiveRatio
	}

	estimatedSpeedup := 0.0
	if sparseOpsTotal > 0 {
		estimatedSpeedup = float64(denseOpsTotal) / float64(sparseOpsTotal)
	}

	mode := "sparse_exact"
	if threshold > 0 {
		mode = "sparse_threshold"
	}

	return compareResult{
		Mode:                   mode,
		Threshold:              threshold,
		Split:                  split,
		Samples:                len(samples),
		Loss:                   loss / float64(len(samples)),
		Accuracy:               confusion.Accuracy(),
		Precision:              confusion.Precision(),
		Recall:                 confusion.Recall(),
		F1:                     confusion.F1(),
		TrueNegative:           confusion.TrueNegative,
		FalsePositive:          confusion.FalsePositive,
		FalseNegative:          confusion.FalseNegative,
		TruePositive:           confusion.TruePositive,
		DurationMilliseconds:   time.Since(startedAt).Milliseconds(),
		DenseOpsTotal:          denseOpsTotal,
		SparseOpsTotal:         sparseOpsTotal,
		EstimatedSpeedup:       estimatedSpeedup,
		ActiveTotal:            activeTotal,
		ActivationSlotsTotal:   activationSlotsTotal,
		AverageActiveCount:     averageActiveCount,
		AverageActiveRatio:     averageActiveRatio,
		AverageSparsity:        averageSparsity,
		AverageActiveByLayer:   layerActivationSummary(layerActiveTotals, layerSlotTotals),
		MaxAbsDiffFromDense:    maxAbsDiff,
		MismatchCountFromDense: mismatchCount,
	}, nil
}

func parseThresholds(raw string) ([]float64, error) {
	parts := strings.Split(raw, ",")
	thresholds := make([]float64, 0, len(parts))
	seen := map[string]bool{}

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		threshold, err := strconv.ParseFloat(part, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid threshold %q: %w", part, err)
		}
		if threshold < 0 {
			return nil, fmt.Errorf("threshold must be non-negative, got %g", threshold)
		}

		key := strconv.FormatFloat(threshold, 'g', -1, 64)
		if seen[key] {
			continue
		}
		seen[key] = true
		thresholds = append(thresholds, threshold)
	}

	if len(thresholds) == 0 {
		thresholds = append(thresholds, 0)
	}

	return thresholds, nil
}

func selectSplit(dataset experiment.DatasetBundle, split string) ([]nn.Sample, string, error) {
	split = strings.ToLower(strings.TrimSpace(split))

	switch split {
	case "train", "training":
		return dataset.Train, "train", nil
	case "validation", "val":
		return dataset.Validation, "validation", nil
	case "test", "":
		return dataset.Test, "test", nil
	default:
		return nil, "", fmt.Errorf("unknown split %q: expected train, validation or test", split)
	}
}

func modelDenseOps(model *nn.MLP) int {
	ops := 0
	for _, layer := range model.Hidden {
		ops += layer.In * layer.Out
	}
	ops += model.Output.In * model.Output.Out
	return ops
}

func fullDenseActivationTotals(model *nn.MLP, samples int) (int, int) {
	perSample := 0
	for _, layer := range model.Hidden {
		perSample += layer.Out
	}
	return perSample * samples, perSample * samples
}

func averageActiveCount(activeTotal, samples, hiddenLayers int) float64 {
	denominator := samples * hiddenLayers
	if denominator <= 0 {
		return 0
	}
	return float64(activeTotal) / float64(denominator)
}

func denseLayerActivationSummary(model *nn.MLP) string {
	parts := make([]string, 0, len(model.Hidden))
	for i, layer := range model.Hidden {
		parts = append(parts, fmt.Sprintf("L%d=%.2f/%d", i, float64(layer.Out), layer.Out))
	}
	return strings.Join(parts, ";")
}

func layerActivationSummary(activeTotals, slotTotals []int) string {
	parts := make([]string, 0, len(activeTotals))
	for i := range activeTotals {
		averageActive := 0.0
		averageSlots := 0.0
		if slotTotals[i] > 0 {
			// slotTotals[i] is samples * layer_size, so active/slot gives ratio.
			// Dividing both totals by the inferred number of samples gives average active and size.
			inferredSamples := float64(slotTotals[i]) / float64(slotTotals[i]/maxInt(slotTotals[i], 1))
			_ = inferredSamples
		}
		parts = append(parts, fmt.Sprintf("L%d=%s/%s", i, formatFloatCompact(float64(activeTotals[i])), formatFloatCompact(float64(slotTotals[i]))))
	}
	return strings.Join(parts, ";")
}

func writeCompareCSV(path string, runID string, runName string, hiddenLabel string, bestEpoch int, results []compareResult) error {
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

	header := []string{
		"run_id",
		"run_name",
		"hidden",
		"best_epoch",
		"mode",
		"threshold",
		"split",
		"samples",
		"loss",
		"accuracy",
		"precision",
		"recall",
		"f1",
		"true_negative",
		"false_positive",
		"false_negative",
		"true_positive",
		"duration_ms",
		"dense_ops_total",
		"sparse_ops_total",
		"estimated_speedup",
		"active_total",
		"activation_slots_total",
		"avg_active_count",
		"avg_active_ratio",
		"avg_sparsity",
		"avg_active_by_layer",
		"max_abs_diff_from_dense",
		"mismatch_count_from_dense",
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	for _, result := range results {
		row := []string{
			runID,
			runName,
			hiddenLabel,
			fmt.Sprintf("%d", bestEpoch),
			result.Mode,
			formatFloat(result.Threshold),
			result.Split,
			fmt.Sprintf("%d", result.Samples),
			formatFloat(result.Loss),
			formatFloat(result.Accuracy),
			formatFloat(result.Precision),
			formatFloat(result.Recall),
			formatFloat(result.F1),
			fmt.Sprintf("%d", result.TrueNegative),
			fmt.Sprintf("%d", result.FalsePositive),
			fmt.Sprintf("%d", result.FalseNegative),
			fmt.Sprintf("%d", result.TruePositive),
			fmt.Sprintf("%d", result.DurationMilliseconds),
			fmt.Sprintf("%d", result.DenseOpsTotal),
			fmt.Sprintf("%d", result.SparseOpsTotal),
			formatFloat(result.EstimatedSpeedup),
			fmt.Sprintf("%d", result.ActiveTotal),
			fmt.Sprintf("%d", result.ActivationSlotsTotal),
			formatFloat(result.AverageActiveCount),
			formatFloat(result.AverageActiveRatio),
			formatFloat(result.AverageSparsity),
			result.AverageActiveByLayer,
			formatFloat(result.MaxAbsDiffFromDense),
			fmt.Sprintf("%d", result.MismatchCountFromDense),
		}
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return writer.Error()
}

func formatActivationFraction(active, slots int) string {
	return fmt.Sprintf("%d/%d", active, slots)
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 8, 64)
}

func formatFloatCompact(value float64) string {
	return strconv.FormatFloat(value, 'f', 2, 64)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func absFloat64(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}
