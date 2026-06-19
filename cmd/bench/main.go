package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/NathanielChristian2077/gomlp/experiment"
	"github.com/NathanielChristian2077/gomlp/nn"
)

type benchResult struct {
	Mode                   string
	Threshold              float64
	Split                  string
	Samples                int
	Repeats                int
	TotalForwards          int
	TotalMilliseconds      int64
	NanosecondsPerForward  float64
	ForwardsPerSecond      float64
	Checksum               float64
	DenseOpsPerPass        int
	SparseOpsPerPass       int
	EstimatedSpeedup       float64
	ActiveTotalPerPass     int
	ActivationSlotsPerPass int
	AverageActiveRatio     float64
	AverageSparsity        float64
	AverageActiveByLayer   string
}

func main() {
	datasetPath := flag.String("dataset", "", "dataset root with train, validation and test folders; empty uses OR synthetic dataset")
	epochs := flag.Int("epochs", 100, "number of training epochs used when no cached run is available")
	hiddenRaw := flag.String("hidden", "128", "hidden architecture, e.g. 64, 256x64 or 512-128")
	batchSize := flag.Int("batch", 0, "mini-batch size; if <= 0, full-batch training is used")
	seed := flag.Int64("seed", 42, "random seed")
	learningRate := flag.Float64("lr", -1, "learning rate; if negative, the dataset default is selected")
	outputHead := flag.String("head", string(nn.OutputHeadSigmoid1), "output head: sigmoid1 or softmax2")
	runsRoot := flag.String("runs", "runs", "root directory used to find or create the dense training run")
	runName := flag.String("name", "dense_bench", "human-readable run name for the dense model")
	checkpointPath := flag.String("checkpoint", "", "optional checkpoint path; if set, training/cache lookup is skipped")
	splitName := flag.String("split", "test", "split to benchmark: train, validation or test")
	thresholdsRaw := flag.String("thresholds", "0", "comma-separated sparse thresholds; 0 is exact DSA")
	repeats := flag.Int("repeat", 100, "number of full split passes measured per mode")
	warmup := flag.Int("warmup", 10, "number of full split passes used as warmup per mode")
	gomaxprocs := flag.Int("gomaxprocs", 0, "optional GOMAXPROCS override; 0 keeps the current runtime value")
	outPath := flag.String("out", "", "CSV output path; default is <run_dir>/bench.csv or bench.csv when using --checkpoint")
	flag.Parse()

	if *repeats <= 0 {
		log.Fatalf("repeat must be positive, got %d", *repeats)
	}
	if *warmup < 0 {
		log.Fatalf("warmup must be non-negative, got %d", *warmup)
	}
	if *gomaxprocs > 0 {
		runtime.GOMAXPROCS(*gomaxprocs)
	}
	if _, err := nn.NormalizeOutputHead(*outputHead); err != nil {
		log.Fatal(err)
	}

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

	model, runID, runDir, bestEpoch, err := loadOrTrainModel(
		*checkpointPath,
		*runName,
		*datasetPath,
		*epochs,
		hiddenSizes,
		*batchSize,
		*seed,
		*learningRate,
		*outputHead,
		*runsRoot,
		dataset,
	)
	if err != nil {
		log.Fatal(err)
	}

	out := *outPath
	if out == "" {
		if runDir != "" {
			out = filepath.Join(runDir, "bench.csv")
		} else {
			out = "bench.csv"
		}
	}

	hiddenLabel := experiment.HiddenSizesLabel(model.HiddenSizes())

	warmupDense(model, samples, *warmup)
	denseResult := benchmarkDense(model, samples, normalizedSplit, *repeats)
	results := []benchResult{denseResult}

	for _, threshold := range thresholds {
		warmupSparse(model, samples, threshold, *warmup)
		result := benchmarkSparse(model, samples, normalizedSplit, threshold, *repeats)
		results = append(results, result)
	}

	if err := writeBenchCSV(out, runID, *runName, hiddenLabel, string(model.Head()), bestEpoch, results); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("run_id=%s run_dir=%s best_epoch=%d head=%s split=%s samples=%d repeat=%d warmup=%d gomaxprocs=%d out=%s\n", runID, runDir, bestEpoch, model.Head(), normalizedSplit, len(samples), *repeats, *warmup, runtime.GOMAXPROCS(0), out)
	fmt.Printf("%-18s %-10s %-12s %-14s %-12s %-12s %-12s %-14s %-10s %-10s\n", "mode", "threshold", "total_ms", "ns_forward", "forward/s", "speedup", "ops_saved", "active", "sparsity", "checksum")
	for _, result := range results {
		opsSaved := 0.0
		if result.DenseOpsPerPass > 0 {
			opsSaved = 1 - float64(result.SparseOpsPerPass)/float64(result.DenseOpsPerPass)
		}
		fmt.Printf(
			"%-18s %-10.6g %-12d %-14.2f %-12.2f %-12.4f %-12.4f %-14s %-10.4f %-10.4f\n",
			result.Mode,
			result.Threshold,
			result.TotalMilliseconds,
			result.NanosecondsPerForward,
			result.ForwardsPerSecond,
			result.EstimatedSpeedup,
			opsSaved,
			formatActivationFraction(result.ActiveTotalPerPass, result.ActivationSlotsPerPass),
			result.AverageSparsity,
			result.Checksum,
		)
		fmt.Printf("  activations_by_layer: %s\n", result.AverageActiveByLayer)
	}
}

func loadOrTrainModel(checkpointPath, runName, datasetPath string, epochs int, hiddenSizes []int, batchSize int, seed int64, learningRate float64, outputHead string, runsRoot string, dataset experiment.DatasetBundle) (*nn.MLP, string, string, int, error) {
	if checkpointPath != "" {
		checkpoint, err := experiment.LoadCheckpoint(checkpointPath)
		if err != nil {
			return nil, "", "", 0, err
		}
		model, err := experiment.RestoreModel(checkpoint)
		if err != nil {
			return nil, "", "", 0, err
		}
		return model, checkpoint.RunID, filepath.Dir(filepath.Dir(checkpointPath)), checkpoint.BestEpoch, nil
	}

	config := experiment.RunConfig{
		Name:         runName,
		DatasetPath:  datasetPath,
		Epochs:       epochs,
		HiddenSizes:  hiddenSizes,
		BatchSize:    batchSize,
		Seed:         seed,
		LearningRate: learningRate,
		OutputHead:   outputHead,
		OutputRoot:   runsRoot,
	}

	result := experiment.RunOrLoadExperiment(config, dataset)
	if !result.Completed {
		return nil, result.RunID, result.RunDirectory, 0, fmt.Errorf("dense run failed: %s", result.Error)
	}

	checkpoint, err := experiment.LoadCheckpoint(filepath.Join(result.RunDirectory, "checkpoints", "best.gob"))
	if err != nil {
		return nil, result.RunID, result.RunDirectory, result.BestEpoch, err
	}
	model, err := experiment.RestoreModel(checkpoint)
	if err != nil {
		return nil, result.RunID, result.RunDirectory, result.BestEpoch, err
	}

	return model, result.RunID, result.RunDirectory, result.BestEpoch, nil
}

func warmupDense(model *nn.MLP, samples []nn.Sample, passes int) float64 {
	checksum := 0.0
	for pass := 0; pass < passes; pass++ {
		for _, sample := range samples {
			checksum += model.ForwardFast(sample.X)
		}
	}
	return checksum
}

func benchmarkDense(model *nn.MLP, samples []nn.Sample, split string, repeats int) benchResult {
	runtime.GC()
	startedAt := time.Now()
	checksum := 0.0
	for pass := 0; pass < repeats; pass++ {
		for _, sample := range samples {
			checksum += model.ForwardFast(sample.X)
		}
	}
	duration := time.Since(startedAt)
	totalForwards := repeats * len(samples)
	denseOpsPerPass := modelDenseOps(model) * len(samples)
	activeTotal, activationSlots := fullDenseActivationTotals(model, len(samples))

	return benchResult{
		Mode:                   "dense",
		Threshold:              0,
		Split:                  split,
		Samples:                len(samples),
		Repeats:                repeats,
		TotalForwards:          totalForwards,
		TotalMilliseconds:      duration.Milliseconds(),
		NanosecondsPerForward:  float64(duration.Nanoseconds()) / float64(totalForwards),
		ForwardsPerSecond:      float64(totalForwards) / duration.Seconds(),
		Checksum:               checksum,
		DenseOpsPerPass:        denseOpsPerPass,
		SparseOpsPerPass:       denseOpsPerPass,
		EstimatedSpeedup:       1,
		ActiveTotalPerPass:     activeTotal,
		ActivationSlotsPerPass: activationSlots,
		AverageActiveRatio:     1,
		AverageSparsity:        0,
		AverageActiveByLayer:   denseLayerActivationSummary(model),
	}
}

func warmupSparse(model *nn.MLP, samples []nn.Sample, threshold float64, passes int) float64 {
	workspace := nn.NewSparseForwardWorkspace(model)
	checksum := 0.0
	for pass := 0; pass < passes; pass++ {
		for _, sample := range samples {
			checksum += model.ForwardSparsePrepared(sample.X, threshold, workspace)
		}
	}
	return checksum
}

func benchmarkSparse(model *nn.MLP, samples []nn.Sample, split string, threshold float64, repeats int) benchResult {
	workspace := nn.NewSparseForwardWorkspace(model)
	runtime.GC()
	startedAt := time.Now()
	checksum := 0.0
	for pass := 0; pass < repeats; pass++ {
		for _, sample := range samples {
			checksum += model.ForwardSparsePrepared(sample.X, threshold, workspace)
		}
	}
	duration := time.Since(startedAt)
	totalForwards := repeats * len(samples)
	stats := collectSparseStats(model, samples, threshold)

	mode := "sparse_exact"
	if threshold > 0 {
		mode = "sparse_threshold"
	}

	return benchResult{
		Mode:                   mode,
		Threshold:              threshold,
		Split:                  split,
		Samples:                len(samples),
		Repeats:                repeats,
		TotalForwards:          totalForwards,
		TotalMilliseconds:      duration.Milliseconds(),
		NanosecondsPerForward:  float64(duration.Nanoseconds()) / float64(totalForwards),
		ForwardsPerSecond:      float64(totalForwards) / duration.Seconds(),
		Checksum:               checksum,
		DenseOpsPerPass:        stats.DenseOpsPerPass,
		SparseOpsPerPass:       stats.SparseOpsPerPass,
		EstimatedSpeedup:       stats.EstimatedSpeedup,
		ActiveTotalPerPass:     stats.ActiveTotal,
		ActivationSlotsPerPass: stats.ActivationSlots,
		AverageActiveRatio:     stats.AverageActiveRatio,
		AverageSparsity:        stats.AverageSparsity,
		AverageActiveByLayer:   stats.AverageActiveByLayer,
	}
}

type sparseStatsSummary struct {
	DenseOpsPerPass      int
	SparseOpsPerPass     int
	EstimatedSpeedup     float64
	ActiveTotal          int
	ActivationSlots      int
	AverageActiveRatio   float64
	AverageSparsity      float64
	AverageActiveByLayer string
}

func collectSparseStats(model *nn.MLP, samples []nn.Sample, threshold float64) sparseStatsSummary {
	workspace := nn.NewSparseForwardWorkspace(model)
	denseOps := 0
	sparseOps := 0
	activeTotal := 0
	activationSlots := 0
	layerActiveTotals := make([]int, len(model.Hidden))
	layerSlotTotals := make([]int, len(model.Hidden))

	for _, sample := range samples {
		_, stats := model.ForwardSparseWithStatsWorkspace(sample.X, threshold, workspace)
		denseOps += stats.DenseOps
		sparseOps += stats.SparseOps
		for _, layerStats := range stats.Hidden {
			activeTotal += layerStats.Active
			activationSlots += layerStats.Size
			layerActiveTotals[layerStats.LayerIndex] += layerStats.Active
			layerSlotTotals[layerStats.LayerIndex] += layerStats.Size
		}
	}

	averageActiveRatio := 0.0
	averageSparsity := 0.0
	if activationSlots > 0 {
		averageActiveRatio = float64(activeTotal) / float64(activationSlots)
		averageSparsity = 1 - averageActiveRatio
	}

	estimatedSpeedup := 0.0
	if sparseOps > 0 {
		estimatedSpeedup = float64(denseOps) / float64(sparseOps)
	}

	return sparseStatsSummary{
		DenseOpsPerPass:      denseOps,
		SparseOpsPerPass:     sparseOps,
		EstimatedSpeedup:     estimatedSpeedup,
		ActiveTotal:          activeTotal,
		ActivationSlots:      activationSlots,
		AverageActiveRatio:   averageActiveRatio,
		AverageSparsity:      averageSparsity,
		AverageActiveByLayer: layerActivationSummary(layerActiveTotals, layerSlotTotals, len(samples)),
	}
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

func denseLayerActivationSummary(model *nn.MLP) string {
	parts := make([]string, 0, len(model.Hidden))
	for i, layer := range model.Hidden {
		parts = append(parts, fmt.Sprintf("L%d=%.2f/%d", i, float64(layer.Out), layer.Out))
	}
	return strings.Join(parts, ";")
}

func layerActivationSummary(activeTotals, slotTotals []int, samples int) string {
	parts := make([]string, 0, len(activeTotals))
	for i := range activeTotals {
		averageActive := 0.0
		averageSlots := 0.0
		if samples > 0 {
			averageActive = float64(activeTotals[i]) / float64(samples)
			averageSlots = float64(slotTotals[i]) / float64(samples)
		}
		parts = append(parts, fmt.Sprintf("L%d=%.2f/%.0f", i, averageActive, averageSlots))
	}
	return strings.Join(parts, ";")
}

func writeBenchCSV(path string, runID string, runName string, hiddenLabel string, outputHead string, bestEpoch int, results []benchResult) error {
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
		"output_head",
		"best_epoch",
		"mode",
		"threshold",
		"split",
		"samples",
		"repeats",
		"total_forwards",
		"total_ms",
		"ns_per_forward",
		"forwards_per_second",
		"checksum",
		"dense_ops_per_pass",
		"sparse_ops_per_pass",
		"estimated_speedup",
		"ops_saved_ratio",
		"active_total_per_pass",
		"activation_slots_per_pass",
		"avg_active_ratio",
		"avg_sparsity",
		"avg_active_by_layer",
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	for _, result := range results {
		opsSaved := 0.0
		if result.DenseOpsPerPass > 0 {
			opsSaved = 1 - float64(result.SparseOpsPerPass)/float64(result.DenseOpsPerPass)
		}
		row := []string{
			runID,
			runName,
			hiddenLabel,
			outputHead,
			fmt.Sprintf("%d", bestEpoch),
			result.Mode,
			formatFloat(result.Threshold),
			result.Split,
			fmt.Sprintf("%d", result.Samples),
			fmt.Sprintf("%d", result.Repeats),
			fmt.Sprintf("%d", result.TotalForwards),
			fmt.Sprintf("%d", result.TotalMilliseconds),
			formatFloat(result.NanosecondsPerForward),
			formatFloat(result.ForwardsPerSecond),
			formatFloat(result.Checksum),
			fmt.Sprintf("%d", result.DenseOpsPerPass),
			fmt.Sprintf("%d", result.SparseOpsPerPass),
			formatFloat(result.EstimatedSpeedup),
			formatFloat(opsSaved),
			fmt.Sprintf("%d", result.ActiveTotalPerPass),
			fmt.Sprintf("%d", result.ActivationSlotsPerPass),
			formatFloat(result.AverageActiveRatio),
			formatFloat(result.AverageSparsity),
			result.AverageActiveByLayer,
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
