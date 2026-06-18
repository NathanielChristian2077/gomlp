package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/NathanielChristian2077/gomlp/experiment"
	"github.com/NathanielChristian2077/gomlp/metrics"
	"github.com/NathanielChristian2077/gomlp/nn"
)

type ensembleModel struct {
	Path       string
	RunID      string
	BestEpoch  int
	Hidden     string
	Model      *nn.MLP
	Workspace  *nn.SparseForwardWorkspace
}

type modeResult struct {
	Mode          string
	Loss          float64
	Accuracy      float64
	Precision     float64
	Recall        float64
	F1            float64
	TrueNegative  int
	FalsePositive int
	FalseNegative int
	TruePositive  int
}

type sampleResult struct {
	Index         int
	Y             float64
	MeanYHat      float64
	WeightedYHat  float64
	StdYHat       float64
	MinYHat       float64
	MaxYHat       float64
	PositiveVotes int
	VoteRatio     float64
	MeanPred      int
	WeightedPred  int
	VotePred      int
	ModelYHats    []float64
}

func main() {
	datasetPath := flag.String("dataset", "", "dataset root with train, validation and test folders; empty uses OR synthetic dataset")
	splitName := flag.String("split", "test", "split to evaluate: train, validation or test")
	checkpointsRaw := flag.String("checkpoints", "", "comma-separated checkpoint paths")
	checkpointGlob := flag.String("checkpoint-glob", "", "glob pattern for checkpoint paths, e.g. runs/dense_sweep/*/checkpoints/best.gob")
	weightsRaw := flag.String("weights", "", "optional comma-separated model weights; default is equal weights")
	threshold := flag.Float64("threshold", nn.DefaultClassificationThreshold, "classification threshold")
	forwardMode := flag.String("forward", "dense", "forward mode: dense, sparse-exact or sparse-threshold")
	sparseThreshold := flag.Float64("sparse-threshold", 0, "threshold used when --forward sparse-threshold")
	outPath := flag.String("out", "runs/ensemble/predictions.csv", "prediction CSV output path")
	summaryPath := flag.String("summary", "runs/ensemble/summary.csv", "summary CSV output path")
	flag.Parse()

	checkpointPaths, err := resolveCheckpointPaths(*checkpointsRaw, *checkpointGlob)
	if err != nil {
		log.Fatal(err)
	}
	if len(checkpointPaths) == 0 {
		log.Fatal("no checkpoints provided; use --checkpoints or --checkpoint-glob")
	}

	weights, err := parseWeights(*weightsRaw, len(checkpointPaths))
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

	models, err := loadModels(checkpointPaths, dataset.InputSize, *forwardMode)
	if err != nil {
		log.Fatal(err)
	}

	startedAt := time.Now()
	rows, summaries, err := evaluateEnsemble(models, weights, samples, *threshold, *forwardMode, *sparseThreshold)
	if err != nil {
		log.Fatal(err)
	}
	duration := time.Since(startedAt)

	if err := writePredictionsCSV(*outPath, models, rows); err != nil {
		log.Fatal(err)
	}
	if err := writeSummaryCSV(*summaryPath, models, weights, normalizedSplit, *forwardMode, *sparseThreshold, *threshold, duration, summaries); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("models=%d split=%s samples=%d forward=%s sparse_threshold=%.6g out=%s summary=%s duration_ms=%d\n", len(models), normalizedSplit, len(samples), *forwardMode, *sparseThreshold, *outPath, *summaryPath, duration.Milliseconds())
	fmt.Printf("%-18s %-10s %-10s %-10s %-10s %-10s %-8s %-8s %-8s %-8s\n", "mode", "loss", "acc", "precision", "recall", "f1", "tn", "fp", "fn", "tp")
	for _, result := range summaries {
		fmt.Printf(
			"%-18s %-10.6f %-10.4f %-10.4f %-10.4f %-10.4f %-8d %-8d %-8d %-8d\n",
			result.Mode,
			result.Loss,
			result.Accuracy,
			result.Precision,
			result.Recall,
			result.F1,
			result.TrueNegative,
			result.FalsePositive,
			result.FalseNegative,
			result.TruePositive,
		)
	}
}

func resolveCheckpointPaths(raw string, pattern string) ([]string, error) {
	seen := map[string]bool{}
	paths := []string{}

	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if !seen[part] {
			paths = append(paths, part)
			seen[part] = true
		}
	}

	pattern = strings.TrimSpace(pattern)
	if pattern != "" {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			return nil, fmt.Errorf("invalid checkpoint glob %q: %w", pattern, err)
		}
		sort.Strings(matches)
		for _, match := range matches {
			if !seen[match] {
				paths = append(paths, match)
				seen[match] = true
			}
		}
	}

	return paths, nil
}

func parseWeights(raw string, expected int) ([]float64, error) {
	if expected <= 0 {
		return nil, fmt.Errorf("invalid model count: %d", expected)
	}

	if strings.TrimSpace(raw) == "" {
		weights := make([]float64, expected)
		for i := range weights {
			weights[i] = 1 / float64(expected)
		}
		return weights, nil
	}

	parts := strings.Split(raw, ",")
	if len(parts) != expected {
		return nil, fmt.Errorf("weights count mismatch: expected %d got %d", expected, len(parts))
	}

	weights := make([]float64, expected)
	sum := 0.0
	for i, part := range parts {
		value, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid weight %q: %w", part, err)
		}
		if value < 0 {
			return nil, fmt.Errorf("weights must be non-negative, got %g", value)
		}
		weights[i] = value
		sum += value
	}
	if sum == 0 {
		return nil, fmt.Errorf("at least one weight must be positive")
	}
	for i := range weights {
		weights[i] /= sum
	}

	return weights, nil
}

func loadModels(paths []string, expectedInputSize int, forwardMode string) ([]ensembleModel, error) {
	models := make([]ensembleModel, 0, len(paths))
	for _, path := range paths {
		checkpoint, err := experiment.LoadCheckpoint(path)
		if err != nil {
			return nil, fmt.Errorf("load checkpoint %q: %w", path, err)
		}
		model, err := experiment.RestoreModel(checkpoint)
		if err != nil {
			return nil, fmt.Errorf("restore checkpoint %q: %w", path, err)
		}
		if model.InputSize() != expectedInputSize {
			return nil, fmt.Errorf("checkpoint %q input size mismatch: expected %d got %d", path, expectedInputSize, model.InputSize())
		}

		entry := ensembleModel{
			Path:      path,
			RunID:     checkpoint.RunID,
			BestEpoch: checkpoint.BestEpoch,
			Hidden:    experiment.HiddenSizesLabel(model.HiddenSizes()),
			Model:     model,
		}
		if forwardMode != "dense" {
			entry.Workspace = nn.NewSparseForwardWorkspace(model)
		}
		models = append(models, entry)
	}
	return models, nil
}

func evaluateEnsemble(models []ensembleModel, weights []float64, samples []nn.Sample, threshold float64, forwardMode string, sparseThreshold float64) ([]sampleResult, []modeResult, error) {
	if len(models) == 0 {
		return nil, nil, fmt.Errorf("no models")
	}
	if len(weights) != len(models) {
		return nil, nil, fmt.Errorf("weights length mismatch: expected %d got %d", len(models), len(weights))
	}
	if len(samples) == 0 {
		return nil, nil, fmt.Errorf("empty split")
	}
	if threshold < 0 || threshold > 1 {
		return nil, nil, fmt.Errorf("classification threshold must be in [0,1], got %g", threshold)
	}
	if sparseThreshold < 0 {
		return nil, nil, fmt.Errorf("sparse threshold must be non-negative, got %g", sparseThreshold)
	}

	individualLosses := make([]float64, len(models))
	individualConfusions := make([]metrics.ConfusionMatrix, len(models))
	for i := range individualConfusions {
		individualConfusions[i] = metrics.NewConfusionMatrix()
	}

	meanLoss := 0.0
	weightedLoss := 0.0
	voteLoss := 0.0
	meanConfusion := metrics.NewConfusionMatrix()
	weightedConfusion := metrics.NewConfusionMatrix()
	voteConfusion := metrics.NewConfusionMatrix()
	rows := make([]sampleResult, 0, len(samples))

	for sampleIndex, sample := range samples {
		yHats := make([]float64, len(models))
		sum := 0.0
		weightedSum := 0.0
		positiveVotes := 0
		minYHat := math.Inf(1)
		maxYHat := math.Inf(-1)

		for modelIndex := range models {
			yHat, err := forward(models[modelIndex], sample.X, forwardMode, sparseThreshold)
			if err != nil {
				return nil, nil, err
			}
			yHats[modelIndex] = yHat
			sum += yHat
			weightedSum += weights[modelIndex] * yHat
			if yHat < minYHat {
				minYHat = yHat
			}
			if yHat > maxYHat {
				maxYHat = yHat
			}
			if metrics.Classify(yHat, threshold) == 1 {
				positiveVotes++
			}

			individualLosses[modelIndex] += nn.BinaryCrossEntropy(yHat, sample.Y)
			individualConfusions[modelIndex].Add(yHat, sample.Y, threshold)
		}

		meanYHat := sum / float64(len(models))
		voteRatio := float64(positiveVotes) / float64(len(models))
		voteYHat := voteRatio

		meanLoss += nn.BinaryCrossEntropy(meanYHat, sample.Y)
		weightedLoss += nn.BinaryCrossEntropy(weightedSum, sample.Y)
		voteLoss += nn.BinaryCrossEntropy(voteYHat, sample.Y)

		meanConfusion.Add(meanYHat, sample.Y, threshold)
		weightedConfusion.Add(weightedSum, sample.Y, threshold)
		voteConfusion.Add(voteYHat, sample.Y, threshold)

		rows = append(rows, sampleResult{
			Index:         sampleIndex,
			Y:             sample.Y,
			MeanYHat:      meanYHat,
			WeightedYHat:  weightedSum,
			StdYHat:       stddev(yHats, meanYHat),
			MinYHat:       minYHat,
			MaxYHat:       maxYHat,
			PositiveVotes: positiveVotes,
			VoteRatio:     voteRatio,
			MeanPred:      metrics.Classify(meanYHat, threshold),
			WeightedPred:  metrics.Classify(weightedSum, threshold),
			VotePred:      metrics.Classify(voteYHat, threshold),
			ModelYHats:    yHats,
		})
	}

	summaries := make([]modeResult, 0, len(models)+3)
	for i, model := range models {
		summaries = append(summaries, newModeResult(fmt.Sprintf("model_%02d_%s", i, model.Hidden), individualLosses[i]/float64(len(samples)), individualConfusions[i]))
	}
	summaries = append(summaries, newModeResult("ensemble_mean", meanLoss/float64(len(samples)), meanConfusion))
	summaries = append(summaries, newModeResult("ensemble_weighted", weightedLoss/float64(len(samples)), weightedConfusion))
	summaries = append(summaries, newModeResult("ensemble_vote", voteLoss/float64(len(samples)), voteConfusion))

	return rows, summaries, nil
}

func forward(model ensembleModel, x []float64, forwardMode string, sparseThreshold float64) (float64, error) {
	switch forwardMode {
	case "dense":
		return model.Model.ForwardFast(x), nil
	case "sparse-exact":
		return model.Model.ForwardSparsePrepared(x, 0, model.Workspace), nil
	case "sparse-threshold":
		return model.Model.ForwardSparsePrepared(x, sparseThreshold, model.Workspace), nil
	default:
		return 0, fmt.Errorf("unknown forward mode %q: expected dense, sparse-exact or sparse-threshold", forwardMode)
	}
}

func newModeResult(mode string, loss float64, confusion metrics.ConfusionMatrix) modeResult {
	return modeResult{
		Mode:          mode,
		Loss:          loss,
		Accuracy:      confusion.Accuracy(),
		Precision:     confusion.Precision(),
		Recall:        confusion.Recall(),
		F1:            confusion.F1(),
		TrueNegative:  confusion.TrueNegative,
		FalsePositive: confusion.FalsePositive,
		FalseNegative: confusion.FalseNegative,
		TruePositive:  confusion.TruePositive,
	}
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

func writePredictionsCSV(path string, models []ensembleModel, rows []sampleResult) error {
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
		"sample_index",
		"y",
		"mean_yhat",
		"weighted_yhat",
		"std_yhat",
		"min_yhat",
		"max_yhat",
		"positive_votes",
		"vote_ratio",
		"mean_pred",
		"weighted_pred",
		"vote_pred",
	}
	for i, model := range models {
		header = append(header, fmt.Sprintf("model_%02d_%s_yhat", i, model.Hidden))
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	for _, row := range rows {
		record := []string{
			strconv.Itoa(row.Index),
			formatFloat(row.Y),
			formatFloat(row.MeanYHat),
			formatFloat(row.WeightedYHat),
			formatFloat(row.StdYHat),
			formatFloat(row.MinYHat),
			formatFloat(row.MaxYHat),
			strconv.Itoa(row.PositiveVotes),
			formatFloat(row.VoteRatio),
			strconv.Itoa(row.MeanPred),
			strconv.Itoa(row.WeightedPred),
			strconv.Itoa(row.VotePred),
		}
		for _, yHat := range row.ModelYHats {
			record = append(record, formatFloat(yHat))
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return writer.Error()
}

func writeSummaryCSV(path string, models []ensembleModel, weights []float64, split string, forwardMode string, sparseThreshold float64, threshold float64, duration time.Duration, results []modeResult) error {
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
		"mode",
		"split",
		"forward",
		"classification_threshold",
		"sparse_threshold",
		"model_count",
		"duration_ms",
		"loss",
		"accuracy",
		"precision",
		"recall",
		"f1",
		"true_negative",
		"false_positive",
		"false_negative",
		"true_positive",
		"checkpoints",
		"weights",
		"run_ids",
		"hidden_architectures",
		"best_epochs",
	}
	if err := writer.Write(header); err != nil {
		return err
	}

	checkpoints := make([]string, len(models))
	runIDs := make([]string, len(models))
	hidden := make([]string, len(models))
	bestEpochs := make([]string, len(models))
	weightParts := make([]string, len(weights))
	for i, model := range models {
		checkpoints[i] = model.Path
		runIDs[i] = model.RunID
		hidden[i] = model.Hidden
		bestEpochs[i] = strconv.Itoa(model.BestEpoch)
		weightParts[i] = formatFloat(weights[i])
	}

	for _, result := range results {
		record := []string{
			result.Mode,
			split,
			forwardMode,
			formatFloat(threshold),
			formatFloat(sparseThreshold),
			strconv.Itoa(len(models)),
			strconv.FormatInt(duration.Milliseconds(), 10),
			formatFloat(result.Loss),
			formatFloat(result.Accuracy),
			formatFloat(result.Precision),
			formatFloat(result.Recall),
			formatFloat(result.F1),
			strconv.Itoa(result.TrueNegative),
			strconv.Itoa(result.FalsePositive),
			strconv.Itoa(result.FalseNegative),
			strconv.Itoa(result.TruePositive),
			strings.Join(checkpoints, ";"),
			strings.Join(weightParts, ";"),
			strings.Join(runIDs, ";"),
			strings.Join(hidden, ";"),
			strings.Join(bestEpochs, ";"),
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return writer.Error()
}

func stddev(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, value := range values {
		diff := value - mean
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(values)))
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 8, 64)
}
