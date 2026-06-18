package main

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
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

var hiddenCandidates = []int{16, 32, 64, 128, 256, 512}
var learningRates = []float64{0, 0.0001, 0.0003, 0.001, 0.003, 0.01}
var batchSizes = []int{0, 16, 32, 64}

const (
	maxEpochs            = 500
	minEpochs            = 30
	patience             = 35
	lowLearningWindow    = 15
	minValidationDelta   = 1e-4
	lowLearningDelta     = 1e-4
	divergentLossLimit   = 10.0
	superSweepConfigID   = "super-sweep-v1"
	defaultDatasetPath   = "./dataset"
	defaultOutputRoot    = "runs/super_sweep"
)

type superConfig struct {
	Name          string    `json:"name"`
	DatasetPath   string    `json:"dataset_path"`
	HiddenSizes   []int     `json:"hidden_sizes"`
	LearningRate  float64   `json:"learning_rate"`
	BatchSize     int       `json:"batch_size"`
	Seed          int64     `json:"seed"`
	MaxEpochs     int       `json:"max_epochs"`
	MinEpochs     int       `json:"min_epochs"`
	Patience      int       `json:"patience"`
	Window        int       `json:"low_learning_window"`
	MinValDelta   float64   `json:"min_validation_delta"`
	LowLearnDelta float64   `json:"low_learning_delta"`
	CreatedBy     string    `json:"created_by"`
}

type superResult struct {
	RunID                   string                  `json:"run_id"`
	Name                    string                  `json:"name"`
	RunDirectory            string                  `json:"run_directory"`
	Completed               bool                    `json:"completed"`
	Cached                  bool                    `json:"cached"`
	Hidden                  string                  `json:"hidden"`
	Depth                   int                     `json:"depth"`
	ParameterCount          int                     `json:"parameter_count"`
	Seed                    int64                   `json:"seed"`
	LearningRate            float64                 `json:"learning_rate"`
	BatchSize               int                     `json:"batch_size"`
	EffectiveBatchSize      int                     `json:"effective_batch_size"`
	EpochsRun               int                     `json:"epochs_run"`
	StopReason              string                  `json:"stop_reason"`
	BestEpoch               int                     `json:"best_epoch"`
	BestTrainLoss           float64                 `json:"best_train_loss"`
	BestTrainAccuracy       float64                 `json:"best_train_accuracy"`
	BestTrainPrecision      float64                 `json:"best_train_precision"`
	BestTrainRecall         float64                 `json:"best_train_recall"`
	BestTrainF1             float64                 `json:"best_train_f1"`
	BestValidationLoss      float64                 `json:"best_validation_loss"`
	BestValidationAccuracy  float64                 `json:"best_validation_accuracy"`
	BestValidationPrecision float64                 `json:"best_validation_precision"`
	BestValidationRecall    float64                 `json:"best_validation_recall"`
	BestValidationF1        float64                 `json:"best_validation_f1"`
	FinalTrainLoss          float64                 `json:"final_train_loss"`
	FinalTrainAccuracy      float64                 `json:"final_train_accuracy"`
	FinalValidationLoss     float64                 `json:"final_validation_loss"`
	FinalValidationAccuracy float64                 `json:"final_validation_accuracy"`
	GeneralizationGap       float64                 `json:"generalization_gap"`
	Confusion               metrics.ConfusionMatrix `json:"validation_confusion"`
	TrainTimeMilliseconds   int64                   `json:"train_time_ms"`
	Error                   string                  `json:"error,omitempty"`
}

func main() {
	datasetPath := flag.String("dataset", defaultDatasetPath, "dataset root with train, validation and test folders; test is loaded but not evaluated")
	workers := flag.Int("workers", 1, "maximum number of experiments running concurrently")
	outputRoot := flag.String("runs", defaultOutputRoot, "root directory for super sweep outputs")
	flag.Parse()

	if *workers <= 0 {
		log.Fatalf("workers must be positive, got %d", *workers)
	}
	if strings.TrimSpace(*outputRoot) == "" {
		log.Fatal("runs cannot be empty")
	}

	dataset, err := experiment.LoadDataset(*datasetPath)
	if err != nil {
		log.Fatal(err)
	}

	configs := buildConfigs(*datasetPath, *outputRoot)
	fmt.Printf("super_sweep=%s architectures=%d configs=%d workers=%d train=%d validation=%d test_loaded_not_used=%d input=%d out=%s\n", superSweepConfigID, len(generateArchitectures()), len(configs), *workers, len(dataset.Train), len(dataset.Validation), len(dataset.Test), dataset.InputSize, *outputRoot)
	fmt.Printf("hidden_candidates=%v depths=1..3 learning_rates=%v batch_sizes=%v seeds=1..42 max_epochs=%d min_epochs=%d patience=%d low_learning_window=%d\n", hiddenCandidates, learningRates, batchSizes, maxEpochs, minEpochs, patience, lowLearningWindow)

	if err := os.MkdirAll(*outputRoot, 0755); err != nil {
		log.Fatal(err)
	}
	summaryPath := filepath.Join(*outputRoot, "summary.csv")
	summaryFile, err := os.Create(summaryPath)
	if err != nil {
		log.Fatal(err)
	}
	defer summaryFile.Close()
	summaryWriter := csv.NewWriter(summaryFile)
	defer summaryWriter.Flush()
	if err := summaryWriter.Write(summaryHeader()); err != nil {
		log.Fatal(err)
	}

	results := runConfigs(configs, dataset, *workers)
	completed := 0
	failed := 0
	cached := 0
	for i := 0; i < len(configs); i++ {
		result := <-results
		if result.Completed {
			completed++
		} else {
			failed++
		}
		if result.Cached {
			cached++
		}
		if err := summaryWriter.Write(summaryRow(result)); err != nil {
			log.Fatal(err)
		}
		summaryWriter.Flush()

		status := "trained"
		if result.Cached {
			status = "cached"
		}
		if !result.Completed {
			status = "failed"
		}
		fmt.Printf(
			"%s %d/%d run=%s hidden=%s lr=%g batch=%d seed=%d epochs=%d stop=%s val_acc=%.4f val_loss=%.6f train_acc=%.4f gap=%.4f dir=%s error=%s\n",
			status,
			i+1,
			len(configs),
			result.RunID,
			result.Hidden,
			result.LearningRate,
			result.BatchSize,
			result.Seed,
			result.EpochsRun,
			result.StopReason,
			result.BestValidationAccuracy,
			result.BestValidationLoss,
			result.BestTrainAccuracy,
			result.GeneralizationGap,
			result.RunDirectory,
			result.Error,
		)
	}

	fmt.Printf("summary=%s completed=%d failed=%d cached=%d\n", summaryPath, completed, failed, cached)
}

func buildConfigs(datasetPath string, outputRoot string) []superConfig {
	architectures := generateArchitectures()
	configs := make([]superConfig, 0, len(architectures)*len(learningRates)*len(batchSizes)*42)

	for _, hidden := range architectures {
		for _, lr := range learningRates {
			for _, batch := range batchSizes {
				for seed := int64(1); seed <= 42; seed++ {
					name := fmt.Sprintf("super_h%s_lr%s_bs%d_seed%d", experiment.HiddenSizesLabel(hidden), learningRateLabel(lr), batch, seed)
					configs = append(configs, superConfig{
						Name:          name,
						DatasetPath:   datasetPath,
						HiddenSizes:   append([]int(nil), hidden...),
						LearningRate:  lr,
						BatchSize:     batch,
						Seed:          seed,
						MaxEpochs:     maxEpochs,
						MinEpochs:     minEpochs,
						Patience:      patience,
						Window:        lowLearningWindow,
						MinValDelta:   minValidationDelta,
						LowLearnDelta: lowLearningDelta,
						CreatedBy:     superSweepConfigID,
					})
				}
			}
		}
	}

	return configs
}

func generateArchitectures() [][]int {
	architectures := make([][]int, 0, len(hiddenCandidates)+len(hiddenCandidates)*len(hiddenCandidates)+len(hiddenCandidates)*len(hiddenCandidates)*len(hiddenCandidates))
	for _, a := range hiddenCandidates {
		architectures = append(architectures, []int{a})
	}
	for _, a := range hiddenCandidates {
		for _, b := range hiddenCandidates {
			architectures = append(architectures, []int{a, b})
		}
	}
	for _, a := range hiddenCandidates {
		for _, b := range hiddenCandidates {
			for _, c := range hiddenCandidates {
				architectures = append(architectures, []int{a, b, c})
			}
		}
	}
	return architectures
}

func runConfigs(configs []superConfig, dataset experiment.DatasetBundle, maxWorkers int) <-chan superResult {
	if maxWorkers > len(configs) && len(configs) > 0 {
		maxWorkers = len(configs)
	}
	jobs := make(chan superConfig)
	results := make(chan superResult)

	for i := 0; i < maxWorkers; i++ {
		go func() {
			for config := range jobs {
				results <- runOrLoadSuper(config, dataset)
			}
		}()
	}

	go func() {
		for _, config := range configs {
			jobs <- config
		}
		close(jobs)
	}()

	return results
}

func runOrLoadSuper(config superConfig, dataset experiment.DatasetBundle) superResult {
	runID, err := configID(config)
	if err != nil {
		return failedSuperResult(config, "", "", err)
	}
	runDir := filepath.Join(configOutputRoot(config), runID+"_"+sanitizeRunName(config.Name))
	summaryPath := filepath.Join(runDir, "summary.json")

	if err := validateSuperConfig(config); err != nil {
		return failedSuperResult(config, runID, runDir, err)
	}

	if _, err := os.Stat(summaryPath); err == nil {
		var summary superResult
		if err := readJSON(summaryPath, &summary); err == nil && summary.Completed {
			summary.Cached = true
			return summary
		}
	}

	if err := os.MkdirAll(filepath.Join(runDir, "checkpoints"), 0755); err != nil {
		return failedSuperResult(config, runID, runDir, err)
	}
	if err := writeJSONAtomic(filepath.Join(runDir, "config.json"), config); err != nil {
		return failedSuperResult(config, runID, runDir, err)
	}

	effectiveBatch := config.BatchSize
	if effectiveBatch <= 0 || effectiveBatch > len(dataset.Train) {
		effectiveBatch = len(dataset.Train)
	}

	model := nn.NewMLPWithHiddenSizes(dataset.InputSize, config.HiddenSizes, config.Seed)
	rng := rand.New(rand.NewSource(config.Seed + 1))

	logger, err := metrics.NewEpochCSVLogger(filepath.Join(runDir, "metrics.csv"))
	if err != nil {
		return failedSuperResult(config, runID, runDir, err)
	}
	defer logger.Close()

	startedAt := time.Now()
	var bestModel *nn.MLP
	bestEpoch := 0
	bestTrain := nn.EpochResult{}
	bestValidation := nn.EpochResult{}
	finalTrain := nn.EpochResult{}
	finalValidation := nn.EpochResult{}
	bestSet := false
	noImprovementEpochs := 0
	stopReason := "max_epochs"
	trainLossHistory := make([]float64, 0, config.MaxEpochs)
	valLossHistory := make([]float64, 0, config.MaxEpochs)

	for epoch := 1; epoch <= config.MaxEpochs; epoch++ {
		trainResult, err := nn.TrainEpochMiniBatch(model, dataset.Train, config.LearningRate, effectiveBatch, rng)
		if err != nil {
			return failedSuperResult(config, runID, runDir, err)
		}
		validationResult, err := nn.Evaluate(model, dataset.Validation)
		if err != nil {
			return failedSuperResult(config, runID, runDir, err)
		}

		finalTrain = trainResult
		finalValidation = validationResult
		trainLossHistory = append(trainLossHistory, trainResult.Loss)
		valLossHistory = append(valLossHistory, validationResult.Loss)

		if isBetterValidation(validationResult, bestValidation, bestSet, config.MinValDelta) {
			bestModel = model.Clone()
			bestEpoch = epoch
			bestTrain = trainResult
			bestValidation = validationResult
			bestSet = true
			noImprovementEpochs = 0
			checkpoint := experiment.NewCheckpoint(runID, epoch, bestEpoch, bestValidation, bestModel)
			if err := experiment.SaveCheckpoint(filepath.Join(runDir, "checkpoints", "best.gob"), checkpoint); err != nil {
				return failedSuperResult(config, runID, runDir, err)
			}
		} else {
			noImprovementEpochs++
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
			return failedSuperResult(config, runID, runDir, err)
		}

		if shouldStopDivergent(epoch, trainResult, validationResult) {
			stopReason = "divergent_or_non_finite"
			break
		}
		if epoch >= config.MinEpochs && noImprovementEpochs >= config.Patience {
			stopReason = "validation_patience"
			break
		}
		if epoch >= config.MinEpochs && hasLowLearning(trainLossHistory, valLossHistory, config.Window, config.LowLearnDelta) && noImprovementEpochs >= config.Window {
			stopReason = "low_learning"
			break
		}
	}

	if !bestSet {
		bestModel = model.Clone()
		bestEpoch = len(trainLossHistory)
		bestTrain = finalTrain
		bestValidation = finalValidation
		checkpoint := experiment.NewCheckpoint(runID, bestEpoch, bestEpoch, bestValidation, bestModel)
		if err := experiment.SaveCheckpoint(filepath.Join(runDir, "checkpoints", "best.gob"), checkpoint); err != nil {
			return failedSuperResult(config, runID, runDir, err)
		}
	}

	result := superResult{
		RunID:                   runID,
		Name:                    config.Name,
		RunDirectory:            runDir,
		Completed:               true,
		Hidden:                  experiment.HiddenSizesLabel(config.HiddenSizes),
		Depth:                   len(config.HiddenSizes),
		ParameterCount:          parameterCount(dataset.InputSize, config.HiddenSizes),
		Seed:                    config.Seed,
		LearningRate:            config.LearningRate,
		BatchSize:               config.BatchSize,
		EffectiveBatchSize:      effectiveBatch,
		EpochsRun:               len(trainLossHistory),
		StopReason:              stopReason,
		BestEpoch:               bestEpoch,
		BestTrainLoss:           bestTrain.Loss,
		BestTrainAccuracy:       bestTrain.Accuracy,
		BestTrainPrecision:      bestTrain.Confusion.Precision(),
		BestTrainRecall:         bestTrain.Confusion.Recall(),
		BestTrainF1:             bestTrain.Confusion.F1(),
		BestValidationLoss:      bestValidation.Loss,
		BestValidationAccuracy:  bestValidation.Accuracy,
		BestValidationPrecision: bestValidation.Confusion.Precision(),
		BestValidationRecall:    bestValidation.Confusion.Recall(),
		BestValidationF1:        bestValidation.Confusion.F1(),
		FinalTrainLoss:          finalTrain.Loss,
		FinalTrainAccuracy:      finalTrain.Accuracy,
		FinalValidationLoss:     finalValidation.Loss,
		FinalValidationAccuracy: finalValidation.Accuracy,
		GeneralizationGap:       bestTrain.Accuracy - bestValidation.Accuracy,
		Confusion:               bestValidation.Confusion,
		TrainTimeMilliseconds:   time.Since(startedAt).Milliseconds(),
	}

	if err := writeJSONAtomic(summaryPath, result); err != nil {
		return failedSuperResult(config, runID, runDir, err)
	}

	return result
}

func validateSuperConfig(config superConfig) error {
	if len(config.HiddenSizes) == 0 {
		return fmt.Errorf("at least one hidden layer is required")
	}
	if len(config.HiddenSizes) > 3 {
		return fmt.Errorf("at most three hidden layers are allowed, got %d", len(config.HiddenSizes))
	}
	for i, size := range config.HiddenSizes {
		if size <= 0 {
			return fmt.Errorf("hidden layer %d must be positive, got %d", i, size)
		}
	}
	if config.LearningRate < 0 {
		return fmt.Errorf("learning rate must be non-negative, got %g", config.LearningRate)
	}
	if config.MaxEpochs <= 0 {
		return fmt.Errorf("max epochs must be positive, got %d", config.MaxEpochs)
	}
	if config.MinEpochs <= 0 || config.MinEpochs > config.MaxEpochs {
		return fmt.Errorf("invalid min epochs: %d", config.MinEpochs)
	}
	if config.Patience <= 0 {
		return fmt.Errorf("patience must be positive, got %d", config.Patience)
	}
	if config.Window <= 0 {
		return fmt.Errorf("low-learning window must be positive, got %d", config.Window)
	}
	return nil
}

func isBetterValidation(candidate nn.EpochResult, current nn.EpochResult, currentSet bool, minDelta float64) bool {
	if !currentSet {
		return true
	}
	if candidate.Accuracy != current.Accuracy {
		return candidate.Accuracy > current.Accuracy
	}
	return candidate.Loss < current.Loss-minDelta
}

func shouldStopDivergent(epoch int, trainResult nn.EpochResult, validationResult nn.EpochResult) bool {
	if math.IsNaN(trainResult.Loss) || math.IsNaN(validationResult.Loss) || math.IsInf(trainResult.Loss, 0) || math.IsInf(validationResult.Loss, 0) {
		return true
	}
	if epoch >= minEpochs && (trainResult.Loss > divergentLossLimit || validationResult.Loss > divergentLossLimit) {
		return true
	}
	return false
}

func hasLowLearning(trainLosses []float64, valLosses []float64, window int, minDelta float64) bool {
	if len(trainLosses) <= window || len(valLosses) <= window {
		return false
	}
	last := len(trainLosses) - 1
	previous := last - window
	trainImprovement := trainLosses[previous] - trainLosses[last]
	valImprovement := valLosses[previous] - valLosses[last]
	return trainImprovement < minDelta && valImprovement < minDelta
}

func failedSuperResult(config superConfig, runID string, runDir string, err error) superResult {
	result := superResult{
		RunID:              runID,
		Name:               config.Name,
		RunDirectory:       runDir,
		Completed:          false,
		Hidden:             experiment.HiddenSizesLabel(config.HiddenSizes),
		Depth:              len(config.HiddenSizes),
		ParameterCount:     parameterCount(0, config.HiddenSizes),
		Seed:               config.Seed,
		LearningRate:       config.LearningRate,
		BatchSize:          config.BatchSize,
		EffectiveBatchSize: config.BatchSize,
		StopReason:         "failed",
	}
	if err != nil {
		result.Error = err.Error()
	}
	return result
}

func parameterCount(inputSize int, hiddenSizes []int) int {
	if inputSize <= 0 || len(hiddenSizes) == 0 {
		return 0
	}
	count := 0
	previous := inputSize
	for _, size := range hiddenSizes {
		count += previous*size + size
		previous = size
	}
	count += previous*1 + 1
	return count
}

func configID(config superConfig) (string, error) {
	payload, err := json.Marshal(config)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])[:12], nil
}

func configOutputRoot(config superConfig) string {
	root := strings.TrimSpace(config.DatasetPath)
	_ = root
	// Output root is intentionally not part of the stable fingerprint. The actual
	// root comes from the generated config name path in buildConfigs.
	return currentOutputRoot
}

var currentOutputRoot string

func init() {
	currentOutputRoot = defaultOutputRoot
}

func sanitizeRunName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	var builder strings.Builder
	lastWasSeparator := false
	for _, r := range name {
		allowed := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if allowed {
			builder.WriteRune(r)
			lastWasSeparator = false
			continue
		}
		if !lastWasSeparator {
			builder.WriteRune('_')
			lastWasSeparator = true
		}
	}
	return strings.Trim(builder.String(), "_")
}

func learningRateLabel(lr float64) string {
	label := strconv.FormatFloat(lr, 'g', -1, 64)
	label = strings.ReplaceAll(label, ".", "p")
	label = strings.ReplaceAll(label, "-", "m")
	return label
}

func summaryHeader() []string {
	return []string{
		"run_id",
		"name",
		"completed",
		"cached",
		"hidden",
		"depth",
		"parameter_count",
		"seed",
		"learning_rate",
		"batch_size",
		"effective_batch_size",
		"epochs_run",
		"stop_reason",
		"best_epoch",
		"best_train_loss",
		"best_train_accuracy",
		"best_train_precision",
		"best_train_recall",
		"best_train_f1",
		"best_val_loss",
		"best_val_accuracy",
		"best_val_precision",
		"best_val_recall",
		"best_val_f1",
		"final_train_loss",
		"final_train_accuracy",
		"final_val_loss",
		"final_val_accuracy",
		"generalization_gap",
		"val_true_negative",
		"val_false_positive",
		"val_false_negative",
		"val_true_positive",
		"train_time_ms",
		"run_directory",
		"error",
	}
}

func summaryRow(result superResult) []string {
	return []string{
		result.RunID,
		result.Name,
		strconv.FormatBool(result.Completed),
		strconv.FormatBool(result.Cached),
		result.Hidden,
		strconv.Itoa(result.Depth),
		strconv.Itoa(result.ParameterCount),
		strconv.FormatInt(result.Seed, 10),
		formatFloat(result.LearningRate),
		strconv.Itoa(result.BatchSize),
		strconv.Itoa(result.EffectiveBatchSize),
		strconv.Itoa(result.EpochsRun),
		result.StopReason,
		strconv.Itoa(result.BestEpoch),
		formatFloat(result.BestTrainLoss),
		formatFloat(result.BestTrainAccuracy),
		formatFloat(result.BestTrainPrecision),
		formatFloat(result.BestTrainRecall),
		formatFloat(result.BestTrainF1),
		formatFloat(result.BestValidationLoss),
		formatFloat(result.BestValidationAccuracy),
		formatFloat(result.BestValidationPrecision),
		formatFloat(result.BestValidationRecall),
		formatFloat(result.BestValidationF1),
		formatFloat(result.FinalTrainLoss),
		formatFloat(result.FinalTrainAccuracy),
		formatFloat(result.FinalValidationLoss),
		formatFloat(result.FinalValidationAccuracy),
		formatFloat(result.GeneralizationGap),
		strconv.Itoa(result.Confusion.TrueNegative),
		strconv.Itoa(result.Confusion.FalsePositive),
		strconv.Itoa(result.Confusion.FalseNegative),
		strconv.Itoa(result.Confusion.TruePositive),
		strconv.FormatInt(result.TrainTimeMilliseconds, 10),
		result.RunDirectory,
		result.Error,
	}
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 8, 64)
}

func readJSON(path string, out any) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return json.NewDecoder(file).Decode(out)
}

func writeJSONAtomic(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	file, err := os.Create(tmp)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func init() {
	sort.Ints(hiddenCandidates)
}
