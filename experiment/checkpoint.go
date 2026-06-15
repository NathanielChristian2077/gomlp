package experiment

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"

	"github.com/NathanielChristian2077/gomlp/nn"
)

type LayerCheckpoint struct {
	In      int       `json:"in"`
	Out     int       `json:"out"`
	Weights []float64 `json:"weights"`
	Biases  []float64 `json:"biases"`
}

type Checkpoint struct {
	RunID                  string          `json:"run_id"`
	Epoch                  int             `json:"epoch"`
	BestEpoch              int             `json:"best_epoch"`
	BestValidationLoss     float64         `json:"best_validation_loss"`
	BestValidationAccuracy float64         `json:"best_validation_accuracy"`
	Hidden                 LayerCheckpoint `json:"hidden"`
	Output                 LayerCheckpoint `json:"output"`
}

func NewCheckpoint(runID string, epoch int, bestEpoch int, bestValidation nn.EpochResult, model *nn.MLP) Checkpoint {
	return Checkpoint{
		RunID:                  runID,
		Epoch:                  epoch,
		BestEpoch:              bestEpoch,
		BestValidationLoss:     bestValidation.Loss,
		BestValidationAccuracy: bestValidation.Accuracy,
		Hidden:                 snapshotLayer(model.Hidden),
		Output:                 snapshotLayer(model.Output),
	}
}

func SaveCheckpoint(path string, checkpoint Checkpoint) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	tmp := path + ".tmp"
	file, err := os.Create(tmp)
	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(checkpoint); err != nil {
		_ = file.Close()
		return err
	}

	if err := file.Close(); err != nil {
		return err
	}

	return os.Rename(tmp, path)
}

func LoadCheckpoint(path string) (Checkpoint, error) {
	file, err := os.Open(path)
	if err != nil {
		return Checkpoint{}, err
	}
	defer file.Close()

	var checkpoint Checkpoint
	if err := gob.NewDecoder(file).Decode(&checkpoint); err != nil {
		return Checkpoint{}, err
	}

	return checkpoint, nil
}

func RestoreModel(checkpoint Checkpoint) (*nn.MLP, error) {
	if err := validateLayerCheckpoint(checkpoint.Hidden); err != nil {
		return nil, fmt.Errorf("invalid hidden checkpoint: %w", err)
	}
	if err := validateLayerCheckpoint(checkpoint.Output); err != nil {
		return nil, fmt.Errorf("invalid output checkpoint: %w", err)
	}
	if checkpoint.Output.In != checkpoint.Hidden.Out {
		return nil, fmt.Errorf("invalid model checkpoint: output input size %d does not match hidden output size %d", checkpoint.Output.In, checkpoint.Hidden.Out)
	}
	if checkpoint.Output.Out != 1 {
		return nil, fmt.Errorf("invalid model checkpoint: output layer must have one neuron, got %d", checkpoint.Output.Out)
	}

	hidden := restoreLayer(checkpoint.Hidden)
	output := restoreLayer(checkpoint.Output)

	return &nn.MLP{
		Hidden: hidden,
		Output: output,

		HiddenZ: make([]float64, hidden.Out),
		HiddenA: make([]float64, hidden.Out),
		OutputZ: make([]float64, output.Out),

		DeltaHidden: make([]float64, hidden.Out),
		DeltaOutput: make([]float64, output.Out),
	}, nil
}

func snapshotLayer(layer nn.DenseLayer) LayerCheckpoint {
	return LayerCheckpoint{
		In:      layer.In,
		Out:     layer.Out,
		Weights: cloneFloat64Slice(layer.Weights),
		Biases:  cloneFloat64Slice(layer.Biases),
	}
}

func restoreLayer(checkpoint LayerCheckpoint) nn.DenseLayer {
	return nn.DenseLayer{
		In:      checkpoint.In,
		Out:     checkpoint.Out,
		Weights: cloneFloat64Slice(checkpoint.Weights),
		Biases:  cloneFloat64Slice(checkpoint.Biases),
		GradW:   make([]float64, checkpoint.In*checkpoint.Out),
		GradB:   make([]float64, checkpoint.Out),
	}
}

func validateLayerCheckpoint(checkpoint LayerCheckpoint) error {
	if checkpoint.In <= 0 || checkpoint.Out <= 0 {
		return fmt.Errorf("invalid shape: in=%d out=%d", checkpoint.In, checkpoint.Out)
	}
	if len(checkpoint.Weights) != checkpoint.In*checkpoint.Out {
		return fmt.Errorf("invalid weights length: expected %d got %d", checkpoint.In*checkpoint.Out, len(checkpoint.Weights))
	}
	if len(checkpoint.Biases) != checkpoint.Out {
		return fmt.Errorf("invalid biases length: expected %d got %d", checkpoint.Out, len(checkpoint.Biases))
	}
	return nil
}

func cloneFloat64Slice(values []float64) []float64 {
	out := make([]float64, len(values))
	copy(out, values)
	return out
}
