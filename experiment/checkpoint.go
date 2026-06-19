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
	RunID                  string            `json:"run_id"`
	Epoch                  int               `json:"epoch"`
	BestEpoch              int               `json:"best_epoch"`
	BestValidationLoss     float64           `json:"best_validation_loss"`
	BestValidationAccuracy float64           `json:"best_validation_accuracy"`
	OutputHead             string            `json:"output_head"`
	Hidden                 []LayerCheckpoint `json:"hidden"`
	Output                 LayerCheckpoint   `json:"output"`
}

func NewCheckpoint(runID string, epoch int, bestEpoch int, bestValidation nn.EpochResult, model *nn.MLP) Checkpoint {
	return Checkpoint{
		RunID:                  runID,
		Epoch:                  epoch,
		BestEpoch:              bestEpoch,
		BestValidationLoss:     bestValidation.Loss,
		BestValidationAccuracy: bestValidation.Accuracy,
		OutputHead:             string(model.Head()),
		Hidden:                 snapshotLayers(model.Hidden),
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
	if len(checkpoint.Hidden) == 0 {
		return nil, fmt.Errorf("invalid model checkpoint: no hidden layers")
	}

	for i, hidden := range checkpoint.Hidden {
		if err := validateLayerCheckpoint(hidden); err != nil {
			return nil, fmt.Errorf("invalid hidden checkpoint %d: %w", i, err)
		}
		if i > 0 && hidden.In != checkpoint.Hidden[i-1].Out {
			return nil, fmt.Errorf("invalid hidden checkpoint %d: input size %d does not match previous output size %d", i, hidden.In, checkpoint.Hidden[i-1].Out)
		}
	}

	if err := validateLayerCheckpoint(checkpoint.Output); err != nil {
		return nil, fmt.Errorf("invalid output checkpoint: %w", err)
	}
	lastHidden := checkpoint.Hidden[len(checkpoint.Hidden)-1]
	if checkpoint.Output.In != lastHidden.Out {
		return nil, fmt.Errorf("invalid model checkpoint: output input size %d does not match last hidden output size %d", checkpoint.Output.In, lastHidden.Out)
	}

	head, err := checkpointOutputHead(checkpoint)
	if err != nil {
		return nil, err
	}
	if checkpoint.Output.Out != head.OutputSize() {
		return nil, fmt.Errorf("invalid model checkpoint: output layer has %d neurons but output head %s expects %d", checkpoint.Output.Out, head, head.OutputSize())
	}

	hidden := restoreLayers(checkpoint.Hidden)
	output := restoreLayer(checkpoint.Output)

	hiddenZ := make([][]float64, len(hidden))
	hiddenA := make([][]float64, len(hidden))
	deltaHidden := make([][]float64, len(hidden))
	for i, layer := range hidden {
		hiddenZ[i] = make([]float64, layer.Out)
		hiddenA[i] = make([]float64, layer.Out)
		deltaHidden[i] = make([]float64, layer.Out)
	}

	return &nn.MLP{
		Hidden:     hidden,
		Output:     output,
		OutputHead: head,

		HiddenZ: hiddenZ,
		HiddenA: hiddenA,
		OutputZ: make([]float64, output.Out),

		DeltaHidden: deltaHidden,
		DeltaOutput: make([]float64, output.Out),
	}, nil
}

func checkpointOutputHead(checkpoint Checkpoint) (nn.OutputHead, error) {
	if checkpoint.OutputHead != "" {
		return nn.NormalizeOutputHead(checkpoint.OutputHead)
	}
	// Compatibilidade com checkpoints antigos que não tinham o campo OutputHead.
	switch checkpoint.Output.Out {
	case 1:
		return nn.OutputHeadSigmoid1, nil
	case 2:
		return nn.OutputHeadSoftmax2, nil
	default:
		return "", fmt.Errorf("cannot infer output head for output size %d", checkpoint.Output.Out)
	}
}

func snapshotLayers(layers []nn.DenseLayer) []LayerCheckpoint {
	out := make([]LayerCheckpoint, len(layers))
	for i, layer := range layers {
		out[i] = snapshotLayer(layer)
	}
	return out
}

func snapshotLayer(layer nn.DenseLayer) LayerCheckpoint {
	return LayerCheckpoint{
		In:      layer.In,
		Out:     layer.Out,
		Weights: cloneFloat64Slice(layer.Weights),
		Biases:  cloneFloat64Slice(layer.Biases),
	}
}

func restoreLayers(checkpoints []LayerCheckpoint) []nn.DenseLayer {
	out := make([]nn.DenseLayer, len(checkpoints))
	for i, checkpoint := range checkpoints {
		out[i] = restoreLayer(checkpoint)
	}
	return out
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
