package experiment

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"unicode"

	"github.com/NathanielChristian2077/gomlp/nn"
)

// RunConfig descreve uma execução treinável da MLP.
// HiddenSizes permite avaliar arquiteturas com uma ou mais camadas ocultas.
type RunConfig struct {
	Name         string  `json:"name"`
	DatasetPath  string  `json:"dataset_path"`
	Epochs       int     `json:"epochs"`
	HiddenSizes  []int   `json:"hidden_sizes"`
	BatchSize    int     `json:"batch_size"`
	Seed         int64   `json:"seed"`
	LearningRate float64 `json:"learning_rate"`
	OutputHead   string  `json:"output_head"`
	OutputRoot   string  `json:"output_root"`
	LogEvery     int     `json:"log_every"`
}

type runFingerprint struct {
	DatasetPath  string  `json:"dataset_path"`
	Epochs       int     `json:"epochs"`
	HiddenSizes  []int   `json:"hidden_sizes"`
	BatchSize    int     `json:"batch_size"`
	Seed         int64   `json:"seed"`
	LearningRate float64 `json:"learning_rate"`
	OutputHead   string  `json:"output_head"`
}

// Normalize aplica apenas defaults seguros e preserva valores inválidos para Validate reportar.
func (c RunConfig) Normalize(defaultLearningRate float64) RunConfig {
	out := c
	if len(out.HiddenSizes) == 0 {
		out.HiddenSizes = []int{128}
	} else {
		out.HiddenSizes = cloneIntSlice(out.HiddenSizes)
	}
	if out.LearningRate < 0 {
		out.LearningRate = defaultLearningRate
	}
	if out.Seed == 0 {
		out.Seed = 42
	}
	if out.OutputHead == "" {
		out.OutputHead = string(nn.OutputHeadSigmoid1)
	}
	if out.OutputRoot == "" {
		out.OutputRoot = "runs"
	}
	if out.Name == "" {
		out.Name = fmt.Sprintf("dense_h%s_%s_lr%g_bs%d_seed%d", HiddenSizesLabel(out.HiddenSizes), out.OutputHead, out.LearningRate, out.BatchSize, out.Seed)
	}
	return out
}

func (c RunConfig) Validate() error {
	if c.Epochs <= 0 {
		return fmt.Errorf("epochs must be positive, got %d", c.Epochs)
	}
	if len(c.HiddenSizes) == 0 {
		return fmt.Errorf("at least one hidden layer is required")
	}
	for i, size := range c.HiddenSizes {
		if size <= 0 {
			return fmt.Errorf("hidden layer %d size must be positive, got %d", i, size)
		}
	}
	if c.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive, got %g", c.LearningRate)
	}
	if _, err := nn.NormalizeOutputHead(c.OutputHead); err != nil {
		return err
	}
	if c.OutputRoot == "" {
		return fmt.Errorf("output root cannot be empty")
	}
	return nil
}

func (c RunConfig) NormalizedOutputHead() string {
	head, err := nn.NormalizeOutputHead(c.OutputHead)
	if err != nil {
		return c.OutputHead
	}
	return string(head)
}

func (c RunConfig) ID() (string, error) {
	fingerprint := runFingerprint{
		DatasetPath:  c.DatasetPath,
		Epochs:       c.Epochs,
		HiddenSizes:  cloneIntSlice(c.HiddenSizes),
		BatchSize:    c.BatchSize,
		Seed:         c.Seed,
		LearningRate: c.LearningRate,
		OutputHead:   c.NormalizedOutputHead(),
	}

	payload, err := json.Marshal(fingerprint)
	if err != nil {
		return "", err
	}

	sum := sha256.Sum256(payload)
	return hex.EncodeToString(sum[:])[:12], nil
}

func (c RunConfig) RunDirectory() (string, error) {
	id, err := c.ID()
	if err != nil {
		return "", err
	}

	name := sanitizeRunName(c.Name)
	if name == "" {
		name = "run"
	}

	return filepath.Join(c.OutputRoot, id+"_"+name), nil
}

func sanitizeRunName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	var builder strings.Builder

	lastWasSeparator := false
	for _, r := range name {
		allowed := unicode.IsLetter(r) || unicode.IsDigit(r)
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

func cloneIntSlice(values []int) []int {
	out := make([]int, len(values))
	copy(out, values)
	return out
}
