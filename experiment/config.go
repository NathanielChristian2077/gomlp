package experiment

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"unicode"
)

// RunConfig descreve uma execução treinável da MLP baseline atual.
// HiddenSizes permite comparar profundidade de forma real, em vez de trocar só
// o tamanho de uma camada e fingir que isso é arquitetura profunda. Pequena vitória
// contra a dramaturgia dos hiperparâmetros.
type RunConfig struct {
	Name         string  `json:"name"`
	DatasetPath  string  `json:"dataset_path"`
	Epochs       int     `json:"epochs"`
	HiddenSizes  []int   `json:"hidden_sizes"`
	BatchSize    int     `json:"batch_size"`
	Seed         int64   `json:"seed"`
	LearningRate float64 `json:"learning_rate"`
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
}

func (c RunConfig) Normalize(defaultLearningRate float64) RunConfig {
	out := c
	if len(out.HiddenSizes) == 0 {
		out.HiddenSizes = []int{128}
	} else {
		out.HiddenSizes = cloneIntSlice(out.HiddenSizes)
	}
	if out.Name == "" {
		out.Name = fmt.Sprintf("dense_h%s_lr%g_bs%d_seed%d", hiddenSizesLabel(out.HiddenSizes), out.LearningRate, out.BatchSize, out.Seed)
	}
	if out.Epochs <= 0 {
		out.Epochs = 1
	}
	for i, size := range out.HiddenSizes {
		if size <= 0 {
			out.HiddenSizes[i] = 128
		}
	}
	if out.Seed == 0 {
		out.Seed = 42
	}
	if out.LearningRate < 0 {
		out.LearningRate = defaultLearningRate
	}
	if out.OutputRoot == "" {
		out.OutputRoot = "runs"
	}
	if out.LogEvery < 0 {
		out.LogEvery = 0
	}
	return out
}

func (c RunConfig) ID() (string, error) {
	fingerprint := runFingerprint{
		DatasetPath:  c.DatasetPath,
		Epochs:       c.Epochs,
		HiddenSizes:  cloneIntSlice(c.HiddenSizes),
		BatchSize:    c.BatchSize,
		Seed:         c.Seed,
		LearningRate: c.LearningRate,
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

func hiddenSizesLabel(hiddenSizes []int) string {
	parts := make([]string, len(hiddenSizes))
	for i, size := range hiddenSizes {
		parts[i] = fmt.Sprintf("%d", size)
	}
	return strings.Join(parts, "x")
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
