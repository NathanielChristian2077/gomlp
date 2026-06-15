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
// Nesta fase a arquitetura ainda usa uma camada oculta; quando o MLP virar N-camadas,
// este tipo pode evoluir de HiddenSize para HiddenSizes sem quebrar o runner inteiro.
type RunConfig struct {
	Name         string  `json:"name"`
	DatasetPath  string  `json:"dataset_path"`
	Epochs       int     `json:"epochs"`
	HiddenSize   int     `json:"hidden_size"`
	BatchSize    int     `json:"batch_size"`
	Seed         int64   `json:"seed"`
	LearningRate float64 `json:"learning_rate"`
	OutputRoot   string  `json:"output_root"`
	LogEvery     int     `json:"log_every"`
}

type runFingerprint struct {
	DatasetPath  string  `json:"dataset_path"`
	Epochs       int     `json:"epochs"`
	HiddenSize   int     `json:"hidden_size"`
	BatchSize    int     `json:"batch_size"`
	Seed         int64   `json:"seed"`
	LearningRate float64 `json:"learning_rate"`
}

func (c RunConfig) Normalize(defaultLearningRate float64) RunConfig {
	out := c
	if out.Name == "" {
		out.Name = fmt.Sprintf("dense_h%d_lr%g_bs%d_seed%d", out.HiddenSize, out.LearningRate, out.BatchSize, out.Seed)
	}
	if out.Epochs <= 0 {
		out.Epochs = 1
	}
	if out.HiddenSize <= 0 {
		out.HiddenSize = 128
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
		HiddenSize:   c.HiddenSize,
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
