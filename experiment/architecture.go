package experiment

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseHiddenArchitectures interpreta uma lista de arquiteturas separadas por ponto e vírgula.
// Exemplos aceitos: "128", "256x64", "512-128", "1024:256:64".
func ParseHiddenArchitectures(raw string) ([][]int, error) {
	parts := strings.Split(raw, ";")
	architectures := make([][]int, 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		architecture, err := ParseHiddenArchitecture(part)
		if err != nil {
			return nil, err
		}
		architectures = append(architectures, architecture)
	}

	if len(architectures) == 0 {
		return nil, fmt.Errorf("empty hidden architecture list")
	}

	return architectures, nil
}

// ParseHiddenArchitecture interpreta uma única arquitetura de camadas ocultas.
func ParseHiddenArchitecture(raw string) ([]int, error) {
	normalized := strings.NewReplacer("x", ",", "X", ",", "-", ",", ":", ",").Replace(raw)
	values, err := parsePositiveInts(normalized)
	if err != nil {
		return nil, fmt.Errorf("invalid hidden architecture %q: %w", raw, err)
	}
	return values, nil
}

// HiddenSizesLabel transforma uma arquitetura em um rótulo estável para logs e nomes de execução.
func HiddenSizesLabel(hiddenSizes []int) string {
	parts := make([]string, len(hiddenSizes))
	for i, size := range hiddenSizes {
		parts[i] = strconv.Itoa(size)
	}
	return strings.Join(parts, "x")
}

func parsePositiveInts(raw string) ([]int, error) {
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
		if value <= 0 {
			return nil, fmt.Errorf("layer size must be positive, got %d", value)
		}

		values = append(values, value)
	}

	if len(values) == 0 {
		return nil, fmt.Errorf("empty integer list")
	}

	return values, nil
}
