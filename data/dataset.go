package data

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/NathanielChristian2077/gomlp/nn"
)

const (
	TrainSplit      = "train"
	ValidationSplit = "validation"
	TestSplit       = "test"

	CatLabel = 0.0
	DogLabel = 1.0
)

type Dataset struct {
	Train      []nn.Sample
	Validation []nn.Sample
	Test       []nn.Sample
}

func LoadDataset(root string) (Dataset, error) {
	train, err := LoadSplit(root, TrainSplit)
	if err != nil {
		return Dataset{}, err
	}

	validation, err := LoadSplit(root, ValidationSplit)
	if err != nil {
		return Dataset{}, err
	}

	test, err := LoadSplit(root, TestSplit)
	if err != nil {
		return Dataset{}, err
	}

	return Dataset{
		Train:      train,
		Validation: validation,
		Test:       test,
	}, nil
}

func LoadSplit(root string, split string) ([]nn.Sample, error) {
	if root == "" {
		return nil, fmt.Errorf("empty dataset root")
	}
	if split == "" {
		return nil, fmt.Errorf("empty dataset split")
	}

	catDir := filepath.Join(root, split, "cat")
	dogDir := filepath.Join(root, split, "dog")

	cats, err := loadClassDir(catDir, CatLabel)
	if err != nil {
		return nil, err
	}

	dogs, err := loadClassDir(dogDir, DogLabel)
	if err != nil {
		return nil, err
	}

	samples := make([]nn.Sample, 0, len(cats)+len(dogs))
	samples = append(samples, cats...)
	samples = append(samples, dogs...)

	return samples, nil
}

func loadClassDir(dir string, label float64) ([]nn.Sample, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	paths := make([]string, 0, len(entries))

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		if isSupportedImage(path) {
			paths = append(paths, path)
		}
	}

	sort.Strings(paths)

	if len(paths) == 0 {
		return nil, fmt.Errorf("no supported images found in %s", dir)
	}

	samples := make([]nn.Sample, 0, len(paths))

	for _, path := range paths {
		vector, err := LoadImageVector(path)
		if err != nil {
			return nil, fmt.Errorf("load image %s: %w", path, err)
		}

		samples = append(samples, nn.Sample{
			X: vector,
			Y: label,
		})
	}

	return samples, nil
}

func isSupportedImage(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png"
}
