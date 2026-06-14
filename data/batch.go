package data

import (
	"math/rand"

	"github.com/NathanielChristian2077/gomlp/nn"
)

func ShuffleSamples(samples []nn.Sample, rng *rand.Rand) {
	if rng == nil || len(samples) < 2 {
		return
	}

	rng.Shuffle(len(samples), func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})
}

func Batches(samples []nn.Sample, batchSize int) [][]nn.Sample {
	if batchSize <= 0 || len(samples) == 0 {
		return nil
	}

	batches := make([][]nn.Sample, 0, (len(samples)+batchSize-1)/batchSize)

	for start := 0; start < len(samples); start += batchSize {
		end := start + batchSize
		if end > len(samples) {
			end = len(samples)
		}

		batches = append(batches, samples[start:end])
	}

	return batches
}
