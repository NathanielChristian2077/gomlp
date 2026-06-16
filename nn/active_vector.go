package nn

import "fmt"

// ActiveVector representa um vetor denso após remoção das ativações inativas.
// Size preserva o tamanho original do vetor antes da compactação.
// Indices guarda as posições originais dos neurônios ativos.
// Values guarda as ativações correspondentes a cada índice.
type ActiveVector struct {
	Size    int
	Indices []int
	Values  []float64
}

func NewActiveVector(size int) ActiveVector {
	if size < 0 {
		panic(fmt.Sprintf("invalid active vector size: %d", size))
	}
	return ActiveVector{
		Size:    size,
		Indices: make([]int, 0, size),
		Values:  make([]float64, 0, size),
	}
}

func (v *ActiveVector) Reset(size int) {
	if size < 0 {
		panic(fmt.Sprintf("invalid active vector size: %d", size))
	}
	v.Size = size
	v.Indices = v.Indices[:0]
	v.Values = v.Values[:0]
}

func (v ActiveVector) ActiveCount() int {
	return len(v.Indices)
}

func (v ActiveVector) ActiveRatio() float64 {
	if v.Size == 0 {
		return 0
	}
	return float64(v.ActiveCount()) / float64(v.Size)
}

func (v ActiveVector) Sparsity() float64 {
	return 1 - v.ActiveRatio()
}

func (v ActiveVector) ToDense() []float64 {
	out := make([]float64, v.Size)
	v.WriteDense(out)
	return out
}

func (v ActiveVector) WriteDense(out []float64) {
	v.mustMatchSize(len(out), "active vector dense output")

	for i := range out {
		out[i] = 0
	}

	for k, index := range v.Indices {
		out[index] = v.Values[k]
	}
}

func (v ActiveVector) mustMatchSize(expectedSize int, name string) {
	if v.Size != expectedSize {
		panic(fmt.Sprintf("invalid %s size: expected %d, got %d", name, expectedSize, v.Size))
	}
	if len(v.Indices) != len(v.Values) {
		panic(fmt.Sprintf("invalid %s: %d indices for %d values", name, len(v.Indices), len(v.Values)))
	}

	for _, index := range v.Indices {
		if index < 0 || index >= v.Size {
			panic(fmt.Sprintf("invalid %s index: index=%d size=%d", name, index, v.Size))
		}
	}
}
