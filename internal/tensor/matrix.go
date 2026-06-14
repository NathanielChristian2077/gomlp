package tensor

import "fmt"

type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

func New(rows, cols int) Matrix {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("Invalid matrix shape: %d%d", rows, cols))
	}

	return Matrix{
		Rows: rows,
		Cols: cols,
		Data: make([]float64, rows*cols),
	}
}

func (m Matrix) At(r, c int) float64 {
	m.mustBeValidIndex(r, c)
	return m.Data[r*m.Cols+c]
}

func (m Matrix) Set(r, c int, value float64) {
	m.mustBeValidIndex(r, c)
	m.Data[r*m.Cols+c] = value
}

func (m Matrix) Clone() Matrix {
	copyMatrix := New(m.Rows, m.Cols)
	copy(copyMatrix.Data, m.Data)
	return copyMatrix
}

func (m Matrix) Shape() (rows, cols int) {
	return m.Rows, m.Cols
}

func (m Matrix) mustBeValidIndex(r, c int) {
	if r < 0 || r >= m.Rows || c < 0 || c >= m.Cols {
		panic(fmt.Sprintf(
			"Matrix index out of range: index=(%d,%d), shape=(%d,%d)",
			r, c, m.Rows, m.Cols,
		))
	}
}
