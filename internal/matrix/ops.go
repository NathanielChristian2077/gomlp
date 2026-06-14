package matrix

import "fmt"

func Dot(a, b Matrix) Matrix {
	if a.Cols != b.Rows {
		panic(fmt.Sprintf(
			"Invalid shapes for dot product: A=(%dx%d), B=(%dx%d)",
			a.Rows, a.Cols, b.Rows, b.Cols,
		))
	}

	out := New(a.Rows, b.Cols)

	for r := 0; r < a.Rows; r++ {
		for c := 0; c < b.Cols; c++ {
			sum := 0.0

			for k := 0; k < a.Cols; k++ {
				sum += a.At(r, k) * b.At(k, c)
			}

			out.Set(r, c, sum)
		}
	}

	return out
}

func Add(a, b Matrix) Matrix {
	mustHaveSameShape(a, b, "add")

	out := New(a.Rows, a.Cols)

	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}

	return out
}

func Sub(a, b Matrix) Matrix {
	mustHaveSameShape(a, b, "sub")

	out := New(a.Rows, a.Cols)

	for i := range a.Data {
		out.Data[i] = a.Data[i] - b.Data[i]
	}

	return out
}

func MulScalar(a Matrix, scalar float64) Matrix {
	out := New(a.Rows, a.Cols)

	for i := range a.Data {
		out.Data[i] = a.Data[i] * scalar
	}

	return out
}

func Apply(a Matrix, fn func(float64) float64) Matrix {
	out := New(a.Rows, a.Cols)

	for i, value := range a.Data {
		out.Data[i] = fn(value)
	}

	return out
}

func Transpose(a Matrix) Matrix {
	out := New(a.Cols, a.Rows)

	for r := 0; r < a.Rows; r++ {
		for c := 0; c < a.Cols; c++ {
			out.Set(c, r, a.At(r, c))
		}
	}

	return out
}

func mustHaveSameShape(a, b Matrix, op string) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(fmt.Sprintf(
			"Invalid shapes for %s: A=(%dx%d), B=(%dx%d)",
			op, a.Rows, a.Cols, b.Rows, b.Cols,
		))
	}
}
