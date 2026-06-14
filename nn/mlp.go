package nn

import "github.com/NathanielChristian2077/gomlp/internal/matrix"

type MLP struct {
	W1 matrix.Matrix
	B1 matrix.Matrix

	W2 matrix.Matrix
	B2 matrix.Matrix

	Cache ForwardCache
}

type ForwardCache struct {
	X  matrix.Matrix
	Z1 matrix.Matrix
	A1 matrix.Matrix
	Z2 matrix.Matrix
	YH matrix.Matrix
}
