package main

import (
	"fmt"

	"github.com/NathanielChristian2077/gomlp/nn"
)

type sample struct {
	x []float64
	y float64
}

func main() {
	samples := []sample{
		{x: []float64{0, 0}, y: 0},
		{x: []float64{0, 1}, y: 1},
		{x: []float64{1, 0}, y: 1},
		{x: []float64{1, 1}, y: 1},
	}

	model := nn.NewMLP(2, 4, 42)

	epochs := 3000
	lr := 0.1

	for epoch := 1; epoch <= epochs; epoch++ {
		model.ZeroGrad()

		loss := 0.0

		for _, s := range samples {
			yHat := model.Forward(s.x)
			loss += nn.BinaryCrossEntropy(yHat, s.y)
			model.Backward(s.x, yHat, s.y)
		}

		model.ApplyGrad(lr, len(samples))

		if epoch == 1 || epoch%500 == 0 {
			fmt.Printf("epoch=%d loss=%.6f\n", epoch, loss/float64(len(samples)))
		}
	}

	fmt.Println("predictions:")

	for _, s := range samples {
		yHat := model.Forward(s.x)
		fmt.Printf("x=%v y=%.0f yHat=%.6f\n", s.x, s.y, yHat)
	}
}
