package data

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
)

const (
	ImageWidth  = 64
	ImageHeight = 64
	ImageSize   = ImageWidth * ImageHeight
)

// LoadImageVector carrega uma imagem usando o tamanho padrão da MLP: 64x64.
func LoadImageVector(path string) ([]float64, error) {
	return LoadImageVectorWithSize(path, ImageWidth, ImageHeight)
}

// LoadImageVectorWithSize abre a imagem, decodifica e transforma em vetor normalizado.
// O vetor resultante é a entrada da MLP.
func LoadImageVectorWithSize(path string, width, height int) ([]float64, error) {
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid target image size: %dx%d", width, height)
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	return ResizeToGrayVector(img, width, height), nil
}

// ResizeToGrayVector redimensiona por vizinho mais próximo, converte para grayscale
// e normaliza cada pixel para o intervalo [0, 1].
func ResizeToGrayVector(img image.Image, width, height int) []float64 {
	bounds := img.Bounds()
	sourceWidth := bounds.Dx()
	sourceHeight := bounds.Dy()
	out := make([]float64, width*height)

	for y := 0; y < height; y++ {
		sourceY := bounds.Min.Y + y*sourceHeight/height

		for x := 0; x < width; x++ {
			sourceX := bounds.Min.X + x*sourceWidth/width
			r, g, b, _ := img.At(sourceX, sourceY).RGBA()

			gray := 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
			out[y*width+x] = gray / 65535.0
		}
	}

	return out
}
