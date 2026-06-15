package metrics

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

type CSVLogger struct {
	file   *os.File
	writer *csv.Writer
}

func NewCSVLogger(path string, header []string) (*CSVLogger, error) {
	if path == "" {
		return nil, fmt.Errorf("empty csv logger path")
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	file, err := os.Create(path)
	if err != nil {
		return nil, err
	}

	logger := &CSVLogger{
		file:   file,
		writer: csv.NewWriter(file),
	}

	if len(header) > 0 {
		if err := logger.WriteRow(header...); err != nil {
			_ = logger.Close()
			return nil, err
		}
	}

	return logger, nil
}

func NewEpochCSVLogger(path string) (*CSVLogger, error) {
	return NewCSVLogger(path, []string{
		"epoch",
		"train_loss",
		"train_accuracy",
		"train_precision",
		"train_recall",
		"train_f1",
		"val_loss",
		"val_accuracy",
		"val_precision",
		"val_recall",
		"val_f1",
		"epoch_time_ms",
	})
}

func (l *CSVLogger) WriteRow(values ...string) error {
	if l == nil || l.writer == nil {
		return fmt.Errorf("nil csv logger")
	}

	if err := l.writer.Write(values); err != nil {
		return err
	}

	l.writer.Flush()
	return l.writer.Error()
}

func (l *CSVLogger) WriteEpoch(epoch int, trainLoss, trainAccuracy, valLoss, valAccuracy float64, epochTimeMS int64) error {
	return l.WriteEpochDetailed(
		epoch,
		trainLoss,
		trainAccuracy,
		0,
		0,
		0,
		valLoss,
		valAccuracy,
		0,
		0,
		0,
		epochTimeMS,
	)
}

func (l *CSVLogger) WriteEpochDetailed(
	epoch int,
	trainLoss float64,
	trainAccuracy float64,
	trainPrecision float64,
	trainRecall float64,
	trainF1 float64,
	valLoss float64,
	valAccuracy float64,
	valPrecision float64,
	valRecall float64,
	valF1 float64,
	epochTimeMS int64,
) error {
	return l.WriteRow(
		strconv.Itoa(epoch),
		formatFloat(trainLoss),
		formatFloat(trainAccuracy),
		formatFloat(trainPrecision),
		formatFloat(trainRecall),
		formatFloat(trainF1),
		formatFloat(valLoss),
		formatFloat(valAccuracy),
		formatFloat(valPrecision),
		formatFloat(valRecall),
		formatFloat(valF1),
		strconv.FormatInt(epochTimeMS, 10),
	)
}

func (l *CSVLogger) Close() error {
	if l == nil {
		return nil
	}

	if l.writer != nil {
		l.writer.Flush()
		if err := l.writer.Error(); err != nil {
			_ = l.file.Close()
			return err
		}
	}

	if l.file == nil {
		return nil
	}

	return l.file.Close()
}

func formatFloat(value float64) string {
	return strconv.FormatFloat(value, 'f', 8, 64)
}
