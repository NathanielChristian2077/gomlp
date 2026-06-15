package experiment

import "testing"

func TestParseHiddenArchitectureAcceptsCommonSeparators(t *testing.T) {
	cases := map[string][]int{
		"128":        {128},
		"256x64":     {256, 64},
		"512-128":    {512, 128},
		"1024:256:64": {1024, 256, 64},
	}

	for raw, expected := range cases {
		actual, err := ParseHiddenArchitecture(raw)
		if err != nil {
			t.Fatalf("parse %q: %v", raw, err)
		}
		if len(actual) != len(expected) {
			t.Fatalf("parse %q: expected %d values, got %d", raw, len(expected), len(actual))
		}
		for i := range expected {
			if actual[i] != expected[i] {
				t.Fatalf("parse %q value %d: expected %d, got %d", raw, i, expected[i], actual[i])
			}
		}
	}
}

func TestParseHiddenArchitectureRejectsInvalidSize(t *testing.T) {
	if _, err := ParseHiddenArchitecture("128x0"); err == nil {
		t.Fatalf("expected invalid layer size error")
	}
}

func TestHiddenSizesLabel(t *testing.T) {
	label := HiddenSizesLabel([]int{256, 64, 16})
	if label != "256x64x16" {
		t.Fatalf("expected label 256x64x16, got %q", label)
	}
}
