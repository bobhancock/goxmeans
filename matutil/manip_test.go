package matutil

import (
	"testing"
	"code.google.com/p/gomatrix/matrix"
)

func TestColSliceValid(t *testing.T) {
	rows := 3
	columns := 2
	mat := matrix.MakeDenseMatrix([]float64{1,2,3,4,5,6}, rows, columns)
	c := ColSlice(mat, 1)
	if len(c) != rows {
		t.Errorf("Returned slice has len=%d instead of %d.", len(c), rows)
	}
}

func TestAppendColInvalid(t *testing.T) {
	rows := 3
	columns := 2
	mat := matrix.MakeDenseMatrix([]float64{1,2,3,4,5,6}, rows, columns)
	col := []float64{1.1,2.2,3.3,4.4}
	mat, err := AppendCol(mat, col)
	if err == nil {
		t.Errorf("AppendCol err=%v", err)
	}
}

func TestAppendColValid(t *testing.T) {
	rows := 3
	columns := 2
	mat := matrix.MakeDenseMatrix([]float64{1,2,3,4,5,6}, rows, columns)
	col := []float64{1.1,2.2,3.3}
	mat, err := AppendCol(mat, col)
	if err != nil {
		t.Errorf("AppendCol err=%v", err)
	}
}
