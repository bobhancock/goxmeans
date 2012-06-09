package matutil

import (
	"code.google.com/p/gomatrix/matrix"
	"fmt"
	"math"
	"testing"
)

func TestColSliceValid(t *testing.T) {
	rows := 3
	columns := 2
	mat := matrix.MakeDenseMatrix([]float64{1, 2, 3, 4, 5, 6}, rows, columns)
	c := ColSlice(mat, 1)
	if len(c) != rows {
		t.Errorf("Returned slice has len=%d instead of %d.", len(c), rows)
	}
}

func TestAppendColInvalid(t *testing.T) {
	rows := 3
	columns := 2
	mat := matrix.MakeDenseMatrix([]float64{1, 2, 3, 4, 5, 6}, rows, columns)
	col := []float64{1.1, 2.2, 3.3, 4.4}
	mat, err := AppendCol(mat, col)
	if err == nil {
		t.Errorf("AppendCol err=%v", err)
	}
}

func TestAppendColValid(t *testing.T) {
	rows := 3
	columns := 2
	mat := matrix.MakeDenseMatrix([]float64{1, 2, 3, 4, 5, 6}, rows, columns)
	col := []float64{1.1, 2.2, 3.3}
	mat, err := AppendCol(mat, col)
	if err != nil {
		t.Errorf("AppendCol err=%v", err)
	}
}

func TestPow(t *testing.T) {
	p00 := float64(3)
	p01 := float64(4)
	mat := matrix.MakeDenseMatrix([]float64{p00, p01}, 1, 2)
	raised := Pow(mat, 2)

	r00 := raised.Get(0, 0)
	if r00 != 9 {
		t.Errorf("TestPow r00 should be 9, but is %f", r00)
	}
	r01 := raised.Get(0, 1)
	if r01 != 16 {
		t.Errorf("TestPow r01 should be 16, but is %f", r01)
	}
}

func TestSumRows(t *testing.T) {
	p00 := 3.0
	p01 := 4.0
	p10 := 3.5
	p11 := 4.6
	mat := matrix.MakeDenseMatrix([]float64{p00, p01, p10, p11}, 2, 2)
	sums := SumRows(mat)

	numRows, numCols := sums.GetSize()
	if numRows != 2 || numCols != 1 {
		t.Errorf("SumRows returned a %dx%d matrix.  It should be 2x1.")
	}
	s00 := sums.Get(0, 0)
	if s00 != (p00 + p01) {
		t.Errorf("SumRows row 0 col 0 is %d.  It should be %d.", s00, p00+p01)
	}
	s10 := sums.Get(1, 0)
	if s10 != (p10 + p11) {
		t.Errorf("SumRows row 1 col 2 is %d.  It should be %d.", s10, p10+p11)
	}
}

func TestSumCols(t *testing.T) {
	p00 := 3.0
	p01 := 4.0
	p10 := 3.5
	p11 := 4.6
	mat := matrix.MakeDenseMatrix([]float64{p00, p01, p10, p11}, 2, 2)
	sums := SumCols(mat)

	numRows, numCols := sums.GetSize()
	if numRows != 1 || numCols != 2 {
		t.Errorf("SumCols returned a %dx%d matrix.  It should be 1x2.")
	}
	s00 := sums.Get(0, 0)
	if s00 != (p00 + p10) {
		t.Errorf("SumCols row 0 col 0 is %d.  It should be %d.", s00, p00+p10)
	}
	s10 := sums.Get(0, 1)
	if s10 != (p01 + p11) {
		t.Errorf("SumCols row 0 col 1 is %d.  It should be %d.", s10, p01+p11)
	}
}

func TestEuclidDist(t *testing.T) {
	rows := 1
	columns := 2
	centroid := matrix.MakeDenseMatrix([]float64{4.6, 9.5}, rows, columns)
	point := matrix.MakeDenseMatrix([]float64{3.0, 4.1}, rows, columns)
	calcEd := EuclidDist(centroid, point)

	expectedEd := 5.632051 //expected value
	epsilon := .000001
	na := math.Nextafter(expectedEd, expectedEd+1)
	diff := math.Abs(calcEd - na)
	fmt.Printf("diff=%f\n", diff)
	if diff > epsilon {
		t.Errorf("diff=%f", diff)
	}

}
