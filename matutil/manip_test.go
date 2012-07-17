package matutil

import (

//	"fmt"
	"testing"
	"math"
	"code.google.com/p/gomatrix/matrix"
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
		t.Errorf("SumRows returned a %dx%d matrix.  It should be 2x1.", numRows, numCols)
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
		t.Errorf("SumCols returned a %dx%d matrix.  It should be 1x2.", numRows, numCols)
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

func TestFiltCol(t *testing.T) {
	mat := matrix.MakeDenseMatrix([]float64{2, 1, 4, 2, 6, 3, 8, 4, 10, 5, 1, 1}, 5, 2)
	// mat, max, min, column
	matches, err := FiltCol(mat, 2.0, 4.0, 1)
	if err != nil {
		t.Errorf("FiltCol returned error: %v", err)
		return
	}
	
	r, _ := matches.GetSize()
	if r != 3 {
		t.Errorf("FiltCol: expected 3 rows and got %d", r)
	}

	m0 := matches.Get(1,1)
	if m0 != 2 {
		t.Errorf("FiltCol: expected row 0 col 1 to be 2, but got %f",m0)
	}

	m1 := matches.Get(2, 1)
	if m1 != 3 {
		t.Errorf("FiltCol: expected row 1 col 1 to be 3, but got %f",m1)
	}

	m2 := matches.Get(3, 1)
	if m2 != 4 {
		t.Errorf("FiltCol: expected row 1 col 1 to be 3, but got %f",m2)
	}
}

func TestFiltColMap(t *testing.T) {
	mat := matrix.MakeDenseMatrix([]float64{2, 1, 4, 2, 6, 3,8, 4, 10, 5, 1, 1}, 5, 2)
	matches, err := FiltColMap(mat, 2.0, 4.0, 1)
	if err != nil {
		t.Errorf("FiltColMap returned error: %v", err)
		return
	}

	if len(matches) != 3 {
		t.Errorf("FiltColMap expecte a map of len 3, but got len %d", len(matches))
	}

	if matches[1] != 2 || matches[2] != 3 || matches[3] != 4 {
		t.Errorf("FiltColMap expected a map with vals 2, 3, 4 but got %v", matches)
	}
}


func TestEuclidDist(t *testing.T) {
	var ed EuclidDist 
	rows := 1
	columns := 2

	centroid := matrix.MakeDenseMatrix([]float64{4.6, 9.5}, rows, columns)
	point := matrix.MakeDenseMatrix([]float64{3.0, 4.1}, rows, columns)
	calcEd, err := ed.CalcDist(centroid, point)
	if err != nil {
		t.Errorf("EuclidDist: returned an error.  err=%v", err)
	}

	expectedEd := 5.632051 //expected value
	epsilon := .000001

	na := math.Nextafter(expectedEd, expectedEd + 1) 
	diff := math.Abs(calcEd - na) 

	if diff > epsilon {
		t.Errorf("EuclidDist: excpected %f but received %f.  The difference %f exceeds epsilon %f", expectedEd, calcEd, diff, epsilon)
	}
}

func BenchmarkEuclidDist(b *testing.B) {
	var ed EuclidDist 
	rows := 1
	columns := 2

	centroid := matrix.MakeDenseMatrix([]float64{4.6, 9.5}, rows, columns)
	point := matrix.MakeDenseMatrix([]float64{3.0, 4.1}, rows, columns)
    for i := 0; i < b.N; i++ {
		_, _ = ed.CalcDist(centroid, point)	
    }
}

func TestManhattanDist(t *testing.T) {
	var md ManhattanDist
	rows := 1
	columns := 2

	a := matrix.MakeDenseMatrix([]float64{4.6, 9.5}, rows, columns)
	b := matrix.MakeDenseMatrix([]float64{3.0, 4.1}, rows, columns)
	
	calcMd, err := md.CalcDist(a, b)
	if err != nil {
		t.Errorf("ManhattandDist: returned an error.  err=%v", err)
	}
	
	// 1.6 + 5.4 = 7.0
	if calcMd != float64(7.0) {
		t.Errorf("ManhattanDist: should be 7.0, but returned %f", calcMd)
	}
}
//TODO: test for MeanCols