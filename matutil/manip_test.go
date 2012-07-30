package matutil

import (

//	"fmt"
	"testing"
	"math"
	"github.com/bobhancock/gomatrix/matrix"
)


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

func TestGetBoundaries(t *testing.T) {
	// first test with single point
	rows := 1
	columns := 2

	a := matrix.MakeDenseMatrix([]float64{4.6, 9.5}, rows, columns)
	xmin, xmax, ymin, ymax := GetBoundaries(a)
	if xmin != 4.6 || xmax != 4.6 || ymin != 9.5 || ymax != 9.5 {
		t.Errorf("GetBoundaries failed on single item matrix")
	}

	b := matrix.MakeDenseMatrix([]float64{3.0, 4.1}, rows, columns)
	c, err := a.Stack(b)
	if err != nil {
		t.Errorf(err.Error())
	}
	xmin, xmax, ymin, ymax = GetBoundaries(c)
	if xmin != 3.0 || xmax != 4.6 || ymin != 4.1 || ymax != 9.5 {
		t.Errorf("GetBoundaries failed on two item matrix")
	}
}