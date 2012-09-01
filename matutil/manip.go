/* 
 Matrix operations and calculations.
*/
package matutil

import (
	"github.com/bobhancock/gomatrix/matrix"
//	"errors"
//	"fmt"
	"math"
)

// Measurer finds the distance between the points in the columns
type VectorMeasurer interface {
	CalcDist(a, b *matrix.DenseMatrix) (dist float64)
}

type vectorDistance struct {}

type EuclidDist vectorDistance

// CalcDist finds the Euclidean distance between points.
// sqrt( \sigma i = 1 to N (q_i - p_i)^2 )
func (ed EuclidDist) CalcDist(p, q *matrix.DenseMatrix) float64 {
	diff := matrix.Difference(q, p)
	sqrd := diff.Pow(2) // square each value in the matrix
	sum := sqrd.SumRows() 
	s := sum.Get(0, 0)
	return math.Sqrt(s)
}

type ManhattanDist struct {}

// CalcDist finds the ManhattanDistance which is the sum of the aboslute 
// difference of the coordinates.   Also known as rectilinear distance, 
// city block distance, or taxicab distance.

func (md ManhattanDist) CalcDist(a, b *matrix.DenseMatrix) float64 {
	return math.Abs(a.Get(0,0) - b.Get(0,0)) + math.Abs(a.Get(0,1) - b.Get(0,1))
}

// GetBoundaries returns the max and min x and y values for a dense matrix
// of shape m x 2.
func GetBoundaries(mat *matrix.DenseMatrix) (xmin, xmax, ymin, ymax float64) {
	rows, cols := mat.GetSize()
	if cols != 2 {
		// TODO - should there be an err return, or should we panic here?
	}
	xmin, ymin = mat.Get(0,0), mat.Get(0,1)
	xmax, ymax = mat.Get(0,0), mat.Get(0,1)
	for i := 1; i < rows; i++ {
		xi, yi := mat.Get(i, 0), mat.Get(i, 1)
		
		if xi > xmax{
			xmax = xi
		} else if xi < xmin {
			xmin = xi
		}

		if yi > ymax{
			ymax = yi
		} else if yi < ymin {
			ymin = yi
		}
	}
	return
}

