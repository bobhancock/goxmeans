/* 
 Matrix operations and calculations.
*/
package matutil

import (
	"github.com/bobhancock/gomatrix/matrix"
	"errors"
	"fmt"
	"math"
)

// Measurer finds the distance between the points in the columns
type VectorMeasurer interface {
	CalcDist(a, b *matrix.DenseMatrix) (dist float64, err error)
}

type VectorDistance struct {}

type EuclidDist VectorDistance

// CalcDist finds the Euclidean distance between a centroid
// a point in the data set.  Arguments are 1x2 matrices.
// All intermediary l-values except s are matricies. The functions that
// operate on them can all take nXn matricies as arguments.
func (ed EuclidDist) CalcDist(centroid, point *matrix.DenseMatrix) (dist float64, err error) {
	err = nil
	diff := matrix.Difference(centroid, point)
	//square the resulting matrix
	sqr := diff.Pow(2)
	// sum of 1x2 matrix 
	sum := sqr.SumRows()
	// square root of sum
	s := sum.Get(0, 0)
	dist = math.Sqrt(s)
	return
}

type ManhattanDist struct {}

// CalcDist finds the ManhattanDistance which is the sum of the aboslute 
// difference of the coordinates.   Also known as rectilinear distance, 
// city block distance, or taxicab distance.
func (md ManhattanDist) CalcDist(a, b *matrix.DenseMatrix) (dist float64, err error) {
	dist = float64(0)
	err = nil
	arows, acols := a.GetSize()
	brows, bcols := b.GetSize()

	if arows != 1 || brows != 1 {
		return dist, errors.New(fmt.Sprintf("matutil: Matrices must contain only 1 row.  a has %d and b has %d.", arows, brows))
	} else if arows != brows {
		return dist, errors.New(fmt.Sprintf("matutil: Matrices must have the same dimensions.  a=%dX%d b=%dX%d", arows, acols, brows, bcols))
	}
	dist = math.Abs(a.Get(0,0) - b.Get(0,0)) + math.Abs(a.Get(0,1) - b.Get(0,1))
	return 
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

