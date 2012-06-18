/* 
 Package matrixutils implements matrix manipulation utilities to augment
 code.google.com/p/gomatrix/matrix.
*/

package matutil

import (
	"code.google.com/p/gomatrix/matrix"
	"errors"
	"fmt"
	"math"
)

// ColSlice gets the values in column i of a matrix as a slice
func ColSlice(mat *matrix.DenseMatrix, col int) []float64 {
	rows, _ := mat.GetSize()
	r := make([]float64, rows)
	for j := 0; j < rows; j++ {
		r[j] = mat.Get(j, col)
	}
	return r
}

// AppendCol appends column to an existing matrix.  If length of column
// is greater than the number of rows in the matrix, and error is returned.
// If the length of column is less than the number of rows, the column is padded
// with zeros.
//
// Returns a new matrix with the column append and leaves the source untouched.
func AppendCol(mat *matrix.DenseMatrix, column []float64) (*matrix.DenseMatrix, error) {
	rows, cols := mat.GetSize()
	var err error = nil
	if len(column) > rows {
		return matrix.Zeros(1, 1), errors.New(fmt.Sprintf("Cannot append a column with %d elements to an matrix with %d rows.", len(column), rows))
	}
	// Put the source array into a slice.
	// If there are R rows and C columns, the first C elements hold the data in
	// the first row, the 2nd C elements hold the data in the 2nd row, etc.
	source := make([]float64, rows*cols+len(column))
	for i := 0; i < rows; i++ {
		j := 0
		for ; j < cols; j++ {
			source[j] = mat.Get(i, j)
		}
		source[j] = column[i]
	}
	return matrix.MakeDenseMatrix(source, rows, cols+1), err
}

// Pow raises every element of the matrix to power.  Returns a new
// matrix
func Pow(mat *matrix.DenseMatrix, power float64) *matrix.DenseMatrix {
	numRows, numCols := mat.GetSize()
	raised := matrix.Zeros(numRows, numCols)

	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			raised.Set(i, j, math.Pow(mat.Get(i, j), power))
		}
	}
	return raised
}

// SumRows takes the sum of each row in a matrix and returns a 1Xn matrix of
// the sums.
func SumRows(mat *matrix.DenseMatrix) *matrix.DenseMatrix {
	numRows, numCols := mat.GetSize()
	sums := matrix.Zeros(numRows, 1)

	for i := 0; i < numRows; i++ {
		j := 0
		s := 0.0
		for ; j < numCols; j++ {
			s += mat.Get(i, j)
		}
		sums.Set(i, 0, s)
	}
	return sums
}

// SumCol calculates the sum of the indicated column and returns a float64
func SumCol(mat *matrix.DenseMatrix, col int) float64 {
	numRows, _ := mat.GetSize()
	sum := float64(0)

	for i := 0; i < numRows; i++ {
		sum += mat.Get(i,col)
	}
	return sum
}

// MeanCols calculates the means of the columns and returns a 1Xn matrix
func MeanCols(mat *matrix.DenseMatrix) *matrix.DenseMatrix {
	numRows, numCols := mat.GetSize()
	sums := SumCols(mat)
	means := matrix.Zeros(1, numCols)
	m := float64(0)

	for j := 0; j < numCols; j++ {
		m = sums.Get(0, j) / float64(numRows)
		means.Set(0, j, m)
	}
	return means
}

// SumCols takes the sum of each column in the matrix and returns a mx1 matrix of
// the sums.
func SumCols(mat *matrix.DenseMatrix) *matrix.DenseMatrix {
	numRows, numCols := mat.GetSize()
	sums := matrix.Zeros(1, numCols)

	for j := 0; j < numCols; j++ {
		i := 0
		s := 0.0
		for ; i < numRows; i++ {
			s += mat.Get(i, j)
		}
		sums.Set(0, j, s)
	}
	return sums
}

// FiltCol find values that match min <= A <= max for specified column.
// Return
//   map[int]float64 - key is the row number in mat, and the value is the value in the column specified by col.
func FiltCol(min, max float64, col int, mat *matrix.DenseMatrix) (map[int]float64, error) {
	r,c := mat.GetSize()
	matches := make(map[int]float64)
	
	if col < 0 || col > c - 1 {
		return matches, errors.New(fmt.Sprintf("matutil: Expected col vaule in range 0 to %d.  Received %d\n", c -1, col))
	}

	for i := 0; i < r; i++ {
		v := mat.Get(i, col)
//		fmt.Printf("i=%d v=%v\n",i, v)
		if v >= min &&  v <= max {
			matches[i] = v
		}
	}
	return matches, nil
 }

// Measurer finds the distance b - a.
type Measurer interface {
	Measure(a, b *matrix.DenseMatrix) (dist float64)
}

func NewVectorMeasurer(m Measurer) Measurer {
	return &VectorMeasurer(m, a, b, 0)
}
	
type VectorMeasurer struct {
	M Measurer // underlying measurer
}

// DistEclidean finds the Euclidean distance between a centroid
// a point in the data set.  Arguments are 1x2 matrices.
// All intermediary l-values except s are matricies. The functions that
// operate on them can all take nXn matricies as arguments.
func (ed *VectorMeasurer) Measure(centroid, point *matrix.DenseMatrix) (dist float64) {
	diff := matrix.Difference(centroid, point)
	//	fmt.Printf("diff=%v\n", diff)
	//square the resulting matrix
	sqr := Pow(diff, 2)
	//	fmt.Printf("sqr=%v\n", sqr)
	// sum of 1x2 matrix 
	sum := SumRows(sqr)
	//	fmt.Printf("sum=%v\n", sum)
	// square root of sum
	s := sum.Get(0, 0)
	//	fmt.Printf("s=%f\n", s)
	dist =  math.Sqrt(s)
	return
}
