/* 
 Package matrixutils implements matrix manipulation utilities to augment
 code.google.com/p/gomatrix/matrix.
*/

package matutil

import (
	"fmt"
	"errors"
	"code.google.com/p/gomatrix/matrix"
)

// ColSlice gets the values in column i of a matrix as a slice
func ColSlice(mat *matrix.DenseMatrix, col int) []float64 {
	rows, _ := mat.GetSize()
	r := make([]float64, rows)
	for j := 0; j <  rows; j++ {
		r[j] =  mat.Get(j, col)
	}
	return r
}

//TODO func DenseMatrixToSlice(mat *DenseMatrix)

// AppendCol appends column to an existing matrix.  If length of column
// is greater than the number of rows in the matrix, and error is returned.
// If the length of column is less than the number of rows, the column is padded
// with zeros.
func AppendCol(mat *matrix.DenseMatrix, column []float64) (*matrix.DenseMatrix, error) {
	rows, cols := mat.GetSize()
	var err error = nil
	if len(column) > rows {
		return matrix.Zeros(1, 1), errors.New(fmt.Sprintf("Cannot append a column with %d elements to an matrix with %d rows.",len(column),rows))
	}
	// Put the source array into a slice.
	// If there are R rows and C columns, the first C elements hold the data in
	// the first row, the 2nd C elements hold the data in the 2nd row, etc.
	source := make([]float64, rows * cols + len(column))
	for i := 0; i < rows; i++ {
		j := 0
		for ; j < cols; j++ {
			source[j] = mat.Get(i, j)
		}
		source[j] = column[i]
	}
	return matrix.MakeDenseMatrix(source, rows, cols + 1), err
}
