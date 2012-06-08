package gomeans

// Matrix manipulation utilities

import (
	"fmt"
	"errors"
	"code.google.com/p/gomatrix/matrix"
)

// ColSlice puts the values in column i of a matrix as a slice
func ColSlice(mat matrix.Matrix, i int) []float64 {
	rows, _ := mat.GetSize()
	r := make([]float64, rows)
	for j := 0; j <  rows; j++ {
		r = append(r, mat.Get(j, i))
	}
	return r
}

// AppendCol appends column to and existing matrix.  If length of column
// is greater than the number of rows in the matrix, and error is returned.
// If the length of column is less than the number of rows, the column is padded
// with zeros.
func AppendCol(mat matrix.Matrix, column []float64) (matrix.Matrix, error) {
	rows, cols := mat.GetSize()
	err := errors.New("")
	if len(column) > rows {
		return matrix.Zeros(1, 1), errors.New(fmt.Sprintf("Cannot append a column with %d elements to an matrix with %d rows.",len(column),rows))
	}
	// Put the source array into a slice.
	// If there are R rows and C columns, the first C elements hold the data in
	// the first row, the 2nd C elements hold the data in the 2nd row, etc.
	source := make([]float64, rows * cols + len(column))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			source = append(source, mat.Get(i,j))
		}
		source = append(source, column[i])
	}
	return matrix.MakeDenseMatrix(source, rows, cols + 1), err
}