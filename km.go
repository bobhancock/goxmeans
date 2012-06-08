/*
 Package gomeans implements a simple library for the xmeans algorithm.

 See Dan Pelleg and Andrew Moore: X-means: Extending K-means with Efficient Estimation of the Number of Clusters. 
*/
package goxmeans

import (
	"fmt"
	"os"
	"bufio"
	"errors"
	"strconv"
	"strings"
	"io"
	"math"
	"math/rand"
	"code.google.com/p/gomatrix/matrix"
)

// Atof64 is shorthand for ParseFloat(s, 64)
func Atof64(s string) (f float64, err error) {
	f64, err := strconv.ParseFloat(s, 64)
	return float64(f64), err
}

// ColSlice puts the values in column i of a matrix as a slice
func ColSlice(mat *matrix.DenseMatrix, col int) []float64 {
	rows, _ := mat.GetSize()
	r := make([]float64, rows)
	for j := 0; j <  rows; j++ {
		r[j] =  mat.Get(j, col)
	}
	return r
}

//TODO func DenseMatrixToSlice(mat *DenseMatrix)

// AppendCol appends column to and existing matrix.  If length of column
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

	
// Load loads a tab delimited text file of floats into a slice.
// Assume last column is the target.
// For now, we limit ourselves to two columns
func Load(fname string) (*matrix.DenseMatrix, error)  {
	datamatrix := matrix.Zeros(1, 1);
	data := make([]float64, 2048) 

	fp, err := os.Open(fname)
	if err != nil {
		return datamatrix, err
	}
	defer fp.Close()

	r := bufio.NewReader(fp)
	linenum := 1
	eof := false
	for !eof {
		var line string
		line, err := r.ReadString('\n')
		if err == io.EOF {
			err = nil
			eof = true
			break
		} else 	if err != nil {
			return datamatrix, errors.New(fmt.Sprintf("means: reading linenum %d: %v", linenum, err))
		} 
//		fmt.Printf("debug: linenum=%d line=%s\n", linenum, line)

		linenum++
		l1 := strings.TrimRight(line, "\n")
		l := strings.Split(l1, "\t")
		if len(l) < 2 {
			return datamatrix, errors.New(fmt.Sprintf("means: linenum %d has only %d elements", linenum, len(line)))
		}

		// for now assume 2 dimensions only
		f0, err := Atof64(string(l[0]))
		if err != nil {
			return datamatrix, errors.New(fmt.Sprintf("means: cannot convert %s to float64.", l[0]))
		}
		f1, err := Atof64(string(l[1]))
		if err != nil {
			return datamatrix, errors.New(fmt.Sprintf("means: cannot convert %s to float64.", l[linenum][1]))
		}
		data = append(data, f0, f1)
	}
	numcols := 2
	datamatrix = matrix.MakeDenseMatrix(data, len(data)/numcols, numcols)
	return datamatrix, nil
}

// RandCentroids picks random centroids based on the min and max values in the matrix.
func RandCentroids(mat *matrix.DenseMatrix, k int) (*matrix.DenseMatrix, error) {
	rows,cols := mat.GetSize()
	centroids := matrix.Zeros(1,1)

	minj := float64(0)
	for j := 0; j <  cols; j++ {
		r := ColSlice(mat, j)
		// min value from column
		for _, val := range r {
			minj = math.Min(minj, val)
		}

		// max value from column
		maxj := float64(0)
		for _, val := range r {
			maxj = math.Max(maxj, val)
		}

		// create a slice of random centroids 
		// based on maxj + minJ * random num to stay in range
		rands := make([]float64, rows)
		for i := 0; i < rows; i++ {
			rands = append(rands,  maxj - minj * rand.Float64())
		}

		centroids, err := AppendCol(mat, rands)
		if err != nil {
			return centroids, errors.New(fmt.Sprintf("means: RandCentroids could not append column.  err=%v", err))
		}
	}
	return centroids, nil
}

