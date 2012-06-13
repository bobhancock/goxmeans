/*
 Package goxmeans implements a simple library for the xmeans algorithm.

 See Dan Pelleg and Andrew Moore: X-means: Extending K-means with Efficient Estimation of the Number of Clusters. 
*/
package goxmeans

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"code.google.com/p/gomatrix/matrix"
	"goxmeans/matutil"
)

// Atof64 is shorthand for ParseFloat(s, 64)
func Atof64(s string) (f float64, err error) {
	f64, err := strconv.ParseFloat(s, 64)
	return float64(f64), err
}

// Load loads a tab delimited text file of floats into a slice.
// Assume last column is the target.
// For now, we limit ourselves to two columns
func Load(fname string) (*matrix.DenseMatrix, error) {
	datamatrix := matrix.Zeros(1, 1)
	data := make([]float64, 2048)
	idx := 0

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
		} else if err != nil {
			return datamatrix, errors.New(fmt.Sprintf("means: reading linenum %d: %v", linenum, err))
		}

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

		if linenum >= len(data) {
			data = append(data, f0, f1)
		} else {
			data[idx] = f0
			idx++
			data[idx] = f1
			idx++
		}
	}
	numcols := 2
	datamatrix = matrix.MakeDenseMatrix(data, len(data)/numcols, numcols)
	return datamatrix, nil
}

// RandCentroids picks random centroids based on the  min and max values in the matrix
// and return a k by cols matrix of the centroids.
func RandCentroids(mat *matrix.DenseMatrix, k int) *matrix.DenseMatrix {
	_, cols := mat.GetSize()
	centroids := matrix.Zeros(k, cols)

	for colnum := 0; colnum < cols; colnum++ {
		r := matutil.ColSlice(mat, colnum)

		minj := float64(0)
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
		// TODO: Better randomization or choose centroids 
		// from datapoints.
		rands := make([]float64, k)
		for i := 0; i < k; i++ {
			randint := float64(rand.Int())
			rf := (maxj - minj) * randint
			for rf > maxj {
				if rf > maxj*3 {
					rf = rf / maxj
				} else {
					rf = rf / 2
				}
			}
			rands[i] = rf
		}
		for h := 0; h < k; h++ {
			centroids.Set(h, colnum, rands[h])
		}
	}
	return centroids
}

// ComputeCentroids Needs comments.
func ComputeCentroid(mat *matrix.DenseMatrix) (*matrix.DenseMatrix, error) {
	rows, _ := mat.GetSize()
	vectorSum := matutil.SumCols(mat)
	if rows == 0 {
		return vectorSum, errors.New("No points inputted")
	}
	vectorSum.Scale(1.0 / float64(rows))
	return vectorSum, nil
}

/* TODO: An interface for all distances 
   should be in a separate distance package
type Distance interface {
	Distance()
}

type CentroidMaker interface {
	MakeCentroids()
	k int // number of centroids
	dataSet *matrix.DenseMatrix  // set of data points
}*/


// TODO: Create Distance interface so that any distance metric, Euclidean, Jacard, etc. can be passed
// kmeans takes a matrix of data points as input dataPoints and attempts to find the best convergence on a set of k centroids.
// func kmeans(data *matrix.DenseMatrix, k int, dist Distance, maker CentroidMaker) (centroids  *matrix.DenseMatrix, clusterAssessment *matrix.DenseMatrix) {
/*func kmeans(dataPoints *matrix.DenseMatrix, k int) (centroids *matrix.DenseMatrix, clusterAssessment *matrix.DenseMatrix) {
	numRows, numCols := dataPoints.GetSize()

	clusterAssessment = matrix.Zeros(numRows, numCols)
	centroids = RandCentroids(dataPoints, k)
	clusterChanged := true

	for ; clusterChanged ; {
	    clusterChanged = false
        for i := 0; i < numRows; {  // assign each data point to a centroid
        	minDist := float64(0)
            minIndex := -1
            for j := 0; j < k; j++ {  // check distance btwn point and each centroid
     	        distJ := matutil.EuclidDist(centroids.GetRowVector(j), dataPoints.GetRowVector(i))
                if distJ < minDist {
                    minDist = distJ
                    minIndex = j
	            } 
            	if clusterAssessment.Get(i, 0) != float64(minIndex) {
	                clusterChanged = true
	            }
                clusterAssessment.Set(i,0, float64(minIndex)) 
	            clusterAssessment.Set(i,1, math.Pow(minDist, 2)) 
	        }
        }
		// Now that you have each data point grouped with a centroid, iterate through the cluster
		// assignment matix and for each centroid retrieve the original ordered pair from dataPoints.
        for c := 0; c < k; k++ {
			// c is the index that identifies the current centroid.
			// d is the index that identifies a row in clusterAssessment.
			// Select all the rows in clusterAssessment whose first col value == c, the second col is the SE distance
			// Get the corresponding data from dataPoints and place it in pointsInCluster.
			var pointsInCluster = matrix.Zeros(numRows,numCols)
			e := 0
			for d := 0; d < numRows; d++ {
				r := clusterAssessment.GetRowVector(d)
				if r.Get(0,0) == float64(c) && r.Get(0,1) != float64(0) {
					pointsInCluster.Set(e, 0, dataPoints.Get(d, 0))
					pointsInCluster.Set(e, 1, dataPoints.Get(d,1))
					e++
				}
				// pointsInClusteris a nX2 matix that contains the data points from dataPoints for the current centroid.
				// Take the mean of each of the 2 cols in pointsInCluster.
				means := matutil.MeanCols(pointsInCluster)  // 1x2 matrix with the mean coordinates of the current centroid
				// Centroids is a kX2 matrix whre the row number corresponds to the centroid and the two
				// columns are the means of the coordinates for that centroid cluster.
				centroids.Set(c, 0, means.Get(c,0))
				centroids.Set(c, 1, means.Get(c,1))
			}
		}
	}
	// N.B. The sum of squared errors in the sum of col 1 in clusterAssessment.
	return
}*/

// dataPoints = [x,y] coordinates of all data points.
// k = number of centroids (i.e., clusters)
//
// Given a matrix of dataPoints return two matricies
// centroids = [x, y] where x and y are the means of distances btwn centroid and points in that cluster
// centroidSqDist = [row number in centroids matrix, squared error btwn centroid and point]
//      This matrix has a 1:1 row correspondence with dataPoints.
func kmeans(dataPoints *matrix.DenseMatrix, k int) (centroids *matrix.DenseMatrix, centroidSqDist *matrix.DenseMatrix, err error) {
	numRows, numCols := dataPoints.GetSize()
	centroidSqDist = matrix.Zeros(numRows, numCols)
	// Intialize centroids with random values.  These will later be over written with
	// the means for each centroid.
	centroids = RandCentroids(dataPoints, k)
	centroidMeans := matrix.Zeros(k, numCols)
	clusterChanged := true

	for ; clusterChanged ; {
	    clusterChanged = false

        for i := 0; i < numRows; {  // assign each data point to a centroid
			//TODO: put each in a goroutine
			dataPoint := dataPoints.GetRowVector(i)
			centroidRowNum, distPointToCentroidSq,err := AssignPointToCentroid(dataPoints, centroids)
			if err != nil {
				return centroids, centroidSqDist, errors.New(fmt.Sprintf("kmeans: AssignPointToCentroid() error on row %d of dataPoint=%v  err=%v", i, dataPoint, err))
			}
			if centroidSqDist.Get(i, 0) != float64(centroidRowNum) {
				clusterChanged = true
			}
			// row num in centroidSqDist == row num in dataPoints
            centroidSqDist.Set(i,0, float64(centroidRowNum)) 
	        centroidSqDist.Set(i, 1, distPointToCentroidSq)  
        }

		// Now that you have each data point grouped with a centroid, iterate through the 
		// centroidSqDist martix and for each centroid retrieve the original ordered pair from dataPoints
		// and place the results in pointsInCuster.  
        for c := 0; c < k; k++ {
			// c is the index that identifies the current centroid.
			// d is the index that identifies a row in centroidSqDist.
			// Select all the rows in centroidSqDist whose first col value == c, the second col is the SE distance
			// Get the corresponding data from dataPoints and place it in pointsInCluster.
			var pointsInCluster = matrix.Zeros(numRows,numCols)
			e := 0
			for d := 0; d < numRows; d++ {
				r := centroidSqDist.GetRowVector(d)
				if r.Get(0,0) == float64(c) {
					pointsInCluster.Set(e, 0, dataPoints.Get(d, 0))
					pointsInCluster.Set(e, 1, dataPoints.Get(d,1))
					e++
				}
				// pointsInClusteris a nX2 matix that contains the data points from dataPoints for the current centroid.
				// Take the mean of each of the 2 cols in pointsInCluster.
				means := matutil.MeanCols(pointsInCluster)  // 1x2 matrix with the mean coordinates of the current centroid
				// Centroids is a kX2 matrix where the row number corresponds to the same row in the centroid matrix
				// and the two columns are the means of the coordinates for that centroid cluster.
				centroidMeans.Set(c, 0, means.Get(c,0))
				centroidMeans.Set(c, 1, means.Get(c,1))
			}
		}
	}
	// N.B. The sum of squared errors in the sum of col 1 in centroidSqDist.
	return
}

// AssignPointToCentroid checks a data point against all centroids and returns the best match.
// The centroid is identified by the row number in the centroid matrix.
// dataPoint is a 1X2 vector representing one data point.
// Return 
//    1. The row number in the centroid matrix.
//    2. (distance between centroid and point)^2
//    3. error
func AssignPointToCentroid(dataPoint, centroids *matrix.DenseMatrix) (float64, float64, error)  {
	r,c := dataPoint.GetSize()
	if r > 1 {
		return float64(-1), float64(-1), errors.New(fmt.Sprintf("dataPoint must be a 1X2 matrix.  It is %dX%d.", r, c))
	}
    distPointToCentroid := math.Inf(1)
    centroidRowNum := float64(-1)
	distSq := float64(0)
	k, _ := centroids.GetSize()

    for j := 0; j < k; j++ {  // check distance btwn point and each centroid
     	distJ := matutil.EuclidDist(centroids.GetRowVector(j), dataPoint)
        if distJ < distPointToCentroid {
            distPointToCentroid = distJ
            centroidRowNum = float64(j)
	    } 
 		distSq = math.Pow(distPointToCentroid, 2)
	  }
	return centroidRowNum, distSq, nil
}