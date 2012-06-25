/*
 Package goxmeans implements a library for the xmeans algorithm.

 See Dan Pelleg and Andrew Moore - X-means: Extending K-means with Efficient Estimation of the Number of Clusters. 
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
	"runtime"
	"log"
	"code.google.com/p/gomatrix/matrix"
	"goxmeans/matutil"
)

var workers = runtime.NumCPU()
//var workers = 1


// minimum returns the smallest int.
func minimum(x int, ys ...int) int {
    for _, y := range ys {
        if y < x {
            x = y
        }
    }
    return x
}

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
		var buf []byte
//		line, err := r.ReadString('\n')
		buf , _, err := r.ReadLine()
		line = string(buf)
	//	fmt.Printf("linenum=%d buf=%v line=%v\n",linenum,buf, line)

		if err == io.EOF {
			err = nil
			eof = true
			break
		} else if err != nil {
			return datamatrix, errors.New(fmt.Sprintf("means.Load: reading linenum %d: %v", linenum, err))
		}

		linenum++
		l1 := strings.TrimRight(line, "\n")
		l := strings.Split(l1, "\t")

		if len(l) < 2 {
			return datamatrix, errors.New(fmt.Sprintf("means.Load: linenum %d has only %d elements", linenum, len(line)))
		}

		// for now assume 2 dimensions only
		f0, err := Atof64(string(l[0]))
		if err != nil {
			return datamatrix, errors.New(fmt.Sprintf("means.Load: cannot convert f0 %s to float64.", l[0]))
		}
		f1, err := Atof64(string(l[1]))

		if err != nil {
			return datamatrix, errors.New(fmt.Sprintf("means.Load: cannot convert f1 %s to float64.",l[1]))
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
	datamatrix = matrix.MakeDenseMatrix(data, linenum - 1, numcols)
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
					rf = rf / 3.14
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

// Kmeansp returns means and distance squared of the coordinates for each 
// centroid using parallel computation.
//
// Input values
//
// datapoints - a kX2 matrix of R^2 coordinates 
//
// centroids - a kX2 matrix of R^2 coordinates for centroids.
//
// measurer - anythng that implements the matutil.VectorMeasurer interface to 
// calculate the distance between a centroid and datapoint. (e.g., Euclidian 
// distance)
//
// Return values
//
// centroidMeans - a kX2 matrix where the row number corresponds to the same 
// row in the centroid matrix and the two columns are the means of the 
// coordinates for that cluster.
//
//  ____      ______
//  | 12.29   32.94 | <-- The mean of coordinates for centroid 0
//  | 4.6     29.22 | <-- The mean of coordinates for centroid 1
//  |_____    ______|
// 
//
// centroidSE - a kX2 matrix where the first column contains a number
// indicating the centroid and the second column contains the sum of  squared 
// errors for all points in this cluster.
//
//  ____      _______
//  | 0        38.01 | <-- Centroid 0, squared error for the coordinates in row 0 of datapoints
//  | 1        23 .21| <-- Centroid 1, squared error for the coordinates in row 1 of datapoints
//  | 0        14.12 | <-- Centroid 0, squared error for the coordinates in row 2 of datapoints
//  _____     _______
func Kmeansp(datapoints, centroids *matrix.DenseMatrix, measurer matutil.VectorMeasurer) (centroidMeans *matrix.DenseMatrix,
    centroidSE *matrix.DenseMatrix,err error) {
	k, _ := centroids.GetSize()
	fp, _ := os.Create("/var/tmp/km.log")
	w := io.Writer(fp)
	log.SetOutput(w)

	numRows, numCols := datapoints.GetSize()
	centroidSE = matrix.Zeros(numRows, numCols)
	centroidMeans = matrix.Zeros(k, numCols)

	jobs := make(chan PairPointCentroidJob, workers)
	results := make(chan PairPointCentroidResult, minimum(1024, numRows))
	done := make(chan struct{}, workers)
	
	go addPairPointCentroidJobs(jobs, datapoints, centroidSE, centroids, measurer, results)
	for i := 0; i < workers; i++ {
		go doPairPointCentroidJobs(done, jobs)
	}
	go awaitPairPointCentroidCompletion(done, results)
	processPairPointToCentroidResults(centroidSE, results) // This blocks so that all the results can be processed

	// Now that you have each data point grouped with a centroid, iterate 
	// through the  centroidSE martix and for each centroid retrieve the 
	// original coordinates from datapoints and place the results in
	// pointsInCuster.
    for c := 0; c < k; c++ {
		// c is the index that identifies the current centroid.
		// d is the index that identifies a row in centroidSE and datapoints.
		// Select all the rows in centroidSE whose first col value == c.
		// Get the corresponding row vector from datapoints and place it in pointsInCluster.
		matches, err :=	matutil.FiltCol(float64(c), float64(c), 0, centroidSE)  //rows with c in column 0.
		if err != nil {
			return centroidMeans, centroidSE, nil
		}
		// It is possible that some centroids will not have any points, so there 
		//may not be any matches in the first column of centroidSE.
		if len(matches) == 0 {
			continue
		}

		pointsInCluster := matrix.Zeros(len(matches), 2) 
		for d, rownum := range matches {
			pointsInCluster.Set(d, 0, datapoints.Get(int(rownum), 0))
			pointsInCluster.Set(d, 1, datapoints.Get(int(rownum), 1))
		}

		// pointsInCluster now contains all the data points for the current 
		// centroid.  Take the mean of each of the 2 cols in pointsInCluster.
		means := matutil.MeanCols(pointsInCluster)
		centroidMeans.Set(c, 0, means.Get(0,0))
		centroidMeans.Set(c, 1, means.Get(0,1))
	}
	return 
}

// CentroidPoint stores the row number in the centroids matrix and
// the distance squared between the centroid as set of coordinates.
type CentroidPoint struct {
	centroidRunNum float64
	distPointToCentroidSq float64
}

// PairPointCentroidJobs stores the elements that defines the job that pairs a 
// set of coordinates (i.e., a data point) with a centroid.
type PairPointCentroidJob struct {
	point, centroids, centroidSE *matrix.DenseMatrix
	results chan<- PairPointCentroidResult
	rowNum int
	measurer matutil.VectorMeasurer
}

// PairPointCentroidResult stores the results of pairing a data point with a 
// centroids.
type PairPointCentroidResult struct {
	centroidRowNum float64
	distSquared float64
	rowNum int
	err error
}


// addPairPointCentroidJobs adds a job to the jobs channel.
func addPairPointCentroidJobs(jobs chan<- PairPointCentroidJob, datapoints, centroids,
	centroidSE *matrix.DenseMatrix, measurer matutil.VectorMeasurer, results chan<- PairPointCentroidResult) {
	numRows, _ := datapoints.GetSize()
    for i := 0; i < numRows; i++ { 
		point := datapoints.GetRowVector(i)
		jobs <- PairPointCentroidJob{point, centroids, centroidSE, results, i, measurer}
	}
	close(jobs)
}

// doPairPointCentroidJobs executes a job from the jobs channel.
func doPairPointCentroidJobs(done chan<- struct{}, jobs <-chan PairPointCentroidJob) {
	for job := range jobs {
		job.PairPointCentroid()
	}
	done <- struct{}{}
}

// awaitPairPointCentroidCompletion waits until all jobs are completed.
func awaitPairPointCentroidCompletion(done <-chan struct{}, results chan PairPointCentroidResult) {
	for i := 0; i < workers; i++ {
		<-done
	}
	close(results)
}

// processPairPointToCentroidResults assigns the results to the centroidSE matrix.
func processPairPointToCentroidResults(centroidSE *matrix.DenseMatrix, results <-chan PairPointCentroidResult)  {
	for result := range results {
	    centroidSE.Set(result.rowNum, 0, result.centroidRowNum)
	    centroidSE.Set(result.rowNum, 1, result.distSquared)  
	}
}
	
// AssignPointToCentroid checks a data point against all centroids and returns the best match.
// The centroid is identified by the row number in the centroid matrix.
func (job PairPointCentroidJob) PairPointCentroid() {
	var err error = nil
    distPointToCentroid := math.Inf(1)
    centroidRowNum := float64(-1)
	squaredErr := float64(0)
	k, _ := job.centroids.GetSize()

	// Find the centroid that is closest to this point.
    for j := 0; j < k; j++ { 
     	distJ, err := job.measurer.CalcDist(job.centroids.GetRowVector(j), job.point)
		if err != nil {
			continue
		}
        if distJ  < distPointToCentroid {
            distPointToCentroid = distJ
            centroidRowNum = float64(j)
		} 
 		squaredErr = math.Pow(distPointToCentroid, 2)
	}	
	job.results <- PairPointCentroidResult{centroidRowNum, squaredErr, job.rowNum, err}
}
