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

var numworkers = runtime.NumCPU()
//var numworkers = 1


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

// RandCentroids picks random centroids based on the min and max values in the matrix
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
// centroidMean - a kX2 matrix where the row number corresponds to the same 
// row in the centroid matrix and the two columns are the means of the 
// coordinates for that cluster.  i.e., the best centroids that could
// be determined.
//
//  ____      ______
//  | 12.29   32.94 | <-- The mean of coordinates for centroid 0
//  | 4.6     29.22 | <-- The mean of coordinates for centroid 1
//  |_____    ______|
// 
//
// centroidSqErr - a kX2 matrix where the first column contains a number
// indicating the centroid and the second column contains the minimum
// distance between centroid and point squared.  (i.e., the squared error)
//
//  ____      _______
//  | 0        38.01 | <-- Centroid 0, squared error for the coordinates in row 0 of datapoints
//  | 1        23 .21| <-- Centroid 1, squared error for the coordinates in row 1 of datapoints
//  | 0        14.12 | <-- Centroid 0, squared error for the coordinates in row 2 of datapoints
//  _____     _______
//func Kmeansp(datapoints, centroids *matrix.DenseMatrix, measurer matutil.VectorMeasurer) (centroidMean, 
func Kmeansp(datapoints *matrix.DenseMatrix, k int, measurer matutil.VectorMeasurer) (centroidMean, 
    centroidSqErr *matrix.DenseMatrix, err error) {
	//k, _ := centroids.GetSize()
	fp, _ := os.Create("/var/tmp/km.log")
	w := io.Writer(fp)
	log.SetOutput(w)

	centroids := RandCentroids(datapoints, k)
	numRows, numCols := datapoints.GetSize()
	centroidSqErr = matrix.Zeros(numRows, numCols)
	centroidMean = matrix.Zeros(k, numCols)

	jobs := make(chan PairPointCentroidJob, numworkers)
	results := make(chan PairPointCentroidResult, minimum(1024, numRows))
	done := make(chan int, numworkers)
	
	go addPairPointCentroidJobs(jobs, datapoints, centroidSqErr, centroids, measurer, results)
	for i := 0; i < numworkers; i++ {
		go doPairPointCentroidJobs(done, jobs)
	}
	go awaitPairPointCentroidCompletion(done, results)
	processPairPointToCentroidResults(centroidSqErr, results) // This blocks so that all the results can be processed

	// Now that you have each data point grouped with a centroid, iterate 
	// through the  centroidSqErr martix and for each centroid retrieve the 
	// original coordinates from datapoints and place the results in
	// pointsInCuster.
    for c := 0; c < k; c++ {
		// c is the index that identifies the current centroid.
		// d is the index that identifies a row in centroidSqErr and datapoints.
		// Select all the rows in centroidSqErr whose first col value == c.
		// Get the corresponding row vector from datapoints and place it in pointsInCluster.
		matches, err :=	matutil.FiltColMap(centroidSqErr, float64(c), float64(c), 0)  //rows with c in column 0.
		if err != nil {
			return centroidMean, centroidSqErr, nil
		}
		// It is possible that some centroids will not have any points, so there 
		//may not be any matches in the first column of centroidSqErr.
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
		centroidMean.Set(c, 0, means.Get(0,0))
		centroidMean.Set(c, 1, means.Get(0,1))
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
	point, centroids, centroidSqErr *matrix.DenseMatrix
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
	centroidSqErr *matrix.DenseMatrix, measurer matutil.VectorMeasurer, results chan<- PairPointCentroidResult) {
	numRows, _ := datapoints.GetSize()
    for i := 0; i < numRows; i++ { 
		point := datapoints.GetRowVector(i)
		jobs <- PairPointCentroidJob{point, centroids, centroidSqErr, results, i, measurer}
	}
	close(jobs)
}

// doPairPointCentroidJobs executes a job from the jobs channel.
func doPairPointCentroidJobs(done chan<- int, jobs <-chan PairPointCentroidJob) {
	for job := range jobs {
		job.PairPointCentroid()
	}
	done <- 1
}

// awaitPairPointCentroidCompletion waits until all jobs are completed.
func awaitPairPointCentroidCompletion(done <-chan int, results chan PairPointCentroidResult) {
	for i := 0; i < numworkers; i++ {
		<-done
	}
	close(results)
}

// processPairPointToCentroidResults assigns the results to the centroidSqErr matrix.
func processPairPointToCentroidResults(centroidSqErr *matrix.DenseMatrix, results <-chan PairPointCentroidResult)  {
	for result := range results {
	    centroidSqErr.Set(result.rowNum, 0, result.centroidRowNum)
	    centroidSqErr.Set(result.rowNum, 1, result.distSquared)  
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

// Kemansbi bisects a given cluster and determines which centroids give the lowest error.
// Take the points in a cluster
// While the number of cluster < k
//
func Kmeansbi(datapoints *matrix.DenseMatrix, k int, measurer matutil.VectorMeasurer) (matCentroidlist, clusterAssignment *matrix.DenseMatrix, err error) {
	numRows, numCols := datapoints.GetSize()
	clusterAssignment = matrix.Zeros(numRows, numCols)
	matCentroidlist = matrix.Zeros(k, numCols)
	centroid0 := matutil.MeanCols(datapoints)
	centroidlist := []*matrix.DenseMatrix{centroid0}

	// Initially create one cluster.
	for j := 0; j < numRows; j++ {
		point := datapoints.GetRowVector(j)
     	distJ, err := measurer.CalcDist(centroid0, point)
		if err != nil {
			return matCentroidlist, clusterAssignment, errors.New(fmt.Sprintf("Kmeansbi: CalcDist returned err=%v", err))
		}
		clusterAssignment.Set(j,1, math.Pow(distJ, 2))
	}

	var bestClusterAssignment, bestNewCentroids *matrix.DenseMatrix
	var bestCentroidToSplit int

	for ; len(centroidlist) < k ;  {
		lowestSSE := math.Inf(1)
		// Try splitting every cluster
		for i, _ := range centroidlist {
			// Get the points in this cluster
			pointsCurCluster, err := matutil.FiltCol(clusterAssignment, float64(i), float64(i), 0)  
			if err != nil {
				return matCentroidlist, clusterAssignment, nil
			}

			centroidMat, splitClusterAssignment, err := Kmeansp(pointsCurCluster, 2, measurer)
			if err != nil {
				return matCentroidlist, clusterAssignment, err
			}

            /* centroidMat is a 2X2 matrix of the best centroids found by kmeans
            
             splitClustAss is and mX2 matrix where col0 is either 0 or 1 and refers to the rows in centroidMat
             and col1 cotains the squared error between a centroid and a point.  The rows here correspond to 
             the rows in ptsInCurrCluster.  For example, if row 2 contains [1, 7.999] this means that centroid 1
             has been paired with the point in row 2 of splitClustAss and that the squared error (distance 
             between centroid and point) is 7.999.
            */

            // Calculate the sum of squared errorsfor each centroid. 
            // This is the sum for both centroids.  This give a statistcal measurement of how good
            // the clustering is for this cluster.
			sseSplit := matutil.SumCol(splitClusterAssignment, 1)
			matches, err := matutil.FiltCol(clusterAssignment, float64(i+1), math.Inf(1), 0)
			if err != nil {
				return matCentroidlist, clusterAssignment, err
			}
			sseNotSplit := matutil.SumCol(matches, 1)

			if sseSplit + sseNotSplit < lowestSSE {
				bestCentroidToSplit = 1
				bestNewCentroids = matrix.MakeDenseCopy(centroidMat)
				bestClusterAssignment =  matrix.MakeDenseCopy(splitClusterAssignment)
			}
		}
	}

	// Applying the split overwrites the existing cluster assginments for the 
	// cluster you have decided to split.  Kmeansp() returned two clusters
	// labeled 0 and 1. Change these cluster numbers to the cluster number
	// you are splitting and the next cluster to be added.
	m, err := matutil.FiltColMap(bestClusterAssignment, 1, 1, 0)
	if err != nil {
		return matCentroidlist, clusterAssignment, err
	}
	for i,_ := range m {
		bestClusterAssignment.Set(i, 0, float64(len(centroidlist)))
	}	

	n, err := matutil.FiltColMap(bestClusterAssignment, 0, 0, 0)
	if err != nil {
		return matCentroidlist, clusterAssignment, err
	}
	for i, _ := range n {
		bestClusterAssignment.Set(i , 1, float64(bestCentroidToSplit))
	}	

	fmt.Printf("Best centroid to split %f\n", bestCentroidToSplit)
	r,_ := bestClusterAssignment.GetSize()
	fmt.Printf("The length of best cluster assesment is %f\n", r)

	// Replace a centroid with the two best centroids from the split.
	centroidlist[bestCentroidToSplit] = bestNewCentroids.GetRowVector(0)
	centroidlist = append(centroidlist, bestNewCentroids.GetRowVector(1))

	// Reassign new clusters and SSE
	rows, _ := clusterAssignment.GetSize()
	for i, j := 0, 0 ; i < rows; i++ {
		if clusterAssignment.Get(i, 0) == float64(bestCentroidToSplit) {
			clusterAssignment.Set(i, 0, bestClusterAssignment.Get(j, 0))
			clusterAssignment.Set(i, 1, bestClusterAssignment.Get(j, 1))
			j++
		}
	}
	
	// make centroidlist into a matrix
	s := make([][]float64, len(centroidlist))
	for i, mat := range centroidlist {
		s[i][0] = mat.Get(0, 0)
		s[i][1] = mat.Get(0, 1)
	}
	matCentroidlist = matrix.MakeDenseMatrixStacked(s)

	return matCentroidlist, clusterAssignment, nil
}