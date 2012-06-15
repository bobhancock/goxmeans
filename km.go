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
	"runtime"
	"code.google.com/p/gomatrix/matrix"
	"goxmeans/matutil"
)

var workers = runtime.NumCPU()

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
//		fmt.Printf("linenum=%d l=%v\n",linenum,l)
		if len(l) < 2 {
			return datamatrix, errors.New(fmt.Sprintf("means.Load: linenum %d has only %d elements", linenum, len(line)))
		}

		// for now assume 2 dimensions only
		f0, err := Atof64(string(l[0]))
		if err != nil {
			return datamatrix, errors.New(fmt.Sprintf("means.Load: cannot convert f0 %s to float64.", l[0]))
		}
		f1, err := Atof64(string(l[1]))
	//	fmt.Printf("f1=%s\n", l[1])
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

// kmeans takes a vector of data points and a number specifying the number of clusters.
// dataPoints = matrix[x,y] coordinates of all data points.
// k = number of centroids (i.e., clusters)
// N.B We are using only R^2 vectors to get something working.
//
// Returns two matricies:
// centroidMeans = [x, y] where each row represents a centroid and x and y are the means of distances btwn centroid and points in that cluster
// centroidSqDist = [row number in centroids matrix, squared error btwn centroid and point]
//      This matrix has a 1:1 row correspondence with dataPoints.
func Kmeans(dataPoints *matrix.DenseMatrix, k int) ( *matrix.DenseMatrix,  *matrix.DenseMatrix,  error) {
	numRows, numCols := dataPoints.GetSize()
	centroidSqDist := matrix.Zeros(numRows, numCols)
	// Intialize centroids with random values.  These will later be over written with
	// the means for each centroid.
	centroids := RandCentroids(dataPoints, k)  //commented out for deterministic testing
	// Testing START
	//centroidsdata := []float64{1.5,1.5,2,2,3,3,0.9,0,9}
	//centroids := matrix.MakeDenseMatrix(centroidsdata, 4,2)
	// Testing END
	centroidMeans := matrix.Zeros(k, numCols)

	clusterChanged := true
	for ; clusterChanged ; {
	    clusterChanged = false

        for i := 0; i < numRows; i++ {  // assign each data point to a centroid
			point := dataPoints.GetRowVector(i)
			centroidRowNum, distPointToCentroidSq ,err := AssignPointToCentroid(point, centroids)
			if err != nil {
				return centroids, centroidSqDist, 
				errors.New(fmt.Sprintf("kmeans: AssignPointToCentroid() error on row %d of dataPoint=%v  err=%v", i, point, err))
			}
			if centroidSqDist.Get(i, 0) != float64(centroidRowNum) {
				clusterChanged = true
			}
			// row num in centroidSqDist == row num in dataPoints
            centroidSqDist.Set(i, 0, float64(centroidRowNum)) 
	        centroidSqDist.Set(i, 1, distPointToCentroidSq)  
        }

		// Now that you have each data point grouped with a centroid, iterate through the 
		// centroidSqDist martix and for each centroid retrieve the original ordered pair from dataPoints
		// and place the results in pointsInCuster.  
        for c := 0; c < k; c++ {
			// c is the index that identifies the current centroid.
			// d is the index that identifies a row in centroidSqDist.
			// Select all the rows in centroidSqDist whose first col value == c.
			// Get the corresponding row vector from dataPoints and place it in pointsInCluster.
			matches, err :=	matutil.FiltCol(float64(c), float64(c), 0, centroidSqDist)  //rows with c in column 0.
			// It is possible that some centroids will not have any points, so there may not be any matches in 
			// the first column of centroidSqDist.
			if err != nil {
					return centroidMeans, centroidSqDist, nil
			}
			if len(matches) == 0 {
				continue
			}
			pointsInCluster := matrix.Zeros(len(matches), 2) //TODO Adapt for more than two columns
			d := 0
			for rownum, _ := range matches {
				pointsInCluster.Set(d, 0, dataPoints.Get(rownum, 0))
				pointsInCluster.Set(d, 1, dataPoints.Get(rownum, 1))
				d++
			}
			// pointsInCluster now contains all the data points for the current centroid.
			// Take the mean of each of the 2 cols in pointsInCluster.
			means := matutil.MeanCols(pointsInCluster)  // 1x2 matrix with the mean coordinates of the current centroid
			// Centroids is a kX2 matrix where the row number corresponds to the same row in the centroid matrix
			// and the two columns are the means of the coordinates for that centroid cluster.
			centroidMeans.Set(c, 0, means.Get(0,0))
			centroidMeans.Set(c, 1, means.Get(0,1))
		}
	}
	return centroidMeans, centroidSqDist, nil
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

//============== Parallel Version ========================================================================
// Parallel version
type CentroidPoint struct {
	centroidRunNum float64
	distPointToCentroidSq float64
}


func Kmeansp(dataPoints *matrix.DenseMatrix, k int)  {
	numRows, numCols := dataPoints.GetSize()
	centroidSqDist := matrix.Zeros(numRows, numCols)
	// Intialize centroids with random values.  These will later be over written with
	// the means for each centroid.
	centroids := RandCentroids(dataPoints, k)  //commented out for deterministic testing
	// Testing START
	//centroidsdata := []float64{1.5,1.5,2,2,3,3,0.9,0,9}
	//centroids := matrix.MakeDenseMatrix(centroidsdata, 4,2)
	// Testing END
	centroidMeans := matrix.Zeros(k, numCols)

	clusterChanged := true
	for ; clusterChanged ; {
	    clusterChanged = false
		jobs := make(chan PairPointToCentroidJob, workers)
		results := make(chan PairPointToCentroidResult, minimum(1024, numRows))
		done := make(chan struct{}, workers)
		
		go addPairPointToCentroidJobs(jobs, dataPoints, centroids, results)
		for i := 0; i < workers; i++ {
			go doPairPointToCentroidJobs(done, jobs)
		}
		go awaitPairPointToCentroidCompletion(done, results)
		processPairPointToCentroiResults(centroidSqDist, results)
	}
	fmt.Printf("centroidMean=%v\n", centroidMeans)
	return

/*        for i := 0; i < numRows; i++ {  // assign each data point to a centroid
			point := dataPoints.GetRowVector(i)
			centroidRowNum, distPointToCentroidSq := AssignPointToCentroid(point, centroids)
			if centroidSqDist.Get(i, 0) != float64(centroidRowNum) {
				clusterChanged = true
			}
			// row num in centroidSqDist == row num in dataPoints
            centroidSqDist.Set(i, 0, float64(centroidRowNum)) 
	        centroidSqDist.Set(i, 1, distPointToCentroidSq)  
        }
*/
		// Now that you have each data point grouped with a centroid, iterate through the 
		// centroidSqDist martix and for each centroid retrieve the original ordered pair from dataPoints
		// and place the results in pointsInCuster.  
        /*for c := 0; c < k; c++ {
			// c is the index that identifies the current centroid.
			// d is the index that identifies a row in centroidSqDist.
			// Select all the rows in centroidSqDist whose first col value == c.
			// Get the corresponding row vector from dataPoints and place it in pointsInCluster.
			matches, err :=	matutil.FiltCol(float64(c), float64(c), 0, centroidSqDist)  //rows with c in column 0.
			// It is possible that some centroids will not have any points, so there may not be any matches in 
			// the first column of centroidSqDist.
			if err != nil {
				return centroidMeans, centroidSqDist, nil
			}
			if len(matches) == 0 {
				continue
			}
			pointsInCluster := matrix.Zeros(len(matches), 2) //TODO Adapt for more than two columns
			d := 0
			for rownum, _ := range matches {
				pointsInCluster.Set(d, 0, dataPoints.Get(rownum, 0))
				pointsInCluster.Set(d, 1, dataPoints.Get(rownum, 1))
				d++
			}
			// pointsInCluster now contains all the data points for the current centroid.
			// Take the mean of each of the 2 cols in pointsInCluster.
			means := matutil.MeanCols(pointsInCluster)  // 1x2 matrix with the mean coordinates of the current centroid
			// Centroids is a kX2 matrix where the row number corresponds to the same row in the centroid matrix
			// and the two columns are the means of the coordinates for that centroid cluster.
			centroidMeans.Set(c, 0, means.Get(0,0))
			centroidMeans.Set(c, 1, means.Get(0,1))
		}
	}
	return CentroidPoint(centroidMeans, centroidSqDist)*/
}


type PairPointToCentroidJob struct {
	point, centroids *matrix.DenseMatrix
	results chan<- PairPointToCentroidResult
	rowNum int
}

type PairPointToCentroidResult struct {
	centroidRowNum float64
	distSquared float64
	rowNum int
}


func addPairPointToCentroidJobs(jobs chan<- PairPointToCentroidJob, dataPoints, centroids *matrix.DenseMatrix, results chan<- PairPointToCentroidResult) {
	numRows, _ := dataPoints.GetSize()
    for i := 0; i < numRows; i++ {  // assign each data point to a centroid
		point := dataPoints.GetRowVector(i)
		jobs <- PairPointToCentroidJob{point, centroids, results, i}
	}
	close(jobs)
}

func doPairPointToCentroidJobs(done chan<- struct{}, jobs <-chan PairPointToCentroidJob) {
	for job := range jobs {
		job.PairPointToCentroid()
	}
	done <- struct{}{}
}

func awaitPairPointToCentroidCompletion(done <-chan struct{}, results chan PairPointToCentroidResult) {
	for i := 0; i < workers; i++ {
		<-done
	}
	close(results)
}

func processPairPointToCentroiResults(centroidSqDist *matrix.DenseMatrix, results <-chan PairPointToCentroidResult) {
	for result := range results {
            centroidSqDist.Set(result.rowNum, 0, result.centroidRowNum)
	        centroidSqDist.Set(result.rowNum, 1, result.distSquared)  
			fmt.Printf("%f, %f, %d\n", result.centroidRowNum, result.distSquared)  
	}
	// How to tell if cluster changed?
}
	
// AssignPointToCentroid checks a data point against all centroids and returns the best match.
// The centroid is identified by the row number in the centroid matrix.
// dataPoint is a 1X2 vector representing one data point.
// Return 
//    1. The row number in the centroid matrix.
//    2. (distance between centroid and point)^2
//    3. error
func (job PairPointToCentroidJob) PairPointToCentroid() {
    distPointToCentroid := math.Inf(1)
    centroidRowNum := float64(-1)
	distSq := float64(0)
	k, _ := job.centroids.GetSize()

    for j := 0; j < k; j++ {  // check distance btwn point and each centroid
     	distJ := matutil.EuclidDist(job.centroids.GetRowVector(j), job.point)
        if distJ < distPointToCentroid {
            distPointToCentroid = distJ
            centroidRowNum = float64(j)
		} 
 		distSq = math.Pow(distPointToCentroid, 2)
	}		
	job.results <- PairPointToCentroidResult{centroidRowNum, distSq, job.rowNum}
}
