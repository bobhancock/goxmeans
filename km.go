/*
 Package goxmeans implements a library for the xmeans algorithm.
 
 See Dan Pelleg and Andrew Moore - X-means: Extending K-means with Efficient Estimation of the Number of Clusters. 

 i = the index of the centroid which is closest to the i-th point.
 D = the input set of points
 Di is a subset of D and is the set of points that have mu_i as their closest centroid.
 R = |D|
 Ri = |Di|
 M = number of dimensions assuming spherical Gaussians.
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
//	"code.google.com/p/gomatrix/matrix"
	"github.com/bobhancock/gomatrix/matrix"
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

type CentroidChooser interface {
	ChooseCentroids(mat *matrix.DenseMatrix, k int) *matrix.DenseMatrix
}

// RandCentroids picks k uniformly distributed points from within the bounds of the dataset
type RandCentroids struct {}

// DataCentroids picks k distinct points from the dataset
type DataCentroids struct {}

// EllipseCentroids lays out the centroids along an elipse inscribed within the boundaries of the dataset
type EllipseCentroids struct {
	frac float64 // must be btw 0 and 1, this will be what fraction of a truly inscribing ellipse this is
}

// Load loads a tab delimited text file of floats into a slice.
// Assume last column is the target.
// For now, we limit ourselves to two columns
func Load(fname string) (*matrix.DenseMatrix, error) {
	z := matrix.Zeros(1,1)

	fp, err := os.Open(fname)
	if err != nil {
		return z, err
	}
	defer fp.Close()

	data := make([]float64, 0)
	cols := 0
	r := bufio.NewReader(fp)
	linenum := 0
	eof := false

	for !eof {
		var line string
		var buf []byte
//		line, err := r.ReadString('\n')
		buf , _, err := r.ReadLine()
		line = string(buf)

		if err == io.EOF {
			err = nil
			eof = true
			break
		} else if err != nil {
			return z, errors.New(fmt.Sprintf("goxmean.Load: reading linenum %d: %v", linenum, err))
		}

		l1 := strings.TrimRight(line, "\n")
		l := strings.Split(l1, "\t")
		// If each line does not have the same number of columns then error
		if linenum == 0 {
			cols = len(l)
		}

		if len(l) != cols {
			return z, errors.New(fmt.Sprintf("Load(): linenum %d has %d columns.  It should have %d columns.", linenum, len(line), cols))
		}	

		if len(l) < 2 {
			return z, errors.New(fmt.Sprintf("Load(): linenum %d has only %d elements", linenum, len(line)))
		}

		linenum++

		// Convert the strings to  float64 and build up the slice t by appending.
		t := make([]float64, 1)

		for i, v := range l {
			f, err := Atof64(string(v))
			if err != nil {
				return z, errors.New(fmt.Sprintf("goxmeanx.Load: cannot convert value %s to float64.", v))
			}
			if i == 0 {
				t[0] = f
			} else {
				t = append(t, f)
			}
			
		}
		data = append(data, t...)
	}
	mat := matrix.MakeDenseMatrix(data, linenum, cols)
	return mat, nil
}

// chooseCentroids picks random centroids based on the min and max values in the matrix
// and return a k by cols matrix of the centroids.
func (c RandCentroids) ChooseCentroids(mat *matrix.DenseMatrix, k int) *matrix.DenseMatrix {
	_, cols := mat.GetSize()
	centroids := matrix.Zeros(k, cols)

	for colnum := 0; colnum < cols; colnum++ {
		r := mat.ColSlice(colnum)

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
		for h := 0; h < k; h++ {
			randInRange := ((maxj - minj) * rand.Float64()) + minj
			centroids.Set(h, colnum, randInRange)
		}
	}
	return centroids
}

// Needs comments
func (c DataCentroids) ChooseCentroids(mat *matrix.DenseMatrix, k int) (*matrix.DenseMatrix, error) {
	// first set up a map to keep track of which data points have already been chosen so we don't dupe
	rows, cols := mat.GetSize()
	centroids := matrix.Zeros(k, cols)
	if k > rows {
		return centroids, errors.New("ChooseCentroids: Can't compute more centroids than data points!")
	}

	chosenIdxs := make(map [int]bool, k)
	for len(chosenIdxs) < k {
		index := rand.Intn(rows)
		chosenIdxs[index] = true 
	}
	i := 0
	for idx, _ := range chosenIdxs {
		centroids.SetRowVector(mat.GetRowVector(idx).Copy(), i)
		i += 1
	}
	return centroids, nil
}

// Needs comments
func (c EllipseCentroids) ChooseCentroids(mat *matrix.DenseMatrix, k int) *matrix.DenseMatrix {
	_, cols := mat.GetSize()
	var xmin, xmax, ymin, ymax = matutil.GetBoundaries(mat) 
	x0, y0 := xmin + (xmax - xmin)/2.0, ymin + (ymax-ymin)/2.0
	centroids := matrix.Zeros(k, cols)
	rx, ry := xmax - x0, ymax - y0  
	thetaInit := rand.Float64() * math.Pi

	for i := 0; i < k; i++ {
		centroids.Set(i, 0, rx * c.frac * math.Cos(thetaInit + float64(i) * math.Pi / float64(k)))
		centroids.Set(i, 1, ry * c.frac * math.Sin(thetaInit + float64(i) * math.Pi / float64(k)))
	}
	return centroids
}

// ComputeCentroids Needs comments.
func ComputeCentroid(mat *matrix.DenseMatrix) (*matrix.DenseMatrix, error) {
	rows, _ := mat.GetSize()
	vectorSum := mat.SumCols()
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
// denoting the centroid and the second column contains the squared
// error between a centroid and a point.
//
//  ____      _______
//  | 0        38.01 | <-- Centroid 0, squared error for the coordinates in row 0 of datapoints
//  | 1        23 .21| <-- Centroid 1, squared error for the coordinates in row 1 of datapoints
//  | 0        14.12 | <-- Centroid 0, squared error for the coordinates in row 2 of datapoints
//  _____     _______
func Kmeansp(datapoints *matrix.DenseMatrix, k int,cc CentroidChooser, measurer matutil.VectorMeasurer) (centroidMean, 
    centroidSqErr *matrix.DenseMatrix, err error) {
	//k, _ := centroids.GetSize()
	fp, _ := os.Create("/var/tmp/km.log")
	w := io.Writer(fp)
	log.SetOutput(w)

	centroids := cc.ChooseCentroids(datapoints, k)
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
		matches, err :=	centroidSqErr.FiltColMap(float64(c), float64(c), 0)  //rows with c in column 0.
		if err != nil {
			return centroidMean, centroidSqErr, nil
		}
		// It is possible that some centroids will not have any points, so there 
		// may not be any matches in the first column of centroidSqErr.
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
		means := pointsInCluster.MeanCols()
		centroidMean.Set(c, 0, means.Get(0,0))
		centroidMean.Set(c, 1, means.Get(0,1))
	}
	return 
}

// CentroidPoint stores the row number in the centroids matrix and
// the distance squared between the centroid.
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
     	distJ := job.measurer.CalcDist(job.centroids.GetRowVector(j), job.point)
        if distJ  < distPointToCentroid {
            distPointToCentroid = distJ
            centroidRowNum = float64(j)
		} 
 		squaredErr = math.Pow(distPointToCentroid, 2)
	}	
	job.results <- PairPointCentroidResult{centroidRowNum, squaredErr, job.rowNum, err}
}

// Kmeansbi bisects a given cluster and determines which centroids give the lowest error.
// Take the points in a cluster
// While the number of cluster < k
//    for every cluster
//        measure total error
//        cacl kmeansp with k=2 on a given cluster
//        measure total error after kmeansp split
//    choose the cluster split with the lowest SSE
//    commit the chosen split
func Kmeansbi(datapoints *matrix.DenseMatrix, k int, cc CentroidChooser, measurer matutil.VectorMeasurer) (matCentroidlist, clusterAssignment *matrix.DenseMatrix, err error) {
	numRows, numCols := datapoints.GetSize()
	clusterAssignment = matrix.Zeros(numRows, numCols)
	matCentroidlist = matrix.Zeros(k, numCols)
	centroid0 := datapoints.MeanCols()
	centroidlist := []*matrix.DenseMatrix{centroid0}

	// Initially create one cluster.
	for j := 0; j < numRows; j++ {
		point := datapoints.GetRowVector(j)
     	distJ := measurer.CalcDist(centroid0, point)
		clusterAssignment.Set(j,1, math.Pow(distJ, 2))
	}

	var bestClusterAssignment, bestNewCentroids *matrix.DenseMatrix
	var bestCentroidToSplit int

	// Find the best centroid configuration.
	for ; len(centroidlist) < k; {
		lowestSSE := math.Inf(1)
		// Split cluster
		for i, _ := range centroidlist {
			// Get the points in this cluster
			pointsCurCluster, err := clusterAssignment.FiltCol(float64(i), float64(i), 0)  
			if err != nil {
				return matCentroidlist, clusterAssignment, err
			}

			centroids, splitClusterAssignment, err := Kmeansp(pointsCurCluster, 2, cc, measurer)
			if err != nil {
				return matCentroidlist, clusterAssignment, err
			}

            /* centroids is a 2X2 matrix of the best centroids found by kmeans
            
             splitClustAssignment is a mX2 matrix where col0 is either 0 or 1 and refers to the rows in centroids
             where col1 cotains the squared error between a centroid and a point.  The rows here correspond to 
             the rows in ptsInCurrCluster.  For example, if row 2 contains [1, 7.999] this means that centroid 1
             has been paired with the point in row 2 of splitClustAssignment and that the squared error (distance 
             between centroid and point) is 7.999.
            */

            // Calculate the sum of squared errors for each centroid. 
            // This give a statistcal measurement of how good
            // the clustering is for this cluster.
			sseSplit := splitClusterAssignment.SumCol(1)
			// Calculate the SSE for the original cluster
			sqerr, err := clusterAssignment.FiltCol(float64(0), math.Inf(1), 0)
			if err != nil {
				return matCentroidlist, clusterAssignment, err
			}
			sseNotSplit := sqerr.SumCol(1)

			if sseSplit + sseNotSplit < lowestSSE {
				bestCentroidToSplit = 1
				bestNewCentroids = matrix.MakeDenseCopy(centroids)
				bestClusterAssignment =  matrix.MakeDenseCopy(splitClusterAssignment)
			}
		}

		// Applying the split overwrites the existing cluster assginments for the 
		// cluster you have decided to split.  Kmeansp() returned two clusters
		// labeled 0 and 1. Change these cluster numbers to the cluster number
		// you are splitting and the next cluster to be added.
		m, err := bestClusterAssignment.FiltColMap(1, 1, 0)
		if err != nil {
			return matCentroidlist, clusterAssignment, err
		}
		for i,_ := range m {
			bestClusterAssignment.Set(i, 0, float64(len(centroidlist)))
		}	

		n, err := bestClusterAssignment.FiltColMap(0, 0, 0)
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
	}
	return matCentroidlist, clusterAssignment, nil
}

// variance is the maximum likelihood estimate (MLE) for the variance, under
// the identical spherical Gaussian assumption.
//
// variance = 	(1 / (R - K) * \sigma for all i  (x_i - mu_(i))^2
// where i indexes the individual points.  
// N.B. mu_i denotes the coordinates of the centroid closest to the i-th data point.  Not
// the mean of the entire cluster.
func variance(points, centroid *matrix.DenseMatrix, K float64, measurer matutil.VectorMeasurer) float64 {
	r, _ := points.GetSize()
	R := float64(r)
	
	// Sum over all points (point_i - mean(i))^2
	sum := float64(0)
	for i := 0; i < r; i++ {
		p := points.GetRowVector(i)
		dist := measurer.CalcDist(centroid, p)
		sum += math.Pow(dist, 2) 
	}
	variance := float64((1 / (R - K))) * sum

	return variance
}

// pointProb calculates the probability of an individual point.
//
// R = |D|
// Ri = |Dn| for the cluster contining the point x_i.
// M = # of dimensions
// V = variance of Dn
// mu(i) =  the coordinates of the centroid closest to the i-th data point.
//
// P(x_i) = [ (Ri / R) * (1 / (sqrt(2 * Pi) * stddev^M) ]^(-(1/2 * sqrt(V) * ||x_i - mean(i)||^2)
//
// This is just the probability that a point exists (Ri / R) times the normal, or Gaussian,  distribution 
// for the number of dimensions.
func pointProb(R, Ri, M, V float64, point, mu *matrix.DenseMatrix, measurer matutil.VectorMeasurer) float64 {
	exists := float64(Ri / R)
	//fmt.Printf("term1=%f\n", term1)

	normdist := normDist(M, V, point, mu, measurer)
	prob := exists * normdist
	return prob
}

// nomrDist calculates the normal distribution for the number of dimesions in a spherical Gaussian.
//
// M = # of dimensions
// V = variance of Dn
// mean(i) =  the mean distance between all points in Dn and a centroid. 
func normDist(M, V float64, point, mean *matrix.DenseMatrix,  measurer matutil.VectorMeasurer) float64 {
	dist := measurer.CalcDist(point, mean)
 	//fmt.Printf("dist=%f\n", dist)
	stddev := math.Sqrt(V)

	sqrt2pi := math.Sqrt(2.0 * math.Pi)
	//fmt.Printf("sqrt2pi=%f\n", sqrt2pi)

	stddevM := math.Pow(stddev, M)
	//fmt.Printf("stddevM=%f\n", stddevM)

	base := 1 / (sqrt2pi * stddevM)
	//fmt.Printf("base=%f\n", base)

	exp := -(1.0/(2.0 * V)) * math.Abs(dist)
	//fmt.Printf("exp=%f\n",exp)

	normdist := math.Pow(base, exp)
	//fmt.Printf("term2=%f\n", term2)
	return normdist
}

// loglikeli is the log likelihood estimate of the data taken at the maximum
// likelihood point.  1 <= n <= K
//
// D = set of points
// R = |D|
// R_n = |D_n|
// M = # of dimensions
// V = unbiased variance of D
// K = number of clusters
//
// l^hat(D) = \sigma n=1 to K [R_n logR_n - R logR - (RM/2log * log(2Pi * V) - 1/2(R - K)
//
// All logs are log e.  The right 3 terms are summed to ts for the loop.
//
// N.B. When applying this to D, then R = Rn.  When bisecting, R refers to the original,
// or parent cluster, Rn is a member of the clusers {R_0, R_1} the two child clusters.
//
// Refer to Notes on Bayesian Information Criterion Calculation equation 23.
func loglikeli(R, M, variance, K float64, Rn []float64) float64 {
	t2 := R * math.Log(R)
//	fmt.Printf("t2=%f\n", t2)

	t3 := ((R * M) / 2.0)  * math.Log(2.0 * math.Pi * variance)
//	fmt.Printf("t3=%f\n",t3)

	t4 := (1 / 2.0) * (R - K)
//	fmt.Printf("t4=%f\n", t4)

	ts := t2 - t3 - t4
//	fmt.Printf("ts=%f\n", ts)

	ll := float64(0)
	for n := 0; n < int(len(Rn)); n++ {
		t1 := Rn[n] * math.Log(Rn[n])
//		fmt.Printf("t1_%d=%f\n", n, t1)
		s := t1 - ts
		ll += s
//		fmt.Printf("lD_n=%f\n", lD)
	}
	return ll
}

// numgreep returns the number of free parameters in the BIC.
//
// K - number of clusters
// M - number of dimensions
//
// (K - 1 class probabilities) + (M * K) + 1 variance estimate.
// Since the variance is a free paramter, identical for every cluster, it
// counts as 1.
func freeparams(K, M float64) float64 {
	return (K - 1.0) + (M * K) + 1
}

// BIC calculated the Bayesian Information Criterion or Schwarz Criterion
// 
// D = set of points

// R = |D|
// M = number of dimesions assuming a spherical Gaussians
// p = number of parameters in Mj
// log is the natural log
// l(D) is the log likelihood of the data of the jth model taken at the 
//  maximum likelihood point.
//
// BIC(M_j) = l_j(D) - freeparams/2 * log R
func BIC(lD, freeparams, R float64) float64 {
	return lD - (freeparams / 2) - math.Log(R)
}