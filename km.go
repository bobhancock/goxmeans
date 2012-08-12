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

type Kmodel struct {
	BIC float64
	centroids *matrix.DenseMatrix
	clusterAssessment *matrix.DenseMatrix
}

// Kmeansmodels runs k-means for k lower bound to k upper bound.
//
// Calling Kmeansmodels(datapoints, 4, 10, cc, measurer) will return a map where
// the index is the number k is a member of {4...10} and the value is a Kmodel 
// with the details of the model.
/*func Kmeansmodels(datapoints *matrix.DenseMatrix, klow, kup int, cc CentroidChooser, measurer matutil.VectorMeasurer) map[int]Kmodel {
	// TODO This is just place holder function at the moment
	for i := klow; i < kup; i++ {
		centroids, clusterAssessment, err := Kmeansp(datapoints, i, cc, measurer)
	}

	m := make(map[int]Kmodel, 10)
	return m
}*/
	

// Kmeansp partitions datapoints into K clusters.  This results in a partitioning of
// the data space into Voronoi cells.  The problem is NP-hard so here we attempt
// to parallelize as many processes as possible to reduce the running time.
//
// 1. Place K points into the space represented by the objects that are being clustered.
// These points represent initial group centroids.
//
// 2. Assign each object to the group that has the closest centroid.
//
// 3. When all objects have been assigned, recalculate the positions of the K centroids
// by calculating the mean of all cooridnates in a group (i.e., cluster) and making that
// the new centroid.
//
// 4. Repeat Steps 2 and 3 until the centroids no longer move.
//
// Return Values
//
// centroids is K x M matrix that cotains the coordinates for the centroids.
// The centroids are indexed by the 0 based rows of this matrix.
//  ____      _________
//  | 12.29   32.94 ... | <-- The coordinates for centroid 0
//  | 4.6     29.22 ... | <-- The coordinates for centroid 1
//  |_____    __________|
// 
//
// clusterAssessment is ax R x 2 matrix.  The rows have a 1:1 relationship to 
// the rows in datapoints.  Column 0 contains the row number in centroids
// that corresponds to the centroid for the datapoint in row i of this matrix.
// Column 1 contains the squared error between the centroid and datapoint(i).
//
//  ____      _______
//  | 3        38.01 | <-- Centroid 3, squared error for the coordinates in row 0 of datapoints
//  | 1        23 .21| <-- Centroid 1, squared error for the coordinates in row 1 of datapoints
//  | 0        14.12 | <-- Centroid 0, squared error for the coordinates in row 2 of datapoints
//  _____     _______
//
func Kmeansp(datapoints *matrix.DenseMatrix, k int, cc CentroidChooser, measurer matutil.VectorMeasurer) (*matrix.DenseMatrix, 
	*matrix.DenseMatrix, error) {
/* centroids                 datapoints                clusterAssessment
                    _____________________________________
   ____   ______    |       ____   ____                __|__   ______
   | ...        |   V       | ...     |               |  ...         |
   | 3     38.1 | row 3     | 3.0  5.1| <-- row i --> |  3     32.12 |
   |___   ______|           |____  ___|               |____    ______|
*/
	fp, _ := os.Create("/var/tmp/km.log")
	w := io.Writer(fp)
	log.SetOutput(w)

	centroids := cc.ChooseCentroids(datapoints, k)
	numRows, numCols := datapoints.GetSize()
	clusterAssessment := matrix.Zeros(numRows, numCols)

	jobs := make(chan PairPointCentroidJob, numworkers)
	results := make(chan PairPointCentroidResult, minimum(1024, numRows))
	done := make(chan int, numworkers)
	
	clusterChanged := true
	for ; clusterChanged == true ; {
		clusterChanged = false
		// Pair each point with its closest centroid.
		go addPairPointCentroidJobs(jobs, datapoints, clusterAssessment, centroids, measurer, results)
		for i := 0; i < numworkers; i++ {
			go doPairPointCentroidJobs(done, jobs)
		}
		go awaitPairPointCentroidCompletion(done, results)
		clusterChanged = assessClusters(clusterAssessment, results) // This blocks so that all the results can be processed
		
		// Now that you have each data point grouped with a centroid,
		for cent := 0; cent < k; cent++ {
			// Select all the rows in clusterAssessment whose first col value == cent.
			// Get the corresponding row vector from datapoints and place it in pointsInCluster.
			// TODO Would it be better to return a slice and use that to create the matrix?
			matches, err :=	clusterAssessment.FiltColMap(float64(cent), float64(cent), 0)  
			// matches - a map[int]float64 where the key is the row number in source 
			//matrix  and the value is the value in the column of the source matrix 
			//specified by col.
			if err != nil {
				return matrix.Zeros(k, numCols), clusterAssessment, err
			}
			// It is possible that some centroids could not have any points, so there 
			// may not be any matches.
			if len(matches) == 0 {
				continue
			}

			pointsInCluster := matrix.Zeros(len(matches), numCols) 
			for d, rownum := range matches {
				pointsInCluster.Set(d, 0, datapoints.Get(int(rownum), 0))
				pointsInCluster.Set(d, 1, datapoints.Get(int(rownum), 1))
			}
			
			// pointsInCluster now contains all the data points for the current 
			// centroid.  The mean of the coordinates for this cluster becomes 
			// the new centroid for this cluster.
			mean := pointsInCluster.MeanCols()
			centroids.SetRowVector(mean, cent)
		}
	}
	return centroids, clusterAssessment, nil
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
	point, centroids, clusterAssessment *matrix.DenseMatrix
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
	clusterAssessment *matrix.DenseMatrix, measurer matutil.VectorMeasurer, results chan<- PairPointCentroidResult) {
	numRows, _ := datapoints.GetSize()
    for i := 0; i < numRows; i++ { 
		point := datapoints.GetRowVector(i)
		jobs <- PairPointCentroidJob{point, centroids, clusterAssessment, results, i, measurer}
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

// assessClusters assigns the results to the clusterAssessment matrix.
func assessClusters(clusterAssessment *matrix.DenseMatrix, results <-chan PairPointCentroidResult) bool {
	change := false
	for result := range results {
		if clusterAssessment.Get(result.rowNum, 0) != result.centroidRowNum {
			change = true
		}
	    clusterAssessment.Set(result.rowNum, 0, result.centroidRowNum)
	    clusterAssessment.Set(result.rowNum, 1, result.distSquared)  
	}
	return change
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

// kmeansbi bisects datapoints with two centroids and performs k-means.  The BIC 
// of the two resulting clusters is compared to the original BIC and whichever
// scores better, wins.
//
// Returns the same values as kmeansp().
func kmeansbi(datapoints *matrix.DenseMatrix,cc CentroidChooser, measurer matutil.VectorMeasurer) (*matrix.DenseMatrix,
	*matrix.DenseMatrix, error) {
	numRows, numCols := datapoints.GetSize()
	clusterAssessment := matrix.Zeros(numRows, numCols)

	centroids, clusterAssessment, err := Kmeansp(datapoints, 2, cc, measurer)
	if err != nil {
		return matrix.Zeros(1,1), matrix.Zeros(1,1), err
	}

	return centroids, clusterAssessment, nil
}


// variance is the maximum likelihood estimate (MLE) for the variance, under
// the identical spherical Gaussian assumption.
//
// points = an R x M matrix of all data point coordinates.
//
// clusterAssessment =  R x 2 matrix.  Column 0 contains the index {0...K} of
// a centroid.  Column 1 contains (datapoint_i - mu(i))^2 
// 
// centroids =  K x M+1 matrix.  Column 0 continas the centroid index {0...K}.
// Columns 1...M contain the centroid coordinates.  (See Kmeansp() for an example.)
//
// variance = (1 / (R - K) * \sigma for all i  (datapoint_i - mu_(i))^2
// where i indexes the individual points.  
//
// N.B. mu_i denotes the coordinates of the centroid closest to the i-th data point.  Not
// the mean of the entire cluster.
func variance(points, centroids, clusterAssessment  *matrix.DenseMatrix, K float64, measurer matutil.VectorMeasurer) (float64, error) {
	rows, _ := points.GetSize()
	R := float64(rows)

	// Sum over all points (point_i - mean(i))^2
	sum := float64(0)
	for i := 0; i < rows; i++ {
		p := points.GetRowVector(i)
		// Find it's centroid
		centindex := clusterAssessment.Get(i, 0)
		mu_i := centroids.GetRowVector(int(centindex))
		dist := measurer.CalcDist(mu_i, p)
		sum += math.Pow(dist, 2) 
	}
	variance := float64((1 / (R - K))) * sum

	return variance, nil
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

// freeparams returns the number of free parameters in the BIC.
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

// BIC calculates the Bayesian Information Criterion or Schwarz Criterion
// 
// D = set of points
//
// R = |D|
//
// M = number of dimesions assuming a spherical Gaussians
//
// p = number of parameters in Mj
//
// log = the natural log
//
// l(D) = the log likelihood of the data of the jth model taken at the 
// maximum likelihood point.
//
// BIC(M_j) = l_j(D) - freeparams/2 * log R
func bic(lD, freeparams, R float64) float64 {
	return lD - (freeparams / 2) - math.Log(R)
}

// calcBIC calculates the Bayesian Information Criterion for the model datapoints
func calcBIC(datapoints, centroids, clusterAssessment *matrix.DenseMatrix, measurer matutil.VectorMeasurer, R, K, M float64, Rn []float64) (float64, error) {
	variance, err := variance(datapoints, centroids, clusterAssessment, K, measurer)
	if err != nil {
		return 0.0, errors.New(fmt.Sprintf("calcBIC: variance returned err = %v\n", err))
	}

	loglikelihood := loglikeli(R, M, variance, K, Rn)
	 
	freeparameters := freeparams(K, M)

	return bic(loglikelihood, freeparameters, R), nil
}
