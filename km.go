/*
 Package goxmeans implements a library for the xmeans algorithm.
 
 See Dan Pelleg and Andrew Moore - X-means: Extending K-means with Efficient Estimation of the Number of Clusters. 

 D = the input set of points

 R = |D| the number of points in a model.

 M = number of dimensions assuming spherical Gaussians.


The algorithm consists of two operations repeated until completion.

1. Improve parameters

2. Improve structure

3. If K > Kmax then stop and return a slice of Models with BIC scores, else goto 1.
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
//	"log"
//	"time"
	"github.com/bobhancock/gomatrix/matrix"
)

var numworkers = runtime.NumCPU()

// minimum returns the smallest int.
func minimum(x int, ys ...int) int {
    for _, y := range ys {
        if y < x {
            x = y
        }
    }
    return x
}

// Load loads a tab delimited text file of floats into a matrix.
func Load(fname, sep string) (*matrix.DenseMatrix, error) {
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
		l := strings.Split(l1, sep)
		
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
		t := make([]float64, 0)

		for _, v := range l {
			v = strings.TrimSpace(v)
			f, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return z, errors.New(fmt.Sprintf("goxmeans.Load: cannot convert value %s to float64.", v))
			}
			t = append(t, f)
		}
		data = append(data, t...)
	}
	mat := matrix.MakeDenseMatrix(data, linenum, cols)
	//fmt.Println(time.Now())n // flag for debugging
	return mat, nil
}

// CentroidChooser is the interface that wraps CentroidChooser function.
//
// CetnroidChooser returns a matrix of K coordinates in M dimensions.
type CentroidChooser interface {
	ChooseCentroids(mat *matrix.DenseMatrix, k int) *matrix.DenseMatrix
}

// RandCentroids picks k uniformly distributed points from within the bounds of the dataset as initial centroids.
type randCentroids struct {}

// DataCentroids picks k distinct points from the dataset as initial centroids.
type DataCentroids struct {}

// EllipseCentroids lays out the initial centroids evenly along an elipse inscribed and centered within the boundaries of the dataset.
// It is only defined for M=2
// * Frac: This must be a float between 0 and 1. It determines the scale of the inscribing ellipse relative to the dataset,
//	so Frac==1.0 produces an ellipse that spans the entire dataset, while Frac==0.5 produces an ellipse spanning half the dataset.
type EllipseCentroids struct {
	Frac float64 // must be btw 0 and 1, this will be what fraction of a truly inscribing ellipse this is
}

// chooseCentroids picks random centroids based on the min and max values in the matrix
// and return a k by m matrix of the centroids.
func (c randCentroids) ChooseCentroids(mat *matrix.DenseMatrix, k int) *matrix.DenseMatrix {
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

// DataCentroids picks k distinct points from the dataset.  If k is > points in
// the matrix then k is set to the number of points.
func (c DataCentroids) ChooseCentroids(mat *matrix.DenseMatrix, k int) (*matrix.DenseMatrix) {
	// first set up a map to keep track of which data points have already been chosen so we don't dupe
	rows, cols := mat.GetSize()
	centroids := matrix.Zeros(k, cols)
	if k > rows {
		k = rows
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
	return centroids
}

// EllipseCentroids lays out the initial centroids evenly along an elipse inscribed and centered within the boundaries of the dataset.
// It is only defined for M=2
// * Frac: This must be a float between 0 and 1. It determines the scale of the inscribing ellipse relative to the dataset,
//	so Frac==1.0 produces an ellipse that spans the entire dataset, while Frac==0.5 produces an ellipse spanning half the dataset.
func (c EllipseCentroids) ChooseCentroids(mat *matrix.DenseMatrix, k int) *matrix.DenseMatrix {
	_, cols := mat.GetSize()
	// TODO Cache boundaries call for each matrix so that it is not called on each bisect
	var xmin, xmax, ymin, ymax = boundaries(mat) 

	x0, y0 := xmin + (xmax - xmin)/2.0, ymin + (ymax-ymin)/2.0
	centroids := matrix.Zeros(k, cols)
	rx, ry := xmax - x0, ymax - y0  
	thetaInit := rand.Float64() * math.Pi

	for i := 0; i < k; i++ {
		centroids.Set(i, 0, rx * c.Frac * math.Cos(thetaInit + float64(i) * 2.0 * math.Pi / float64(k)))
		centroids.Set(i, 1, ry * c.Frac * math.Sin(thetaInit + float64(i) * 2.0 * math.Pi / float64(k)))
	}
	return centroids
}

// Measurer finds the distance between the points in the columns
type VectorMeasurer interface {
	CalcDist(a, b *matrix.DenseMatrix) (dist float64)
}

type vectorDistance struct {}

type EuclidDist vectorDistance

// CalcDist finds the Euclidean distance between points.
func (ed EuclidDist) CalcDist(p, q *matrix.DenseMatrix) float64 {
	diff := matrix.Difference(q, p)
	diff.Sqrm() // square each value in the matrix in place
	// To avoid allocating a new matrix the sum is placed in the 
	// first column of each row.  N.B. this modifies the matrix directly.
	diff.SumRowsM() 
	// Each point coordinate consists of 1 row and M cols.  So,
	// the sum will be at [0, 0]
	s := diff.Get(0, 0)

	return math.Sqrt(s)
}

type ManhattanDist struct {}

// CalcDist finds the ManhattanDistance which is the sum of the aboslute 
// difference of the coordinates.   Also known as rectilinear distance, 
// city block distance, or taxicab distance.
func (md ManhattanDist) CalcDist(a, b *matrix.DenseMatrix) float64 {
	return math.Abs(a.Get(0,0) - b.Get(0,0)) + math.Abs(a.Get(0,1) - b.Get(0,1))
}

// boundaries returns the max and min x and y values for a dense matrix
// of shape m x m.
func boundaries(mat *matrix.DenseMatrix) (xmin, xmax, ymin, ymax float64) {
	rows, _ := mat.GetSize()
	xmin, ymin = mat.Get(0,0), mat.Get(0,1)
	xmax, ymax = mat.Get(0,0), mat.Get(0,1)

	for i := 1; i < rows; i++ {
		xi, yi := mat.Get(i, 0), mat.Get(i, 1)
		
		if xi > xmax{
			xmax = xi
		} else if xi < xmin {
			xmin = xi
		}

		if yi > ymax{
			ymax = yi
		} else if yi < ymin {
			ymin = yi
		}
	}
	return
}

// Model is a statistical model with a BIC score and a collection of clusters.
type Model struct {
	Bic float64
	Clusters []cluster
}

func (m Model) Numcentroids() int {
	c := 0
	for _, clust := range m.Clusters {
		c += clust.Numcentroids()
	}
	return c
}

// cluster models an individual cluster.
type cluster struct {
	Points *matrix.DenseMatrix
	Centroid  *matrix.DenseMatrix
	Variance float64
}

// numpoints returns the number of points in a cluster.
func (c cluster) Numpoints() int {
	r, _ := c.Points.GetSize()
	return r
}

func (c cluster) Dim() int {
	_, d := c.Points.GetSize()
	return d
}

// numcentroids returns the number of centroids for a cluster.  This should normally be 1.
func (c cluster) Numcentroids() int {
	r, _ := c.Centroid.GetSize()
	return r
}

var centroids *matrix.DenseMatrix

// Xmeans runs k-means for k lower bound to k upper bound on a data set.
// Once the k centroids have converged each cluster is bisected and the BIC
// of the orginal cluster (parent = a model with one centroid) to the 
// the bisected model which consists of two centroids and whichever is greater
// is committed to the set of clusters for this larger model k.
// 
func Xmeans(datapoints, centroids *matrix.DenseMatrix, k, kmax int,  cc, bisectcc CentroidChooser, measurer VectorMeasurer) ([]Model, map[string] error) {
	var err error

// Uncomment logging code as well as the import statement above if you want simple logging to the elapsed
// time between major events.
/*	logname := "/var/tmp/xmeans.log"
	fp, err :=  os.OpenFile(logname, os.O_RDWR|os.O_APPEND, 0666)
	if err != nil {
		if os.IsNotExist(err) {
			fp, err = os.Create(logname)
    		if err != nil {
	    		fmt.Printf("Xmeans: cannot open %s for logging.\n", logname)
			}
		} 
	}

	log.SetOutput(io.Writer(fp))
*/	
	if k > kmax {
		m := make([]Model,0)
		e := map[string] error {
			"k": errors.New(fmt.Sprintf("k must be <= kmax.  Received k=%d and kmax=%d.", k, kmax)),
		}
	    return m, e
	}

//	log.Printf("Start k=%d kmax=%d\n", k, kmax)
	
	R, M := datapoints.GetSize()
	errs := make(map[string]error)
	runtime.GOMAXPROCS(numworkers)
	models := make([]Model, 0)

	for  k <= kmax {
//		log.Printf("kmeans started k=%d\n", k)
		model := kmeans(datapoints, centroids, measurer)
		
		// Bisect the returned clusters
//		log.Println("bisect started")
		bimodel := bisect(model.Clusters, R, M, bisectcc, measurer)
		numCentroids := len(bimodel.Clusters)
//		log.Printf("bisect returned %d clusters\n", numCentroids)
		
		models = append(models, bimodel)

		var cent *matrix.DenseMatrix
				
		if numCentroids <= kmax {
			for  rowexists := true; rowexists == true; {
				cent = cc.ChooseCentroids(datapoints, 1)
				rowexists = centroids.RowExists(cent)
			}
		
			centroids, err  = centroids.AppendRow(cent)
			if err != nil {
				errs["ApppendRow"] = err
				break
			} 
		}
		k = numCentroids
	}
		
//	log.Println("Finished")
	return models, errs
}

// bisect takes a slice of clusters, bisects them attempting to create a better
// model with two new centroids.
func bisect(clustersToBisect []cluster, R, M int, bisectcc CentroidChooser,  measurer VectorMeasurer) Model {
	// bisect each clusters and see if you can get a better BIC
	// You are comparing ModelParent with the one centroid to ModelChild
	// bisected with two centroids.
	bufsize := 0.0
	for _, clust := range clustersToBisect {
		numRows, _ := clust.Points.GetSize()
		bufsize = math.Max(bufsize, float64(numRows))
	}
	
	bufclusters := make([]cluster, 0)

	// You would do this if you wanted to keep bisecting until only
	// the parent won.
	//for len(clustersToBisect) > 0  {
	bijobs := make(chan bisectJob, 1024)
	biresults := make(chan bisectResult, 1024)
	bidone := make(chan int, 1024)
		
	go addBisectJobs(bijobs, clustersToBisect, bisectcc, measurer, biresults)
	for i := 0; i < numworkers; i++ {
		go doBisectJob(bidone, bijobs)
	}
	go awaitBisectJobsCompletion(bidone, biresults)
	
	for biresult := range biresults {
		bufclusters = append(bufclusters, biresult.clusters...)
	}
	
	modelbic := calcbic(R, M, bufclusters) 
	model := Model{modelbic, bufclusters}
	return model
}

// kmeans partitions datapoints into K clusters.  This results in a partitioning of
// the data space into Voronoi cells.  The problem is NP-hard so here we attempt
// to parallelize or make concurrent as many processes as possible to reduce the 
// running time.
//
// 1. Place K points into the space represented by the objects that are being clustered.
// These points represent initial group centroids.
//
// 2. Assign each object to the group that has the closest centroid.
//
// 3. When all objects have been assigned, recalculate the positions of the K centroids
// by calculating the mean of all cooridnates in a cluster and making that
// the new centroid.
//
// 4. Repeat Steps 2 and 3 until the centroids no longer move.
//
// centroids is K x M matrix that cotains the coordinates for the centroids.
// The centroids are indexed by the 0 based rows of this matrix.
//  ____      _________
//  | 12.29   32.94 ... | <-- The coordinates for centroid 0
//  | 4.6     29.22 ... | <-- The coordinates for centroid 1
//  |_____    __________|
// 
//
// CentPointDist is ax R x M matrix.  The rows have a 1:1 relationship to 
// the rows in datapoints.  Column 0 contains the row number in centroids
// that corresponds to the centroid for the datapoint in row i of this matrix.
// Column 1 contains (x_i - mu(i))^2.
//  ____      _______
//  | 3        38.01 | <-- Centroid 3, squared error for the coordinates in row 0 of datapoints
//  | 1        23 .21| <-- Centroid 1, squared error for the coordinates in row 1 of datapoints
//  | 0        14.12 | <-- Centroid 0, squared error for the coordinates in row 2 of datapoints
//  _____     _______
//
func kmeans(datapoints, centroids *matrix.DenseMatrix, measurer VectorMeasurer) Model {
/*  datapoints				  CentPoinDist            centroids				  
                                 ________________
   ____	  ____				  __|__	  ______	 |	  ____	___________	  
   | ...	 |				 |	...			|	 V	 | ...		       |	  
   | 3.0  5.1| <-- row i --> |	3	  32.12 |  row 3 | 3	 38.1, ... |		  
   |____  ___|				 |____	  ______|	     |___	__________ |			  
*/
	R, M := datapoints.GetSize()
	CentPointDist := matrix.Zeros(R,2)
	k, _ := centroids.GetSize()

	clusterChanged := true
	var clusters []cluster

	for ; clusterChanged == true ; {
		clusterChanged = false
		clusters = make([]cluster, 0)

		jobs := make(chan PairPointCentroidJob, 1024)
		results := make(chan PairPointCentroidResult, 1024)
		done := make(chan int, 1024)

		// Pair each point with its closest centroid.
		go addPairPointCentroidJobs(jobs, datapoints, centroids, measurer, results)
		for i := 0; i < numworkers; i++ {
			go doPairPointCentroidJobs(done, jobs)
		}
		go awaitPairPointCentroidCompletion(done, results)

		clusterChanged = assessClusters(CentPointDist, results) // This blocks so that all the results can be processed

		// You have each data point grouped with a centroid,
		for idx, cent := 0, 0; cent < k; cent++ {
			// Select all the rows in CentPointDist whose first col value == cent.
			// Get the corresponding row vector from datapoints and place it in pointsInCluster.
			r, _ := CentPointDist.GetSize()
			matches := make([]int, 0)

			for i := 0; i < r; i++ {
				v := CentPointDist.Get(i, 0)
				if v == float64(cent) {
					matches = append(matches, i)
				}
			}

			// It is possible that some centroids may have zero points, so there 
			// may not be any matches.
			if len(matches) == 0 {
				continue
			}

			pointsInCluster := matrix.Zeros(len(matches), M) 
			i := 0

			for _, rownum := range matches {
				pointsInCluster.Set(i, 0, datapoints.Get(int(rownum), 0))
				pointsInCluster.Set(i, 1, datapoints.Get(int(rownum), 1))
				i++
			}

			// pointsInCluster now contains all the data points for the current 
			// centroid.  The mean of the coordinates for this cluster becomes 
			// the new centroid for this cluster.
			mean := pointsInCluster.MeanCols()
			centroids.SetRowVector(mean, cent)

			clust := cluster{pointsInCluster, mean, 0}
			clust.Variance = variance(clust, measurer)
			clusters = append(clusters, clust)
			idx++
		}
	}
	modelbic := calcbic(R, M, clusters) 
	model := Model{modelbic, clusters}
	return model
}
 
// CentroidPoint stores the row number in the centroids matrix and
// the distance squared between the centroid and the point.
type CentroidPoint struct {
	centroidRunNum float64
	distPointToCentroidSq float64
}

// PairPointCentroidJobs stores the elements that define the job that pairs a 
// point with a centroid.
type PairPointCentroidJob struct {
	point, centroids *matrix.DenseMatrix
	results chan<- PairPointCentroidResult
	rowNum int
	measurer VectorMeasurer
}

// PairPointCentroidResult stores the results of pairing a point with a 
// centroid.
type PairPointCentroidResult struct {
	centroidRowNum float64
	distSquared float64
	rowNum int
}

// addPairPointCentroidJobs adds a job to the jobs channel.
func addPairPointCentroidJobs(jobs chan<- PairPointCentroidJob, datapoints, 
	centroids *matrix.DenseMatrix,	measurer VectorMeasurer, results chan<- PairPointCentroidResult) {
	numRows, _ := datapoints.GetSize()
    for i := 0; i < numRows; i++ { 
		point := datapoints.GetRowVector(i)
		jobs <- PairPointCentroidJob{point, centroids, results, i, measurer}
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

// PairPointCentroid pairs a point with the closest centroids.
func (job PairPointCentroidJob) PairPointCentroid() {
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
// 		squaredErr = math.Pow(distPointToCentroid, 2)
 		squaredErr = distPointToCentroid * distPointToCentroid
	}	
	job.results <- PairPointCentroidResult{centroidRowNum, squaredErr, job.rowNum}
}

// awaitPairPointCentroidCompletion waits until all jobs are completed.
func awaitPairPointCentroidCompletion(done <-chan int, results chan PairPointCentroidResult) {
	for i := 0; i < numworkers; i++ {
		<-done
	}
	close(results)
}

// assessClusters assigns the results to the CentPointDist matrix.
func assessClusters(CentPointDist *matrix.DenseMatrix, results <-chan PairPointCentroidResult) bool {
	change := false
	for result := range results {
		if CentPointDist.Get(result.rowNum, 0) != result.centroidRowNum {
			change = true
		}
	    CentPointDist.Set(result.rowNum, 0, result.centroidRowNum)
	    CentPointDist.Set(result.rowNum, 1, result.distSquared)  
	}
	return change
}
	
type bisectJob struct {
	clust cluster
	bcc CentroidChooser
	measurer VectorMeasurer
	results chan<- bisectResult 
}

type bisectResult struct {
	centroids *matrix.DenseMatrix
	bic float64
	clusters []cluster
}

func addBisectJobs(jobs chan<- bisectJob, clusters []cluster, cc CentroidChooser,
	measurer VectorMeasurer,  results chan<- bisectResult) {
	for _, clust := range clusters {
		jobs <- bisectJob{clust, cc, measurer, results}
	}
	close(jobs)
}

func doBisectJob(done chan<- int, jobs <-chan bisectJob) {
	for job := range jobs {
		job.bisectCluster()
	}
	done <- 1
}

func (job bisectJob) bisectCluster() {
	R, M := job.clust.Points.GetSize()
	parentCluster := make([]cluster,0)
	parentCluster = append(parentCluster, job.clust)

	centroids := job.bcc.ChooseCentroids(job.clust.Points, 2)
	model := kmeans(job.clust.Points, centroids, job.measurer)

	parentBIC := calcbic(R, M, parentCluster)
	childBIC := calcbic(R, M, model.Clusters)
	if childBIC > parentBIC {
		job.results <- bisectResult{centroids, childBIC, model.Clusters}
	} else {
		job.results <- bisectResult{centroids, parentBIC, parentCluster}
	}
}

func awaitBisectJobsCompletion(done <- chan int, results chan bisectResult) {
	for i := 0; i < numworkers; i++ {
		<-done
	}
	close(results)
}
	
// variance is the maximum likelihood estimate (MLE) for the variance, under
// the identical spherical Gaussian assumption.
//
// points = an R x M matrix of all point coordinates.
//
// CentPointDist =  R x M+1 matrix.  Column 0 contains the index {0...K-1} of
// a centroid.  Column 1 contains (datapoint_i - mu(i))^2 
// 
// centroids =  K x M+1 matrix.  Column 0 continas the centroid index {0...K}.
// Columns 1...M contain the centroid coordinates.  (See kmeans() for an example.)
//
//    1        __                 2
// ------  *  \     (x   -  mu   ) 
// R - K      /__ i   i       (i)  
//
// where i indexes the individual points.  
//
// N.B. mu_(i) denotes the coordinates of the centroid closest to the i-th data point.  Not
// the mean of the entire cluster.
//
// TODO would it be more efficient to calculate it in one pass instead of pre-calculating the
// mean?  Or will we always have to pre-calc to fill the cluster?
//
//   1    __  2       1    / __    \2 
// ----- \   x  - -------- |\   x  |  
// R - K /__  i          2 \/__  i /  
//                (R - K)      
//    
func variance(c cluster, measurer VectorMeasurer) float64 {
	if matrix.Equals(c.Points, c.Centroid) == true {
		return 0.0
	}

	sum := float64(0)
	denom := float64(c.Numpoints() - c.Numcentroids())

	for i := 0; i < c.Numpoints(); i++ {
		p := c.Points.GetRowVector(i)
		mu_i := c.Centroid.GetRowVector(0)
		dist := measurer.CalcDist(mu_i, p)
		sum += dist * dist
	}
	v := (1.0 / denom) * sum
	return v
}

// pointProb calculates the probability of an individual point.
//
// R = |D|
// Ri = |Dn| for the cluster contining the point x_i.
// M = # of dimensions
// V = variance of D
// mu(i) =  the coordinates of the centroid closest to the i-th data point.
//
//           /R                    \                                       
//           | (i)          1      |   /         1                     2\  
// P(x )  =  |---- * --------------|exp| - ------------  * ||x  - mu || |  
//    i      |  R    2 ___________M|   \   2 * variance       i     i   /  
//           \       |/Pi * stddev /                                       
//
//
// This is just the probability that a point exists (Ri / R) times the normal, or Gaussian,  distribution 
// for the number of dimensions.
func pointProb(R, Ri, M, V float64, point, mu *matrix.DenseMatrix, measurer VectorMeasurer) float64 {
	exists := float64(Ri / R)
	normdist := normDist(M, V, point, mu, measurer)
	prob := exists * normdist
	return prob
}

// nomrDist calculates the normal distribution for the number of dimesions in a spherical Gaussian.
//
// M = # of dimensions
// V = variance of Dn
// mean(i) =  the mean distance between all points in Dn and a centroid. 
func normDist(M, V float64, point, mean *matrix.DenseMatrix,  measurer VectorMeasurer) float64 {
	dist := measurer.CalcDist(point, mean)
	stddev := math.Sqrt(V)
	sqrt2pi := math.Sqrt(2.0 * math.Pi)
	stddevM := math.Pow(stddev, M)
	base := 1 / (sqrt2pi * stddevM)
	exp := -(1.0/(2.0 * V)) * math.Abs(dist)
	normdist := math.Pow(base, exp)

	return normdist
}

// loglikelih is the log likelihood estimate of the data taken at the maximum
// likelihood point.  1 <= n <= K
//
// D = total set of points which belong to the centroids under consideration.
// R = |D|
// R-n = |D-n| - n is the nth cluster in the model.
// M = # of dimensions
// V = unbiased variance of Dn
// K = number of centroids under consideration.
//
// All logs are log e.  The right 3 terms are summed to ts for the loop.
//
// N.B. When applying this to a model with no parent cluster as in evaluating 
// the model for D, then R = Rn and hence [[R_n logR_n - R logR] = 0.
//
// Refer to Notes on Bayesian Information Criterion Calculation equation
// for derivation.
//
//          /                   R M                               \ 
//  __ K    |                    n                       1        | 
// \        |R logR  - R logR - ---log(2pi * variance) - -(R  - 1)| 
// /__ n = 1\ n    n    n        2                       2  n     / 
//
func loglikelih(R int, c []cluster) float64 {
	ll := float64(0)

	for i := 0; i < int(len(c)); i++ {
		fRn := float64(c[i].Numpoints())
		t1 := fRn * math.Log(fRn)
		t2 :=  fRn * math.Log(float64(R))
		// This is the Bob's Your Uncle smoothing factor.  If the variance is 
		// zero , the fit can't be any better and will drive the log
		// likelihood to Infinity.
		if c[i].Variance == 0 {
			c[i].Variance = math.Nextafter(0, 1)
		} 
		t3 := ((fRn * float64(c[i].Dim())) / 2)  * math.Log((2 * math.Pi) * c[i].Variance)
		t4 := ((fRn - 1) / 2)

		ll += (t1 - t2 - t3 - t4)
	}
	return ll
}

// freeparams returns the number of free parameters in the BIC.
//
// K - number of clusters
// M - number of dimensions
//
// Since the variance is a free paramter, identical for every cluster, it
// counts as 1.
//
// (K - 1 class probabilities) + (M * K) + 1 variance estimate.
func freeparams(K, M int) int {
	return (K - 1) + (M * K) + 1
}

// calcBIC calculates the Bayesian Information Criterion or Schwarz Criterion
// 
// D = set of points
//
// R = |D|
//
// M = number of dimesions assuming spherical Gaussians
//
// p_j = number of parameters in Mj
//
// log = the natural log
//
// l(D) = the log likelihood of the data of the jth model taken at the 
// maximum likelihood point.
//
//                       p        
//              ^         j       
// BIC(M )  =  l (D)  -  -- * logR
//      j       j         2       
//
func bic(loglikelih float64, numparams, R int) (float64) {
	return loglikelih - (float64(numparams) / 2.0) * math.Log(float64(R))
}

// calcbic calculates BIC from R, M, and a slice of clusters
func calcbic(R, M int, clusters []cluster) float64 {
	ll := loglikelih(R, clusters)
	numparams := freeparams(len(clusters), M)
	return  bic(ll, numparams, R)
}

