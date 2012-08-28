/*
 Package goxmeans implements a library for the xmeans algorithm.
 
 See Dan Pelleg and Andrew Moore - X-means: Extending K-means with Efficient Estimation of the Number of Clusters. 

 D = the input set of points

 R = |D| the number of points in a model.

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
//	"runtime"
//	"log"
	"github.com/bobhancock/gomatrix/matrix"
	"goxmeans/matutil"
)

//var numworkers = runtime.NumCPU()
var numworkers = 1

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

type Model struct {
	bic float64
	clusters []cluster
}

// cluster models an individual cluster.
type cluster struct {
	points *matrix.DenseMatrix
	centroid  *matrix.DenseMatrix
	dim int // number of dimensions
	variance float64
}

func (c cluster) numpoints() int {
	r, _ := c.points.GetSize()
	return r
}

func (c cluster) numcentroids() int {
	r, _ := c.centroid.GetSize()
	return r
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

// DataCentroids picks k distinct points from the dataset
func (c DataCentroids) ChooseCentroids(mat *matrix.DenseMatrix, k int) (*matrix.DenseMatrix) {
	// first set up a map to keep track of which data points have already been chosen so we don't dupe
	rows, cols := mat.GetSize()
	centroids := matrix.Zeros(k, cols)
	// Can't return error of it does not implement CentroidChooser
	// TODO How to deal with this? Should CentroidChooser return err?
	if k > rows {
//		return centroids, errors.New("ChooseCentroids: Can't compute more centroids than data points!")
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

// EllipseCentroids lays out the centroids along an elipse inscribed within the boundaries of the dataset
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

// Models runs k-means for k lower bound to k upper bound on a data set.
// Once the k centroids have converged each cluster is bisected and the BIC
// of the orginal cluster (parent = a model with one centroid) to the 
// the bisected model which consists of two centroids and whichever is greater
// is committed to the set of clusters for this larger model k.
// 
// TODO How many bisections should be tried?
// TODO Parallelize bisection of clusters
// TODO Allow spearate CentroidChoosers for parent and child models.
//
// Why do some models have zero clusters?
// With a large number of centroids why is variance inf?
// Use a different centroid chooser for main process and bisection?
func Models(datapoints *matrix.DenseMatrix, klow, kup int, cc CentroidChooser, measurer matutil.VectorMeasurer) ([]Model, map[string]error) {
	R, M := datapoints.GetSize()
	models := make([]Model, kup)
	errs := make(map[string]error)

	fmt.Printf("=======Models klow=%d kup=%d\n", klow, kup)
	for k := klow; k <= kup; k++ {
		fmt.Printf("\nKloop Top: k=%d\n", k)
		bufclusters := make([]cluster, 0)

		fmt.Printf("Kloop: Before Kmeansp k=%d\n", k)
		clusters, err := Kmeansp(datapoints, k, cc, measurer)
		if err != nil {
			errs[strconv.Itoa(k)] = err
		}
		fmt.Printf("Kloop: After Kmeansp len(clusters)=%d\n", len(clusters))

		
		// clusters is a []cluster. 
		// bisect each clusters and see if you can get a better BIC
		// You are comparing ModelA with the one centroid to ModelA_1
		// bisected with two centroids.
		for j, clust := range clusters {
			fmt.Printf("Biloop: Top j=%d\n", j)
			clust.variance = variance(clust, measurer)

			parentbic := calcbic(clust.numpoints(), M, []cluster{clust})
			fmt.Printf("Biloop: parentbic=%f\n", parentbic)

			fmt.Printf("Biloop: Before Kmeansp: j=%d clust.points=%v  centroid=%v  clust.variance=%f\n", j, clust.points, clust.centroid, clust.variance)
			biclusters, berr := Kmeansp(clust.points, 2, cc, measurer)
			if berr != nil {
				idx := strconv.Itoa(k)+"."+strconv.Itoa(j)
				errs[idx] = berr
				continue
			}
			for _, biclust := range biclusters {
			    biclust.variance = variance(biclust, measurer)
			}

			fmt.Printf("Biloop: After Kmeansp: j=%d len(biclusters)=%d\n", j, len(biclusters))
			for i, clust := range biclusters {
				fmt.Printf("Biloop: After Kmeansp: bicluster[%d]=%v\n variance=%f\n", i, clust.points, clust.variance)
			}
		
			//Compare the BIC of this model to the parent
			childbic := calcbic(clust.numpoints(), M, biclusters)
			
			fmt.Printf("Biloop: j=%d parentbic=%f childbic=%f\n", j, parentbic, childbic)
			// Whichever model is better goes into the array of clusters
			// for this model k.
			if parentbic >= childbic { 
				bufclusters = append(bufclusters, clust)
			}

			if childbic > parentbic {
					bufclusters = append(bufclusters, biclusters...)
			} 
		}
		// Add this model to the model slice
		modelbic := calcbic(R, M, bufclusters) //<==ERROR
		m := Model{modelbic, bufclusters}
		if k == klow {
			models[0] = m
		} else {
			models = append(models, m)
		}
	}
	return models, errs
}
	
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
// centroids is K x M matrix that cotains the coordinates for the centroids.
// The centroids are indexed by the 0 based rows of this matrix.
//  ____      _________
//  | 12.29   32.94 ... | <-- The coordinates for centroid 0
//  | 4.6     29.22 ... | <-- The coordinates for centroid 1
//  |_____    __________|
// 
//
// CentPointDist is ax R x 2 matrix.  The rows have a 1:1 relationship to 
// the rows in datapoints.  Column 0 contains the row number in centroids
// that corresponds to the centroid for the datapoint in row i of this matrix.
// Column 1 contains (x_i - mu(i))^2.
//  ____      _______
//  | 3        38.01 | <-- Centroid 3, squared error for the coordinates in row 0 of datapoints
//  | 1        23 .21| <-- Centroid 1, squared error for the coordinates in row 1 of datapoints
//  | 0        14.12 | <-- Centroid 0, squared error for the coordinates in row 2 of datapoints
//  _____     _______
//
func Kmeansp(datapoints *matrix.DenseMatrix, k int, cc CentroidChooser, measurer matutil.VectorMeasurer) ([]cluster, error) {
/*  datapoints				  CentPoinDist            centroids				  
                                 ________________
   ____	  ____				  __|__	  ______	 |	  ____	___________	  
   | ...	 |				 |	...			|	 V	 | ...		       |	  
   | 3.0  5.1| <-- row i --> |	3	  32.12 |  row 3 | 3	 38.1, ... |		  
   |____  ___|				 |____	  ______|	     |___	__________ |			  
*/
/*	fp, _ := os.Create("/var/tmp/km.log")
	w := io.Writer(fp)
	log.SetOutput(w)
*/

	centroids := cc.ChooseCentroids(datapoints, k)
	
	numRows, M := datapoints.GetSize()
	CentPointDist := matrix.Zeros(numRows, M)

	clusterChanged := true
	var clusters []cluster

	for ; clusterChanged == true ; {
		clusterChanged = false
		clusters = make([]cluster, 0)

		jobs := make(chan PairPointCentroidJob, numworkers)
		results := make(chan PairPointCentroidResult, minimum(1024, numRows))
		done := make(chan int, numworkers)

		// Pair each point with its closest centroid.
		go addPairPointCentroidJobs(jobs, datapoints, centroids, CentPointDist, measurer, results)
		for i := 0; i < numworkers; i++ {
			go doPairPointCentroidJobs(done, jobs)
		}
		go awaitPairPointCentroidCompletion(done, results)

		clusterChanged = assessClusters(CentPointDist, results) // This blocks so that all the results can be processed
//		fmt.Printf("kmeansp: clusterChanged=%v\n", clusterChanged)
		
		// Now that you have each data point grouped with a centroid,
		for idx, cent := 0, 0; cent < k; cent++ {
			// Select all the rows in CentPointDist whose first col value == cent.
			// Get the corresponding row vector from datapoints and place it in pointsInCluster.
			//fmt.Printf("kmeansp: cent=%d: clusterAss=%v\n", cent, CentPointDist)
			matches, err :=	CentPointDist.FiltColMap(float64(cent), float64(cent), 0)  

			// matches - a map[int]float64 where the key is the row number in source 
			//matrix  and the value is the value in the column of the source matrix 
			//specified by col.  Here the value is the centroid paired with the point.
			if err != nil {
				return clusters, err
			}
			// It is possible that some centroids could not have any points, so there 
			// may not be any matches.
			if len(matches) == 0 {
				continue
			}

			pointsInCluster := matrix.Zeros(len(matches), M) 
			i := 0
			for rownum, _ := range matches {
				//fmt.Printf("kmeansp: cent=%d centroid=%f rownum=%d\n", cent, centroid, rownum)
				pointsInCluster.Set(i, 0, datapoints.Get(int(rownum), 0))
				pointsInCluster.Set(i, 1, datapoints.Get(int(rownum), 1))
				i++
			}
//			fmt.Printf("kmeansp: cent=%d pointsInCluster=%v\n", cent, pointsInCluster)
			// pointsInCluster now contains all the data points for the current 
			// centroid.  The mean of the coordinates for this cluster becomes 
			// the new centroid for this cluster.
			mean := pointsInCluster.MeanCols()
			centroids.SetRowVector(mean, cent)
//			fmt.Printf("kmeansp: cent=%d centroids=%v\n", cent, centroids)

			clust := cluster{pointsInCluster, mean, M, 0}
			clust.variance = variance(clust, measurer)
			clusters = append(clusters, clust)
			idx++
		}
	}
	//return centroids, CentPointDist, nil
	return clusters, nil
}

// CentroidPoint stores the row number in the centroids matrix and
// the distance squared between the centroid and the point.
type CentroidPoint struct {
	centroidRunNum float64
	distPointToCentroidSq float64
}

// PairPointCentroidJobs stores the elements that defines the job that pairs a 
// point (i.e., a data point) with a centroid.
type PairPointCentroidJob struct {
	point, centroids, CentPointDist *matrix.DenseMatrix
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
	CentPointDist *matrix.DenseMatrix, measurer matutil.VectorMeasurer, results chan<- PairPointCentroidResult) {
	numRows, _ := datapoints.GetSize()
    for i := 0; i < numRows; i++ { 
		point := datapoints.GetRowVector(i)
//		fmt.Printf("398: i=%d numRows=%d\n",i,numRows)
		jobs <- PairPointCentroidJob{point, centroids, CentPointDist, results, i, measurer}
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
		//fmt.Printf("squaredErr=%f\n", squaredErr)
	}	
	job.results <- PairPointCentroidResult{centroidRowNum, squaredErr, job.rowNum, err}
}

// variance is the maximum likelihood estimate (MLE) for the variance, under
// the identical spherical Gaussian assumption.
//
// points = an R x M matrix of all data point coordinates.
//
// CentPointDist =  R x 2 matrix.  Column 0 contains the index {0...K} of
// a centroid.  Column 1 contains (datapoint_i - mu(i))^2 
// 
// centroids =  K x M+1 matrix.  Column 0 continas the centroid index {0...K}.
// Columns 1...M contain the centroid coordinates.  (See Kmeansp() for an example.)
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
func variance(c cluster, measurer matutil.VectorMeasurer) float64 {
	if matrix.Equals(c.points, c.centroid) == true {
		return 0.0
	}

	sum := float64(0)
	denom := float64(c.numpoints() - c.numcentroids())

	for i := 0; i < c.numpoints(); i++ {
		p := c.points.GetRowVector(i)
		mu_i := c.centroid.GetRowVector(0)
		dist := measurer.CalcDist(mu_i, p)
		sum += math.Pow(dist, 2) 
	}
	//fmt.Printf("denom=%f sum=%f\n", denom, sum)
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
func pointProb(R, Ri, M, V float64, point, mu *matrix.DenseMatrix, measurer matutil.VectorMeasurer) float64 {
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
func normDist(M, V float64, point, mean *matrix.DenseMatrix,  measurer matutil.VectorMeasurer) float64 {
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
// the model for D, then R = Rn and [[R_n logR_n - R logR] = 0.
//
// Refer to Notes on Bayesian Information Criterion Calculation equation.
//
//          /                   R M                               \ 
//  __ K    |                    n                       1        | 
// \        |R logR  - R logR - ---log(2pi * variance) - -(R  - 1)| 
// /__ n = 1\ n    n    n        2                       2  n     / 
//
func loglikelih(R int, c []cluster) float64 {
	ll := float64(0)
	
	for i := 0; i < int(len(c)); i++ {
		fRn := float64(c[i].numpoints())
//		fmt.Printf("fRn=%f\n", fRn)

		t1 := fRn * math.Log(fRn)
//		fmt.Printf("t1=%f\n", t1)

		t2 :=  fRn * math.Log(float64(R))
//		fmt.Printf("t2=%f\n", t2)

//		fmt.Printf("t3: c[%d].dim=%d  c[%d].variance=%f\n", i,c[i].dim, i, c[i].variance)
		t3 := ((fRn * float64(c[i].dim)) / 2)  * math.Log((2 * math.Pi) * c[i].variance)
//		fmt.Printf("t3=%f\n", t3)

		t4 := ((fRn - 1) / 2)
//		fmt.Printf("t4=%f\n", t4)

		ll += (t1 - t2 - t3 - t4)
//		fmt.Printf("loglikelih: ll=%f\n", ll)
	}
	return ll
}

// As stated in the Moore-Pellig paper.
// R            R  * M                R  - K                     
//  n            n                     n                         
// --log(2Pi) - ------log(variance) - ------ + R logr{n} - R log 
//  2              2                     2      n           n   n
//
func loglikelihMoore(R, K int, c []cluster) float64 {
	ll := 0.0

	for _, clust := range c {
		Rn := float64(clust.numpoints())
		t1 := (Rn / 2) * math.Log(2 * math.Pi)
		t2 := ((Rn * float64(clust.dim)) / 2) * math.Log(clust.variance)
		t3 := (Rn - float64(K)) / 2
		t4 := Rn * math.Log(Rn)
		t5 := Rn * math.Log(float64(R))

		ll += -t1 - t2 - t3 + t4 - t5
		fmt.Printf("loglikelih: ll=%f\n", ll)
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
	return loglikelih - (float64(numparams) / 2.0) - math.Log(float64(R))
}

// calcbic calculate BIC from R, M, and a slice of clusters
func calcbic(R, M int, clusters []cluster) float64 {
	ll := loglikelih(R, clusters)
	numparams := freeparams(len(clusters), M)
	return  bic(ll, numparams, R)
}
