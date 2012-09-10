 package goxmeans

import (
	"bufio"
	"fmt"
	"os"
	"math"
	"testing"
	"github.com/bobhancock/gomatrix/matrix"
)

var DATAPOINTS = matrix.MakeDenseMatrix([]float64{3.0,2.0,
	-3.0,2.0,
	0.355083,-3.376585,   
	1.852435,3.547351,
	-2.078973,2.552013,
	-0.993756,-0.884433,
	2.682252,4.007573,
	-3.087776,2.878713,
	-1.565978,-1.256985,
	2.441611,0.444826,
	10.29,20.6594,
	12.93,23.3988,
	120.1, 202.18}, 13, 2)

var CENTROIDS = matrix.MakeDenseMatrix([]float64{ 4.5,   11.3,
    6.1,  12.0,
    12.1,   9.6}, 3, 2)

var DATAPOINTS_D = matrix.MakeDenseMatrix( []float64{2,3, 3,2, 3,4, 4,3, 8,7, 9,6, 9,8, 10,7}, 8,2)
var CENTROIDS_D = matrix.MakeDenseMatrix([]float64{6,7}, 1,2)

var DATAPOINTS_D0 = matrix.MakeDenseMatrix( []float64{2,3, 3,2, 3,4, 4,3}, 4,2)
var CENTROID_D0 =  matrix.MakeDenseMatrix([]float64{3,3}, 1,2) 

var DATAPOINTS_D1 = matrix.MakeDenseMatrix( []float64{8,7, 9,6, 9,8, 10,7}, 4,2)
var CENTROID_D1 =  matrix.MakeDenseMatrix([]float64{9,7}, 1,2) 

func makeCentPointDist(datapoints, centroids *matrix.DenseMatrix) *matrix.DenseMatrix {
	r, c := datapoints.GetSize()
	CentPointDist := matrix.Zeros(r, c)

	done := make(chan int)
	jobs := make(chan PairPointCentroidJob, r)
	results := make(chan PairPointCentroidResult, minimum(1024, r))
	var ed EuclidDist

	go addPairPointCentroidJobs(jobs, datapoints, centroids, ed, results)
		
	for i := 0; i < r; i++ {
		go doPairPointCentroidJobs(done, jobs)
	}
	go awaitPairPointCentroidCompletion(done, results)
	
	clusterChanged := assessClusters(CentPointDist, results)
	
	if clusterChanged == true || clusterChanged == false {
	}
	//fmt.Printf("clusterchanged=%v\n", clusterChanged)
	return CentPointDist
}

func TestAtof64Invalid(t *testing.T) {
	s := "xyz"
	if _, err := Atof64(s); err == nil {
		t.Errorf("err == nil with invalid input %s.", s)
	}
}

func TestAtof64Valid(t *testing.T) {
	s := "1234.5678"
	if f64, err := Atof64(s); err != nil {
		t.Errorf("err != nil with input %s. Returned f64=%f,err= %v.", s, f64, err)
	}
}

func TestFileNotExistsLoad(t *testing.T) {
	f := "filedoesnotexist"
	if _, err := Load(f); err == nil {
		t.Errorf("err == nil with file that does not exist.  err=%v.", err)
	}
}

func createtestfile(fname, record string) (int, error) {
	fp, err := os.Create(fname)
	if err != nil {
		return 0, err
	}
	defer fp.Close()

	w := bufio.NewWriter(fp)
	i, err := w.WriteString(record)
	if err != nil {
		return i, err
	}
	w.Flush()

	return i, err
}

// Does the input line contain < 2 elements
func TestInputInvalid(t *testing.T) {
	fname := "inputinvalid"
	_, err := createtestfile(fname, "123\n")
	if err != nil {
		t.Errorf("Could not create test file. err=%v", err)
	}
	defer os.Remove(fname)

	if _, err := Load(fname); err == nil {
		t.Errorf("err: %v", err)
	}
}

func TestValidReturnLoad(t *testing.T) {
	fname := "inputvalid"
	record := fmt.Sprintf("123\t456\n789\t101")
	_, err := createtestfile(fname, record)
	if err != nil {
		t.Errorf("Could not create test file %s err=%v", fname, err)
	}
	defer os.Remove(fname)

	if _, err := Load(fname); err != nil {
		t.Errorf("Load(%s) err=%v", fname, err)
	}
}

func TestRandCentroids(t *testing.T) {
	rows := 3
	cols := 3
	k := 2
	data := []float64{1, 2.0, 3, -4.945, 5, -6.1, 7, 8, 9}
	mat := matrix.MakeDenseMatrix(data, rows, cols)
	choosers := []CentroidChooser{randCentroids{}, DataCentroids{}, EllipseCentroids{0.5}}
	for _, cc := range choosers{
		centroids := cc.ChooseCentroids(mat, k)

		r, c := centroids.GetSize()
		if r != k || c != cols {
			t.Errorf("Returned centroid was %dx%d instead of %dx%d", r, c, rows, cols)
		}
	}

	// This section of the test places ellipse centroids at fraction 1 on a 2x2 box,
	// and asserts that there is a distance of 2 between them (ie, they are diametrically opposite)
	data2 := []float64{1.0, 1.0, -1.0, -1.0}
	mat2 := matrix.MakeDenseMatrix(data2, 2, 2)
	newCentroids := EllipseCentroids{1.0}.ChooseCentroids(mat2, 2)
	dist := EuclidDist{}.CalcDist(newCentroids.GetRowVector(0), newCentroids.GetRowVector(1))
	expectedEd := 2.0 //expected value
	epsilon := .000001
	diff := math.Abs(dist - expectedEd)
	if diff > epsilon {
		t.Errorf("EuclidDist: excpected %f but received %f.  The difference %f exceeds epsilon %f", expectedEd, dist, diff, epsilon)
	}

}



func TestComputeCentroid(t *testing.T) {
	empty := matrix.Zeros(0, 0)
	_, err := ComputeCentroid(empty)
	if err == nil {
		t.Errorf("Did not raise error on empty matrix")
	}
	twoByTwo := matrix.Ones(2, 2)
	centr, err := ComputeCentroid(twoByTwo)
	if err != nil {
		t.Errorf("Could not compute centroid, err=%v", err)
	}
	expected := matrix.MakeDenseMatrix([]float64{1.0, 1.0}, 1, 2)
	if !matrix.Equals(centr, expected) {
		t.Errorf("Incorrect centroid: was %v, should have been %v", expected, centr)
	}
	twoByTwo.Set(0, 0, 3.0)
	expected.Set(0, 0, 2.0)
	centr, err = ComputeCentroid(twoByTwo)
	if err != nil {
		t.Errorf("Could not compute centroid, err=%v", err)
	}
	if !matrix.Equals(centr, expected) {
		t.Errorf("Incorrect centroid: was %v, should have been %v", expected, centr)
	}
}


func TestKmeansp(t *testing.T) {
//	dataPoints, err := Load("./testSetSmall.txt")
//	if err != nil {
//		t.Errorf("Load returned: %v", err)
//		return
//	}
	
	var ed EuclidDist
//	var cc randCentroids
	var cc DataCentroids
//	cc := EllipseCentroids{0.1}

	datapoints := matrix.MakeDenseMatrix( []float64{2,3, 3,2, 3,4, 4,3, 8,7, 9,6, 9,8, 10,7, 3, 5}, 9,2)
	fmt.Printf("datapoints=%v\n", datapoints)
	clusters, err := kmeansp(datapoints, 2, cc, ed)
	if err != nil {
		t.Errorf("Kmeans returned: %v", err)
		return
	}

	if len(clusters) != 2 {
		t.Errorf("TestKemansp: expected 2 clusters and received %d.", len(clusters))
	}

	variances := make([]float64, 2)
	for i, clust := range clusters {
		variances[i] = clust.variance
	}

	//TODO Test that the cluster members and the variances are correct.
	//N.B. The order of returned clusters is not deterministic.
/*	E := 1.8
	v := clusters[0].variance 
	epsilon := .000001
	na := math.Nextafter(E, E + 1) 
	diff := math.Abs(v - na) 

	
	if diff > epsilon {
		t.Errorf("TestKmeansp: expected variance of %f but received %f.  The difference %f exceeds epsilon %f", E, v, diff, epsilon)
	}
*/
/*
	y := clusters[0].centroid.Get(0,1)
	expect = 3.4
	if y != expect {
		t.Error("TestKmeansp: first centroid y coordinate is %f instead of %f.", x, expect)
	}

	
	x = clusters[1].centroid.Get(0,0)
	expect = 9.0
	if x != expect {
		t.Error("TestKmeansp: second centroid x coordinate is %f instead of %f.", x, expect)
	}

	y = clusters[1].centroid.Get(0,1)
	expect = 7.0
	if y != expect {
		t.Error("TestKmeansp: second centroid y coordinate is %f instead of %f.", x, expect)
	}
*/
	for i, clust := range clusters {
		fmt.Printf("%d: points=%v\n",i, clust.points)
		fmt.Printf("%d: centroid=%v\n", i, clust.centroid)
		fmt.Printf("%d: variance:%f\n\n", i, clust.variance)
	}
}
   
func TestAddPairPointToCentroidJob(t *testing.T) {
	r := 4
	c := 2
	jobs := make(chan PairPointCentroidJob, r)
	results := make(chan PairPointCentroidResult, minimum(1024, r))
	dataPoints := matrix.Zeros(r, c)
//	centroidSqDist := matrix.Zeros(r, c)
	centroids := matrix.Zeros(r, c)

	var ed EuclidDist
	
	go addPairPointCentroidJobs(jobs, dataPoints, centroids, ed ,results)
	i := 0
	for ; i < r; i++ {
        <-jobs 
		//fmt.Printf("Drained %d\n", i)
    }

	if i  != r {
		t.Errorf("addPairPointToCentroidJobs number of jobs=%d.  Should be %d", i, r)
	}
}
	
func TestDoPairPointCentroidJobs(t *testing.T) {
	r := 4
	c := 2
	dataPoints := matrix.Zeros(r, c)
//	centroidSqDist := matrix.Zeros(r, c)
	centroids := matrix.Zeros(r, c)

	done := make(chan int)
	jobs := make(chan PairPointCentroidJob, r)
	results := make(chan PairPointCentroidResult, minimum(1024, r))

	var md ManhattanDist

	go addPairPointCentroidJobs(jobs, dataPoints, centroids,  md, results)

	for i := 0; i < r; i++ {
		go doPairPointCentroidJobs(done, jobs)
	}

	j := 0
	for ; j < r; j++ {
        <- done
    }

	if j  != r {
		t.Errorf("doPairPointToCentroidJobs jobs processed=%d.  Should be %d", j, r)
	}
}

func TestAssessClusters(t *testing.T) {
	r, c := DATAPOINTS.GetSize()
	CentPointDist := matrix.Zeros(r, c)

	done := make(chan int)
	jobs := make(chan PairPointCentroidJob, r)
	results := make(chan PairPointCentroidResult, minimum(1024, r))

	var md ManhattanDist
	go addPairPointCentroidJobs(jobs, DATAPOINTS, CENTROIDS,  md, results)

	for i := 0; i < r; i++ {
		go doPairPointCentroidJobs(done, jobs)
	}
	go awaitPairPointCentroidCompletion(done, results)

    clusterChanged := assessClusters(CentPointDist, results)
	if clusterChanged != true {
		t.Errorf("TestAssessClusters: clusterChanged=%b and should be true.", clusterChanged)
	}

	if CentPointDist.Get(9, 0) != 0 || CentPointDist.Get(10, 0) != 1 {
		t.Errorf("TestAssessClusters: rows 9 and 10 should have 0 and 1 in column 0, but received %v", CentPointDist)
	}
}

func TestPointProb(t *testing.T) {
	R := 10010.0
	Ri := 100.0
	M := 2.0
	V := 20.000000

	point := matrix.MakeDenseMatrix([]float64{5, 7},
		1,2)

	mean := matrix.MakeDenseMatrix([]float64{6, 8},
		1,2)

	var ed EuclidDist 

	//	pointProb(R, Ri, M int, V float64, point, mean *matrix.DenseMatrix, measurer VectorMeasurer) (float64, error) 
	pp := pointProb(R, Ri, M, V, point, mean, ed)

	E :=  0.011473
	epsilon := .000001
	na := math.Nextafter(E, E + 1) 
	diff := math.Abs(pp - na) 

	if diff > epsilon {
		t.Errorf("TestPointProb: expected %f but received %f.  The difference %f exceeds epsilon %f", E, pp, diff, epsilon)
	}
}

func TestFreeparams(t *testing.T) {
	K := 6
	M := 3

	r := freeparams(K, M)
	if r != 24 {
		t.Errorf("TestFreeparams: Expected 24 but received %f.", r)
	}
}

func TestVariance(t *testing.T) {
	var ed EuclidDist
	_, dim := DATAPOINTS_D.GetSize()
	// Model D
	c := cluster{DATAPOINTS_D, CENTROIDS_D, dim, 0}
	v := variance(c, ed)
	
	E :=  20.571429
	epsilon := .000001
	na := math.Nextafter(E, E + 1) 
	diff := math.Abs(v - na) 

	if diff > epsilon {
		t.Errorf("TestVariance: for model D excpected %f but received %f.  The difference %f exceeds epsilon %f.", E, v, diff, epsilon)
	}

	// Variance a cluster with a perfectly centered centroids
	_, dim0 := DATAPOINTS_D0.GetSize()
	c0 := cluster{DATAPOINTS_D0, CENTROID_D0, dim0, 0}
	v0 := variance(c0, ed)
	
	E = 1.333333
	na = math.Nextafter(E, E + 1) 
	diff = math.Abs(v0 - na) 

	if diff > epsilon {
		t.Errorf("TestVariance: for model D excpected %f but received %f.  The difference %f exceeds epsilon %f.", E, v0, diff, epsilon)
	}
}


func TestLogLikelih(t *testing.T) {
	// Model D - one cluster
	R, M := DATAPOINTS_D.GetSize()
	var ed EuclidDist

	cd := cluster{DATAPOINTS_D, CENTROIDS_D, M, 0}
	vard := variance(cd, ed)
	cd.variance = vard

	cslice := make([]cluster, 1)
	cslice[0] = cd

	ll := loglikelih(R, cslice)

	epsilon := .000001
	E := -42.394242
	na := math.Nextafter(E, E + 1) 
	diff := math.Abs(ll - na) 

	if diff > epsilon {
		t.Errorf("TestLoglikeli: For model D expected %f but received %f.  The difference %f exceeds epsilon %f", E, ll, diff, epsilon)
	}

	// Model Dn - two clusters
	c0 := cluster{DATAPOINTS_D0, CENTROID_D0, M, 0}
	v0 := variance(c0, ed)
	c0.variance = v0


	c1 := cluster{DATAPOINTS_D1, CENTROID_D1, M, 0}
	v1 := variance(c1, ed)
	c1.variance = v1

	cslicen := []cluster{c0, c1}

	ll_n := loglikelih(R, cslicen)

	E = -25.549651
	na = math.Nextafter(E, E + 1) 
	diff = math.Abs(ll_n - na) 

	if diff > epsilon {
		t.Errorf("TestLoglikeli: For model Dn expected %f but received %f.  The difference %f exceeds epsilon %f", E, ll_n, diff, epsilon)
	}
}


// Create two tight clusters and test the scores for a model with 1 centroid 
// that is equidistant between the two and a model with 2 centroids where 
// the centroids are in the center of each cluster.
// 
// The BIC of the second should always be better.
//
//Model 1
//                                     *
//                                  *     *
//                                     *
//                        +
//
//           *
//        *     *
//           *
//
//Model 2
//                                     *
//                                  *  +  *
//                                     *
//                        
//
//           *
//        *  +  *
//           *
//
func TestBic(t *testing.T) {
	// Model D - 1 cluster
	R, M := DATAPOINTS_D.GetSize()
	K, _ := CENTROIDS_D.GetSize()
	numparams := freeparams(K, M)
	var ed EuclidDist

	c := cluster{DATAPOINTS_D, CENTROIDS_D, M, 0}
	vard := variance(c, ed)
	c.variance = vard

	cslice := []cluster{c}

	lld := loglikelih(R, cslice)

	bic1 := bic(lld, numparams, R)
//	fmt.Printf("bic1=%f\n", bic1)
	
	// Model 2
	K = 1
	numparamsn := freeparams(K, M)

	c0:= cluster{DATAPOINTS_D0, CENTROID_D0, M, 0}
	var0 := variance(c0, ed)
	c0.variance = var0

	c1:= cluster{DATAPOINTS_D1, CENTROID_D1, M, 0}
	var1 := variance(c1, ed)
	c1.variance = var1

	cslicen := []cluster{c0, c1}

	loglikehn := loglikelih(R, cslicen)
//	fmt.Printf("loglikelihood2 = %f\n", loglikeh2)

	bic2 := bic(loglikehn, numparamsn, R)
//	fmt.Printf("bic2=%f\n", bic2)

	if bic1 >= bic2 {
		t.Errorf("TestBicComp: bic2 should be greater than bic1, but received bic1=%f and bic2=%f", bic1, bic2)
	}
}

func TestCalcbic(t *testing.T) {
	var ed EuclidDist
	R, M := DATAPOINTS_D.GetSize()
	
	c := cluster{DATAPOINTS_D, CENTROIDS_D, M, 0}
	vard := variance(c, ed)
	c.variance = vard
	cslice := []cluster{c}

	bic := calcbic(R, M, cslice)

	epsilon := .000001
	E := -45.513404
	na := math.Nextafter(E, E + 1) 
	diff := math.Abs(bic - na) 

	if diff > epsilon {
		t.Errorf("TestCalcbic: Expected %f but received %f.  The difference %f exceeds epsilon %f", E, bic, diff, epsilon)
	}
} 

func TestModels(t *testing.T) {
	var ed EuclidDist
	bisectcc := EllipseCentroids{0.5}
	var cc DataCentroids
//	var cc randCentroids
	klow := 2
	kup := 3
	models, errs := Models(DATAPOINTS_D, klow, kup, cc, bisectcc, ed)
	fmt.Printf("============Test\n")
	for i := 0; i < len(models); i++ {
		fmt.Printf("\nModel i=%d numclusters=%d bic=%f\n", i, len(models[i].clusters), models[i].bic)
		for j := 0; j < len(models[i].clusters); j++ {
			fmt.Printf("cluster %d\n", j)
			fmt.Printf("\tpoints=%v\n", models[i].clusters[j].points)
			fmt.Printf("\tcentroid=%v\n", models[i].clusters[j].centroid)
			fmt.Printf("\tdim=%v\n", models[i].clusters[j].dim)
			fmt.Printf("\tvariance=%v\n", models[i].clusters[j].variance)
		}
	}

	fmt.Printf("\nerrs: %v\n", errs)
	fmt.Printf("models=%v\n", models)
}

func TestZarc(t *testing.T) {
	var ed  EuclidDist
	points := DATAPOINTS_D
	centroid := matrix.MakeDenseMatrix([]float64{6,7}, 1,2)
	c0 := cluster{points, centroid, 2, 0}
//	fmt.Printf("c0.points=%v\n", c0.points)
	c0.variance = variance(c0, ed)
	fmt.Printf("variance=%f\n", c0.variance)

	clusters := []cluster{c0}
	R, _ := points.GetSize()
	ll := loglikelih(R, clusters)
	fmt.Printf("\nll=%f\n", ll)

//	bic := calcbic(R, 2, clusters)
//	fmt.Printf("bic=%f\n", bic)
}

func TestZarcEllipse(t *testing.T) {
	ec := EllipseCentroids{0.1}
	centroids := ec.ChooseCentroids(DATAPOINTS_D, 2)
	fmt.Printf("centroids=%v\n", centroids)
	fmt.Printf("row1=%v\n", centroids.GetRowVector(1))
}
