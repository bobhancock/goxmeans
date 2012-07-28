package goxmeans

import (
	"bufio"
//	"code.google.com/p/gomatrix/matrix"
	"fmt"
	"os"
	"testing"
	"github.com/bobhancock/gomatrix/matrix"
	"goxmeans/matutil"
)

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
	choosers := []CentroidChooser{RandCentroids{}, DataCentroids{}, EllipseCentroids{0.5}}
	for _, cc := range choosers{
		centroids := cc.ChooseCentroids(mat, k)

		r, c := centroids.GetSize()
		if r != k || c != cols {
			t.Errorf("Returned centroid was %dx%d instead of %dx%d", r, c, rows, cols)
		}
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
	dataPoints, err := Load("./testSetSmall.txt")
	if err != nil {
		t.Errorf("Load returned: %v", err)
		return
	}
	
	var ed matutil.EuclidDist
	var cc RandCentroids
	//centroidsdata := []float64{1.5,1.5,2,2,3,3,0.9,0,9}
	//centroids := matrix.MakeDenseMatrix(centroidsdata, 4,2)

	centroidMeans, centroidSqDist, err := Kmeansp(dataPoints, 4, cc, ed)
	if err != nil {
		t.Errorf("Kmeans returned: %v", err)
		return
	}

	if 	a, b := centroidMeans.GetSize(); a == 0 || b == 0 {
		t.Errorf("Kmeans centroidMeans is of size %d, %d.", a,b)
	}

	if c, d := centroidSqDist.GetSize(); c == 0 || d == 0 {
		t.Errorf("Kmeans centroidSqDist is of size %d, %d.", c,d)
	}
}
   
func TestAddPairPointToCentroidJob(t *testing.T) {
	r := 4
	c := 2
	jobs := make(chan PairPointCentroidJob, r)
	results := make(chan PairPointCentroidResult, minimum(1024, r))
	dataPoints := matrix.Zeros(r, c)
	centroidSqDist := matrix.Zeros(r, c)
	centroids := matrix.Zeros(r, c)

	var ed matutil.EuclidDist
	
	go addPairPointCentroidJobs(jobs, dataPoints, centroids, centroidSqDist,ed ,results)
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
	centroidSqDist := matrix.Zeros(r, c)
	centroids := matrix.Zeros(r, c)

	done := make(chan int)
	jobs := make(chan PairPointCentroidJob, r)
	results := make(chan PairPointCentroidResult, minimum(1024, r))

	var md matutil.ManhattanDist

	go addPairPointCentroidJobs(jobs, dataPoints, centroids, centroidSqDist, md, results)

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

func TestProcessPairPointToCentroidResults(t *testing.T) {
	r := 4
	c := 2
	dataPoints := matrix.Zeros(r, c)
	centroidSqDist := matrix.Zeros(r, c)
	centroids := matrix.Zeros(r, c)

	done := make(chan int)
	jobs := make(chan PairPointCentroidJob, r)
	results := make(chan PairPointCentroidResult, minimum(1024, r))

	var md matutil.ManhattanDist
	go addPairPointCentroidJobs(jobs, dataPoints,  centroids, centroidSqDist, md, results)

	for i := 0; i < r; i++ {
		go doPairPointCentroidJobs(done, jobs)
	}
	go awaitPairPointCentroidCompletion(done, results)

    //TODO check deterministic results of centroidDistSq
     processPairPointToCentroidResults(centroidSqDist, results)

}


func TestKmeansbi(t *testing.T) {
	dataPoints, err := Load("./testSetSmall.txt")
	if err != nil {
		t.Errorf("Load returned: %v", err)
		return
	}
	
	var ed matutil.EuclidDist
	var cc RandCentroids

	matCentroidlist, clusterAssignment, err := Kmeansp(dataPoints, 4, cc, ed)
	if err != nil {
		t.Errorf("Kmeans returned: %v", err)
		return
	}

	if 	a, b := matCentroidlist.GetSize(); a == 0 || b == 0 {
		t.Errorf("Kmeans centroidMeans is of size %d, %d.", a,b)
	}

	if c, d := clusterAssignment.GetSize(); c == 0 || d == 0 {
		t.Errorf("Kmeans centroidSqDist is of size %d, %d.", c,d)
	}
}
   