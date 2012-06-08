package goxmeans

import (
	"testing"
	"os"
	"bufio"
	"fmt"
	"code.google.com/p/gomatrix/matrix"
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

func TestColSliceValid(t *testing.T) {
	rows := 3
	columns := 2
	mat := matrix.MakeDenseMatrix([]float64{1,2,3,4,5,6}, rows, columns)
	c := ColSlice(mat, 1)
	if len(c) != rows {
		t.Errorf("Returned slice has len=%d instead of %d.", len(c), rows)
	}
}

func TestAppendColInvalid(t *testing.T) {
	rows := 3
	columns := 2
	mat := matrix.MakeDenseMatrix([]float64{1,2,3,4,5,6}, rows, columns)
	col := []float64{1.1,2.2,3.3,4.4}
	mat, err := AppendCol(mat, col)
	if err == nil {
		t.Errorf("AppendCol err=%v", err)
	}
}

func TestAppendColValid(t *testing.T) {
	rows := 3
	columns := 2
	mat := matrix.MakeDenseMatrix([]float64{1,2,3,4,5,6}, rows, columns)
	col := []float64{1.1,2.2,3.3}
	mat, err := AppendCol(mat, col)
	if err != nil {
		t.Errorf("AppendCol err=%v", err)
	}
}

func TestFileNotExistsLoad(t *testing.T) {
	f := "filedoesnotexist"
	if _, err := Load(f); err == nil {
		t.Errorf("err == nil with file that does not exist.  err=%v.", err)
	}
}

func createtestfile(fname, record string)(int, error) {
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

	if _, err := Load(fname);  err == nil {
		t.Errorf("err: %v", err)
	}
}

func TestValidReturnLoad(t *testing.T) {
	fname := "inputvalid"
	record := fmt.Sprintf("123\t456\n789\t101")
	_, err := createtestfile(fname, record)
	if err != nil {
		t.Errorf("Could not create test file %s err=%v", err)
	}
	defer os.Remove(fname)
	
	if _, err := Load(fname); err != nil {
		t.Errorf("Load(%s) err=%v", fname, err)
	}
}