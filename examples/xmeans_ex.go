package main

import (
	"fmt"
	"goxmeans"
	"os"
	"flag"
	"runtime/pprof"
	"log"
	"strconv"
	"math"
)


var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var c = flag.Int("centroids", 1234, "number of centroids")

func main() {
	usage := "usage: xmeans_ex k kmax"
	if len(os.Args) < 3 {
		fmt.Println(usage)
		return
	}
	
	k, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Printf("Could not convert arg %s to int.\n", os.Args[1])
		return
	}
	kmax, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Printf("Could not convert arg %s to int.\n", os.Args[2])
		return
	}

	flag.Parse()
    if *cpuprofile != "" {
        f, err := os.Create(*cpuprofile)
        if err != nil {
            log.Fatal(err)
        }
		pprof.StartCPUProfile(f)
        defer pprof.StopCPUProfile()
	}

	// Load data set.
	data, err := goxmeans.Load("dataset", ",")
	if err != nil {
		fmt.Println("Load: ", err)
		return
	}
	fmt.Println("Load complete")

	// Type of data measurement between points.
	var measurer goxmeans.EuclidDist

	// How to select your initial centroids.
	var cc goxmeans.DataCentroids

	// How to select centroids for bisection.
	bisectcc := goxmeans.EllipseCentroids{0.5}

	// Initial matrix of centroids to use
	centroids := cc.ChooseCentroids(data, k)

	models, errs := goxmeans.Xmeans(data, centroids, kmax, cc, bisectcc, measurer)
	if len(errs) > 0 {
		for k, v := range errs {
			fmt.Printf("%s: %v\n", k, v)
		}
		return
	}

	// Find and display the best model
	bestbic := math.Inf(-1)
	bestidx := 0
	for i, m := range models {
		if  m.Bic > bestbic {
			bestbic = m.Bic
			bestidx = i
		}
		fmt.Printf("%d: #centroids=%d BIC=%f\n", i, m.Numcentroids(), m.Bic)
	}

	fmt.Printf("\nBest fit:[ %d: #centroids=%d BIC=%f]\n",  bestidx, models[bestidx].Numcentroids(), models[bestidx].Bic)
	bestm := models[bestidx]
	for i, c := range bestm.Clusters {
		fmt.Printf("cluster-%d: numpoints=%d  variance=%f\n", i, c.Numpoints(), c.Variance)
	}
}