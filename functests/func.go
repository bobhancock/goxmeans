package main

import (
	"fmt"
	"goxmeans"
	"os"
	"flag"
	"runtime/pprof"
	"log"
	"strconv"
)


var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
//var memprofile = flag.String("memprofile", "", "write memory profile to this file")
//var klower = flag.Int("klower", "", "klower bound"
var c = flag.Int("centroids", 1234, "number of centroids")



func main() {
	usage := "usage: func numcentroids"
	if len(os.Args) < 1 {
		fmt.Println(usage)
		return
	}

	k, _ := strconv.Atoi(os.Args[1])

   flag.Parse()
    if *cpuprofile != "" {
        f, err := os.Create(*cpuprofile)
        if err != nil {
            log.Fatal(err)
        }
		pprof.StartCPUProfile(f)
        defer pprof.StopCPUProfile()
	}

	data, err := goxmeans.Load("randints")
	if err != nil {
		fmt.Println("Load: ", err)
		return
	}
	fmt.Println("Load complete")
	fmt.Println(data.GetSize())

	var measurer goxmeans.EuclidDist
	var cc goxmeans.DataCentroids
	//cc := goxmeans.EllipseCentroids{0.5}
//	bisectcc := goxmeans.EllipseCentroids{0.5}

//	model, errs := goxmeans.Xmean(data, k, cc, bisectcc, measurer)
	model, errs := goxmeans.Xmean(data, k, cc, cc, measurer)
	if len(errs) > 0 {
		for k, v := range errs {
			fmt.Printf("%s: %v\n", k, v)
		}
	}
	
	fmt.Println("goxmeans complete")
	fmt.Printf("original num centroids=%d  bic=%f  numclusters=%d\n", model.Numcentroids, model.Bic, len(model.Clusters))
	for i, clust := range model.Clusters {
		fmt.Printf("cluster_id=%d numpoints=%d variance=%f  centroid=%v\n", i, clust.Numpoints(), clust.Variance, clust.Centroid)
	}
}