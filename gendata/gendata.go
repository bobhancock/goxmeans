package main

import (
	"fmt"
	"math/rand"
	"os"
	"flag"
	"strconv"
)

var maxrows = flag.String("maxrows", "", "maximum rows to write")

func main() {
	mrows := 940000

	flag.Parse()
    if *maxrows != "" {
		mrows, _ = strconv.Atoi(*maxrows)
	}

	fints, err := os.Create("randints")
	if err != nil {
		fmt.Println("randints: ", err)
		return
	}
	defer fints.Close()

	for i := 0; i < mrows; i++ {
		i0 := rand.Int()
		i1 := rand.Int()
		_, err := fints.WriteString(fmt.Sprintf("%d\t%d\n", i0, i1))

		if err != nil {
			fmt.Println("WriteSrting to fints: ", err)
			return
		}
	}
}