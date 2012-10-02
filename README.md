Overview
=========
An implementation of the x-means algorithm in Go.  See Dan Pelleg and Andrew Moore
- X-Means: Extending K-means with Efficient Estimation of the Number of Clusters.

The goal of goxmeans is to provide a fast and efficient way to cluster 
unstructured data on commodity hardware.  The use of concurrency speeds up the 
process of model construction and the use of the Bayesian Information Criterion
gives a mathematically sound measure of quality.  See the included 
doc/BIC_notes.pdf for an explanation of how we apply this formula.

At its core, goxmeans provides a faster implementation of the classic k-means 
algorithm, see http://en.wikipedia.org/wiki/Kmeans for an explanation.  
kmeans is normally a sequential process of O(n^2) and is run multiple times.  
Upon each invocation, the user specifies the number of centroids to use and a
single model is returned. 

We allow the user to specify the number of centroids, k, to start with and
an upper bound, kmax, which when exceeded will cause the program to stop.  

The algorithm consists of two operation repeated until completion.

1. Improve parameters.

2. Improve structure.

3. If k > kmax then stop and return a slice of models with BIC scores, else go to 1.

An initial kmeans is run with k centroids and the model is assigned a BIC score. 
Each of the clusters is then bisected and a BIC score is assigned to this new 
model.  Whichever has a better a BIC score, the parent with one centroid, or 
the child with two centroids, is kept.

Once k exceeds kmax, a slice of models is returned.  Each model has k to kmax + 1
centroids and a BIC score which will tell you how well the clusters fit.  See 
the file in the example sub-directory.

The major acceleration is accomplished via the use of the built in concurrency 
functions provided by Go.  Instead of calculating the distance between each 
point and each centroid sequentially, we construct a job and send it down a 
request channel where there as many goroutines to perform the calculations as 
there are CPUs on the machine so the computations are performed concurrently 
and sent to a results channel. 

Most of the time is currently spent on allocation and garbage collection, so we 
can improve performance by optimizing the gomatrix library to minimize 
allocations.  Using kdtrees to cache statistical data based on hyperrectangles 
and their relationship to centroids will avoid redundant calculations.  This 
library is planned for the next major release.  

There may be some room for memoization or caching during centroid choice, but 
efficient caching in a concurrent environment may require a separate library.

Visualization is imperative.  We are working on this and expect to release this
functionality before the next major release.

We provide two example data sets in the examples sub-directory.  On our hardware,
floats with mantissas of length of six can take up to 10 times as long as the 
integer data set.  This is a function of the way the CPUs process floating point 
numbers.

N.B. This has only been tested on Linux.


Installation
=============
Install Go.  See golang.org/doc/install

Once you are sure that Go is correctly installed and your GOPATH is correctly 
set, execute:

go get github.com/bobhancock/goxmeans


Test
====
cd to $GOPATH/github.com/bobhancock/goxmeans/examples and execute:

go run ./xmeans_ex.go 2 3

to test your installation.  Depending on the speed of your machine, this could
take up to three minutes.  Your output will be something like:

Load complete                         <-- Finished loading data from file
0: #centroids=2 BIC=-10935854.304219  <-- Model 0 with 2 centroids
1: #centroids=3 BIC=-10918521.272880  <-- Model 1 with 3 centroids

Best fit:[ 1: #centroids=3 BIC=-10918521.272880] <-- Best model
cluster-0: numpoints=77908  variance=259369997592716064.000000 <-- Individual clusters
cluster-1: numpoints=94254  variance=381034319072047744.000000 <-- of model.
cluster-2: numpoints=77838  variance=258988118576487776.000000

We write to stdout for demonstration purposes.  In the real world, you would call 
this from a Go program that would take the contents of the individual clusters of
the best model and continue processing.  For example, you may want to display the 
points of each cluster on a scatter plot where each cluster is a different
color.


Contributors
=============
Bob Hancock

Dan Frank

Anthony Foglia, Ph.D

Ralph Yozzo


Versions
=========
0.1  Initial release.
