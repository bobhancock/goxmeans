Overview
=========

An implementation of the x-means algorithm in Go.  See Dan Pelleg and Andrew Moore - X-Means:
Extending K-means with Efficient Estimation of the Number of Clusters.

The goal of goxmeans is to provide a fast and efficient way to cluster unstructured data on 
commodity hardware.  The use of concurrency speeds up the process of distance calculation and the
use of the Bayesian Information Criterion, instead of the average of the sum of the squared errors,
gives a better indication of how efficient the clustering is.  You should be able to run multiple
models in a short period of time to give you a reliable idea of how well your data can be modeled.

At its core, goxmeans provides a faster implementation of the classic k-means algorithm, 
see http://en.wikipedia.org/wiki/Kmeans for an explanation.  kmeans is normally a sequential
process of O(n^2) and is run multiple times.  Upon each invocation, the user specifies the 
number of centroids to use and a single model is returned.

Here, we allow the user to specify the number of centroids, k, to start with and a maximum number,
kmax, which when exceeded will cause the program to stop.  

The algorithm consists of two oeprations repeated until completion.

1. Improve parameters.

2. Improve structure.

3. If k > kmax then stop and return a slice of models with BIC scores, else go to 1.

An initial kmeans is run with k centroids and the model is assigned a BIC score.  (See the BIC notes in a separate file for
the derivation of the equation used to compute the Bayesian Information Criterion.)  Each of the clusters is then bisected
and a BIC score is assigned to this model.  Whichever has a better a BIC score, the parent with one centroid, or the child with
two centroids, is kept.

Once k exceeds kmax, a slice of models is returned.  Each model has k to kmax + 1 centroids and a BIC score which will tell you
how well the clusters fit.  See the file in the example sub-directory.

The major acceleration is accomplised viat the use of the built in concurrency functions provided by Go.  Instead of 
calculating the distance between each point and each centroid sequentially, we construct a job and send it down a 
request channel where there as many goroutines to perform the calculations as there are CPUs on the machine so the
computations are performed concurrently and sent to a results channel. 

Most of the time is currently spent on allocation and garbage collection, so we can improve performance by optimizing the
gomatrix library to minimize allocations.  Using kdtrees to cache statistcal data based on hyper-rectangles and their
relationship to centroids will avoid redundant calculations.  This library is planned for the next major release.
There may be some room for memoization or caching of centroid choice, but efficient caching in a concurrent environment 
may require a separate library.

We provide two example data sets.  On our hardware, the floats with mantissas of length of six can take up to 10 times as 
long as the integer data set.  This is a function of the way the CPUs process floating point numbers.


Installation
=============
Install Go.  See golang.org/doc/install


Dependencies
============
gomatrix at https://github.com/bobhancock/gomatrix.git


Members
========
Bob Hancock

Dan Frank

Anthony Foglia, Phd

Ralph Yozzo

Versions
=========
0.1  Initial release.
