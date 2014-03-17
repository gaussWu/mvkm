#!/usr/bin/env python
# coding=utf-8
'''this module provide some utilites tools'''
import math
import numpy as np
cimport numpy as np
DTYPE = np.float
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t INT_t
ctypedef np.float64_t DOUBLE
ctypedef np.int64_t INT
cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int N, double *X, int incX, double *Y, int incY)

from itertools import product, combinations
def labels2Clusters(labels):
    rd = {}
    for i, v in enumerate(labels):
        rd.setdefault(v, set()).add(i)
    return rd

def clusters2labels(clusters, size):
    '''return cluster index labels according an clustering'''
    labels = [0] * size 
    for key, values in clusters.iteritems():
        for v in values:
            labels[v] = key
    return labels

def sym(np.ndarray[DTYPE_t, ndim=2] X):
    '''make a matrix symmetrical, where x is a numpy array'''
    return (X + X.T) / 2

def x2sim(np.ndarray[DTYPE_t, ndim=2] x, 
          np.ndarray[DTYPE_t, ndim=2] sim_x, 
          int method=1, kernel='gauss', float alpha=0.5):
    '''compute similarity of samples'''
    cdef int n = x.shape[0]
    cdef int d = x.shape[1]
    cdef int i, j
    cdef DTYPE_t sim
    if n != sim_x.shape[0]:
        raise ValueError("rows must be same.")
    else:
        for i, j in combinations(xrange(n), 2):
            sim = similarity(x[i,:], x[j,:])
            if kernel == 'gauss':
                sim_x[i, j] =  math.exp(-alpha * sim ** 2)
            else:
                sim_x[i, j] = sim
            sim_x[j, i] = sim_x[i, j]
    
def similarity(np.ndarray[DTYPE_t] x, np.ndarray[DTYPE_t] y, 
               int method=1):
    '''return similarity of two arrays

        Paraments
        ---------
        x : ndarray
            input array
        y : ndarray
            input array
        method : int
            optionl value and meaning are: 
            ==   ==================
            id   similarity metrics
            ==   ==================
            1       Euclid
            2       Cosine  
            3       City


    '''
    if method == 1:
        return math.sqrt(np.dot(x - y, x - y))
    elif method == 2:
        return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))
    else:
        pass

def normalize(np.ndarray[DOUBLE, ndim=2] X, int axis=0):
    '''normalize x along one dimension'''
    cdef int n = X.shape[0]
    cdef int m = X.shape[1]
    cdef int i
    if axis == 0:
        for i in range(X.shape[0]):
            #X[i,:] /= math.sqrt(ddot(m, &X[i,0], 1, &X[i,0], 1))
            X[i,:] /= math.sqrt(np.dot(X[i,:], X[i,:]))
            assert(np.dot(X[i,:], X[i,:]) <= 1.1)
    else:
        for i in range(X.shape[1]):
            X[:,i] /= math.sqrt(ddot(n, &X[0,i], 1,  &X[0,i], 1))

def norm(np.ndarray[DTYPE_t, ndim=2] X):
    '''return norm of each row of X, X is a 2d numpy array'''
    cdef int n = X.shape[0]
    cdef int k = X.shape[1]
    rv = np.array([math.sqrt(ddot(k, &X[i,0], 1, &X[i,0], 1))
                   for i in range(n)], dtype=np.float64)
    return rv

def accu_centroid_distance(np.ndarray[DTYPE_t, ndim=2] dis, 
             int n_clusters, 
             np.ndarray[DTYPE_t, ndim=2] x, 
             np.ndarray[DTYPE_t, ndim=2] y,
             np.ndarray[DTYPE_t, ndim=1] d,
             float weight):
    """This method servers only for the MVKM clustering
    """
    cdef int n = x.shape[0]
    cdef int i, j 
    for i, j in product(range(n), range(n_clusters)):
        dis[i, j] += np.dot(x[i] - y[j], x[i] - y[j]) * d[i] * weight

def accu_indictor_distance(np.ndarray[DTYPE_t, ndim=2] dis,
                      int n_clusters, np.ndarray[np.int_t, ndim=1] x, 
                      float alpha):
    """This method servers only for the MVKM clustering
    """
    cdef int n = x.shape[0]
    cdef int i, j
    for i, j in product(range(n), range(n_clusters)):
        dis[i, j] += alpha * (j != x[i])

def indictor_distance(np.ndarray[np.int_t] x,
                      np.ndarray[np.int_t] y,
                      float alpha):
    cdef int n = x.shape[0]
    cdef int i
    cdef float rv = 0
    for i in xrange(n):
        rv += alpha * (x[i] != [y[i]])
    return rv

def compute_x_norm(np.ndarray[DOUBLE] x_norm, 
                   np.ndarray[DOUBLE, ndim=2] x):
    """compute norm of each sample"""
    cdef int n = x.shape[0]
    cdef int k = x.shape[1]
    cdef int i
    for i in xrange(n):
        x_norm[i] = ddot(k, &x[i,0], 1, &x[i,0], 1)
                    
def new_accu_centroid_distance(np.ndarray[DTYPE_t, ndim=2] dis, 
             int n_clusters, 
             np.ndarray[DTYPE_t, ndim=2] x, 
             np.ndarray[DTYPE_t, ndim=2] y,
             np.ndarray[DTYPE_t, ndim=1] d,
             np.ndarray[DOUBLE] x_norm,
             np.ndarray[DOUBLE] centroids_norm,
             float weight):
    """This method servers only for the MVKM clustering
    """
    cdef int n = x.shape[0]
    cdef int k = x.shape[1]
    cdef int i, j
    cdef double *px
    cdef double xn, di
    #for i, j in product(range(n), range(n_clusters)):
    for i in xrange(n):
        xn = x_norm[i]
        px = &x[i,0]
        di = d[i]
        for j in xrange(n_clusters):
            #dis[i, j] += ((x_norm[i] + centroids_norm[j]
            #          - 2 * ddot(k, &x[i,0], 1, &y[j, 0], 1)) * d[i] * weight)
            dis[i, j] += ((xn + centroids_norm[j]
                      - 2 * ddot(k, px, 1, &y[j, 0], 1)) * di * weight)

def centroid_distance_util(np.ndarray[DOUBLE] dis,
                           np.ndarray[DOUBLE, ndim=2] x,
                           np.ndarray[DOUBLE, ndim=2] centroids,
                           np.ndarray[INT] index,
                           np.ndarray[DOUBLE] x_norm,
                           np.ndarray[DOUBLE] centroids_norm):
    cdef int n = x.shape[0]
    cdef int m = x.shape[1]
    cdef int i, j
    for i in xrange(n):
        j = index[i]
        dis[i] = 0.5 / math.sqrt(x_norm[i] + centroids_norm[j]
                                 - 2 * ddot(m, &x[i,0], 1,
                                # plus a small number to prevent divide by zero
                                 &centroids[j, 0], 1) + 0.00000000001)

def centroids_update_util(np.ndarray[DOUBLE] centroids,
                          np.ndarray[DOUBLE, ndim=2] x,
                          np.ndarray[DOUBLE] distance ):
    cdef int n = x.shape[0]
    cdef int m = x.shape[1]
    cdef int i
    for i in xrange(m):
        centroids[i]  += ddot(n, &distance[0], 1, &x[0,0], 1)
    cdef double tot = math.sqrt(ddot(m, &centroids[0], 1, &centroids[0], 1))
    for i in xrange(m):
        centroids[i] /= tot 

