#!/usr/bin/env python
# coding=utf-8
"""
this module contains several evaluate methods for 
clustering task. Currently, we implement Precision,
Recall, F-score, NMI(normalized mututal information)
and Purity. Additionly, we provide a method to test
these methods. All the evaluate methods accept same 
parameters as below:


Parameters
----------
result : a list whose i'th element is the cluster label
          of i'th object.
truth  : a list whose i'th element is the true label
          of i'th object.

"""
from __future__ import division
import math
import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float64_t DOUBLE
ctypedef np.int64_t INT 

def precision(np.ndarray[INT] result,
              np.ndarray[INT] truth):
    cdef int n = len(result)
    cdef int i, j
    cdef float crt, tot
    cdef np.ndarray[DOUBLE] prec = np.zeros(n, dtype=np.float64)
    for i in xrange(n):
        crt, tot = 0.0, 0
        for j in xrange(n):
            if i == j:
                continue
            elif result[i] == result[j]:
                tot += 1
                if truth[i] == truth[j]:
                    crt += 1
        if tot:
            prec[i] = crt / tot
    return sum(prec) / n 

def recall(np.ndarray[INT] result,
            np.ndarray[INT] truth):
    cdef int n = len(result)
    cdef int i, j, tot
    cdef float crt
    cdef np.ndarray[DOUBLE] rc = np.zeros(n, dtype=np.float64)
    for i in xrange(n):
        crt, tot = 0.0, 0
        for j in xrange(n):
            if i == j:
                continue
            elif truth[i] == truth[j]:
                tot += 1
                if result[i] == result[j]:
                    crt += 1
        if tot:
            rc[i] = crt / tot
    return sum(rc) / n 

def fscore(result, truth):
    pc = precision(result, truth)
    rc = recall(result, truth)
    return 2 * (pc * rc) / (pc + rc)

def nmi(result, truth):
    N = len(truth) + 0.0
    clusters, classes = (labels2Clusters(result),
                         labels2Clusters(truth)) 

    inf = 0.0
    for k in clusters:
        for j in classes:
            il = len(clusters[k] & classes[j])
            ul = len(clusters[k]) * len(classes[j])
            tmp = N * il / ul
            if tmp and il:
                inf += (il / N) * math.log(N * il / ul)

    clsh = entropy(clusters, N)
    clah = entropy(classes, N) 
    
    return 2 * inf / abs(clsh + clah)
    
def randIndex(result, truth):
    pass

def purity(result, truth):
    N = len(truth)
    clusters, classes = {}, {}

    for i, v in enumerate(result):
        clusters.setdefault(v, set()).add(i)
    for i, v in enumerate(truth):
        classes.setdefault(v, set()).add(i)

    pu = 0.0
    for k in clusters:
        max_num = 0.0
        for j in classes:
            l =  len(clusters[k] & classes[j])
            if l > max_num:
                max_num = l
        pu += max_num
        
    return pu / N

def entropy(dist, n):
    '''return entropy of a distribution'''
    ep = 0.0
    for k in dist:
        ep -= len(dist[k]) / n * math.log(len(dist[k]) / n )
    return ep

def labels2Clusters(labels):
    rd = {}
    for i, v in enumerate(labels):
        rd.setdefault(v, set()).add(i)
    return rd

def test():
    clst = [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3]
    clss = [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3]
    clst1 = [1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3]
    clss1 = [1,1,1,1,1,2,2,2,2,2,1,3,3,3,3,1,1]
    print precision(clst, clss)
    print recall(clst, clss)
    print fscore(clst, clss)
    print nmi(clst, clss)
    print purity(clst, clss)

if __name__ == '__main__':
    test()
