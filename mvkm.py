#!/usr/bin/env python
# coding=utf-8
"""
An implementation of MVKM(Multi-View-Kmeans-Clustering)

Author : wuguoxin gxwu@insun.hit.edu.cn
License: GPLv2

"""
import numpy as np
import cu_v3 as cu
import ev
import dataset as ds
from numpy.random import random, random_integers
from math import sqrt

class MVKM(object):
    """implementation of Multi-View-Kmeans-Clustering

        Parameters
        ----------
        X : list of numpy 2d-array
            A list whose i-th element is the input matrix of
            i-th view. Each line of this matrix represents a sample.

        n_samples : int
            The number of samples

        n_clusters : int
            The number of clusters to find.

        groups : list of tuple 
            A list whose i-th element is a tuple whose elements are
            view ids which in the i-th group.

        interacts : list of tuple 
            A list whose i-th element is a tuple whose elements are
            view ids which iteract with the i-th view.
        
        truth : array-like
            The truth class of samples

        gamma : float, default is 2
            A hyper-parameter of this model.

        alpha : float, default is 0.01
            A hyper-parameter of this model.

        iter_max : int, default is 50
            Maximum iteration times.


        Attributes
        ----------
        `clusters` : list of dict
            A dict whose key is cluster id and value is a list consist of
            sample id belong to this cluster.

        `cluster_indictors` : list of array 
            A list whose i-th element is the cluster indictor of i-th group,
            whose i-th element is the cluster id of i-th sample.
        
        `cluster_centorids` : list of numpy 2d array
            A list whose i-th element is the centroids matrix of i-the view.
        
        `centroids_distance` : list of array 
            A list whose i-th element is an array record the distance from
            sample to it's cluster centorid.

        `weights` : array-like
            An array whose i-th element is the weight of i-th view.


        Note
        ----
        the input matrix x in X must be normalized into [-1.0, 1.0].

        """ 

    def __init__(self, config_file): 
        self.groups = [] 
        self.interacts = [] 
        self.config_file = config_file
        self.clusters = {}
        self.objs = {}
        self.cur_tot_obj = 1
        self.prev_tot_obj = 0
        self.eps = 1.0e-5
        # read config and set attributes
        self._read_config() 

        # intialize cluster indictors randomly
        self.cluster_indictors = [random_integers(0, self.n_clusters-1, 
                                                  self.n_samples)
                                  for i in range(self.n_groups)]
        # intialize cluster centroids randomly 
        self.cluster_centroids = [random((self.n_clusters, self.dims[i]))
                                 for i in range(self.n_views)]
        # intialize cluster centroids norm
        self.centroids_norm = [np.zeros(self.n_clusters, dtype=np.float64) 
                               for i in range(self.n_views)]
        # intialize centorids distance to be ones 
        self.centroids_distances = [np.ones(self.n_samples, dtype=np.float64)
                                   for i in range(self.n_views)]
        # intialize average centroids distance 
        self.avr_centroids_distances = [np.ones(self.n_samples, dtype=np.float64)
                                      for i in range(self.n_groups)]
        # intialize weights of views to be same
        self.weights = (1.0 / self.n_views) * np.ones(self.n_views, 
                                                      dtype=np.float64) 

        # generate clusters from cluster indictors 
        self.indictors2clusters()

    def run(self):
        iter_times = 0
        while (iter_times < self.max_iters): #and (abs(
            #self.cur_tot_obj - self.prev_tot_obj) > self.eps):
            self._update_once()
            iter_times += 1
        self._evaluate()
        return self.evaluate_result

    def _read_config(self):
        with open(self.config_file, 'r') as config:
            for line in config:
                if line.startswith('#'):
                    continue
                elif line.startswith('[PARAS]'):
                    self.alpha, self.gamma, self.max_iters = \
                            line.strip()[7:].strip().split()
                    self.alpha = float(self.alpha)
                    self.gamma = float(self.gamma)
                    self.max_iters = int(self.max_iters)

                elif line.startswith('[DATASET]'):
                    data = line.strip()[9:].strip().split()
                    data_name = data[0]
                    data_views = [int(v) for v in data[1:]]
                    self.X, self.truth, self.n_clusters, self.n_samples = \
                           ds.get_dataset(name=data_name, views=data_views)
                    # get number of views
                    self.n_views = len(self.X)
                    # get dimension of each view
                    self.dims = [x.shape[1] for x in self.X]

                elif line.startswith('[GROUPS]'):
                    tmp_list = line.strip()[8:].split(';')
                    self.group_size = [0] * self.n_views;
                    #print self.n_views
                    for tl in tmp_list:
                        self.groups.append(tuple([int(t) 
                                for t in tl.strip().split()]))
                        lg = len(self.groups[-1])
                        for view in self.groups[-1]:
                            #print view
                            self.group_size[view] = lg 
                    # get number of groups
                    self.n_groups = len(self.groups)

                elif line.startswith('[INTERACTS]'):
                    tmp_list = line.strip()[11:].split(';')
                    for tl in tmp_list:
                        self.interacts.append(tuple([int(t) 
                                for t in tl.strip().split()]));
                    #print self.interacts
                    self._collect_interacts_items()

    def _update_once(self):
        """update once"""
        for group_id in range(self.n_groups):
            # first update cluster centroids in this group
            self._update_centroids(group_id)
            # second update cluster indictor of this group
            self._update_indictors(group_id)
            # thirdly update centords distance in this group
            self._update_distances(group_id)
            
        # update objective functions
        self._update_objs()
        #print "current totol objective is %f"%self.cur_tot_obj
        # finally update weights 
        self._update_weights()

    def _update_centroids(self, group_id):
        for view_id in self.groups[group_id]:
           self._update_centroid(view_id, group_id) 

    def _update_indictors(self, group_id):
        '''find best cluster id for each sample in this group'''
        dis = np.zeros((self.n_samples, self.n_clusters), dtype=np.float64)
        for view_id in self.groups[group_id]:
            cu.new_accu_centroid_distance(dis,
                    self.n_clusters, 
                    self.X[view_id], 
                    self.cluster_centroids[view_id],
                    self.centroids_distances[view_id],
                    self.weights[view_id] ** self.gamma)

        for gi in self.interacts[group_id]:
            cu.accu_indictor_distance(dis, self.n_clusters,
                                self.cluster_indictors[group_id],
                                self.avr_centroids_distances[gi],
                                self.avr_centroids_distances[group_id],
                                self.alpha)
        self.cluster_indictors[group_id] = np.argmin(dis, axis=1)
        # update clusters
        self.indictors2clusters()

    def _update_distances(self, group_id):
        for view_id in self.groups[group_id]:
            cu.centroid_distance_util(self.centroids_distances[view_id],
                                      self.X[view_id],
                                      self.cluster_centroids[view_id],
                                      self.cluster_indictors[group_id])
        self._update_avr_distances(group_id)

    def _update_avr_distances(self, group_id):
        for i in range(self.n_samples):
            #self.groups[group_id][i] = 0
            # reset to all 0
            self.avr_centroids_distances[group_id][i] = 0

        for view_id in self.groups[group_id]:
            cu.accu_avr_centroid_distance(self.n_samples, 
                                        1.0/self.group_size[group_id], 
                                        self.avr_centroids_distances[group_id],
                                        self.centroids_distances[view_id])

    def _update_objs(self):
        # reset to 0
        self.prev_tot_obj, self.cur_tot_obj = self.cur_tot_obj, 0 
        for view_id in range(self.n_views):
            distances = self.centroids_distances[view_id]
            self.objs[view_id] = sum(.5 / distances) #/ sqrt(self.dims[view_id])
            self.cur_tot_obj += (self.weights[view_id] 
                                 ** self.gamma) * self.objs[view_id]

        for gi in self.bi_interacts:
            self.cur_tot_obj += cu.indictor_distance(
                        self.cluster_indictors[gi[0]],
                        self.cluster_indictors[gi[1]],
                        self.alpha
                        )
        #print "current totol objective is %f"%self.cur_tot_obj

    def _update_weights(self):
        weights_sum = 0
        for view_id in range(self.n_views):
            self.weights[view_id] = (self.gamma * self.objs[view_id]
                                    ) ** (1.0 / (1.0 - self.gamma))
            weights_sum += self.weights[view_id]
        # norm to (0, 1)
        for view_id in range(self.n_views):
            self.weights[view_id] /= weights_sum #* self.group_size[view_id]
            #print "view %d's weight is %f"%(view_id, self.weights[view_id])

    def _update_centroid(self, view_id, group_id):
        # this is a numpy with shape (self.n_samples, dim)
        x = self.X[view_id]
        # this is a numpy array with shape (self.n_clusters, dim)
        centroids = self.cluster_centroids[view_id]
        # get distances
        distances = self.centroids_distances[view_id]
        # get clusters
        clusters = self.clusters[group_id]
        # update
        for cluster_id in range(self.n_clusters):
            tmp_list = np.array(list(clusters[cluster_id]), dtype=np.int64)
            centroids[cluster_id] = np.dot(distances[tmp_list], x[tmp_list])
            # normalized
            nrm2 = sqrt(np.dot(centroids[cluster_id], 
                              centroids[cluster_id]))
            if nrm2 > 1e-10:
                centroids[cluster_id] /= nrm2 

    def indictors2clusters(self):
        for group_id in range(self.n_groups):
            self.clusters[group_id] = cu.labels2Clusters(
                self.cluster_indictors[group_id])

    def _evaluate(self):
        '''evalute the clusterings'''
        self.evaluate_result = []
        for group_id in range(self.n_groups):
            label = self.cluster_indictors[group_id]
            result = [ev.nmi(label, self.truth),ev.purity(label, self.truth),
                     ev.precision(label, self.truth),ev.recall(label, self.truth), 0]
            result[-1] = 2 * result[2] * result[3] / (result[2] + result[3])
            self.evaluate_result.extend(result)
            #self._print_result(group_id, result)

    def _print_result(self, group_id, result):
        print self.weights[list(self.groups[group_id])]
        print "%d'th group's evaluate results are:"%group_id
        print "%-9s%-9s%-13s%-9s%-9s\n%-9.4f%-9.4f%-13.4f%-9.4f%-9.4f\n"%(
                    'NMI', 'Purity', 'Precision', 'Recall', 'F-score',
                    result[0], result[1], result[2], result[3], result[4])
    
    def _collect_interacts_items(self):
        """convert the interacts into a set of bi-tuple"""
        self.bi_interacts = set() 
        for i, t in enumerate(self.interacts):
            for j in t:
                self.bi_interacts.add((i, j) if i < j else (j, i)) 

#===================== test for this class ============================
def test():
       mk = MVKM('config_rtbt.txt') 
       print mk.run()

if __name__ == "__main__":
    test()


