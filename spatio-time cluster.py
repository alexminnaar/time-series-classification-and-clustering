from __future__ import division
import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn import preprocessing

class ts_cluster(object):
    def __init__(self,num_clust=100):
        '''
        num_clust is the number of clusters for the k-means algorithm
        assignments holds the assignments of data points (indices) to clusters
        centroids holds the centroids of the clusters
        '''
        self.num_clust=num_clust
        self.assignments={}
        self.centroids=[]
    
    def compa_clust(self,s1,centroid,w):        
        centroid_part = centroid[:,:5]
        self.assign = pd.Series([[] for i in range(len(centroid))],index=np.arange(len(centroid)))

        for ind,i in enumerate(s1):
            min_dist=float('inf')
            #closest_clust=None
            for c_ind,j in enumerate(centroid_part):
                    
                if self.LB_Keogh(i,j,5)<min_dist:
                    cur_dist=self.SpatioTemporalDis(i, j, w)
                        
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            
            
            self.assign[closest_clust].append(ind)

        print self.assign
        #print s1[1]
        self.s2 = pd.Series([[] for i in range(len(s1))],index=np.arange(len(s1)))
        #print self.s2
        for key in self.assign.index:
            for k in self.assign[key]:
                self.s2[k] = centroid[key,6:].tolist()
                #print self.s2[k]
        self.s2 = np.array(self.s2)
        return self.s2
        
    
    
    def k_means_clust(self,data,num_iter,w,progress=True):
        
        '''
        k-means clustering algorithm for time series data.  dynamic time warping Euclidean distance
         used as default similarity measure. 
        '''
        self.centroids=random.sample(data,self.num_clust)
        print len(self.centroids)
        for n in range(num_iter):
            if progress:
                print 'iteration '+str(n+1)
            
            self.assignments={}
            for ind,i in enumerate(data):
                min_dist=float('inf')
                #closest_clust=None
                for c_ind,j in enumerate(self.centroids):
                    if self.LB_Keogh(i,j,5)<min_dist:
                        cur_dist=self.SpatioTemporalDis(i, j, w)
                        
                        if cur_dist<min_dist:
                            min_dist=cur_dist
                            closest_clust=c_ind
                
                if closest_clust in self.assignments:
                    self.assignments[closest_clust].append(ind)
                else:
                    self.assignments[closest_clust]=[]
                    
            print len(self.assignments)
            #recalculate centroids of clusters
            for key in self.assignments:
                clust_sum=0
                for k in self.assignments[key]:
                    clust_sum=clust_sum+data[k]
                self.centroids[key]=[m/len(self.assignments[key]) for m in clust_sum]
            
    def get_centroids(self):
        return self.centroids
        
    def get_assignments(self):
        return self.assignments
        
    def plot_centroids(self):
        for i in self.centroids:
            plt.plot(i)
        plt.show()
        
    def SpatioTemporalDis(self,s1,s2,w):
        
        f1,f2 = s1[0:2],s2[0:2]
        v1,v2 = s1[2:],s2[2:]
        disInvari = self.InvarDistance(f1, f2)
        disVari   = self.DTWDistance(v1, v2, w)
        
        return np.sqrt(disInvari+disVari)
    
    
    def InvarDistance(self,f1,f2):
        '''calculates the invariant features distance using Euclidean distance'''
         
        dis = spatial.distance.euclidean(f1, f2)
        
        return dis
        
    def DTWDistance(self,s1,s2,w=None):
        '''
        Calculates dynamic time warping Euclidean distance between two
        sequences. Option to enforce locality constraint for window w.
        '''
        DTW={}
        
        if w:
            w = max(w, abs(len(s1)-len(s2)))
    
            for i in range(-1,len(s1)):
                for j in range(-1,len(s2)):
                    DTW[(i, j)] = float('inf')
            
        else:
            for i in range(len(s1)):
                DTW[(i, -1)] = float('inf')
            for i in range(len(s2)):
                DTW[(-1, i)] = float('inf')
        
        DTW[(-1, -1)] = 0
        
        for i in range(len(s1)):
            if w:
                for j in range(max(0, i-w), min(len(s2), i+w)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            else:
                for j in range(len(s2)):
                    dist= (s1[i]-s2[j])**2
                    DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
            
        return (DTW[len(s1)-1, len(s2)-1])      
        
    def LB_Keogh(self,s1,s2,r):
        '''
        Calculates LB_Keough lower bound to dynamic time warping. Linear
        complexity compared to quadratic complexity of dtw.
        '''
        LB_sum=0
        for ind,i in enumerate(s1):
            
            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            
            if i>upper_bound:
                LB_sum=LB_sum+(i-upper_bound)**2
            elif i<lower_bound:
                LB_sum=LB_sum+(i-lower_bound)**2
            
        return np.sqrt(LB_sum)
