import numpy as np
import random
import itertools
import math
from math import*

def good_indx_clust(a,prdct):
    centroids = [max(a),min(a)]
    indexs = []
    cv = []
    cva_01 = []
    clusters = {}
    cluster0_elem = 0
    for k in range(2):
        clusters[k] = [ ]

    # if the cluster centres of current round epi_rew not close
    #if max(a)-min(a) >10:
    for data in range (len(a)):
        euc_dist = []
        for i in range (len(centroids)): # The two cluster centers are the 'Max' and 'Min' episodes in the round
            euc_dist.append(np.linalg.norm(a[data]-centroids[i])) #Check distance from cluster centers i=0 and i=1

        #we get distance of the 'data' from two cluster centers and assign to the cluster it is nearest to
        clusters[euc_dist.index(min(euc_dist))].append(a[data])

        if (len(clusters[0])-cluster0_elem) == 1: # if added to cluster 0
            if a[data] >= prdct  or math.isclose(a[data], prdct, abs_tol = 100): # 100: Ipd, if also exceeds the predicted SMA, then return index and values, Wak: 100
                indexs.append(data) #data holds the current index of the element of 'a'
            else:
                pass
            cluster0_elem = len(clusters[0])
        else:
            pass

    # returns the best idexes and the corresponding
    # cumulative rewards from good cluster
    return indexs, clusters[0]

def best_indx(a):
    indxx = a.index(max(a))
    return indxx, max(a)

def knearest(a,k):
    edist = []
    ret_indx = []
    ret_data = []
    compa = max(a)
    for val in range(len(a)):
        #Get Euclidean distance
        edist.append(np.linalg.norm(a[val]-compa))
    #return the indices of the k smallest/ nearest values
    indxed_data = list(enumerate(edist))
    indxed_data = sorted(indxed_data, key=lambda x:x[1], reverse= False) #sorted indexed data
    #get k nearest terms
    for st in range(k+1):
        ret_indx.append(indxed_data[st][0])

    return ret_indx

class similarity():
    '''
    Accepts two vectors of equal dimension and returns their Cosine similarity measure
    '''
    def square_rooted(self,x):

        return round(sqrt(sum([a*a for a in x])),3)

    def cosine_similarity(self,x,y):

        numerator = sum(a*b for a,b in zip(x,y))
        denominator = self.square_rooted(x)*self.square_rooted(y)
        return round(numerator/float(denominator),3)
