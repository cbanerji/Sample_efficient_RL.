# Generate novelty value using autoencoder--
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import random
import gym
import numpy as np
import random
import math

device = torch.device("cpu")

def init_model():

    #-----Redefine the Reacher AE architecture----------------------------------------------------------------------
    class AE(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(11, 64),
                torch.nn.ELU(),
                torch.nn.Linear(64,16),
                torch.nn.ELU(),
                torch.nn.Linear(16,3),
                torch.nn.ELU(),
                torch.nn.Sigmoid(),
            )

            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(3,16),
                torch.nn.ELU(),
                torch.nn.Linear(16,64),
                torch.nn.ELU(),
                torch.nn.Linear(64, 11),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    model = AE() # model initialization
    model.load_state_dict(torch.load("/home/Hopper_pretrain10k.pth")) # load state dict from saved AE model
    return model

def get_dense(can_pk,ng,k):
    iter_dist = []
    for pck in ng:
        # Calculate Euclidean distance between two points
        euc = np.linalg.norm(can_pk-pck)
        wt = np.exp(-euc) # get weight, more attention to closer voters
        iter_dist.append(wt*euc) # we weight the values according to their distance from the candidate
    den_score = math.exp(-np.sum(iter_dist)/k) # calculate Density score
    return den_score


def get_den_all(candt_pk, bufr_enc, k,rept=100):
    score_arr = np.zeros([len(candt_pk), rept]) #container

    for rp in range (rept): #repeat
        ng = random.sample(bufr_enc, k) #get k sized random sampled voters from buffer

        for can_id, can_pk in enumerate(candt_pk):
            #2d array, can_id: score_rept1, score_rept2, score_rept3
            score_arr[can_id][rp] = get_dense(can_pk,ng,k) # get density value for this candidate

    #take row-wise average, i.e. average density value for each canditate
    can_avg_den = np.average(score_arr,axis=1)
    max_indx = np.where(can_avg_den == max(can_avg_den))
    max_indx = np.squeeze(max_indx)
    max_indx = max_indx.item()
    den_peak = candt_pk[max_indx]
    return den_peak


# Calculate density peak
def get_HVD_point(candt,k, bufr_enc):
    coll = []
    # Sample 'candt' points at random
    candt_pk = random.sample(bufr_enc, candt)
    den_pk = get_den_all(candt_pk,bufr_enc, k)

    return den_pk
