import numpy as np
import torch

def split_by_clip(n_clips, *, train_fraction):    
    n_train = int(n_clips * train_fraction)
    return range(n_train), range(n_train, n_clips)

def squasher(values):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    std  = values.std(axis = 0)
    #slid = []
    #for i in values:
        #slid.append(i - mean)
    std[std == 0] = 1.0
    return mean, std

def apply_squash(scaler, values):
    mean, std = scaler
    return (values - mean) / std

def cancel_squash(scaler, values):
    mean , std = scaler
    return values * std + mean



def skill(prediction, truth, baseline):
    model_error = ((prediction - truth) ** 2).sum()
    copier_error = ((baseline - truth) ** 2).sum()
    return 1 - model_error / copier_error #please be better

class Model(torch.nn.Module):
    def __init__(self, context, n_features): #192*24
        super().__init__()
        self.layer = torch.nn.Linear(context * n_features, n_features)

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer(x)
        return x

def build_model(*, context, n_features): #callable, everywhere
    return Model(context, n_features)  






#no clip overlapping, and complete splitted coverage
#  train_x, train_y = build_windows("data/clips", context=8, clips=train_clips)
    #test_x,  test_y  = build_windows("data/clips", context=8, clips=test_clips)