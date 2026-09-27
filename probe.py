import numpy as np


def fit_probe(features, targets, *, alpha):
    #features: (pages, boxes), targets: (pages, 2), the real push for each page
    #every box on the same scale, like squasher in train.py
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std==0] = 1.0
    x = (features - mean) /std
    #the average push is the bias, the multipliers explain the rest
    bias = targets.mean(axis=0)
    y = targets - bias
    #the best multipliers, with a penalty of alpha on big ones
    n_boxes = x.shape[1]
    weights = np.linalg.solve(x.T @ x + alpha * np.eye(n_boxes), x.T @ y)
    return {"mean": mean, "std": std, "weights": weights, "bias": bias}


def probe_predict(probe, features):
    x = (features - probe["mean"]) / probe["std"]#new pages, scaled with the TRAINING mean and std
    return x @ probe["weights"] + probe["bias"]


def probe_r2(probe, features, targets):
    guess = probe_predict(probe, features)
    miss = ((guess - targets) **2).sum()  #the probe's squared miss, everything added up
    spread = ((targets - targets.mean(axis=0)) ** 2).sum() #guessing average wrongness
    return float(1 - miss / spread)