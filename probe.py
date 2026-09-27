import numpy as np
import torch
import train
import flocker
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


def read_drafts(checkpoint, position_windows, *, physics):
    #every bird's draft in every window, from one snapshot, as pages for the probe
    if checkpoint["config"]["kind"] != "rel_per_bird":
        raise ValueError("model must be 'rel_per_bird'")
    model = checkpoint["model"]
    n_drafts = len(model.blocks) + 1
    pages = []
    for k in range(n_drafts):
        pages.append([]) #one pile per draft

    for start in range(0, len(position_windows), 256): #in batches
        chunk = position_windows[start:start + 256]
        x = torch.from_numpy(train.apply_squash(checkpoint["x_scaler"], train.to_moves(chunk))).float()
        pair = torch.from_numpy(train.pair_geometry(chunk, bounds=physics["bounds"], scale=physics["separation_radius"]))

        with torch.no_grad():
            drafts = model.hidden_states(x, pair) 
        for k in range(n_drafts):
            n_windows, n_birds, n_boxes = drafts[k].shape
            pages[k].append(drafts[k].numpy().reshape(n_windows * n_birds, n_boxes))              #to numpy, then reshape to (n_windows * n_birds, n_boxes)

    for k in range(n_drafts):
        pages[k] = np.concatenate(pages[k])        #glue the batches back into one pile
    return pages