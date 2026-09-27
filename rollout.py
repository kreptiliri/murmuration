import numpy as np
import torch

import train
from boids import get_torus

#autoregressive closed loop
def rollout(checkpoint, start, *, n_ticks, physics):
    #let one saved model fly the flock by itself: starting from real ticks then
    #every new tick comes from the model's own last guesses
    #the checkpoint has to come from a --residual none run
    model = checkpoint["model"]
    context = checkpoint["config"]["context"]
    relative = checkpoint["config"]["kind"] == "rel_per_bird"
    bounds, dt = physics["bounds"], physics["dt"]

    start = np.asarray(start, dtype=float)
    if len(start) != context:
        raise ValueError(f"start must hold exactly {context} ticks, the model's context, got {len(start)}")
    n_birds = start.shape[1]

    ticks = list(start)
    for t in range(n_ticks):
        #the last context ticks, repaired like build_windows does it; the first
        #tick as it is, then each bird follows its own moves, even across the edge
        last = ticks[-context:]
        repaired = [last[0]]
        #unwrapping periodic torus boundary
        for k in range(1, context):
            move = get_torus(last[k], last[k - 1], bounds)
            repaired.append(repaired[-1] + move)
        window = np.array(repaired).reshape(1, context, n_birds * 2)

        #same squashing as training, so the model gets numbers it knows
        x = torch.from_numpy(train.apply_squash(checkpoint["x_scaler"], train.to_moves(window))).float()
        with torch.no_grad(): #predict change
            if relative:
                pair = torch.from_numpy(train.pair_geometry(window, bounds=bounds, scale=physics["separation_radius"]))
                out = model(x, pair)
            else:
                out = model(x)
        change = train.cancel_squash(checkpoint["y_scaler"], out.numpy()).reshape(n_birds, 2)#back

        #same order as boids.step: new velocity first, then move with it
        velocity = (repaired[-1] - repaired[-2]) / dt
        velocity = velocity + change
        ticks.append((ticks[-1] + velocity * dt) % bounds)

    return np.array(ticks)
#(context+nticks,nbirds,2)