import numpy as np
from boids import step


def polarisation(velocities):
    #mean heading length, 0 is chaos, 1 is one flock
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True) 
    safe_speeds = np.where(speeds == 0, 1.0, speeds)  #a stationary bird stays the zero vector instead of becoming NaN
    unit_headings = velocities / safe_speeds                    
    return np.linalg.norm(unit_headings.mean(axis=0))           


def generate_clip(physics, *, n_slots, n_live, n_ticks, seed):
    rng = np.random.default_rng(seed)
    #0th snapshot
    positions_0 = rng.uniform(0.0, physics.bounds, size=(n_live, 2)) #uniform distribution of random floats in (x,y)
    headings_0 = rng.uniform(0.0, 2 * np.pi, size=n_live)
    speeds_0 = rng.uniform(physics.min_speed, physics.max_speed, size=n_live)
    vx = speeds_0 * np.cos(headings_0)
    vy = speeds_0 * np.sin(headings_0)
    velocities_0 = np.column_stack((vx, vy))
    #simulation loop
    position_hist = [positions_0]
    velocity_hist = [velocities_0]
    sep_hist = []
    ali_hist = []
    coh_hist = []
    for t in range(n_ticks):
        curr_pos = position_hist[-1] #always referring to last step
        curr_vel = velocity_hist[-1]
        next_pos, next_vel, forces = step(curr_pos, curr_vel, physics)
        position_hist.append(next_pos)
        velocity_hist.append(next_vel)
        sep_hist.append(forces["separation"])#this is weighted already
        ali_hist.append(forces["alignment"])
        coh_hist.append(forces["cohesion"])

    stacked_pos = np.stack(position_hist)#shape :(ticks, live birds. 2)
    stacked_vel = np.stack(velocity_hist)
    positions = np.full((n_ticks + 1, n_slots, 2), np.nan) #fills the matrix with NaN
    positions[:, :n_live] = stacked_pos #copies data into matrix while leaving other NaN untouched
    velocities = np.full((n_ticks + 1, n_slots, 2), np.nan)
    velocities[:, :n_live] = stacked_vel

    if n_ticks > 0:
        separation = np.full((n_ticks, n_slots, 2), np.nan)
        alignment  = np.full((n_ticks, n_slots, 2), np.nan)
        cohesion   = np.full((n_ticks, n_slots, 2), np.nan)
        separation[:, :n_live] = np.stack(sep_hist)
        alignment[:, :n_live]  = np.stack(ali_hist)
        cohesion[:, :n_live]   = np.stack(coh_hist) #same concept, could be compacted
    else:
        separation = np.full((0, n_slots, 2), np.nan)
        alignment  = np.full((0, n_slots, 2), np.nan)
        cohesion   = np.full((0, n_slots, 2), np.nan)

    mask = np.zeros((n_ticks + 1, n_slots), dtype=bool)#false
    mask[:, :n_live] = True

    kind = np.zeros(n_slots, dtype=np.int8)

    settle_tick = -1  #stays -1 if the flock never orders within the clip

    for tick in range(n_ticks + 1):                          
        live_velocities = velocities[tick, :n_live]           
        if polarisation(live_velocities) > 0.99:       
            settle_tick = tick                               
            break                                            
    return {
    "positions": positions,
    "velocities": velocities,
    "separation": separation,
    "alignment": alignment,
    "cohesion": cohesion,
    "mask": mask,
    "kind": kind,
    "settle_tick": settle_tick}




        

    
    
