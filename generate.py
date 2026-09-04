import subprocess
from pathlib import Path

import numpy as np
from boids import step

import dataclasses
import json


_REPO = Path(__file__).resolve().parent


def _git(*args):
    return subprocess.run(["git", "-C", str(_REPO), *args],
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def git_is_dirty():
    return bool(_git("status", "--porcelain"))


def git_provenance():
    return {"commit": _git("rev-parse", "HEAD"),
            "tag": _git("describe", "--tags", "--always"),
            "dirty": git_is_dirty()}


def polarisation(velocities):
    #mean heading length, 0 is chaos, 1 is one flock
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True) 
    safe_speeds = np.where(speeds == 0, 1.0, speeds)  #a stationary bird stays the zero vector instead of becoming NaN
    unit_headings = velocities / safe_speeds                    
    return np.linalg.norm(unit_headings.mean(axis=0))           

def clip_seed(base_seed, index):
    #(entropy, spawn key) is what identifies each spawned child, every
    #child of spawn() inherits the parent's entropy alone
    return np.random.SeedSequence(entropy=base_seed, spawn_key=(index,))

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


def build_dataset(physics, *, n_clips, n_slots, n_live, n_ticks, base_seed, out_dir, allow_dirty=False):
    #refuse to run now, before generating
    if git_is_dirty() and not allow_dirty:
        raise RuntimeError(
            "the working tree has uncommitted changes, so the commit recorded "
            "in config.json would not describe the running code, commit "
            "first, or pass allow_dirty=True to record the dataset as "
            "provisional.")

    provenance = git_provenance()  #captured once, here, so every clip in this dataset carries the same SHA
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for i in range(n_clips):
        clips.append(generate_clip(physics, n_slots=n_slots, n_live=n_live, n_ticks=n_ticks, seed=clip_seed(base_seed, i)))

    stacked = {}
    float_keys = ("positions", "velocities", "separation", "alignment", "cohesion")
    for key in float_keys:
        arrays = []
        for c in clips:
            arrays.append(c[key])
        stacked[key] = np.stack(arrays).astype(np.float32)

    mask_arrays = []
    for c in clips:
        mask_arrays.append(c["mask"])
    stacked["mask"] = np.stack(mask_arrays)
    kind_arrays = []
    for c in clips:
        kind_arrays.append(c["kind"])
    stacked["kind"] = np.stack(kind_arrays)
    # Stack settle_tick (int32)
    settle_list = []
    for c in clips:
        settle_list.append(c["settle_tick"])
    stacked["settle_tick"] = np.array(settle_list, dtype=np.int32)
    #build and write config.json
    np.savez_compressed(out_dir / "clips.npz", **stacked)
    config_data = {
        "physics": dataclasses.asdict(physics),
        "generation": {
            "n_clips": n_clips,
            "n_slots": n_slots,
            "n_live": n_live,
            "n_ticks": n_ticks,
            "base_seed": base_seed,
            "allow_dirty": allow_dirty,
        },
        "provenance": provenance,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2)


    return out_dir

        




    
    
