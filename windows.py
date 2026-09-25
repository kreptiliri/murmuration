import json
import numpy as np
from pathlib import Path

from boids import get_torus



def build_windows(dataset_dir, *, context, clips=None):
    #check the arguments before touching the file, so a bad call fails at the
    #argument instead of somewhere inside numpy
    if context < 1:
        raise ValueError(f"context must be at least 1, got {context}")
    dataset_dir = Path(dataset_dir)
    #the world's width, read from the dataset's own record rather than assumed -
    #a dataset generated with different physics must repair against its own bounds
    with open(dataset_dir / "config.json") as config_file:
        bounds = json.load(config_file)["physics"]["bounds"]
    with np.load(dataset_dir / "clips.npz") as clips_data:
        positions = clips_data["positions"]
        velocities = clips_data["velocities"]
        mask = clips_data["mask"]

    if clips is None:
        clip_indices = range(positions.shape[0]) #defaults to total count in dataset
    elif isinstance(clips, int): 
        clip_indices = [clips] #zero clips would be [], not 0, good
    else:
        clip_indices = list(clips) #a generator would be consumed by the first use
    if not clip_indices:
        raise ValueError("no clips selected - an empty clip list builds no windows")
    if not mask[clip_indices].all():   
        raise ValueError("dataset contains dead slots")
    #n_clips0, n_states1, n_slots2):
    n_states = positions.shape[1]
    n_windows_clip = n_states - context
    #need at MOST (nstates-1) contexts (max = nstates -1)
    if n_windows_clip <= 0: #n states<context
        raise ValueError("context is too large for the dataset state length")
    inputs = []
    targets = []
    for c in clip_indices:
        for offset in range(n_windows_clip):
            window_in = positions[c, offset : offset + context] #slicing, taking in the offset+context positions
            #what each tick's move REALLY was. get_torus wraps the subtraction,
            #so "99.995, 0.151" comes back as -0.156 instead of +99.844
            real_moves = get_torus(window_in[1:], window_in[:-1], bounds)
            #how far the bird has travelled from the window's first tick by
            #each later tick
            travelled = np.concatenate([np.zeros_like(window_in[:1]),
                                        np.cumsum(real_moves, axis=0)])
            #start at the stored first tick and follow those moves, so
            #a flock circling the world forever never drifts off to infinity
            window_in = window_in[:1] + travelled
            target_velo = (velocities[c, offset + context] - velocities[c, offset + context -1]).astype(np.float32) #comparing CHANGE in velocities, harder to copy, not good enough though (t+1 - t)
            window_in_flat = window_in.reshape(context, -1)
            target_velo_flat = target_velo.reshape(-1)
            inputs.append(window_in_flat)
            targets.append(target_velo_flat)

    return np.array(inputs), np.array(targets)


def settled_mask(dataset_dir, *, context, clips=None):
    #one flag per window
    if context < 1:
        raise ValueError(f"context must be at least 1, got {context}")
    with np.load(Path(dataset_dir) / "clips.npz") as clips_data:
        settle_tick = clips_data["settle_tick"]
        n_states = clips_data["positions"].shape[1]

    if clips is None:
        clip_indices = range(len(settle_tick))
    elif isinstance(clips, int):
        clip_indices = [clips]
    else:
        clip_indices = list(clips)
    n_windows_clip = n_states - context
    if n_windows_clip <= 0:
        raise ValueError("context is too large for the dataset state length")

    #last input tick per window
    last_input = np.arange(n_windows_clip) + context - 1
    flags = []
    for c in clip_indices:
        #-1 means never settled
        settled_here = (settle_tick[c] >= 0) & (last_input >= settle_tick[c])
        flags.append(settled_here)

    return np.concatenate(flags)
