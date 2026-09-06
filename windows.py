import json
import numpy as np
from pathlib import Path



def build_windows(dataset_dir, *, context, clips=None):
    #check the arguments before touching the file, so a bad call fails at the
    #argument instead of somewhere inside numpy
    if context < 1:
        raise ValueError(f"context must be at least 1, got {context}")
    dataset_dir = Path(dataset_dir)
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
            target_velo = (velocities[c, offset + context] - velocities[c, offset + context -1]).astype(np.float32) #comparing CHANGE in velocities, harder to copy, not good enough though (t+1 - t)
            window_in_flat = window_in.reshape(context, -1)
            target_velo_flat = target_velo.reshape(-1)
            inputs.append(window_in_flat)
            targets.append(target_velo_flat)

    return np.array(inputs), np.array(targets)
