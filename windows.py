import json
import numpy as np
from pathlib import Path



def build_windows(dataset_dir, *, context, clips=None):
    dataset_dir = Path(dataset_dir)
    clips_data = np.load(dataset_dir / "clips.npz")
    
    positions = clips_data["positions"]
    velocities = clips_data["velocities"]
    mask = clips_data["mask"] 

    with open(dataset_dir / "config.json", "r") as f:
        config = json.load(f)

    dt = config["physics"]["dt"]

    if clips is None:
        clip_indices = range(positions.shape[0]) #defaults to total count in dataset
    elif isinstance(clips, int):
        clip_indices = [clips] #user calls clips = x
    else:
        clip_indices = clips
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
            target_velo = (velocities[c, offset + context] * dt).astype(np.float32) #taking in the last context + 1 scaled vector 
            window_in_flat = window_in.reshape(context, -1)
            target_velo_flat = target_velo.reshape(-1)
            inputs.append(window_in_flat)
            targets.append(target_velo_flat)

    return np.array(inputs), np.array(targets)
