from dataclasses import replace

import numpy as np

from boids import Physics

# PHYSICS_CRUISING is the shipped configuration, taken from boids.py rather
# than restated here. Everything below derives from it.
#
# The previous version declared BOUNDS, ALIGNMENT_RADIUS and the rest as
# literals and fed them back into Physics(...), which is a second source of
# truth for the same quantities - and they had already drifted apart, with
# alignment_radius 35.0 here against 30.0 in boids.py. Deriving makes that
# class of divergence impossible rather than merely fixing today's instance.
PHYSICS_CRUISING = Physics()

# The world as frozen at sim-v1: floor on, hard radius edges, no per-rule force
# cap. For tests about windows, datasets and baselines rather than the physics,
# which were written against it and need its flocks (they settle sooner, and a
# copier cannot predict them almost perfectly). Fields Physics() does not have
# yet are skipped, so this file still imports while boids.py is being changed;
# test_the_test_baseline_tracks_the_shipped_config fails until they exist.
_V1_OFF = {"edge_softness": 0.0, "max_force": None}
PHYSICS_V1 = replace(PHYSICS_CRUISING, **{k: v for k, v in _V1_OFF.items()
                                          if hasattr(PHYSICS_CRUISING, k)})

# The same world with the speed floor switched off as well, so that every test
# written before the floor, the soft edges and the cap existed still means
# exactly what it meant then. Tests that vary one constant use
# dataclasses.replace(PHYSICS, thing=...).
PHYSICS = replace(PHYSICS_V1, min_speed=0.0)

BOUNDS = PHYSICS.bounds
SEPARATION_RADIUS = PHYSICS.separation_radius
ALIGNMENT_RADIUS = PHYSICS.alignment_radius
COHESION_RADIUS = PHYSICS.cohesion_radius
MAX_SPEED = PHYSICS.max_speed
MIN_SPEED = PHYSICS_CRUISING.min_speed
DT = PHYSICS.dt


def random_flock(seed, n_birds):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, BOUNDS, size=(n_birds, 2))
    velocities = rng.uniform(-MAX_SPEED, MAX_SPEED, size=(n_birds, 2))
    return positions, velocities
