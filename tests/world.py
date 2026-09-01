"""The world every test in this suite runs in.

Shared so that the five test modules cannot drift into disagreeing about the
size of the world or how fast a bird may fly. Tests that need to vary one
constant use dataclasses.replace(PHYSICS, thing=...), which changes that field
and leaves the other nine exactly as they are here.
"""

import numpy as np

from boids import Physics

# a 100x100 torus
BOUNDS = 100.0
SEPARATION_RADIUS = 20.0
ALIGNMENT_RADIUS = 35.0
COHESION_RADIUS = 40.0
MAX_SPEED = 5.0
DT = 0.1

PHYSICS = Physics(
    dt=DT,
    bounds=BOUNDS,
    max_speed=MAX_SPEED,
    separation_radius=SEPARATION_RADIUS,
    alignment_radius=ALIGNMENT_RADIUS,
    cohesion_radius=COHESION_RADIUS,
)


def random_flock(seed, n_birds):
    """A deterministic starting world. Same seed, same flock, always."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, BOUNDS, size=(n_birds, 2))
    velocities = rng.uniform(-MAX_SPEED, MAX_SPEED, size=(n_birds, 2))
    return positions, velocities
