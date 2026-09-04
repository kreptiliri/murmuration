"""calculate_alignment: match the mean heading of nearby neighbours."""

import numpy as np

from boids import calculate_alignment
from world import ALIGNMENT_RADIUS, BOUNDS


def test_alignment_steers_toward_the_neighbour_mean():
    """Three birds in a tight cluster, all headed differently.

    Bird 0 sees velocities [0,1] and [-1,0], mean [-0.5, 0.5], and is itself
    doing [1,0]. The steering force is the difference: [-1.5, 0.5].
    """
    positions = np.array([[50.0, 50.0],
                          [52.0, 50.0],
                          [54.0, 50.0]])
    velocities = np.array([[1.0, 0.0],
                           [0.0, 1.0],
                           [-1.0, 0.0]])

    force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [-1.5, 0.5])

def test_alignment_switches_off_when_the_flock_already_matches():
    """The trap this rule exists around.

    Returning the raw neighbour mean instead of the difference from your own
    velocity gives an already-aligned flock a permanent force of [3,0] - it
    would accelerate forever into the speed cap and never settle. A matched
    flock must feel nothing.
    """
    positions = np.array([[50.0, 50.0],
                          [52.0, 50.0],
                          [54.0, 50.0]])
    velocities = np.array([[3.0, 0.0],
                           [3.0, 0.0],
                           [3.0, 0.0]])

    force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)

def test_alignment_ignores_distance_inside_the_radius():
    """No 1/d falloff - that is separation's law, not this one.

    A neighbour at d = 5 and the same neighbour at d = 25 must produce an
    identical force. This is what catches alignment written by copy-pasting
    calculate_separation and forgetting to strip the distance weighting.

    The far distance is 25 rather than 30 because ALIGNMENT_RADIUS now derives
    from boids.py and is 30.0; the comparison is a strict <, so a neighbour at
    exactly 30 would fall outside and this would be testing nothing.
    """
    velocities = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
    near = np.array([[50.0, 50.0], [55.0, 50.0]])   # d = 5
    far = np.array([[50.0, 50.0], [75.0, 50.0]])    # d = 25, inside 30

    force_near = calculate_alignment(0, near, velocities, ALIGNMENT_RADIUS, BOUNDS)
    force_far = calculate_alignment(0, far, velocities, ALIGNMENT_RADIUS, BOUNDS)

    np.testing.assert_allclose(force_near, [-1.0, 1.0])
    np.testing.assert_allclose(force_far, force_near)

def test_alignment_selects_neighbours_by_position_not_velocity():
    """Regression test for a real bug: `enumerate(velocities)`.

    Iterating the wrong array made `neighbor_pos` a velocity, so the radius
    check measured the distance from a position at [50,50] to a point at
    [0,1] - about 70 units, outside every sane radius. Every bird's alignment
    force came out as exactly [0,0], in every flock, and nothing raised: the
    summation line indexed `velocities[i]` correctly, so the function was
    averaging a legitimate set of velocities. Just the empty one.
    """
    positions = np.array([[50.0, 50.0],
                          [52.0, 50.0]])       # 2 units apart, plainly neighbours
    velocities = np.array([[1.0, 0.0],
                           [0.0, 1.0]])        # 70 units apart as coordinates

    force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [-1.0, 1.0])
    assert np.linalg.norm(force) > 0, (
        "zero force for adjacent birds means neighbours were selected by "
        "velocity rather than by position"
    )

def test_alignment_neighbour_across_the_seam_counts():
    """x = 2 and x = 98 are 4 units apart on a torus, not 96."""
    positions = np.array([[2.0, 50.0],
                          [98.0, 50.0]])
    velocities = np.array([[1.0, 0.0],
                           [0.0, 2.0]])

    force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [-1.0, 2.0])

def test_alignment_isolated_bird_feels_nothing():
    """No neighbours means no force - not minus its own velocity.

    A lone bird coasts. Returning `-my_v` for an empty neighbourhood would
    make emptiness act as a brake, which is a different rule than the one
    being modelled.
    """
    positions = np.array([[10.0, 10.0],
                          [50.0, 50.0]])   # torus distance 56.6, outside 35
    velocities = np.array([[4.0, 4.0],
                           [0.0, -1.0]])

    force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_array_equal(force, [0.0, 0.0])
