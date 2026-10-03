import numpy as np

from boids import calculate_alignment
from world import ALIGNMENT_RADIUS, BOUNDS


def test_alignment_steers_toward_the_neighbour_mean():
    positions = np.array([[50.0, 50.0],
                          [52.0, 50.0],
                          [54.0, 50.0]])
    velocities = np.array([[1.0, 0.0],
                           [0.0, 1.0],
                           [-1.0, 0.0]])

    force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [-1.5, 0.5])

def test_alignment_switches_off_when_the_flock_already_matches():
    positions = np.array([[50.0, 50.0],
                          [52.0, 50.0],
                          [54.0, 50.0]])
    velocities = np.array([[3.0, 0.0],
                           [3.0, 0.0],
                           [3.0, 0.0]])

    force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)

def test_alignment_ignores_distance_inside_the_radius():
    velocities = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
    near = np.array([[50.0, 50.0], [55.0, 50.0]])   # d = 5
    far = np.array([[50.0, 50.0], [75.0, 50.0]])    # d = 25, inside 30

    force_near = calculate_alignment(0, near, velocities, ALIGNMENT_RADIUS, BOUNDS)
    force_far = calculate_alignment(0, far, velocities, ALIGNMENT_RADIUS, BOUNDS)

    np.testing.assert_allclose(force_near, [-1.0, 1.0])
    np.testing.assert_allclose(force_far, force_near)

def test_alignment_selects_neighbours_by_position_not_velocity():
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
    positions = np.array([[2.0, 50.0],
                          [98.0, 50.0]])
    velocities = np.array([[1.0, 0.0],
                           [0.0, 2.0]])

    force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [-1.0, 2.0])

def test_alignment_isolated_bird_feels_nothing():
    positions = np.array([[10.0, 10.0],
                          [50.0, 50.0]])   # torus distance 56.6, outside 35
    velocities = np.array([[4.0, 4.0],
                           [0.0, -1.0]])

    force = calculate_alignment(0, positions, velocities, ALIGNMENT_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_array_equal(force, [0.0, 0.0])
