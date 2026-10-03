import warnings

import numpy as np
import pytest

from boids import calculate_cohesion
from world import BOUNDS, COHESION_RADIUS


def test_cohesion_pulls_toward_a_lone_neighbour():
    positions = np.array([[50.0, 50.0],
                          [60.0, 50.0]])   # neighbour 10 units in +x

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.25, 0.0])   # 10 / 40
    assert force[0] > 0, (
        "cohesion must pull toward the neighbour; a negative x here means the "
        "arguments to get_torus are the wrong way round and this is separation"
    )


def test_cohesion_averages_offsets_not_positions():
    positions = np.array([[0.0, 50.0],
                          [2.0, 50.0],
                          [99.0, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0125, 0.0])
    assert force[0] > 0, "a negative force here means positions were averaged"


def test_cohesion_neighbour_across_the_seam_counts():
    positions = np.array([[2.0, 50.0],
                          [98.0, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [-0.1, 0.0])   # -4 / 40


def test_cohesion_switches_off_at_the_centre_of_mass():
    positions = np.array([[50.0, 50.0],
                          [45.0, 50.0],
                          [55.0, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)


def test_cohesion_is_small_just_off_the_centre():
    positions = np.array([[50.0, 50.0],
                          [45.0, 50.0],
                          [55.04, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0005, 0.0])
    assert np.linalg.norm(force) < 0.001, (
        "a magnitude near 1.0 means the mean offset was normalised, which "
        "makes the force discontinuous at the centre of mass"
    )


def test_cohesion_grows_linearly_with_displacement():
    forces = [
        calculate_cohesion(
            0,
            np.array([[50.0, 50.0], [50.0 + d, 50.0]]),
            COHESION_RADIUS,
            BOUNDS,
        )
        for d in (4.0, 20.0, 39.0)
    ]
    magnitudes = [np.linalg.norm(f) for f in forces]

    assert magnitudes == pytest.approx([0.1, 0.5, 0.975])
    assert magnitudes[0] < magnitudes[1] < magnitudes[2]


def test_cohesion_magnitude_never_reaches_one():
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.0, BOUNDS, size=(30, 2))

    magnitudes = np.array([
        np.linalg.norm(calculate_cohesion(i, positions, COHESION_RADIUS, BOUNDS))
        for i in range(len(positions))
    ])

    assert magnitudes.max() < 1.0
    # premise guard: a function returning zeros would satisfy the bound
    assert magnitudes.max() > 0.0


def test_cohesion_excludes_neighbours_outside_the_radius():
    with_far_bird = np.array([[50.0, 50.0],
                              [60.0, 50.0],
                              [95.0, 50.0]])   # 45 units away, outside 40

    force = calculate_cohesion(0, with_far_bird, COHESION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.25, 0.0])


def test_cohesion_counts_a_coincident_neighbour():
    positions = np.array([[50.0, 50.0],
                          [50.0, 50.0],
                          [60.0, 50.0]])

    force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_allclose(force, [0.125, 0.0])   # 5 / 40


def test_cohesion_isolated_bird_feels_nothing_and_divides_by_nothing():
    positions = np.array([[10.0, 10.0],
                          [50.0, 50.0]])   # torus distance 56.6, outside 40

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any RuntimeWarning fails this test
        force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_array_equal(force, [0.0, 0.0])


def test_cohesion_single_bird_world():
    positions = np.array([[10.0, 10.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        force = calculate_cohesion(0, positions, COHESION_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_array_equal(force, [0.0, 0.0])
