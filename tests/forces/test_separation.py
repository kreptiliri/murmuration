import numpy as np
import pytest

from boids import calculate_separation
from world import BOUNDS, SEPARATION_RADIUS


def test_force_points_away_from_neighbour():
    positions = np.array([[10.0, 10.0],
                          [15.0, 10.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    assert force[0] < 0, "bird 0 must flee in -x, away from the bird at +x"
    assert force[1] == pytest.approx(0.0), "no y neighbour, so no y force"
    # magnitude is 1/d, and the birds are d = 5 apart
    assert np.linalg.norm(force) == pytest.approx(0.2)

def test_force_points_away_from_2nd_bird():
    positions = np.array([[10.0, 10.0],
                          [15.0, 10.0]])

    force = calculate_separation(1, positions, SEPARATION_RADIUS, BOUNDS)

    assert force[0] > 0, "bird 1 must flee in +x, away from the bird at -x"
    assert force[1] == pytest.approx(0.0), "no y neighbour, so no y force"
    # magnitude is 1/d, and the birds are d = 5 apart
    assert np.linalg.norm(force) == pytest.approx(0.2)

def test_isolated_bird_feels_nothing():
    positions = np.array([[10.0, 10.0],
                          [80.0, 80.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    assert not np.isnan(force).any(), "NaN would silently poison every label"
    np.testing.assert_array_equal(force, [0.0, 0.0])

def test_neighbour_across_the_seam_still_counts():
    positions = np.array([[2.0, 50.0],
                          [98.0, 50.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    # d = 4, so magnitude 1/4; bird 0 flees in +x, through the seam.
    np.testing.assert_allclose(force, [0.25, 0.0])

def test_closer_neighbour_pushes_harder():
    far = np.array([[10.0, 50.0],
                    [18.0, 50.0]])   # d = 8
    near = np.array([[10.0, 50.0],
                     [14.0, 50.0]])  # d = 4

    force_far = calculate_separation(0, far, SEPARATION_RADIUS, BOUNDS)
    force_near = calculate_separation(0, near, SEPARATION_RADIUS, BOUNDS)

    assert np.linalg.norm(force_far) == pytest.approx(1 / 8)
    assert np.linalg.norm(force_near) == pytest.approx(1 / 4)
    assert np.linalg.norm(force_near) > np.linalg.norm(force_far)

def test_symmetric_neighbours_cancel():
    positions = np.array([[5.0, 50.0],
                          [10.0, 50.0],
                          [15.0, 50.0]])

    force = calculate_separation(1, positions, SEPARATION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)

def test_forces_are_averaged_not_summed():
    positions = np.array([[10.0, 50.0],
                          [15.0, 50.0],
                          [20.0, 50.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [-0.15, 0.0])

def test_single_bird_world():
    positions = np.array([[10.0, 10.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_array_equal(force, [0.0, 0.0])

def test_force_cap_limits_magnitude_without_rotating_it():
    positions = np.array([[10.0, 50.0],
                          [10.5, 50.0]])

    uncapped = calculate_separation(
        0, positions, SEPARATION_RADIUS, BOUNDS, sep_force_cap=10.0
    )
    capped = calculate_separation(
        0, positions, SEPARATION_RADIUS, BOUNDS, sep_force_cap=1.0
    )

    # Guard the premise: if the cap never bites, this test proves nothing.
    assert np.linalg.norm(uncapped) == pytest.approx(2.0)

    np.testing.assert_allclose(capped, [-1.0, 0.0])
    assert np.linalg.norm(capped) == pytest.approx(1.0), (
        "magnitude should be the cap exactly - 0.5 means `dist` was clamped "
        "and `direction` is no longer a unit vector"
    )

def test_force_cap_leaves_distant_neighbours_alone():
    positions = np.array([[10.0, 50.0],
                          [15.0, 50.0]])

    force = calculate_separation(
        0, positions, SEPARATION_RADIUS, BOUNDS, sep_force_cap=1.0
    )

    np.testing.assert_allclose(force, [-0.2, 0.0])

def test_a_coincident_neighbour_is_ignored_not_infinite():
    positions = np.array([[10.0, 10.0],
                          [10.0, 10.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    assert not np.isnan(force).any(), "NaN would silently poison every label"
    np.testing.assert_array_equal(force, [0.0, 0.0])
