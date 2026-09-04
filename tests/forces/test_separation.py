"""calculate_separation: push away from close neighbours, 1/d, capped."""

import numpy as np
import pytest

from boids import calculate_separation
from world import BOUNDS, SEPARATION_RADIUS


def test_force_points_away_from_neighbour():
    """Bird 0 has a neighbour 5 units away in +x, so it must be pushed in -x.

    A flipped sign here turns separation into attraction. Nothing raises, the
    flock still looks like a flock, and every f_sep label in the dataset is
    wrong - so this assertion is the one holding the convention in place.
    """
    positions = np.array([[10.0, 10.0],
                          [15.0, 10.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    assert force[0] < 0, "bird 0 must flee in -x, away from the bird at +x"
    assert force[1] == pytest.approx(0.0), "no y neighbour, so no y force"
    # magnitude is 1/d, and the birds are d = 5 apart
    assert np.linalg.norm(force) == pytest.approx(0.2)

def test_force_points_away_from_2nd_bird():
    """The mirror of the test above: same world, other bird.

    Not redundant. A bug that reads `positions[0]` instead of
    `positions[boid_id]` passes the test above and fails this one.
    """
    positions = np.array([[10.0, 10.0],
                          [15.0, 10.0]])

    force = calculate_separation(1, positions, SEPARATION_RADIUS, BOUNDS)

    assert force[0] > 0, "bird 1 must flee in +x, away from the bird at -x"
    assert force[1] == pytest.approx(0.0), "no y neighbour, so no y force"
    # magnitude is 1/d, and the birds are d = 5 apart
    assert np.linalg.norm(force) == pytest.approx(0.2)

def test_isolated_bird_feels_nothing():
    """Nearest neighbour is ~42 away against a radius of 20, so nothing counts.

    Must return exactly zero - and must not return NaN, which is what a
    divide-by-zero produces here without raising anything.
    """
    positions = np.array([[10.0, 10.0],
                          [80.0, 80.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    assert not np.isnan(force).any(), "NaN would silently poison every label"
    np.testing.assert_array_equal(force, [0.0, 0.0])

def test_neighbour_across_the_seam_still_counts():
    """Birds at x=2 and x=98 on a 100-wide torus are 4 apart, not 96.

    An implementation that lost the wrap would see nothing inside the radius
    and return zero here. This is the test that proves the world is a torus.
    """
    positions = np.array([[2.0, 50.0],
                          [98.0, 50.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    # d = 4, so magnitude 1/4; bird 0 flees in +x, through the seam.
    np.testing.assert_allclose(force, [0.25, 0.0])

def test_closer_neighbour_pushes_harder():
    """Magnitude must fall off as 1/d.

    This is the test that catches the cancellation bug: if the direction
    vector is not normalised before scaling by 1/d, every force comes out
    with magnitude exactly 1 regardless of distance, and both worlds below
    would return the same answer.
    """
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
    """Equal neighbours either side leave the middle bird with no net force.

    Catches sign errors and accumulation errors at once. Note it would also
    pass for a function that always returned zero - which is exactly why it
    is an addition to the suite rather than a replacement for it.
    """
    positions = np.array([[5.0, 50.0],
                          [10.0, 50.0],
                          [15.0, 50.0]])

    force = calculate_separation(1, positions, SEPARATION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [0.0, 0.0], atol=1e-12)

def test_forces_are_averaged_not_summed():
    """Two neighbours on the same side, at d = 5 and d = 10.

    Individual magnitudes are 1/5 and 1/10, both pointing -x, so the sum is
    -0.3 and the average is -0.15. This pins the averaging convention, which
    is a choice - and one the reconstruction test will have to agree with.
    """
    positions = np.array([[10.0, 50.0],
                          [15.0, 50.0],
                          [20.0, 50.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    np.testing.assert_allclose(force, [-0.15, 0.0])

def test_single_bird_world():
    """A world with one bird. No neighbours to skip, nothing to divide by.

    The degenerate case a refactor is most likely to forget, and it must
    return zero rather than raising or producing NaN.
    """
    positions = np.array([[10.0, 10.0]])

    force = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    assert not np.isnan(force).any()
    np.testing.assert_array_equal(force, [0.0, 0.0])

def test_force_cap_limits_magnitude_without_rotating_it():
    """Two birds 0.5 apart. Uncapped the 1/d law gives magnitude 2.0.

    The cap must bound the magnitude and leave the heading alone - exactly
    what `velocity_capper` does for speed. The failure this exists to catch
    is clamping `dist` itself instead of the magnitude expression: that makes
    `offset / dist` a half-length vector rather than a unit one, so the force
    comes out at 0.5 instead of the capped 1.0, and inside the cap radius the
    repulsion would grow with distance instead of shrinking. Nothing raises.
    """
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
    """A neighbour at d = 5 gives magnitude 0.2, far under the cap.

    The cap must be inert everywhere except very close range, or it is
    quietly rescaling every label in the dataset.
    """
    positions = np.array([[10.0, 50.0],
                          [15.0, 50.0]])

    force = calculate_separation(
        0, positions, SEPARATION_RADIUS, BOUNDS, sep_force_cap=1.0
    )

    np.testing.assert_allclose(force, [-0.2, 0.0])
