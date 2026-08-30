"""Tests for the boids simulator.

These forces are the ground-truth labels for every probe in the project, so a
silent error here corrupts every downstream result without raising anything.
That is what this file exists to prevent.
"""

import numpy as np
import pytest

from boids import calculate_separation, get_torus, step, velocity_capper

# Every test in this file runs in a 100x100 torus.
BOUNDS = 100.0
SEPARATION_RADIUS = 20.0
MAX_SPEED = 5.0
DT = 0.1


# --- get_torus ---------------------------------------------------------------

def test_short_delta_passes_through_untouched():
    """Birds close together: the direct route is already the shortest one,
    so nothing should wrap."""
    a = np.array([10.0, 10.0])
    b = np.array([15.0, 10.0])

    delta = get_torus(a, b, BOUNDS)

    # x: 10 - 15 = -5, y: 0. Both well inside half a screen (50).
    np.testing.assert_allclose(delta, [-5.0, 0.0])


def test_wraps_positive_delta():
    """A raw delta of +90 on a 100-wide torus is really -10: the short way
    round runs in the opposite direction."""
    a = np.array([95.0, 50.0])
    b = np.array([5.0, 50.0])

    delta = get_torus(a, b, BOUNDS)

    np.testing.assert_allclose(delta, [-10.0, 0.0])


def test_wraps_negative_delta():
    """The mirror of the case above. An implementation that only corrects one
    sign gets this wrong, and gets it wrong silently."""
    a = np.array([5.0, 50.0])
    b = np.array([95.0, 50.0])

    delta = get_torus(a, b, BOUNDS)

    np.testing.assert_allclose(delta, [10.0, 0.0])


def test_wrap_is_decided_per_axis():
    """x is more than half a screen apart and must wrap; y is not and must not.
    A single `if` on the whole vector cannot express two different decisions."""
    a = np.array([95.0, 15.0])
    b = np.array([5.0, 5.0])

    delta = get_torus(a, b, BOUNDS)

    # x: raw 90 -> short way round is 10 the other direction -> -10
    # y: raw 10 -> already shortest -> unchanged
    np.testing.assert_allclose(delta, [-10.0, 10.0])


def test_handles_position_wound_past_the_bounds():
    """A position outside [0, bounds) still resolves correctly.

    Positions should be kept wrapped by the integrator, but the modulo form
    does not depend on that invariant holding, and this pins that guarantee
    so a future "simplification" cannot quietly remove it.
    """
    a = np.array([230.0, 0.0])
    b = np.array([0.0, 0.0])

    delta = get_torus(a, b, BOUNDS)

    # 230 is 2.3 screens out; 230 mod 100 = 30, which is under half a screen.
    np.testing.assert_allclose(delta, [30.0, 0.0])


# --- calculate_separation ----------------------------------------------------

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


# --- velocity_capper ---------------------------------------------------------

def test_capper_leaves_slow_velocity_untouched():
    """Under the cap the velocity must come back completely unchanged."""
    v = np.array([1.0, 2.0])  # speed ~2.24, well under 5

    np.testing.assert_array_equal(velocity_capper(v, MAX_SPEED), v)


def test_capper_shortens_the_arrow_without_turning_it():
    """Capping limits magnitude only - heading must survive untouched.

    An implementation that returns `max_speed` itself passes a magnitude
    check by accident while pointing the bird somewhere it was never going,
    so the direction assertion is the one doing the work here.
    """
    v = np.array([6.0, 8.0])  # speed exactly 10
    out = velocity_capper(v, MAX_SPEED)

    assert np.linalg.norm(out) == pytest.approx(MAX_SPEED)
    np.testing.assert_allclose(out / np.linalg.norm(out), v / np.linalg.norm(v))
    np.testing.assert_allclose(out, [3.0, 4.0])


def test_capper_leaves_velocity_exactly_at_the_cap():
    """The boundary case: speed == max_speed must not be rescaled."""
    v = np.array([3.0, 4.0])  # speed exactly 5

    np.testing.assert_allclose(velocity_capper(v, MAX_SPEED), v)


def test_capper_handles_a_motionless_bird():
    """A stationary bird has speed 0. Normalising it would divide by zero and
    produce NaN without raising, so the branch must not be taken at all."""
    out = velocity_capper(np.array([0.0, 0.0]), MAX_SPEED)

    assert not np.isnan(out).any(), "NaN velocity would spread to every later tick"
    np.testing.assert_array_equal(out, [0.0, 0.0])


# --- step: the reconstruction invariant --------------------------------------

def _reconstruct_velocity(velocities, forces, boid_id, dt, max_speed):
    """Re-derive one bird's new velocity from the RECORDED forces alone.

    This is the whole point of the exercise: if the numbers stored in the
    forces dict cannot reproduce the velocity update the simulator actually
    performed, then those numbers are not labels for this simulation, and
    every probe trained on them is fitting something that never happened.
    """
    total = (forces["separation"][boid_id]
             + forces["alignment"][boid_id]
             + forces["cohesion"][boid_id])
    return velocity_capper(velocities[boid_id] + total * dt, max_speed)


def test_recorded_forces_reconstruct_the_velocity_update():
    """The uncapped case: recorded forces sum straight to the velocity change."""
    positions = np.array([[10.0, 10.0],
                          [15.0, 10.0],
                          [80.0, 80.0]])
    velocities = np.array([[1.0, 0.0],
                           [-1.0, 0.0],
                           [0.5, 0.5]])

    _, new_velocities, forces = step(
        positions, velocities, DT, SEPARATION_RADIUS, BOUNDS, MAX_SPEED
    )

    for i in range(len(positions)):
        np.testing.assert_allclose(
            _reconstruct_velocity(velocities, forces, i, DT, MAX_SPEED),
            new_velocities[i],
            err_msg=f"recorded forces do not reproduce bird {i}'s velocity update",
        )


def test_reconstruction_holds_when_the_speed_cap_fires():
    """The case that actually matters.

    Once the cap engages, recorded forces no longer sum to the velocity
    change - the relationship is `cap(v_old + sum(f) * dt) == v_new`, not a
    plain sum. A reconstruction test that never trips the cap verifies only
    the easy half and would pass against a simulator whose labels are wrong
    in exactly the regime where birds are moving fastest.
    """
    max_speed = 0.5  # low enough that every bird below exceeds it
    positions = np.array([[10.0, 10.0],
                          [15.0, 10.0],
                          [80.0, 80.0]])
    velocities = np.array([[1.0, 0.0],
                           [-1.0, 0.0],
                           [0.5, 0.5]])

    _, new_velocities, forces = step(
        positions, velocities, DT, SEPARATION_RADIUS, BOUNDS, max_speed
    )

    # Guard the test's own premise: if the cap never fires this proves nothing.
    uncapped = np.array([
        velocities[i] + (forces["separation"][i]
                         + forces["alignment"][i]
                         + forces["cohesion"][i]) * DT
        for i in range(len(positions))
    ])
    assert np.any(np.linalg.norm(uncapped, axis=1) > max_speed), (
        "premise failed: the cap never fired, so this test verifies nothing"
    )

    for i in range(len(positions)):
        np.testing.assert_allclose(
            _reconstruct_velocity(velocities, forces, i, DT, max_speed),
            new_velocities[i],
            err_msg=f"reconstruction broke for bird {i} once the cap fired",
        )
        assert np.linalg.norm(new_velocities[i]) <= max_speed + 1e-12


def test_positions_stay_inside_the_world():
    """`pos %= bounds` every tick is the invariant get_torus depends on.

    Both birds start within half a step of an edge and are moving straight at
    it, so each must wrap round rather than leaving the box.
    """
    positions = np.array([[99.9, 50.0],
                          [50.0, 99.9]])
    velocities = np.array([[10.0, 0.0],
                           [0.0, 10.0]])

    new_positions, _, _ = step(
        positions, velocities, DT, SEPARATION_RADIUS, BOUNDS, MAX_SPEED
    )

    assert np.all(new_positions >= 0.0)
    assert np.all(new_positions < BOUNDS)
    # confirm they wrapped rather than simply not reaching the edge
    assert new_positions[0][0] < 1.0, "bird 0 should have crossed the x seam"
    assert new_positions[1][1] < 1.0, "bird 1 should have crossed the y seam"
