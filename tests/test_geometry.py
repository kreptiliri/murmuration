"""The primitives every rule is built on: torus distance, the speed cap,
and the Physics config object's own validation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from boids import Physics, get_torus, velocity_capper
from world import BOUNDS, MAX_SPEED, MIN_SPEED


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

def test_physics_is_immutable():
    """A rollout must not be able to change the physics halfway through."""
    physics = Physics()

    with pytest.raises(FrozenInstanceError):
        physics.dt = 0.2

def test_physics_rejects_a_radius_at_or_beyond_half_the_world():
    """The minimum-image convention returns at most bounds/2 along each axis.

    A radius beyond that makes every bird a neighbour along that axis, so the
    rule stops being local while still producing entirely plausible numbers.
    """
    with pytest.raises(ValueError, match="alignment_radius"):
        Physics(bounds=100.0, alignment_radius=50.0)

    with pytest.raises(ValueError, match="separation_radius"):
        Physics(bounds=100.0, separation_radius=60.0)

def test_physics_rejects_nonsense_scales():
    with pytest.raises(ValueError):
        Physics(bounds=-1.0)
    with pytest.raises(ValueError):
        Physics(dt=0.0)
    with pytest.raises(ValueError):
        Physics(max_speed=-5.0)


# --- velocity_capper: the speed floor ----------------------------------------

def test_floor_raises_a_slow_velocity_without_turning_it():
    """The mirror of the cap: rescale the magnitude, keep the heading.

    Speed 1.0 against a floor of 2.0 must double the vector, not rotate it.
    """
    out = velocity_capper(np.array([0.6, 0.8]), MAX_SPEED, MIN_SPEED)

    np.testing.assert_allclose(out, [1.2, 1.6])
    assert np.linalg.norm(out) == pytest.approx(MIN_SPEED)


def test_floor_leaves_a_fast_enough_velocity_untouched():
    velocity = np.array([3.0, 0.0])

    np.testing.assert_array_equal(
        velocity_capper(velocity, MAX_SPEED, MIN_SPEED), velocity)


def test_floor_leaves_velocity_exactly_at_the_floor():
    """`0 < speed < min_speed` is strict at both ends, deliberately."""
    velocity = np.array([MIN_SPEED, 0.0])

    np.testing.assert_array_equal(
        velocity_capper(velocity, MAX_SPEED, MIN_SPEED), velocity)


def test_floor_leaves_a_motionless_bird_at_rest():
    """A zero vector has no direction to rescale, so it stays put.

    Measured over 60,000 bird-ticks: speed is exactly zero if and only if a
    bird has no neighbour in any radius AND started at rest, which never
    happens at ten birds or more. This branch is a guard against dividing by
    zero, not a behaviour anything relies on.
    """
    out = velocity_capper(np.array([0.0, 0.0]), MAX_SPEED, MIN_SPEED)

    assert not np.isnan(out).any(), "a zero vector must not be rescaled"
    np.testing.assert_array_equal(out, [0.0, 0.0])


def test_a_floor_of_zero_disables_it():
    """min_speed = 0.0 must reproduce the behaviour from before the floor."""
    velocity = np.array([0.001, 0.0])

    np.testing.assert_array_equal(
        velocity_capper(velocity, MAX_SPEED, 0.0), velocity)


def test_floor_equal_to_cap_gives_constant_speed():
    """The self-propelled-particle setup: every bird flies at exactly one speed."""
    slow = velocity_capper(np.array([0.5, 0.0]), 2.0, 2.0)
    fast = velocity_capper(np.array([9.0, 0.0]), 2.0, 2.0)

    np.testing.assert_allclose(slow, [2.0, 0.0])
    np.testing.assert_allclose(fast, [2.0, 0.0])


def test_physics_allows_a_floor_of_zero_and_a_floor_equal_to_the_cap():
    """Both ends of the range are legal and both are useful.

    min_speed = 0 means no floor, which is what ablations and any test that
    needs the floor out of the way will ask for. min_speed = max_speed is the
    constant-speed flocking model. Neither is an error.
    """
    assert Physics(min_speed=0.0).min_speed == 0.0
    assert Physics(min_speed=5.0, max_speed=5.0).min_speed == 5.0


def test_physics_rejects_a_floor_above_the_cap():
    with pytest.raises(ValueError):
        Physics(min_speed=6.0, max_speed=5.0)
    with pytest.raises(ValueError):
        Physics(min_speed=-1.0)
