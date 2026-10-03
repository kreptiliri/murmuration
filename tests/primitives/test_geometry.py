from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

from boids import Physics, get_torus, velocity_capper
from world import (ALIGNMENT_RADIUS, BOUNDS, COHESION_RADIUS, DT, MAX_SPEED,
                   MIN_SPEED, PHYSICS, PHYSICS_CRUISING, PHYSICS_V1,
                   SEPARATION_RADIUS)


def test_short_delta_passes_through_untouched():
    a = np.array([10.0, 10.0])
    b = np.array([15.0, 10.0])

    delta = get_torus(a, b, BOUNDS)

    # x: 10 - 15 = -5, y: 0. Both well inside half a screen (50).
    np.testing.assert_allclose(delta, [-5.0, 0.0])

def test_wraps_positive_delta():
    a = np.array([95.0, 50.0])
    b = np.array([5.0, 50.0])

    delta = get_torus(a, b, BOUNDS)

    np.testing.assert_allclose(delta, [-10.0, 0.0])

def test_wraps_negative_delta():
    a = np.array([5.0, 50.0])
    b = np.array([95.0, 50.0])

    delta = get_torus(a, b, BOUNDS)

    np.testing.assert_allclose(delta, [10.0, 0.0])

def test_wrap_is_decided_per_axis():
    a = np.array([95.0, 15.0])
    b = np.array([5.0, 5.0])

    delta = get_torus(a, b, BOUNDS)

    # x: raw 90 -> short way round is 10 the other direction -> -10
    # y: raw 10 -> already shortest -> unchanged
    np.testing.assert_allclose(delta, [-10.0, 10.0])

def test_handles_position_wound_past_the_bounds():
    a = np.array([230.0, 0.0])
    b = np.array([0.0, 0.0])

    delta = get_torus(a, b, BOUNDS)

    # 230 is 2.3 screens out; 230 mod 100 = 30, which is under half a screen.
    np.testing.assert_allclose(delta, [30.0, 0.0])

def test_capper_leaves_slow_velocity_untouched():
    v = np.array([1.0, 2.0])  # speed ~2.24, well under 5

    np.testing.assert_array_equal(velocity_capper(v, MAX_SPEED), v)

def test_capper_shortens_the_arrow_without_turning_it():
    v = np.array([6.0, 8.0])  # speed exactly 10
    out = velocity_capper(v, MAX_SPEED)

    assert np.linalg.norm(out) == pytest.approx(MAX_SPEED)
    np.testing.assert_allclose(out / np.linalg.norm(out), v / np.linalg.norm(v))
    np.testing.assert_allclose(out, [3.0, 4.0])

def test_capper_leaves_velocity_exactly_at_the_cap():
    v = np.array([3.0, 4.0])  # speed exactly 5

    np.testing.assert_allclose(velocity_capper(v, MAX_SPEED), v)

def test_capper_handles_a_motionless_bird():
    out = velocity_capper(np.array([0.0, 0.0]), MAX_SPEED)

    assert not np.isnan(out).any(), "NaN velocity would spread to every later tick"
    np.testing.assert_array_equal(out, [0.0, 0.0])

def test_physics_is_immutable():
    physics = Physics()

    with pytest.raises(FrozenInstanceError):
        physics.dt = 0.2

def test_physics_rejects_a_radius_at_or_beyond_half_the_world():
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
    heading = np.array([0.6, 0.8])           # a unit vector, chosen to be exact
    out = velocity_capper(heading * MIN_SPEED / 2, MAX_SPEED, MIN_SPEED)

    np.testing.assert_allclose(out, heading * MIN_SPEED)
    assert np.linalg.norm(out) == pytest.approx(MIN_SPEED)


def test_floor_leaves_a_fast_enough_velocity_untouched():
    velocity = np.array([3.0, 0.0])

    np.testing.assert_array_equal(
        velocity_capper(velocity, MAX_SPEED, MIN_SPEED), velocity)


def test_floor_leaves_velocity_exactly_at_the_floor():
    velocity = np.array([MIN_SPEED, 0.0])

    np.testing.assert_array_equal(
        velocity_capper(velocity, MAX_SPEED, MIN_SPEED), velocity)


def test_floor_leaves_a_motionless_bird_at_rest():
    out = velocity_capper(np.array([0.0, 0.0]), MAX_SPEED, MIN_SPEED)

    assert not np.isnan(out).any(), "a zero vector must not be rescaled"
    np.testing.assert_array_equal(out, [0.0, 0.0])


def test_a_floor_of_zero_disables_it():
    velocity = np.array([0.001, 0.0])

    np.testing.assert_array_equal(
        velocity_capper(velocity, MAX_SPEED, 0.0), velocity)


def test_floor_equal_to_cap_gives_constant_speed():
    slow = velocity_capper(np.array([0.5, 0.0]), 2.0, 2.0)
    fast = velocity_capper(np.array([9.0, 0.0]), 2.0, 2.0)

    np.testing.assert_allclose(slow, [2.0, 0.0])
    np.testing.assert_allclose(fast, [2.0, 0.0])


def test_physics_allows_a_floor_of_zero_and_a_floor_equal_to_the_cap():
    assert Physics(min_speed=0.0).min_speed == 0.0
    assert Physics(min_speed=5.0, max_speed=5.0).min_speed == 5.0


def test_physics_rejects_a_floor_above_the_cap():
    with pytest.raises(ValueError):
        Physics(min_speed=6.0, max_speed=5.0)
    with pytest.raises(ValueError):
        Physics(min_speed=-1.0)


def test_the_test_baseline_tracks_the_shipped_config():
    assert PHYSICS_CRUISING == Physics(), (
        "PHYSICS_CRUISING must be boids.py's Physics() unchanged")

    shipped, baseline = asdict(Physics()), asdict(PHYSICS)
    differing = {k for k in shipped if shipped[k] != baseline[k]}
    assert differing == {"min_speed", "edge_softness", "max_force"}, (
        f"the test baseline differs from the shipped config in {differing}, "
        "but min_speed, edge_softness and max_force are the only differences "
        "it is allowed to have"
    )
    assert (PHYSICS.min_speed, PHYSICS.edge_softness, PHYSICS.max_force) == (0.0, 0.0, None)
    assert PHYSICS_V1 == replace(Physics(), edge_softness=0.0, max_force=None), (
        "PHYSICS_V1 must be the shipped config with only the soft edges and the cap off")

    # And the module-level names must be the config's own values, not copies.
    # Checking only the Physics objects leaves the original bug available: a
    # literal reintroduced here would disagree with PHYSICS.alignment_radius
    # while both objects stayed correct. A mutation test found exactly that
    # hole in the first version of this test.
    assert (BOUNDS, SEPARATION_RADIUS, ALIGNMENT_RADIUS,
            COHESION_RADIUS, MAX_SPEED, DT) == (
        PHYSICS.bounds, PHYSICS.separation_radius, PHYSICS.alignment_radius,
        PHYSICS.cohesion_radius, PHYSICS.max_speed, PHYSICS.dt), (
        "a world.py constant has been restated as a literal instead of derived")
    assert MIN_SPEED == PHYSICS_CRUISING.min_speed
