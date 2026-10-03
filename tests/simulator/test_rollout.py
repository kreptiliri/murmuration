from dataclasses import replace

import numpy as np

from boids import Physics, step, velocity_capper
from world import BOUNDS, DT, MAX_SPEED, PHYSICS, PHYSICS_CRUISING, random_flock

TICKS = 200
N_BIRDS = 20


def test_no_nan_ever_appears():
    positions, velocities = random_flock(seed=0, n_birds=N_BIRDS)

    for tick in range(TICKS):
        positions, velocities, forces = step(positions, velocities, PHYSICS)
        assert not np.isnan(positions).any(), f"NaN position at tick {tick}"
        assert not np.isnan(velocities).any(), f"NaN velocity at tick {tick}"
        for rule, force in forces.items():
            assert not np.isnan(force).any(), f"NaN {rule} force at tick {tick}"


def test_positions_stay_inside_the_world_for_every_tick():
    positions, velocities = random_flock(seed=1, n_birds=N_BIRDS)

    for tick in range(TICKS):
        positions, velocities, _ = step(positions, velocities, PHYSICS)
        assert np.all(positions >= 0.0), f"bird left the world at tick {tick}"
        assert np.all(positions < BOUNDS), f"bird left the world at tick {tick}"


def test_speed_never_exceeds_the_cap():
    positions, velocities = random_flock(seed=2, n_birds=N_BIRDS)

    for tick in range(TICKS):
        positions, velocities, _ = step(positions, velocities, PHYSICS)
        fastest = np.linalg.norm(velocities, axis=1).max()
        assert fastest <= MAX_SPEED + 1e-9, (
            f"speed {fastest} exceeded the cap at tick {tick}"
        )


def _assert_reconstruction_holds(physics, seed, ticks=TICKS):
    positions, velocities = random_flock(seed=seed, n_birds=N_BIRDS)
    ticks_where_the_cap_fired = 0

    for tick in range(ticks):
        old_velocities = velocities.copy()
        positions, velocities, forces = step(positions, velocities, physics)

        totals = forces["separation"] + forces["alignment"] + forces["cohesion"]
        uncapped = old_velocities + totals * physics.dt
        if np.linalg.norm(uncapped, axis=1).max() > physics.max_speed:
            ticks_where_the_cap_fired += 1

        for i in range(N_BIRDS):
            np.testing.assert_allclose(
                velocity_capper(uncapped[i], physics.max_speed, physics.min_speed),
                velocities[i],
                err_msg=f"reconstruction broke for bird {i} at tick {tick}",
            )

    return ticks_where_the_cap_fired


def test_the_reconstruction_invariant_holds_on_every_tick():
    _assert_reconstruction_holds(PHYSICS, seed=3)


def test_the_reconstruction_invariant_holds_when_the_cap_fires_constantly():
    physics = replace(PHYSICS, max_speed=0.3, w_sep=5.0)

    ticks_where_the_cap_fired = _assert_reconstruction_holds(physics, seed=3)

    assert ticks_where_the_cap_fired == TICKS, (
        f"premise failed: the cap fired on only {ticks_where_the_cap_fired} of "
        f"{TICKS} ticks, so this barely tested the capped regime"
    )


def test_the_flock_settles_rather_than_being_pinned_at_the_cap():
    positions, velocities = random_flock(seed=4, n_birds=N_BIRDS)
    for _ in range(TICKS):
        positions, velocities, _ = step(positions, velocities, PHYSICS)

    speeds = np.linalg.norm(velocities, axis=1)
    at_the_cap = (speeds > MAX_SPEED - 1e-9).mean()

    assert at_the_cap < 0.1, (
        f"{at_the_cap:.0%} of birds are pinned at the speed cap; the flock is "
        "being driven rather than settling"
    )


def test_the_cohesion_spring_does_not_gain_energy():
    physics = replace(PHYSICS, w_sep=0.0, w_ali=0.0, w_coh=1.0)
    positions = np.array([[45.0, 50.0], [55.0, 50.0]])
    velocities = np.zeros((2, 2))

    separations = []
    for _ in range(4000):
        positions, velocities, _ = step(positions, velocities, physics)
        offset = (positions[1] - positions[0] + BOUNDS / 2) % BOUNDS - BOUNDS / 2
        separations.append(np.linalg.norm(offset))
    separations = np.array(separations)

    early = separations[:400].max()
    late = separations[-400:].max()

    # the cap must stay out of the way or it would mask any growth
    assert np.linalg.norm(velocities, axis=1).max() < MAX_SPEED
    assert late < early * 1.01, (
        f"oscillation grew from {early:.3f} to {late:.3f} over 4000 ticks - "
        "check that positions advance using the new velocity, not the old"
    )


def test_a_rollout_is_reproducible():
    first_positions, first_velocities = random_flock(seed=5, n_birds=10)
    second_positions, second_velocities = random_flock(seed=5, n_birds=10)

    for _ in range(50):
        first_positions, first_velocities, first_forces = step(
            first_positions, first_velocities, PHYSICS)
        second_positions, second_velocities, second_forces = step(
            second_positions, second_velocities, PHYSICS)

    np.testing.assert_array_equal(first_positions, second_positions)
    np.testing.assert_array_equal(first_velocities, second_velocities)
    for rule in first_forces:
        np.testing.assert_array_equal(first_forces[rule], second_forces[rule])


def test_two_birds_beyond_every_radius_just_drift():
    positions = np.array([[10.0, 10.0], [60.0, 60.0]])
    velocities = np.array([[1.0, 0.5], [-2.0, 0.0]])
    starting_velocities = velocities.copy()
    expected = positions.copy()

    for _ in range(100):
        positions, velocities, forces = step(positions, velocities, PHYSICS)
        expected = (expected + starting_velocities * DT) % BOUNDS

    np.testing.assert_allclose(velocities, starting_velocities)
    np.testing.assert_allclose(positions, expected)
    for rule, force in forces.items():
        np.testing.assert_array_equal(force, np.zeros((2, 2)), err_msg=rule)


def test_the_reconstruction_invariant_holds_with_the_speed_floor():
    _assert_reconstruction_holds(PHYSICS_CRUISING, seed=3)


def test_a_cruising_flock_keeps_moving():
    positions, velocities = random_flock(seed=0, n_birds=N_BIRDS)
    for _ in range(TICKS):
        positions, velocities, _ = step(positions, velocities, PHYSICS_CRUISING)
    cruising = np.linalg.norm(velocities, axis=1)

    positions, velocities = random_flock(seed=0, n_birds=N_BIRDS)
    for _ in range(TICKS):
        positions, velocities, _ = step(positions, velocities, PHYSICS)
    drifting = np.linalg.norm(velocities, axis=1)

    assert cruising.min() >= PHYSICS_CRUISING.min_speed - 1e-9
    assert cruising.mean() < PHYSICS_CRUISING.min_speed * 1.5
    assert drifting.mean() < PHYSICS_CRUISING.min_speed / 2, (
        "premise failed: the flock did not slow down without the floor, so "
        "this comparison shows nothing"
    )


def test_speed_stays_between_the_floor_and_the_cap_on_every_tick():
    physics = PHYSICS_CRUISING
    positions, velocities = random_flock(seed=11, n_birds=N_BIRDS)
    floor_engaged = 0

    for tick in range(500):
        positions, velocities, _ = step(positions, velocities, physics)
        speeds = np.linalg.norm(velocities, axis=1)
        assert speeds.min() >= physics.min_speed - 1e-9, (
            f"bird below the floor at tick {tick}: {speeds.min()}")
        assert speeds.max() <= physics.max_speed + 1e-9, (
            f"bird above the cap at tick {tick}: {speeds.max()}")
        floor_engaged += int(np.isclose(speeds.min(), physics.min_speed))

    # The guard exists so this test cannot go quietly vacuous if the floor
    # stops engaging at all. Measured at N_BIRDS, seed 11, 500 ticks: 24 of 500
    # ticks at min_speed 1.0, against roughly 150 of 500 when the floor was 2.0.
    # Lowering the floor is what moved it, and 24 still exercises the branch.
    assert floor_engaged > 10, (
        f"premise failed: the floor only engaged on {floor_engaged} of 500 "
        "ticks, so this mostly tested a flock that never needed it"
    )


def test_the_shipped_defaults_produce_a_sane_rollout():
    physics = Physics()
    positions, velocities = random_flock(seed=13, n_birds=N_BIRDS)

    for tick in range(TICKS):
        positions, velocities, forces = step(positions, velocities, physics)
        speeds = np.linalg.norm(velocities, axis=1)
        assert not np.isnan(positions).any(), f"NaN position at tick {tick}"
        assert np.all(positions >= 0.0) and np.all(positions < physics.bounds)
        assert physics.min_speed - 1e-9 <= speeds.min()
        assert speeds.max() <= physics.max_speed + 1e-9
        for rule, force in forces.items():
            assert not np.isnan(force).any(), f"NaN {rule} force at tick {tick}"
