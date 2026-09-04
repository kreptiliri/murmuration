"""The simulator run forwards, rather than one tick at a time.

Every other file in this suite checks a single call. These check what happens
over hundreds of them, which is the regime the dataset is actually generated
in and where a different class of bug lives: a NaN that appears on tick 300
and spreads, a position that drifts outside the world, an integrator that
quietly pumps energy into the flock until every bird is pinned at the speed
cap. None of those can be caught by looking at one tick.
"""

from dataclasses import replace

import numpy as np

from boids import Physics, step, velocity_capper
from world import BOUNDS, DT, MAX_SPEED, PHYSICS, PHYSICS_CRUISING, random_flock

TICKS = 200
N_BIRDS = 20


def test_no_nan_ever_appears():
    """The failure this whole project is most exposed to.

    A NaN in a force is not an error - it propagates into the velocity, then
    the position, then every subsequent tick, and the run completes. The
    dataset would be written, the model would train on it, and the labels
    would be garbage from some arbitrary tick onwards.
    """
    positions, velocities = random_flock(seed=0, n_birds=N_BIRDS)

    for tick in range(TICKS):
        positions, velocities, forces = step(positions, velocities, PHYSICS)
        assert not np.isnan(positions).any(), f"NaN position at tick {tick}"
        assert not np.isnan(velocities).any(), f"NaN velocity at tick {tick}"
        for rule, force in forces.items():
            assert not np.isnan(force).any(), f"NaN {rule} force at tick {tick}"


def test_positions_stay_inside_the_world_for_every_tick():
    """`pos %= bounds` every tick is the invariant get_torus depends on.

    If a position ever escapes [0, bounds), the minimum-image convention
    silently starts returning the wrong displacement and every rule that uses
    it is computed against a world that has come apart.
    """
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
    """Run a flock forwards, re-deriving every velocity from recorded forces.

    Returns the number of ticks on which the speed cap actually engaged, so
    the caller can guard its own premise.
    """
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
    """The load-bearing test of this project, run forwards.

    The single-tick version proves the recorded forces reproduce one velocity
    update in a hand-built world. This proves it for every bird on every one
    of two hundred ticks of a real flock - through seam crossings, through
    birds that lose all their neighbours, through the whole settling process.

    If this ever fails, the numbers in the forces dict are not labels for this
    simulation, and every probe and ablation result downstream is fitted to
    something that never happened.
    """
    _assert_reconstruction_holds(PHYSICS, seed=3)


def test_the_reconstruction_invariant_holds_when_the_cap_fires_constantly():
    """The same invariant in the regime where forces do not simply sum.

    Once the speed cap engages, the relationship is cap(v + sum(f)*dt) == v',
    not v + sum(f)*dt == v'. A settled flock almost never reaches its cap -
    measured, 1 tick in 200 - so an ordinary rollout barely tests this at all.
    Strong separation against a low cap saturates it on every single tick.
    """
    physics = replace(PHYSICS, max_speed=0.3, w_sep=5.0)

    ticks_where_the_cap_fired = _assert_reconstruction_holds(physics, seed=3)

    assert ticks_where_the_cap_fired == TICKS, (
        f"premise failed: the cap fired on only {ticks_where_the_cap_fired} of "
        f"{TICKS} ticks, so this barely tested the capped regime"
    )


def test_the_flock_settles_rather_than_being_pinned_at_the_cap():
    """A flock that has found its shape should be moving gently, not flat out.

    If the integrator pumped energy in - which plain forward Euler does to a
    spring force like cohesion - every bird would end up saturated at the
    speed cap and stay there. Measured: mean speed falls from ~3.7 to ~0.6 and
    the fraction of birds at the cap goes to zero.
    """
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
    """Cohesion is Hooke's law, and springs are where integrators go wrong.

    Two birds, cohesion only, oscillating about their shared centre of mass.
    Under plain forward Euler the amplitude would grow every period until the
    speed cap took over. It does not, because step() advances position using
    the *new* velocity - semi-implicit (symplectic) Euler - which conserves
    energy for oscillators. That one line is load-bearing and this pins it.
    """
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
    """Same seed, same physics, identical trajectory - bit for bit.

    Without this the dataset cannot be regenerated from its own metadata, and
    a checkpoint sweep cannot be compared against a rerun of the world it was
    trained on.
    """
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
    """No neighbours, no forces, pure translation.

    The two birds here are 70.7 units apart on the torus, outside all three
    radii. Their velocities must not change at all, and their positions must
    advance by exactly v*dt each tick. Any drift means a rule is firing when
    nothing is in range.
    """
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
    """The load-bearing invariant under the new clamp.

    Adding the floor changed the function on the outside of
    `clamp(v + sum(f)*dt) == v'`. The recorded forces must still reproduce the
    update, on every bird on every tick, with the floor engaged.
    """
    _assert_reconstruction_holds(PHYSICS_CRUISING, seed=3)


def test_a_cruising_flock_keeps_moving():
    """Why the floor exists at all.

    Without it, alignment steers each bird toward its neighbours' mean
    *vector*, and the mean of scattered vectors is short - so the flock
    organises its headings and brakes to a crawl in the same motion. Measured
    without the floor: mean speed falls to about 0.27 and stays there.
    """
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
    """The screenshot, as an assertion.

    Random headings, 500 ticks, floor engaged: every bird must sit inside
    [min_speed, max_speed] at every single tick, not merely at the end. The
    picture shows this as every arrow being the same length from tick 25
    onwards; this is that claim in a form that fails loudly.

    It would also catch a bird stalling to exactly zero, which the floor cannot
    rescue because a zero vector has no direction - measured to happen only in
    flocks too small to be flocks, and this pins that it stays that way.
    """
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

    # Measured at this density: the floor holds at least one bird on about
    # 30% of ticks. At 60 birds it is every tick; at 20 the flock is sparse
    # enough that many birds drift above it on their own.
    assert floor_engaged > 50, (
        f"premise failed: the floor only engaged on {floor_engaged} of 500 "
        "ticks, so this mostly tested a flock that never needed it"
    )


def test_the_shipped_defaults_produce_a_sane_rollout():
    """Physics() with no arguments is what a careless caller gets.

    Since world.py began deriving from Physics() this is nearly what
    PHYSICS_CRUISING already is, so the gap this test was written to cover has
    mostly closed. It stays because it constructs Physics() literally: if a
    future default were changed without world.py being re-read, this is the
    test that runs against the new value rather than a remembered one.
    """
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
