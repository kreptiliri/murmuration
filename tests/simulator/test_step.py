from dataclasses import replace

import numpy as np

from boids import (calculate_alignment, calculate_cohesion,
                   calculate_separation, step, velocity_capper)
from world import (BOUNDS, DT, MAX_SPEED, MIN_SPEED, PHYSICS,
                   PHYSICS_CRUISING, SEPARATION_RADIUS, random_flock)


def _reconstruct_velocity(velocities, forces, boid_id, physics):
    total = (forces["separation"][boid_id]
             + forces["alignment"][boid_id]
             + forces["cohesion"][boid_id])
    return velocity_capper(velocities[boid_id] + total * physics.dt,
                           physics.max_speed, physics.min_speed)

def test_recorded_forces_reconstruct_the_velocity_update():
    positions = np.array([[10.0, 10.0],
                          [15.0, 10.0],
                          [80.0, 80.0]])
    velocities = np.array([[1.0, 0.0],
                           [-1.0, 0.0],
                           [0.5, 0.5]])

    _, new_velocities, forces = step(positions, velocities, PHYSICS)

    # Guard the premise. The invariant holds trivially if a rule contributes
    # nothing, so this world must actually exercise all three of them.
    for rule in ("separation", "alignment", "cohesion"):
        assert np.any(forces[rule] != 0.0), (
            f"{rule} is all zeros here, so this world does not test it"
        )

    for i in range(len(positions)):
        np.testing.assert_allclose(
            _reconstruct_velocity(velocities, forces, i, PHYSICS),
            new_velocities[i],
            err_msg=f"recorded forces do not reproduce bird {i}'s velocity update",
        )

def test_reconstruction_holds_when_the_speed_cap_fires():
    physics = replace(PHYSICS, max_speed=0.5)  # every bird below exceeds it
    positions = np.array([[10.0, 10.0],
                          [15.0, 10.0],
                          [80.0, 80.0]])
    velocities = np.array([[1.0, 0.0],
                           [-1.0, 0.0],
                           [0.5, 0.5]])

    _, new_velocities, forces = step(positions, velocities, physics)

    # Guard the test's own premise: if the cap never fires this proves nothing.
    uncapped = np.array([
        velocities[i] + (forces["separation"][i]
                         + forces["alignment"][i]
                         + forces["cohesion"][i]) * physics.dt
        for i in range(len(positions))
    ])
    assert np.any(np.linalg.norm(uncapped, axis=1) > physics.max_speed), (
        "premise failed: the cap never fired, so this test verifies nothing"
    )

    for i in range(len(positions)):
        np.testing.assert_allclose(
            _reconstruct_velocity(velocities, forces, i, physics),
            new_velocities[i],
            err_msg=f"reconstruction broke for bird {i} once the cap fired",
        )
        assert np.linalg.norm(new_velocities[i]) <= physics.max_speed + 1e-12

def test_positions_stay_inside_the_world():
    positions = np.array([[99.9, 50.0],
                          [50.0, 99.9]])
    velocities = np.array([[10.0, 0.0],
                           [0.0, 10.0]])

    new_positions, _, _ = step(positions, velocities, PHYSICS)

    assert np.all(new_positions >= 0.0)
    assert np.all(new_positions < BOUNDS)
    # confirm they wrapped rather than simply not reaching the edge
    assert new_positions[0][0] < 1.0, "bird 0 should have crossed the x seam"
    assert new_positions[1][1] < 1.0, "bird 1 should have crossed the y seam"

def test_step_reads_the_separation_weight():
    positions = np.array([[10.0, 50.0],
                          [15.0, 50.0]])
    velocities = np.zeros((2, 2))

    raw = calculate_separation(0, positions, SEPARATION_RADIUS, BOUNDS)

    _, _, forces_1x = step(positions, velocities, replace(PHYSICS, w_sep=1.0))
    _, _, forces_2x = step(positions, velocities, replace(PHYSICS, w_sep=2.0))

    # weight 1.0 must leave the raw force untouched: it multiplies, not adds
    np.testing.assert_allclose(forces_1x["separation"][0], raw)
    # and doubling the weight must double the recorded force
    np.testing.assert_allclose(forces_2x["separation"][0],
                               2.0 * forces_1x["separation"][0])

def test_step_reads_the_separation_force_cap():
    positions = np.array([[10.0, 50.0],
                          [10.5, 50.0]])   # d = 0.5, uncapped magnitude 2.0
    velocities = np.zeros((2, 2))

    _, _, tight = step(positions, velocities,
                       replace(PHYSICS, sep_force_cap=1.0, w_sep=1.0))
    _, _, loose = step(positions, velocities,
                       replace(PHYSICS, sep_force_cap=10.0, w_sep=1.0))

    np.testing.assert_allclose(tight["separation"][0], [-1.0, 0.0])
    np.testing.assert_allclose(loose["separation"][0], [-2.0, 0.0])

def test_step_reads_the_alignment_weight():
    positions = np.array([[50.0, 50.0],
                          [52.0, 50.0]])
    velocities = np.array([[1.0, 0.0],
                           [0.0, 1.0]])

    _, _, forces_1x = step(positions, velocities, replace(PHYSICS, w_ali=1.0))
    _, _, forces_2x = step(positions, velocities, replace(PHYSICS, w_ali=2.0))

    np.testing.assert_allclose(forces_1x["alignment"][0], [-1.0, 1.0])
    np.testing.assert_allclose(forces_2x["alignment"][0],
                               2.0 * forces_1x["alignment"][0])

def test_step_reads_the_alignment_radius():
    positions = np.array([[50.0, 50.0],
                          [75.0, 50.0]])   # d = 25, inside the configured 30
    velocities = np.array([[1.0, 0.0],
                           [0.0, 1.0]])

    _, _, wide = step(positions, velocities, PHYSICS)
    _, _, narrow = step(positions, velocities, replace(PHYSICS, alignment_radius=10.0))

    np.testing.assert_allclose(wide["alignment"][0], [-1.0, 1.0])
    np.testing.assert_allclose(narrow["alignment"][0], [0.0, 0.0])

def test_positions_wrap_the_lower_seam():
    positions = np.array([[0.1, 50.0]])
    velocities = np.array([[-10.0, 0.0]])   # capped to -5.0, so -0.5 per tick

    new_positions, _, _ = step(positions, velocities, PHYSICS)

    np.testing.assert_allclose(new_positions[0], [99.6, 50.0])
    assert np.all(new_positions >= 0.0)

def test_step_does_not_mutate_its_inputs():
    positions = np.array([[10.0, 10.0],
                          [15.0, 10.0]])
    velocities = np.array([[1.0, 0.0],
                           [-1.0, 0.0]])
    positions_before = positions.copy()
    velocities_before = velocities.copy()

    step(positions, velocities, PHYSICS)

    np.testing.assert_array_equal(positions, positions_before)
    np.testing.assert_array_equal(velocities, velocities_before)


def test_step_reads_the_cohesion_weight():
    positions = np.array([[50.0, 50.0],
                          [60.0, 50.0]])
    velocities = np.zeros((2, 2))

    _, _, forces_1x = step(positions, velocities, replace(PHYSICS, w_coh=1.0))
    _, _, forces_2x = step(positions, velocities, replace(PHYSICS, w_coh=2.0))

    np.testing.assert_allclose(forces_1x["cohesion"][0], [0.25, 0.0])
    np.testing.assert_allclose(forces_2x["cohesion"][0],
                               2.0 * forces_1x["cohesion"][0])


def test_step_reads_the_cohesion_radius():
    positions = np.array([[50.0, 50.0],
                          [80.0, 50.0]])   # 30 apart
    velocities = np.zeros((2, 2))

    _, _, wide = step(positions, velocities,
                      replace(PHYSICS, cohesion_radius=40.0, w_coh=1.0))
    _, _, narrow = step(positions, velocities,
                        replace(PHYSICS, cohesion_radius=10.0, w_coh=1.0))

    np.testing.assert_allclose(wide["cohesion"][0], [0.75, 0.0])
    np.testing.assert_allclose(narrow["cohesion"][0], [0.0, 0.0])


def test_step_records_all_three_rules_separately():
    positions = np.array([[50.0, 50.0],
                          [55.0, 50.0],
                          [50.0, 58.0]])
    velocities = np.array([[1.0, 0.0],
                           [0.0, 2.0],
                           [-1.0, 1.0]])

    _, _, forces = step(positions, velocities, PHYSICS)

    for rule in ("separation", "alignment", "cohesion"):
        assert np.any(forces[rule] != 0.0), f"{rule} did not fire"

    assert not np.allclose(forces["separation"], forces["alignment"])
    assert not np.allclose(forces["separation"], forces["cohesion"])
    assert not np.allclose(forces["alignment"], forces["cohesion"])


def test_step_reads_the_min_speed():
    positions = np.array([[10.0, 10.0], [60.0, 60.0]])
    velocities = np.array([[1.0, 0.0], [-1.0, 0.0]])

    _, drifting, _ = step(positions, velocities, PHYSICS)
    _, cruising, _ = step(positions, velocities, PHYSICS_CRUISING)

    np.testing.assert_allclose(drifting, velocities)
    np.testing.assert_allclose(cruising, [[MIN_SPEED, 0.0], [-MIN_SPEED, 0.0]])


def test_step_reads_the_dt():
    # neighbours in range, so the forces are non-zero and dt has to scale them
    positions = np.array([[50.0, 50.0], [55.0, 50.0]])
    velocities = np.zeros((2, 2))

    p_slow, v_slow, f_slow = step(positions, velocities, replace(PHYSICS, dt=0.1))
    p_fast, v_fast, f_fast = step(positions, velocities, replace(PHYSICS, dt=0.2))

    # the forces themselves do not depend on dt - only how much of them is applied
    np.testing.assert_allclose(f_slow["separation"], f_fast["separation"])
    # twice the timestep, twice the velocity change from the same force
    np.testing.assert_allclose(v_fast, 2.0 * v_slow)
    # and the position update uses it too: dx = v_new * dt, so four times as far
    np.testing.assert_allclose(p_fast - positions, 4.0 * (p_slow - positions))


def test_step_reads_the_bounds():
    small_world = replace(PHYSICS, bounds=50.0, separation_radius=20.0,
                          alignment_radius=20.0, cohesion_radius=20.0)
    positions = np.array([[49.9, 25.0]])
    velocities = np.array([[5.0, 0.0]])

    new_positions, _, _ = step(positions, velocities, small_world)

    np.testing.assert_allclose(new_positions[0], [0.4, 25.0])


def test_recorded_forces_match_the_rules_applied_to_the_input_state():
    # the shipped world: soft radius edges and the per-rule cap, so each label
    # is its rule at the shipped softness, times its weight, then capped
    from boids import force_capper

    physics = PHYSICS_CRUISING
    positions, velocities = random_flock(seed=7, n_birds=12)

    _, _, forces = step(positions, velocities, physics)

    soft, cap = physics.edge_softness, physics.max_force
    for i in range(len(positions)):
        np.testing.assert_allclose(
            forces["separation"][i],
            force_capper(physics.w_sep * calculate_separation(
                i, positions, physics.separation_radius, physics.bounds,
                physics.sep_force_cap, softness=soft), cap),
            err_msg=f"separation label wrong for bird {i}")
        np.testing.assert_allclose(
            forces["alignment"][i],
            force_capper(physics.w_ali * calculate_alignment(
                i, positions, velocities, physics.alignment_radius, physics.bounds,
                softness=soft), cap),
            err_msg=f"alignment label wrong for bird {i}")
        np.testing.assert_allclose(
            forces["cohesion"][i],
            force_capper(physics.w_coh * calculate_cohesion(
                i, positions, physics.cohesion_radius, physics.bounds,
                softness=soft), cap),
            err_msg=f"cohesion label wrong for bird {i}")

    # premise guard: a frozen flock would satisfy all of the above trivially
    for rule in ("separation", "alignment", "cohesion"):
        assert np.linalg.norm(forces[rule], axis=1).max() > 0, f"{rule} never fired"


# --- dtype: what survives being written to disk and read back ----------------

def test_step_accepts_integer_input_without_truncating():
    positions = np.array([[10, 10], [15, 10], [80, 80]])
    velocities = np.array([[1, 0], [-1, 0], [0, 1]])
    assert positions.dtype == np.int64 and velocities.dtype == np.int64

    new_positions, new_velocities, forces = step(positions, velocities, PHYSICS)

    assert new_positions.dtype == np.float64
    assert new_velocities.dtype == np.float64
    # bird 0 moves 1.0 * dt = 0.1 in x, plus a small push from its neighbour
    assert new_positions[0][0] != int(new_positions[0][0]), (
        "position landed on a whole number: the output array inherited an "
        "integer dtype and the world has been quantised"
    )


def test_step_always_computes_in_float64():
    positions, velocities = random_flock(seed=0, n_birds=10)
    positions, velocities = positions.astype(np.float32), velocities.astype(np.float32)

    new_positions, new_velocities, forces = step(positions, velocities, PHYSICS_CRUISING)

    assert new_positions.dtype == np.float64
    assert new_velocities.dtype == np.float64
    for rule, force in forces.items():
        assert force.dtype == np.float64, f"{rule} came back as {force.dtype}"


def test_the_reconstruction_invariant_is_exact_in_float64():
    physics = PHYSICS_CRUISING
    positions, velocities = random_flock(seed=3, n_birds=15)

    for tick in range(50):
        old_velocities = velocities.copy()
        positions, velocities, forces = step(positions, velocities, physics)
        total = forces["separation"] + forces["alignment"] + forces["cohesion"]

        for i in range(len(positions)):
            np.testing.assert_array_equal(
                velocity_capper(old_velocities[i] + total[i] * physics.dt,
                                physics.max_speed, physics.min_speed),
                velocities[i],
                err_msg=f"reconstruction is no longer exact for bird {i} at tick {tick}")


def test_the_reconstruction_invariant_survives_float32_storage():
    physics = PHYSICS_CRUISING
    positions, velocities = random_flock(seed=3, n_birds=15)
    worst = 0.0

    for _ in range(100):
        old_velocities = velocities.copy()
        positions, velocities, forces = step(positions, velocities, physics)
        total = forces["separation"] + forces["alignment"] + forces["cohesion"]

        def stored(a):                      # write to disk as float32, read back
            return a.astype(np.float32).astype(np.float64)

        rebuilt = np.array([
            velocity_capper(stored(old_velocities)[i] + stored(total)[i] * physics.dt,
                            physics.max_speed, physics.min_speed)
            for i in range(len(positions))])
        worst = max(worst, np.abs(rebuilt - stored(velocities)).max())

    assert worst < 1e-6, f"float32 round trip lost {worst:.3e}, more than the cast alone"
    assert worst > 0.0, (
        "premise failed: the float32 round trip was lossless, so this test did "
        "not exercise the precision loss it exists to bound"
    )
