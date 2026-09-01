"""step(): the integrator, and the reconstruction invariant.

The invariant is the load-bearing test in this project. Every probe and every
ablation is trained against the numbers in the forces dict, so if those numbers
cannot reproduce the velocity update the simulator actually performed, they are
not labels for this simulation and every downstream result is fitted to
something that never happened.
"""

from dataclasses import replace

import numpy as np

from boids import calculate_separation, step, velocity_capper
from world import BOUNDS, DT, MAX_SPEED, PHYSICS, SEPARATION_RADIUS


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

    _, new_velocities, forces = step(positions, velocities, PHYSICS)

    # Guard the premise. The invariant holds trivially if a rule contributes
    # nothing, so this world must actually exercise all three of them.
    for rule in ("separation", "alignment", "cohesion"):
        assert np.any(forces[rule] != 0.0), (
            f"{rule} is all zeros here, so this world does not test it"
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
        positions, velocities, replace(PHYSICS, max_speed=max_speed)
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

    new_positions, _, _ = step(positions, velocities, PHYSICS)

    assert np.all(new_positions >= 0.0)
    assert np.all(new_positions < BOUNDS)
    # confirm they wrapped rather than simply not reaching the edge
    assert new_positions[0][0] < 1.0, "bird 0 should have crossed the x seam"
    assert new_positions[1][1] < 1.0, "bird 1 should have crossed the y seam"

def test_step_reads_the_separation_weight():
    """`w_sep` must scale the recorded force, and proportionally.

    Deleting `* w_sep` from `step` passed all twenty of the tests above. The
    weights are not decoration - NOTES.md makes unequal weights the mechanism
    that spreads the three rules' acquisition times apart, so a silently
    ignored weight does not crash anything, it just deletes the experiment's
    independent variable.

    This asserts the ratio rather than a hardcoded 1.5, so retuning the
    weights later does not turn this into a false alarm.
    """
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
    """`sep_force_cap` must reach `calculate_separation`.

    A parameter that `step` accepts and then drops is invisible: the callee's
    default silently sets the physics, the top-level config that gets recorded
    alongside the dataset says nothing about it, and the run is not
    reproducible from its own metadata.
    """
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
    """The same pair of birds, neighbours under one radius and not the other."""
    positions = np.array([[50.0, 50.0],
                          [80.0, 50.0]])   # d = 30
    velocities = np.array([[1.0, 0.0],
                           [0.0, 1.0]])

    _, _, wide = step(positions, velocities, replace(PHYSICS, alignment_radius=35.0))
    _, _, narrow = step(positions, velocities, replace(PHYSICS, alignment_radius=10.0))

    np.testing.assert_allclose(wide["alignment"][0], [-1.0, 1.0])
    np.testing.assert_allclose(narrow["alignment"][0], [0.0, 0.0])

def test_positions_wrap_the_lower_seam():
    """Crossing x = 0 travelling in -x.

    The suite already covers the upper seam (99.9 -> 0.4). The lower one
    relies on Python's modulo returning a non-negative result for a positive
    modulus - which is a language guarantee, not a universal one; C's fmod
    would leave -0.4 here. `get_torus` assumes every position lives in
    [0, bounds), so that assumption is worth pinning.
    """
    positions = np.array([[0.1, 50.0]])
    velocities = np.array([[-10.0, 0.0]])   # capped to -5.0, so -0.5 per tick

    new_positions, _, _ = step(positions, velocities, PHYSICS)

    np.testing.assert_allclose(new_positions[0], [99.6, 50.0])
    assert np.all(new_positions >= 0.0)

def test_step_does_not_mutate_its_inputs():
    """Double-buffering: every force for tick t comes from the state at t.

    `step` currently builds fresh arrays, so this passes today. It is here
    because the obvious optimisation later - doing `positions %= bounds` in
    place instead of allocating - would compute some birds' forces against a
    half-updated world. The forces would still be internally consistent, so
    the reconstruction test would keep passing while the labels described a
    world that never existed.
    """
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
    """The same pair of birds, inside one radius and outside the other."""
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
    """The forces dict is the label set. Each key must carry its own rule.

    A world arranged so that all three rules fire at once, with three
    distinct answers: if any two keys ever hold the same array, the probes
    would be trained to distinguish rules the simulator never separated.
    """
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
