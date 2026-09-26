import numpy as np
import pytest

from boids import (calculate_alignment, calculate_cohesion,
                   calculate_separation, velocity_capper)
from world import PHYSICS_CRUISING

generate = pytest.importorskip(
    "generate",
    reason="generate.py does not exist yet, this suite is its specification")

PHYSICS = PHYSICS_CRUISING      # the shipped config, floor on
N_SLOTS = 12
N_TICKS = 20                    # short on purpose: contract tests, not rollouts


def clip(n_live=N_SLOTS, n_ticks=N_TICKS, seed=0, n_slots=N_SLOTS, physics=PHYSICS):
    return generate.generate_clip(physics, n_slots=n_slots, n_live=n_live,
                                  n_ticks=n_ticks, seed=seed)


def polarisation(velocities):
    """Length of the mean unit heading: 0 is chaos, 1 is one flock."""
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    return np.linalg.norm((velocities / np.where(speeds == 0, 1, speeds)).mean(axis=0))


# --- polarisation ------------------------------------------------------------

def test_polarisation_is_one_when_every_bird_flies_the_same_way():
    v = np.array([[3.0, 0.0], [3.0, 0.0], [3.0, 0.0]])

    assert generate.polarisation(v) == pytest.approx(1.0)


def test_polarisation_ignores_speed_entirely():
    """The definition is the mean *unit* heading, so speed must not vote.

    Same three headings as above at wildly different speeds. If this returns
    anything but 1.0, the function is averaging velocities rather than
    directions.
    """
    v = np.array([[2.0, 0.0], [5.0, 0.0], [2.5, 0.0]])

    assert generate.polarisation(v) == pytest.approx(1.0)


def test_polarisation_is_zero_for_two_birds_flying_at_each_other():
    """The regression test for a bug this function actually had.

    The first implementation was |mean velocity| / mean speed, which lets a fast
    bird outvote a slow one: these two score 0.43 under it and 0.0 under the
    definition. Two birds flying directly at each other are not 43% ordered.
    """
    v = np.array([[5.0, 0.0], [-2.0, 0.0]])

    assert generate.polarisation(v) == pytest.approx(0.0)


def test_polarisation_of_perpendicular_headings():
    """A value that is neither 0 nor 1, so the scale is checked and not just its ends."""
    v = np.array([[4.0, 0.0], [0.0, 1.0]])

    assert generate.polarisation(v) == pytest.approx(1 / np.sqrt(2))


def test_polarisation_survives_a_stationary_bird():
    """Normalising divides by each bird's speed, and a still bird's speed is 0.

    min_speed keeps this out of generated clips, but the function is called on
    hand-built flocks too, and a NaN here would propagate silently through every
    mean downstream rather than raising.
    """
    v = np.array([[3.0, 0.0], [0.0, 0.0]])
    result = generate.polarisation(v)

    assert not np.isnan(result)
    assert result == pytest.approx(0.5), "the still bird should count as no vote, not as a heading"


def test_polarisation_does_not_depend_on_flock_orientation():
    """Rotating the whole flock cannot change how ordered it is."""
    v = np.array([[2.0, 1.0], [3.0, -1.0], [2.5, 0.5]])
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    assert generate.polarisation(v @ rotation.T) == pytest.approx(generate.polarisation(v))


# --- the shape contract ------------------------------------------------------

def test_the_shape_contract():
    """One more state than there are forces, and the asymmetry stays visible.

    n_ticks steps produce n_ticks force sets and n_ticks + 1 states. Padding the
    force arrays to match would fabricate a label for tick 0, where no force has
    been computed yet.
    """
    c = clip()

    assert c["positions"].shape == (N_TICKS + 1, N_SLOTS, 2)
    assert c["velocities"].shape == (N_TICKS + 1, N_SLOTS, 2)
    for rule in ("separation", "alignment", "cohesion"):
        assert c[rule].shape == (N_TICKS, N_SLOTS, 2), rule
    assert c["mask"].shape == (N_TICKS + 1, N_SLOTS)
    assert c["kind"].shape == (N_SLOTS,)
    assert isinstance(int(c["settle_tick"]), int)


def test_arrays_are_float64_in_memory():
    """float32 is the writer's decision, made once, at the point of saving.

    Casting here would lose precision before anything has been written, and
    would weaken the tick-alignment test from exact equality to a tolerance.
    """
    c = clip()

    for key in ("positions", "velocities", "separation", "alignment", "cohesion"):
        assert c[key].dtype == np.float64, f"{key} is {c[key].dtype}"
    assert c["mask"].dtype == np.bool_
    assert c["kind"].dtype == np.int8


def test_more_live_birds_than_slots_is_refused():
    """Nowhere to put them. Better a loud failure than a truncated flock."""
    with pytest.raises((ValueError, AssertionError)):
        clip(n_live=N_SLOTS + 1, n_slots=N_SLOTS)


def test_the_sizing_arguments_are_keyword_only():
    """Three ints in a row is a transposition waiting to happen.

    `generate_clip(physics, 12, 10, 300, 0)` and `(physics, 10, 12, 300, 0)`
    both run. One of them silently produces a clip with more live birds than
    slots, or a differently shaped world. Keywords make it impossible rather
    than unlikely.
    """
    with pytest.raises(TypeError):
        generate.generate_clip(PHYSICS, N_SLOTS, N_SLOTS, N_TICKS, 0)


# --- the two invariants that can silently corrupt every label ----------------

def test_forces_produce_the_next_velocity_exactly():
    """Tick alignment, stated as an equation.

    `forces[t]` must be what acted on `velocities[t]` to give `velocities[t+1]`.
    Record them one iteration apart and every label in the dataset describes the
    wrong state, and the result still looks entirely plausible, because the
    physics is self-consistent, just shifted.

    Exact rather than approximate: same operations, same order, same float64
    values, so any drift here means something was recomputed rather than
    recorded.
    """
    c = clip()
    total = c["separation"] + c["alignment"] + c["cohesion"]

    for t in range(N_TICKS):
        for i in range(N_SLOTS):
            np.testing.assert_array_equal(
                velocity_capper(c["velocities"][t][i] + total[t][i] * PHYSICS.dt,
                                PHYSICS.max_speed, PHYSICS.min_speed),
                c["velocities"][t + 1][i],
                err_msg=f"tick {t}, slot {i}: forces do not produce the next velocity")


def test_forces_are_the_rules_applied_to_the_state_at_that_tick():
    """The strongest statement available about the labels.

    Each recorded force must equal its own rule function, at its own radius,
    times its own weight, evaluated on the state stored at the *same* index.
    That pins the rule, the constant, the weight, the slot index and the tick
    together. Nothing else can: forces taken from the following state still
    satisfy the equation above, because that only checks self-consistency.
    """
    # shipped world: each label is its rule at the shipped softness, times its
    # weight, then capped - the force that actually moved the bird
    from boids import force_capper

    c = clip()
    ph = PHYSICS
    soft, cap = ph.edge_softness, ph.max_force

    for t in (0, N_TICKS // 2, N_TICKS - 1):
        positions, velocities = c["positions"][t], c["velocities"][t]
        for i in range(N_SLOTS):
            np.testing.assert_allclose(
                c["separation"][t][i],
                force_capper(ph.w_sep * calculate_separation(
                    i, positions, ph.separation_radius, ph.bounds,
                    ph.sep_force_cap, softness=soft), cap),
                err_msg=f"separation label wrong at tick {t}, slot {i}")
            np.testing.assert_allclose(
                c["alignment"][t][i],
                force_capper(ph.w_ali * calculate_alignment(
                    i, positions, velocities, ph.alignment_radius, ph.bounds,
                    softness=soft), cap),
                err_msg=f"alignment label wrong at tick {t}, slot {i}")
            np.testing.assert_allclose(
                c["cohesion"][t][i],
                force_capper(ph.w_coh * calculate_cohesion(
                    i, positions, ph.cohesion_radius, ph.bounds,
                    softness=soft), cap),
                err_msg=f"cohesion label wrong at tick {t}, slot {i}")

    for rule in ("separation", "alignment", "cohesion"):
        assert np.linalg.norm(c[rule], axis=2).max() > 0, f"{rule} never fired"


# --- the initial state must be one the simulator could have produced --------

def test_initial_velocities_are_legal():
    """Tick 0 is a state, and it has to obey the same clamp as every other.

    `tests/world.py:random_flock` draws velocity components uniform in
    ±max_speed, which puts 33.2% of its birds outside `[min_speed, max_speed]`
    (200 seeds × 12 birds). That is deliberate there, it exercises the clamp.
    Here it would write a first row the simulator can never reach.

    Draw a heading uniform in [0, 2π) and a speed uniform in the clamp instead.
    """
    speeds = np.linalg.norm(clip()["velocities"][0], axis=1)

    assert speeds.min() >= PHYSICS.min_speed - 1e-12
    assert speeds.max() <= PHYSICS.max_speed + 1e-12


def test_headings_are_not_all_the_same_at_tick_zero():
    """A premise guard for the test above.

    Clamping every initial speed to exactly `min_speed` would satisfy the
    previous test while producing a degenerate flock. Speeds should vary, and
    headings should cover the circle rather than clustering.
    """
    velocities = clip(n_ticks=1)["velocities"][0]

    assert np.std(np.linalg.norm(velocities, axis=1)) > 0.1
    assert polarisation(velocities) < 0.9


def test_the_world_invariants_hold_at_every_tick():
    c = clip()
    live = c["mask"]

    for t in range(N_TICKS + 1):
        positions = c["positions"][t][live[t]]
        speeds = np.linalg.norm(c["velocities"][t][live[t]], axis=1)
        assert np.all(positions >= 0.0) and np.all(positions < PHYSICS.bounds), t
        assert speeds.min() >= PHYSICS.min_speed - 1e-9, t
        assert speeds.max() <= PHYSICS.max_speed + 1e-9, t


def test_no_nan_in_live_slots():
    c = clip()
    live = c["mask"]

    assert not np.isnan(c["positions"][live]).any()
    assert not np.isnan(c["velocities"][live]).any()
    for rule in ("separation", "alignment", "cohesion"):
        assert not np.isnan(c[rule][live[:-1]]).any(), rule


# --- slots, masks and the birds that are not there --------------------------

def test_every_slot_is_live_when_n_live_equals_n_slots():
    c = clip(n_live=N_SLOTS)

    assert c["mask"].all()
    np.testing.assert_array_equal(c["kind"], np.zeros(N_SLOTS, dtype=np.int8))


def test_dead_slots_are_nan_and_exactly_complement_the_mask():
    """NaN in dead slots is a checksum on the mask, not decoration.

    With zeros a dead slot is indistinguishable from a bird at the origin, and 0.0
    and -1.0 are both legal velocities. With NaN the two encodings must agree
    exactly, and disagreeing is a test failure rather than a silent one.
    """
    c = clip(n_live=8)
    dead = ~c["mask"]

    assert dead.any(), "premise failed: nothing was masked off"
    np.testing.assert_array_equal(np.isnan(c["positions"][..., 0]), dead)
    np.testing.assert_array_equal(np.isnan(c["velocities"][..., 0]), dead)
    for rule in ("separation", "alignment", "cohesion"):
        np.testing.assert_array_equal(np.isnan(c[rule][..., 0]), dead[:-1], rule)


def test_masked_slots_exert_no_force_on_live_birds():
    """The quietest way to produce a plausible dataset of the wrong physics.

    `step` uses `N = len(positions)`, so a dead slot handed to it is simply a
    bird, it pushes on its neighbours and enters their averages, and nothing
    raises. Simulate an (n_live, 2) array and scatter into slots afterwards.

    This also pins the initialiser's draw *size*: the two clips below only match
    if the random stream is identical, which requires drawing exactly `n_live`
    positions and velocities rather than `n_slots` with some discarded.
    """
    wide = clip(n_live=9, n_slots=12, seed=7)
    narrow = clip(n_live=9, n_slots=9, seed=7)

    live = wide["mask"][0]
    np.testing.assert_array_equal(wide["positions"][:, live, :], narrow["positions"])
    np.testing.assert_array_equal(wide["velocities"][:, live, :], narrow["velocities"])
    for rule in ("separation", "alignment", "cohesion"):
        np.testing.assert_array_equal(wide[rule][:, live, :], narrow[rule], rule)


# --- reproducibility ---------------------------------------------------------

def test_the_same_seed_reproduces_the_clip_bit_for_bit():
    """Without this the dataset cannot be regenerated from its own metadata."""
    first, second = clip(seed=99), clip(seed=99)

    for key in ("positions", "velocities", "separation", "alignment", "cohesion"):
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)
    np.testing.assert_array_equal(first["mask"], second["mask"])


def test_a_seed_sequence_is_accepted_and_distinguishes_clips():
    """Clips are seeded by `SeedSequence(entropy=base, spawn_key=(i,))`.

    Children of `spawn()` all inherit the parent's `.entropy`, so seeding from
    that alone gives every clip the same stream. The spawn_key is what
    distinguishes them, and it must survive being passed in here.
    """
    base = 4242
    a = clip(seed=np.random.SeedSequence(entropy=base, spawn_key=(0,)))
    b = clip(seed=np.random.SeedSequence(entropy=base, spawn_key=(1,)))
    a_again = clip(seed=np.random.SeedSequence(entropy=base, spawn_key=(0,)))

    np.testing.assert_array_equal(a["positions"], a_again["positions"])
    assert not np.array_equal(a["positions"], b["positions"])


def test_different_seeds_give_different_clips():
    assert not np.array_equal(clip(seed=1)["positions"], clip(seed=2)["positions"])


# --- the settling tick -------------------------------------------------------

def test_settle_tick_matches_a_recomputation_from_the_stored_velocities():
    """Defined by polarisation, not by a force threshold.

    Polarisation is dimensionless and bounded, so it survives any future
    reweighting of the rules; an absolute force threshold would need retuning.
    """
    c = clip(n_ticks=400)
    live = c["mask"][0]

    expected = -1
    for t in range(len(c["velocities"])):
        if polarisation(c["velocities"][t][live]) > 0.99:
            expected = t
            break

    assert int(c["settle_tick"]) == expected


def test_settle_tick_is_minus_one_when_the_flock_never_orders():
    """A clip too short to settle must say so, not report its last tick."""
    c = clip(n_ticks=2)
    live = c["mask"][0]

    assert max(polarisation(c["velocities"][t][live])
               for t in range(len(c["velocities"]))) <= 0.99, (
        "premise failed: this clip did settle, so it cannot test the -1 case")
    assert int(c["settle_tick"]) == -1
