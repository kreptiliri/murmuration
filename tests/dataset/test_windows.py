import numpy as np
import pytest

import boids
from world import PHYSICS_V1

generate = pytest.importorskip("generate")
windows = pytest.importorskip(
    "windows",
    reason="windows.py does not exist yet, this suite is its specification")

# sim-v1's world: in the capped world a copier predicts the velocity change
# almost perfectly (0.99), so the target guard below could not tell targets apart
PHYSICS = PHYSICS_V1
N_CLIPS = 4
N_SLOTS = 6
N_TICKS = 150 # long enough that some birds cross the seam
CONTEXT = 8
BASE_SEED = 20260904


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("windows")
    generate.build_dataset(PHYSICS, n_clips=N_CLIPS, n_slots=N_SLOTS,
                           n_live=N_SLOTS, n_ticks=N_TICKS,
                           base_seed=BASE_SEED, out_dir=out, allow_dirty=True)
    return out


@pytest.fixture(scope="module")
def masked(tmp_path_factory):
    out = tmp_path_factory.mktemp("windows_masked")
    generate.build_dataset(PHYSICS, n_clips=2, n_slots=N_SLOTS, n_live=4,
                           n_ticks=20, base_seed=BASE_SEED, out_dir=out,
                           allow_dirty=True)
    return out


def arrays(dataset_dir):
    with np.load(dataset_dir / "clips.npz") as z:
        return {name: z[name] for name in z.files}


def repaired(window):
    moves = boids.get_torus(window[1:], window[:-1], PHYSICS.bounds)
    out = np.empty_like(window)
    out[0] = window[0]
    for i in range(1, len(window)):
        out[i] = out[i - 1] + moves[i - 1]
    return out


def windows_per_clip(n_ticks=N_TICKS, context=CONTEXT):
    #the last window must leave a tick after it to be the target
    return n_ticks + 1 - context


# --- the shape contract ------------------------------------------------------

def test_the_shapes(dataset):
    inputs, targets = windows.build_windows(dataset, context=CONTEXT)

    assert inputs.shape == (N_CLIPS * windows_per_clip(), CONTEXT, N_SLOTS * 2)
    assert targets.shape == (N_CLIPS * windows_per_clip(), N_SLOTS * 2)


def test_the_arrays_are_float32(dataset):
    inputs, targets = windows.build_windows(dataset, context=CONTEXT)

    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32


def test_the_window_count_is_exact(dataset):
    for context in (1, 2, CONTEXT, 32):
        inputs, _ = windows.build_windows(dataset, context=context)
        assert len(inputs) == N_CLIPS * windows_per_clip(context=context), context


# --- trap 1: windows must not straddle clips ---------------------------------

def test_no_window_straddles_two_clips(dataset):
    a = arrays(dataset)
    inputs, _ = windows.build_windows(dataset, context=CONTEXT)

    per_clip = windows_per_clip()
    for c in range(N_CLIPS):
        for offset in range(per_clip):
            expected = repaired(
                a["positions"][c, offset:offset + CONTEXT]).reshape(CONTEXT, -1)
            np.testing.assert_allclose(
                inputs[c * per_clip + offset], expected, atol=1e-3,
                err_msg=f"clip {c}, offset {offset}")


# --- trap 2: the target -------------------------------------------------------

def test_the_target_is_neither_a_position_nor_a_displacement(dataset):
    a = arrays(dataset)
    _, targets = windows.build_windows(dataset, context=CONTEXT)
    sizes = np.linalg.norm(targets.reshape(len(targets), N_SLOTS, 2), axis=-1)

    assert sizes.max() <= 2 * PHYSICS.max_speed
    assert sizes.min() < PHYSICS.min_speed * PHYSICS.dt, (
        "no target smaller than one tick of travel - this looks like a "
        "displacement target, not a velocity change")

    displacement = a["velocities"][:, CONTEXT] * PHYSICS.dt
    assert not np.allclose(targets[0], displacement[0].reshape(-1), atol=1e-4)


def test_the_target_is_the_change_in_the_stored_velocities(dataset):
    a = arrays(dataset)
    _, targets = windows.build_windows(dataset, context=CONTEXT)

    per_clip = windows_per_clip()
    for c in range(N_CLIPS):
        for offset in (0, 37, per_clip - 1):
            last_tick = offset + CONTEXT - 1
            expected = (a["velocities"][c, last_tick + 1]
                        - a["velocities"][c, last_tick])
            np.testing.assert_allclose(
                targets[c * per_clip + offset].reshape(N_SLOTS, 2),
                expected, rtol=0, atol=1e-6,
                err_msg=f"clip {c}, offset {offset}")


def test_the_seam_does_not_reach_the_target(dataset):
    a = arrays(dataset)
    positions = a["positions"].astype(np.float64)
    jumps = np.abs(np.diff(positions, axis=1))
    wrapped = (jumps > PHYSICS.bounds / 2).any(axis=-1)
    assert wrapped.any(), "premise failed: no bird crossed the seam in this fixture"

    _, targets = windows.build_windows(dataset, context=CONTEXT)

    assert np.abs(targets).max() <= 2 * PHYSICS.max_speed


def test_a_copier_cannot_score_well_on_this_target(dataset):
    a = arrays(dataset)
    velocities = a["velocities"].astype(np.float64)
    _, targets = windows.build_windows(dataset, context=CONTEXT)

    per_clip = windows_per_clip()
    ticks = np.arange(CONTEXT - 1, CONTEXT - 1 + per_clip)
    previous_change = (velocities[:, ticks] - velocities[:, ticks - 1])
    previous_change = previous_change.reshape(len(targets), -1)

    def skill(prediction, truth):
        return 1 - ((prediction - truth) ** 2).mean() / truth.var()

    on_target = skill(previous_change, targets.astype(np.float64))
    displacement = (velocities[:, ticks + 1] * PHYSICS.dt).reshape(len(targets), -1)
    previous_move = (velocities[:, ticks] * PHYSICS.dt).reshape(len(targets), -1)
    on_displacement = skill(previous_move, displacement)

    assert on_displacement > 0.99, (
        f"premise failed: a copier only scored {on_displacement:.4f} against a "
        "displacement target, so this fixture does not demonstrate the leak")
    assert 0.5 < on_target < 0.95, (
        f"a copier scores {on_target:.4f} against the current target. Above 0.95 "
        "is displacement-target territory and leaves nothing for the model to "
        "learn; below 0.5 means the target is not a velocity change at all, "
        "since repeating the previous change should be a decent guess")


# --- what the input is --------------------------------------------------------

def test_the_input_is_raw_unnormalised_positions(dataset):
    inputs, _ = windows.build_windows(dataset, context=CONTEXT)
    drift = (CONTEXT - 1) * PHYSICS.max_speed * PHYSICS.dt

    assert inputs.min() >= -drift
    assert inputs.max() <= PHYSICS.bounds + drift
    assert inputs.max() - inputs.min() > PHYSICS.bounds / 2, "looks normalised"


# --- trap 3: the seam in the input --------------------------------------------

def test_no_window_contains_a_teleport(dataset):
    a = arrays(dataset)
    raw_jumps = np.abs(np.diff(a["positions"].astype(np.float64), axis=1))
    assert (raw_jumps > PHYSICS.bounds / 2).any(), (
        "premise failed: no bird crossed the seam in this fixture, so this "
        "test would pass without any repair happening")

    inputs, _ = windows.build_windows(dataset, context=CONTEXT)
    per_tick = inputs.reshape(len(inputs), CONTEXT, N_SLOTS, 2)
    steps = np.abs(np.diff(per_tick.astype(np.float64), axis=1))

    assert steps.max() <= PHYSICS.max_speed * PHYSICS.dt + 1e-3, (
        f"a window contains a {steps.max():.2f}-unit step; the speed cap allows "
        f"{PHYSICS.max_speed * PHYSICS.dt}")


def test_each_step_in_a_window_is_the_stored_velocity(dataset):
    a = arrays(dataset)
    velocities = a["velocities"].astype(np.float64)
    inputs, _ = windows.build_windows(dataset, context=CONTEXT)
    per_tick = inputs.reshape(len(inputs), CONTEXT, N_SLOTS, 2).astype(np.float64)

    per_clip = windows_per_clip()
    for c in range(N_CLIPS):
        for offset in range(per_clip):
            window = per_tick[c * per_clip + offset]
            moved = np.diff(window, axis=0)
            expected = velocities[c, offset + 1:offset + CONTEXT] * PHYSICS.dt
            np.testing.assert_allclose(
                moved, expected, atol=1e-3,
                err_msg=f"clip {c}, offset {offset}")


def test_the_window_is_anchored_at_its_first_stored_tick(dataset):
    a = arrays(dataset)
    inputs, _ = windows.build_windows(dataset, context=CONTEXT)
    per_clip = windows_per_clip()

    for c in range(N_CLIPS):
        for offset in range(per_clip):
            np.testing.assert_array_equal(
                inputs[c * per_clip + offset][0],
                a["positions"][c, offset].reshape(-1),
                err_msg=f"clip {c}, offset {offset}")


def test_a_window_that_never_crosses_is_the_raw_slice(dataset):
    a = arrays(dataset)
    positions = a["positions"]
    inputs, _ = windows.build_windows(dataset, context=CONTEXT)
    per_clip = windows_per_clip()

    untouched = 0
    for c in range(N_CLIPS):
        for offset in range(per_clip):
            slice_ = positions[c, offset:offset + CONTEXT]
            if (np.abs(np.diff(slice_.astype(np.float64), axis=0))
                    > PHYSICS.bounds / 2).any():
                continue
            np.testing.assert_allclose(
                inputs[c * per_clip + offset], slice_.reshape(CONTEXT, -1),
                atol=2e-3, err_msg=f"clip {c}, offset {offset}")
            untouched += 1

    assert untouched > 0, "premise failed: every window in this fixture crossed"


def test_the_fixture_actually_exercises_the_repair(dataset):
    a = arrays(dataset)
    positions = a["positions"]
    inputs, _ = windows.build_windows(dataset, context=CONTEXT)
    per_clip = windows_per_clip()

    biggest = 0.0
    for c in range(N_CLIPS):
        for offset in range(per_clip):
            raw = positions[c, offset:offset + CONTEXT].reshape(CONTEXT, -1)
            difference = np.abs(inputs[c * per_clip + offset].astype(np.float64)
                                - raw.astype(np.float64)).max()
            biggest = max(biggest, difference)

    assert biggest > PHYSICS.bounds / 2, (
        f"the largest correction applied anywhere was {biggest:.3f}; a real "
        "crossing moves a coordinate by nearly a full bounds, so either the "
        "fixture contains none or the repair is not happening")


# --- clip selection, which is how splits happen -------------------------------

def test_clips_none_uses_every_clip(dataset):
    all_windows, _ = windows.build_windows(dataset, context=CONTEXT)
    explicit, _ = windows.build_windows(dataset, context=CONTEXT,
                                        clips=range(N_CLIPS))

    np.testing.assert_array_equal(all_windows, explicit)


def test_a_subset_selects_only_those_clips(dataset):
    a = arrays(dataset)
    inputs, _ = windows.build_windows(dataset, context=CONTEXT, clips=[2])

    assert len(inputs) == windows_per_clip()
    np.testing.assert_allclose(
        inputs[0], repaired(a["positions"][2, :CONTEXT]).reshape(CONTEXT, -1),
        atol=1e-3)


def test_a_generator_selects_the_same_clips_as_a_list(dataset):
    from_list, from_list_t = windows.build_windows(
        dataset, context=CONTEXT, clips=[0, 2])
    from_gen, from_gen_t = windows.build_windows(
        dataset, context=CONTEXT, clips=iter([0, 2]))

    np.testing.assert_array_equal(from_gen, from_list)
    np.testing.assert_array_equal(from_gen_t, from_list_t)


def test_two_disjoint_clip_lists_partition_the_whole_dataset(dataset):
    full, full_t = windows.build_windows(dataset, context=CONTEXT)
    first, first_t = windows.build_windows(dataset, context=CONTEXT, clips=[0, 1])
    second, second_t = windows.build_windows(dataset, context=CONTEXT, clips=[2, 3])

    np.testing.assert_array_equal(np.concatenate([first, second]), full)
    np.testing.assert_array_equal(np.concatenate([first_t, second_t]), full_t)


# --- refusals -----------------------------------------------------------------

def test_masked_slots_are_refused(masked):
    with pytest.raises(ValueError):
        windows.build_windows(masked, context=4)


def test_a_context_longer_than_the_clip_is_refused(dataset):
    with pytest.raises(ValueError):
        windows.build_windows(dataset, context=N_TICKS + 5)


@pytest.mark.parametrize("context", [0, -1])
def test_a_context_below_one_is_refused(dataset, context):
    with pytest.raises(ValueError, match="context"):
        windows.build_windows(dataset, context=context)


def test_an_empty_clip_selection_is_refused(dataset):
    with pytest.raises(ValueError, match="clip"):
        windows.build_windows(dataset, context=CONTEXT, clips=[])


# --- determinism ---------------------------------------------------------------

def test_two_calls_give_identical_arrays(dataset):
    first = windows.build_windows(dataset, context=CONTEXT)
    second = windows.build_windows(dataset, context=CONTEXT)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
