"""The contract for `build_windows()` — written before it exists.

`clips.npz` holds whole clips. A transformer eats fixed-length windows. This is
the function in between, and it is where the project's two known traps live:
the leaky target, and the torus seam.

    build_windows(dataset_dir, *, context, clips=None) -> (inputs, targets)

    inputs   (n_windows, context, n_slots * 2)  float32  raw positions
    targets  (n_windows, n_slots * 2)           float32  displacement, next tick

One window is `context` consecutive ticks from a single clip; its target is how
far each bird moves on the tick after the window ends.

Seven decisions are baked in here. Say so and I will change them:

1. **Positions only in the input.** `SPEC.md` §5: "One tick = N birds × 2
   values". Velocity is recoverable from consecutive ticks, so feeding it too
   would hand the model a quantity it could derive — and quietly change what a
   probe for "does it represent velocity" is even asking.
2. **The target is a displacement, not the next position.** Predicting the next
   position is answerable to R² = 1.000000 by copying the input: a bird moves
   0.2–0.5 units per tick in a 100-unit world. `NOTES.md` records that baseline.
   A displacement of ~0.3 cannot be produced without modelling the forces.
3. **The displacement comes from `velocities[t+1] * dt`, never from subtracting
   two stored positions.** Subtraction is wrong across the seam by exactly
   `bounds`, and lossy near 100.0 in float32. The stored velocity carries it
   exactly, and `step` advances position with the *new* velocity, so
   `v[t+1] * dt` is the displacement by construction.
4. **Raw coordinates, not normalised.** Scaling is the model's business, and a
   loader that silently divides by 100 makes every later "what units is this
   probe in" question harder than it needs to be.
5. **numpy out, not torch.** Keeps this testable without importing torch, and
   leaves dtype and device to the training script where they belong.
6. **Clip selection by argument, not a returned index.** `clips=range(0, 180)`
   and `clips=range(180, 200)` is a split by clip. Getting a tick-level split
   out of this API takes deliberate effort, which is the point — adjacent ticks
   are nearly identical and a tick split leaks the test set into training.
7. **Each tick flattened to `n_slots * 2` numbers**, one token per tick, as §5
   describes. Per-bird structure is not preserved here.
"""

import numpy as np
import pytest

from boids import get_torus
from world import PHYSICS_CRUISING

generate = pytest.importorskip("generate")
windows = pytest.importorskip(
    "windows",
    reason="windows.py does not exist yet — this suite is its specification")

PHYSICS = PHYSICS_CRUISING
N_CLIPS = 4
N_SLOTS = 6
N_TICKS = 150               # long enough that some birds cross the seam
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


def windows_per_clip(n_ticks=N_TICKS, context=CONTEXT):
    #the last window must leave a tick after it to be the target
    return n_ticks + 1 - context


# --- the shape contract ------------------------------------------------------

def test_the_shapes(dataset):
    inputs, targets = windows.build_windows(dataset, context=CONTEXT)

    assert inputs.shape == (N_CLIPS * windows_per_clip(), CONTEXT, N_SLOTS * 2)
    assert targets.shape == (N_CLIPS * windows_per_clip(), N_SLOTS * 2)


def test_the_arrays_are_float32(dataset):
    """float32 all the way to the model, matching what is on disk."""
    inputs, targets = windows.build_windows(dataset, context=CONTEXT)

    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32


def test_the_window_count_is_exact(dataset):
    """One window per legal starting tick, and no window without a target.

    A clip of 301 states yields 301 - context windows, not 301. Getting this
    wrong by one either drops data silently or indexes past the end of the clip
    and wraps into the next one.
    """
    for context in (1, 2, CONTEXT, 32):
        inputs, _ = windows.build_windows(dataset, context=context)
        assert len(inputs) == N_CLIPS * windows_per_clip(context=context), context


# --- trap 1: windows must not straddle clips ---------------------------------

def test_no_window_straddles_two_clips(dataset):
    """Reconstructed from the source array, window by window.

    A flat view of the positions array would let a window begin near the end of
    one clip and run into the start of the next — splicing together two
    unrelated flocks. Every value in that window is a legal position, nothing
    raises, and the model is asked to predict a discontinuity that never
    happened.
    """
    a = arrays(dataset)
    inputs, _ = windows.build_windows(dataset, context=CONTEXT)

    per_clip = windows_per_clip()
    for c in range(N_CLIPS):
        for offset in range(per_clip):
            expected = a["positions"][c, offset:offset + CONTEXT].reshape(CONTEXT, -1)
            np.testing.assert_array_equal(
                inputs[c * per_clip + offset], expected,
                err_msg=f"clip {c}, offset {offset}")


# --- trap 2: the target -------------------------------------------------------

def test_the_target_is_a_displacement_not_the_next_position(dataset):
    """The leak, stated as a bound.

    Every target must be a per-tick move: at most `max_speed * dt` = 0.5 in this
    config. A next-position target would be somewhere in [0, 100), so this fails
    immediately and loudly rather than after a training run that looked great.
    """
    _, targets = windows.build_windows(dataset, context=CONTEXT)
    step_sizes = np.linalg.norm(targets.reshape(len(targets), N_SLOTS, 2), axis=-1)

    assert step_sizes.max() <= PHYSICS.max_speed * PHYSICS.dt + 1e-5
    assert step_sizes.min() >= PHYSICS.min_speed * PHYSICS.dt - 1e-5


def test_the_target_is_the_stored_velocity_times_dt(dataset):
    """Where the displacement must come from.

    `step` advances position with the *new* velocity, so the move from tick t to
    t+1 is exactly `velocities[t+1] * dt`. Recomputing it any other way is at
    best redundant and at worst wrong at the seam.
    """
    a = arrays(dataset)
    _, targets = windows.build_windows(dataset, context=CONTEXT)

    per_clip = windows_per_clip()
    for c in range(N_CLIPS):
        for offset in (0, 37, per_clip - 1):
            last_tick = offset + CONTEXT - 1
            expected = a["velocities"][c, last_tick + 1] * PHYSICS.dt
            np.testing.assert_allclose(
                targets[c * per_clip + offset].reshape(N_SLOTS, 2),
                expected, rtol=0, atol=1e-6,
                err_msg=f"clip {c}, offset {offset}")


def test_the_target_is_torus_correct_across_the_seam(dataset):
    """The test that separates a correct loader from a plausible one.

    Subtracting two stored positions gives ~99.8 for a bird that wrapped, rather
    than ~0.2 — a target 300x larger than any real one, on roughly 1 tick in 30
    at this configuration. It would dominate the loss and teach the model that
    birds teleport. The premise assertion below fails if the fixture happens to
    contain no crossings, so this cannot pass vacuously.
    """
    a = arrays(dataset)
    positions = a["positions"].astype(np.float64)
    jumps = np.abs(np.diff(positions, axis=1))
    crossings = int((jumps > PHYSICS.bounds / 2).any(axis=-1).sum())
    assert crossings > 0, "premise failed: no bird crossed the seam in this fixture"

    _, targets = windows.build_windows(dataset, context=CONTEXT)

    assert np.abs(targets).max() <= PHYSICS.max_speed * PHYSICS.dt + 1e-5


def test_the_target_agrees_with_the_torus_delta_of_the_positions(dataset):
    """Two independent derivations of the same displacement must agree.

    `v[t+1] * dt` is what the loader should use; `get_torus` on the two stored
    positions is the other route to the same quantity. Agreement means the
    stored velocities, the stored positions and the wrap all describe one world.
    """
    a = arrays(dataset)
    positions = a["positions"].astype(np.float64)
    _, targets = windows.build_windows(dataset, context=CONTEXT)

    per_clip = windows_per_clip()
    for c in range(N_CLIPS):
        for offset in (0, 91, per_clip - 1):
            last_tick = offset + CONTEXT - 1
            for i in range(N_SLOTS):
                expected = get_torus(positions[c, last_tick, i],
                                     positions[c, last_tick + 1, i],
                                     PHYSICS.bounds)
                np.testing.assert_allclose(
                    targets[c * per_clip + offset].reshape(N_SLOTS, 2)[i],
                    expected, rtol=0, atol=1e-4,
                    err_msg=f"clip {c}, offset {offset}, bird {i}")


# --- what the input is --------------------------------------------------------

def test_the_input_is_raw_unnormalised_positions(dataset):
    """A loader that silently divides by bounds makes every later unit question
    harder, and hides the seam rather than fixing it."""
    inputs, _ = windows.build_windows(dataset, context=CONTEXT)

    assert inputs.min() >= 0.0
    assert inputs.max() <= PHYSICS.bounds
    assert inputs.max() - inputs.min() > PHYSICS.bounds / 2, "looks normalised"


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
    np.testing.assert_array_equal(
        inputs[0], a["positions"][2, :CONTEXT].reshape(CONTEXT, -1))


def test_two_disjoint_clip_lists_partition_the_whole_dataset(dataset):
    """The structural guarantee that makes a split-by-clip safe.

    Train and validation are two calls with disjoint clip lists. If together
    they reproduce the full set exactly, then no window is in both and none is
    lost - which is the property a tick-level split violates.
    """
    full, full_t = windows.build_windows(dataset, context=CONTEXT)
    first, first_t = windows.build_windows(dataset, context=CONTEXT, clips=[0, 1])
    second, second_t = windows.build_windows(dataset, context=CONTEXT, clips=[2, 3])

    np.testing.assert_array_equal(np.concatenate([first, second]), full)
    np.testing.assert_array_equal(np.concatenate([first_t, second_t]), full_t)


# --- refusals -----------------------------------------------------------------

def test_masked_slots_are_refused(masked):
    """NaN into a model is a NaN loss, discovered an hour into a run.

    The perturbation split has dead slots by design, and it is the obvious thing
    to point this function at. Better a refusal here than a training curve that
    goes flat and takes an afternoon to explain.
    """
    with pytest.raises(ValueError):
        windows.build_windows(masked, context=4)


def test_a_context_longer_than_the_clip_is_refused(dataset):
    """Zero windows returned silently is worse than an error: the training loop
    would run, produce nothing, and report a fine-looking loss of nan or 0."""
    with pytest.raises(ValueError):
        windows.build_windows(dataset, context=N_TICKS + 5)


# --- determinism ---------------------------------------------------------------

def test_two_calls_give_identical_arrays(dataset):
    first = windows.build_windows(dataset, context=CONTEXT)
    second = windows.build_windows(dataset, context=CONTEXT)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
