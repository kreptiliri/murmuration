"""The contract for `build_windows()` — written before it exists.

`clips.npz` holds whole clips. A transformer eats fixed-length windows. This is
the function in between, and it is where the project's two known traps live:
the leaky target, and the torus seam.

    build_windows(dataset_dir, *, context, clips=None) -> (inputs, targets)

    inputs   (n_windows, context, n_slots * 2)  float32  raw positions
    targets  (n_windows, n_slots * 2)           float32  velocity change, next tick

One window is `context` consecutive ticks from a single clip; its target is how
much each bird's velocity changes on the tick after the window ends -
`velocities[t+1] - velocities[t]`.

Seven decisions are baked in here. Say so and I will change them:

1. **Positions only in the input.** `SPEC.md` §5: "One tick = N birds × 2
   values". Velocity is recoverable from consecutive ticks, so feeding it too
   would hand the model a quantity it could derive — and quietly change what a
   probe for "does it represent velocity" is even asking.
2. **The target is the change in velocity, not the next position and not the
   displacement.** Both of those are answerable by copying. A do-nothing
   predictor scores R² = 1.000000 against the next position; a predictor that
   repeats the previous move scores **+0.999809** against the displacement,
   leaving 0.019% of the variance for anything a model might actually learn.
   Against the velocity change the same copier scores **+0.924143** - still a
   lot, which is why `NOTES.md` also requires reporting skill against
   persistence rather than against the mean, but four hundred times more room.
   Measured over 200 clips, 300 ticks, N=12, context 8.
3. **The target comes from the stored velocities, never from positions.**
   `velocities[t+1] - velocities[t]`. Velocities do not wrap, so unlike a
   displacement target this one has no seam in it at all. Note the target is
   *not* the force: the speed floor fires on 31% of bird-ticks, so it is
   force-then-clamped, and `corr(dv, force*dt) = 0.765`.
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

from world import PHYSICS_CRUISING

generate = pytest.importorskip("generate")
windows = pytest.importorskip(
    "windows",
    reason="windows.py does not exist yet — this suite is its specification")

PHYSICS = PHYSICS_CRUISING
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

def test_the_target_is_neither_a_position_nor_a_displacement(dataset):
    """Both of the rejected targets, ruled out by their own arithmetic.

    A position target lands in [0, 100). A displacement target has every entry
    between min_speed*dt and max_speed*dt, so its magnitude is never below 0.2.
    A velocity change is bounded by 2*max_speed and routinely near zero - a
    settled bird barely changes velocity at all.
    """
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
    """The definition, checked against the file it came from.

    The window's last input tick is `offset + context - 1`; the change asked
    about is the one into the tick after that. Off by one here and every target
    describes the wrong tick while remaining a plausible small number.
    """
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
    """Velocities do not wrap, so the target cannot be corrupted by a crossing.

    This is the property that made the velocity change the right quantity to ask
    for, quite apart from the leak. A displacement target has to be derived
    torus-correctly or a wrapping bird contributes a 99.8-unit move; a velocity
    change is immune by construction. The premise assertion below fails if the
    fixture contains no crossings, so it cannot pass vacuously.
    """
    a = arrays(dataset)
    positions = a["positions"].astype(np.float64)
    jumps = np.abs(np.diff(positions, axis=1))
    wrapped = (jumps > PHYSICS.bounds / 2).any(axis=-1)
    assert wrapped.any(), "premise failed: no bird crossed the seam in this fixture"

    _, targets = windows.build_windows(dataset, context=CONTEXT)

    assert np.abs(targets).max() <= 2 * PHYSICS.max_speed


def test_a_copier_cannot_score_well_on_this_target(dataset):
    """The reason the target changed, as a regression guard.

    "The same change as last tick" is the dumbest predictor that could plausibly
    win here. On the full dataset it scores R2 +0.924 against this target and
    +0.9998 against a displacement target - so if someone quietly switches the
    target back, this fails. Both numbers are computed here rather than
    hardcoded, and the fixture is smaller than the dataset, so the thresholds
    are loose on purpose.
    """
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


def test_a_generator_selects_the_same_clips_as_a_list(dataset):
    """`clips` is consumed twice — the mask check, then the window loop — so
    a one-shot iterator would arrive empty at the second use. The
    implementation converts to a list first; this pins that a generator
    selects exactly what the equivalent list does."""
    from_list, from_list_t = windows.build_windows(
        dataset, context=CONTEXT, clips=[0, 2])
    from_gen, from_gen_t = windows.build_windows(
        dataset, context=CONTEXT, clips=iter([0, 2]))

    np.testing.assert_array_equal(from_gen, from_list)
    np.testing.assert_array_equal(from_gen_t, from_list_t)


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


@pytest.mark.parametrize("context", [0, -1])
def test_a_context_below_one_is_refused(dataset, context):
    """A window of zero ticks predicts from nothing.

    Today context=0 dies inside numpy's reshape with a message about
    (0,newaxis), not at the argument — and a refactor of the flattening below
    would turn it silent entirely: offset=0 then reads velocities[c, -1], the
    clip's *last* tick, into its *first* target. Refuse at the door, naming it.
    """
    with pytest.raises(ValueError, match="context"):
        windows.build_windows(dataset, context=context)


def test_an_empty_clip_selection_is_refused(dataset):
    """clips=[] builds zero windows and returns shape-(0,) float64 arrays —
    not even the documented shapes. A split that accidentally selects nothing
    would then train on air. Same philosophy as the context refusal above:
    an empty training set must be an error, never a quiet empty array."""
    with pytest.raises(ValueError, match="clip"):
        windows.build_windows(dataset, context=CONTEXT, clips=[])


# --- determinism ---------------------------------------------------------------

def test_two_calls_give_identical_arrays(dataset):
    first = windows.build_windows(dataset, context=CONTEXT)
    second = windows.build_windows(dataset, context=CONTEXT)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
