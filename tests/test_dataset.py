"""The contract for the dataset writer — written before it exists.

`generate_clip()` is done and green. This file specifies everything that turns
clips into a dataset on disk, and it skips as one unit until `build_dataset`
exists. Build into it.

    build_dataset(physics, *, n_clips, n_slots, n_live, n_ticks,
                  base_seed, out_dir, allow_dirty=False) -> pathlib.Path

    clip_seed(base_seed, index) -> np.random.SeedSequence
    git_is_dirty() -> bool
    git_provenance() -> dict   {"commit": str, "tag": str, "dirty": bool}

`build_dataset` writes exactly two files into `out_dir` and returns it:

    clips.npz     positions    (n_clips, n_ticks+1, n_slots, 2)  float32
                  velocities   (n_clips, n_ticks+1, n_slots, 2)  float32
                  separation   (n_clips, n_ticks,   n_slots, 2)  float32
                  alignment    (n_clips, n_ticks,   n_slots, 2)  float32
                  cohesion     (n_clips, n_ticks,   n_slots, 2)  float32
                  mask         (n_clips, n_ticks+1, n_slots)     bool
                  kind         (n_clips, n_slots)                int8
                  settle_tick  (n_clips,)                        int32

    config.json   {"physics": {...}, "generation": {...}, "provenance": {...}}

Four decisions are baked in here. Say so and I will change them:

1. **No `seeds` column.** A clip is identified by `(base_seed, its index)`:
   the base seed is in `config.json` and the index is the array index it already
   sits at. The plan left the column optional on the condition that it hold
   something that actually regenerates a clip — and a column that must *not*
   contain `SeedSequence.entropy`, while looking exactly like it should, is a
   trap for whoever reads this next. `clip_seed` is the regeneration path.
2. **The float32 cast lives here and only here.** `generate_clip` stays float64,
   so the exact tick-alignment test keeps its exactness and this file measures
   what the cast costs.
3. **Fixed filenames**, so nothing downstream takes a path template as an
   argument.
4. **`build_dataset` returns the directory, not the arrays.** Reading the
   dataset back is `np.load`; a function that both writes and returns would let
   every check downstream quietly test the in-memory copy instead of the file.
"""

import json
import subprocess

import numpy as np
import pytest

from boids import Physics, velocity_capper
from world import PHYSICS_CRUISING

generate = pytest.importorskip("generate")

pytestmark = pytest.mark.skipif(
    not hasattr(generate, "build_dataset"),
    reason="generate.build_dataset() does not exist yet — this suite is its "
           "specification")

PHYSICS = PHYSICS_CRUISING
N_CLIPS = 4
N_SLOTS = 6
N_TICKS = 15                    # short on purpose: contract tests, not rollouts
SETTLED_TICKS = 300             # long enough for some flocks to order - see the settled fixture
BASE_SEED = 20260902


def build(tmp_dir, *, n_live=N_SLOTS, n_clips=N_CLIPS, n_slots=N_SLOTS,
          n_ticks=N_TICKS, base_seed=BASE_SEED):
    # allow_dirty=True throughout: the working tree during a test run is
    # routinely dirty, and refusing to run is the behaviour under test in
    # test_a_dirty_tree_is_refused_unless_allowed, not a thing to fight here.
    return generate.build_dataset(PHYSICS, n_clips=n_clips, n_slots=n_slots,
                                  n_live=n_live, n_ticks=n_ticks,
                                  base_seed=base_seed, out_dir=tmp_dir,
                                  allow_dirty=True)


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    """Every slot live — the shape the training split has."""
    return build(tmp_path_factory.mktemp("full"))


@pytest.fixture(scope="module")
def settled(tmp_path_factory):
    """Long enough that some flocks actually order.

    At N_TICKS none of them do, so `settle_tick` is -1 everywhere and any test
    of it agrees with itself. Measured at n_slots=6, base seed 20260902: at 15
    and at 100 ticks every clip reports -1; at 300 the four clips report
    285, -1, 104, 176. Only the last of those can catch a wrong answer.
    """
    return build(tmp_path_factory.mktemp("settled"), n_ticks=SETTLED_TICKS)


@pytest.fixture(scope="module")
def partial(tmp_path_factory):
    """Four live birds in six slots — the perturbation split's shape."""
    return build(tmp_path_factory.mktemp("partial"), n_live=4)


def arrays(out_dir):
    with np.load(out_dir / "clips.npz") as z:
        return {name: z[name] for name in z.files}


def config(out_dir):
    return json.loads((out_dir / "config.json").read_text())


def polarisation(velocities):
    """Reimplemented here on purpose — see the note in test_generate.py.

    Recomputing an expected value with the implementation's own function proves
    only that the function is deterministic. This has to be an independent
    statement of the definition or it is evidence of nothing.
    """
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    return np.linalg.norm((velocities / np.where(speeds == 0, 1, speeds)).mean(axis=0))


def upcast(a):
    """float32 off disk, float64 for arithmetic — the cast is what we measure."""
    return a.astype(np.float64)


# --- the files ---------------------------------------------------------------

def test_the_writer_produces_the_two_files(full):
    assert (full / "clips.npz").is_file()
    assert (full / "config.json").is_file()


def test_build_dataset_returns_the_directory_it_wrote(tmp_path):
    assert build(tmp_path) == tmp_path


# --- the shape contract, one level up ----------------------------------------

def test_the_stored_shapes_carry_a_clips_axis(full):
    a = arrays(full)

    assert a["positions"].shape == (N_CLIPS, N_TICKS + 1, N_SLOTS, 2)
    assert a["velocities"].shape == (N_CLIPS, N_TICKS + 1, N_SLOTS, 2)
    assert a["mask"].shape == (N_CLIPS, N_TICKS + 1, N_SLOTS)
    assert a["kind"].shape == (N_CLIPS, N_SLOTS)
    assert a["settle_tick"].shape == (N_CLIPS,)


def test_the_force_arrays_are_never_padded_to_match_the_states(full):
    """The 301/300 asymmetry is the truth and belongs in the shape.

    Padding forces[0] with zeros to square the arrays off fabricates a label for
    a tick where no force has been computed — and it is invisible afterwards,
    because zero is a perfectly ordinary force.
    """
    a = arrays(full)

    for rule in ("separation", "alignment", "cohesion"):
        assert a[rule].shape == (N_CLIPS, N_TICKS, N_SLOTS, 2), rule


def test_the_stored_dtypes(full):
    """float32 for the floats, and nothing silently widened back to float64."""
    a = arrays(full)

    for key in ("positions", "velocities", "separation", "alignment", "cohesion"):
        assert a[key].dtype == np.float32, f"{key} is {a[key].dtype}"
    assert a["mask"].dtype == np.bool_
    assert a["kind"].dtype == np.int8
    assert a["settle_tick"].dtype == np.int32


def test_kind_is_all_birds_for_now(full):
    assert np.array_equal(arrays(full)["kind"], np.zeros((N_CLIPS, N_SLOTS), np.int8))


# --- config.json -------------------------------------------------------------

def test_the_physics_round_trips_field_for_field(full):
    """Values alone do not identify a dataset, but they are half of it.

    Reconstructing the dataclass rather than comparing dicts also proves the
    stored keys are the field names, so a later rename fails here rather than
    quietly producing a config nothing can load.
    """
    assert Physics(**config(full)["physics"]) == PHYSICS


def test_the_generation_parameters_are_recorded(full):
    gen = config(full)["generation"]

    assert gen["n_clips"] == N_CLIPS
    assert gen["n_slots"] == N_SLOTS
    assert gen["n_live"] == N_SLOTS
    assert gen["n_ticks"] == N_TICKS
    assert gen["base_seed"] == BASE_SEED


def test_provenance_is_recorded(full):
    """The other half. Values say what was asked for; the SHA says what ran.

    A dataset generated by code that has since changed is not reproducible from
    its parameters, and the failure is silent — the numbers still look fine.
    """
    prov = config(full)["provenance"]

    assert prov["commit"], "no commit SHA recorded"
    assert prov["commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=True).stdout.strip()
    assert "tag" in prov
    assert isinstance(prov["dirty"], bool)


def test_the_allow_dirty_flag_is_recorded_even_when_the_tree_is_clean(full):
    """Provenance that can be silently wrong is decoration."""
    assert config(full)["generation"]["allow_dirty"] is True


def test_a_dirty_tree_is_refused_unless_allowed(tmp_path, monkeypatch):
    """Refuse by default, and only the flag gets past it.

    A dataset generated from uncommitted code records a SHA that does not
    describe the code that ran. That is worse than no provenance, because it
    reads as trustworthy.
    """
    monkeypatch.setattr(generate, "git_is_dirty", lambda: True)

    with pytest.raises(RuntimeError):
        generate.build_dataset(PHYSICS, n_clips=1, n_slots=N_SLOTS,
                               n_live=N_SLOTS, n_ticks=2, base_seed=0,
                               out_dir=tmp_path)

    generate.build_dataset(PHYSICS, n_clips=1, n_slots=N_SLOTS, n_live=N_SLOTS,
                           n_ticks=2, base_seed=0, out_dir=tmp_path,
                           allow_dirty=True)
    assert config(tmp_path)["provenance"]["dirty"] is True


# --- the seeds ---------------------------------------------------------------

def test_clip_seed_reproduces_the_spawned_child_and_does_not_depend_on_n():
    """`SeedSequence(entropy=base, spawn_key=(i,))`, not `spawn(n)[i].entropy`.

    Every child of `spawn()` inherits the parent's entropy, so a seeds column
    holding `child.entropy` writes the same number into every row and reseeding
    from it gives one identical stream for all clips — a different stream from
    the one that generated any of them. What identifies a child is the pair
    (entropy, spawn_key). Verified equal for two different n, so the identity
    does not depend on how many clips were asked for.
    """
    for n in (4, 9):
        for i in range(n):
            np.testing.assert_array_equal(
                generate.clip_seed(BASE_SEED, i).generate_state(4),
                np.random.SeedSequence(BASE_SEED).spawn(n)[i].generate_state(4),
                err_msg=f"clip {i} of {n}")


def test_clip_seed_gives_every_clip_a_different_stream():
    states = {tuple(generate.clip_seed(BASE_SEED, i).generate_state(4))
              for i in range(N_CLIPS)}

    assert len(states) == N_CLIPS, "some clips share a random stream"


def test_a_stored_clip_regenerates_from_the_base_seed_and_its_index(full):
    """The claim that makes the seeds column unnecessary.

    If this passes, `(base_seed, index)` is enough to reconstruct any clip in
    the dataset exactly, and `config.json` plus the array index carry both.
    """
    a = arrays(full)
    index = 2
    again = generate.generate_clip(PHYSICS, n_slots=N_SLOTS, n_live=N_SLOTS,
                                   n_ticks=N_TICKS,
                                   seed=generate.clip_seed(BASE_SEED, index))

    for key in ("positions", "velocities", "separation", "alignment", "cohesion"):
        np.testing.assert_array_equal(
            a[key][index], again[key].astype(np.float32), err_msg=key)
    assert int(a["settle_tick"][index]) == again["settle_tick"]


def test_the_clips_are_not_all_the_same_flock(full):
    """Cheap, and it catches seeding every clip from the base seed itself."""
    positions = arrays(full)["positions"]

    for i in range(1, N_CLIPS):
        assert not np.array_equal(positions[0], positions[i]), f"clip {i} == clip 0"


# --- the invariants, restated against what is actually on disk ---------------

def test_forces_reconstruct_velocities_after_the_float32_round_trip(full):
    """Tick alignment, surviving the cast — the one that matters most here.

    Exactness is gone by definition once the arrays are float32, so this is a
    tolerance. The measured worst case for this round trip is 2.9e-07, about
    2.4x float32 epsilon (tests/test_step.py), so a failure at 1e-6 means
    something is misaligned rather than that the bound is tight.

    Checked against `velocities[t]` and `velocities[t+1]` on purpose: an
    internally consistent but shifted recording passes every check that looks
    only at the forces.
    """
    a = arrays(full)
    v = upcast(a["velocities"])
    total = upcast(a["separation"]) + upcast(a["alignment"]) + upcast(a["cohesion"])

    worst = 0.0
    for c in range(N_CLIPS):
        for t in range(N_TICKS):
            for i in range(N_SLOTS):
                expected = velocity_capper(v[c, t, i] + total[c, t, i] * PHYSICS.dt,
                                           PHYSICS.max_speed, PHYSICS.min_speed)
                worst = max(worst, float(np.abs(expected - v[c, t + 1, i]).max()))

    assert worst < 1e-6, f"round trip lost {worst:.3e} — more than the cast alone"


def test_mask_and_nan_are_exact_complements_on_disk(partial):
    """The checksum a mask alone cannot provide.

    With zero-filled dead slots there is no way to tell an empty slot from a
    bird at the origin, and 0.0 is a legal velocity component too. NaN makes the
    two representations checkable against each other, which is why both are
    stored.
    """
    a = arrays(partial)
    dead = ~a["mask"]

    assert dead.any(), "premise failed: nothing was masked off"
    np.testing.assert_array_equal(np.isnan(a["positions"][..., 0]), dead)
    np.testing.assert_array_equal(np.isnan(a["velocities"][..., 0]), dead)
    for rule in ("separation", "alignment", "cohesion"):
        np.testing.assert_array_equal(np.isnan(a[rule][..., 0]), dead[:, :-1], rule)


def test_no_nan_in_live_slots(partial):
    """NaN shouts on comparisons and whispers on aggregates.

    Every check in this file is written over live slots explicitly rather than
    trusting NaN to surface a mistake, because `mean()` propagates it in
    silence. This test is the one that makes that safe to rely on.
    """
    a = arrays(partial)
    live = a["mask"]

    assert not np.isnan(a["positions"][live]).any()
    assert not np.isnan(a["velocities"][live]).any()
    for rule in ("separation", "alignment", "cohesion"):
        assert not np.isnan(a[rule][live[:, :-1]]).any(), rule


def test_positions_and_speeds_are_legal_including_tick_0(full):
    """Tick 0 is the one the simulator never produced, so it is the one to check.

    The upper bound on position is closed rather than half-open here on purpose:
    a float64 coordinate a hair under `bounds` can round *up* to exactly
    `bounds` in float32. That is a property of the cast, not a wrapping bug, and
    asserting `< bounds` would make this test fail at random.
    """
    a = arrays(full)
    positions = upcast(a["positions"])[a["mask"]]
    speeds = np.linalg.norm(upcast(a["velocities"])[a["mask"]], axis=-1)

    assert positions.min() >= 0.0
    assert positions.max() <= PHYSICS.bounds
    assert speeds.min() >= PHYSICS.min_speed - 1e-5, f"slowest {speeds.min()}"
    assert speeds.max() <= PHYSICS.max_speed + 1e-5, f"fastest {speeds.max()}"


def test_settle_tick_matches_a_recomputation_from_the_stored_velocities(settled):
    """Recomputed from the file, not from the generator's own bookkeeping.

    Deliberately on the long fixture. At the short one no flock reaches
    polarisation 0.99, so both sides of the comparison are -1 and an
    implementation that never looks at the velocities passes - which is what
    this test did until the premise assertion below was added.
    """
    a = arrays(settled)

    computed = []
    for c in range(N_CLIPS):
        live = a["mask"][c, 0]
        expected = -1
        for t in range(SETTLED_TICKS + 1):
            if polarisation(upcast(a["velocities"][c, t][live])) > 0.99:
                expected = t
                break
        computed.append(expected)
        assert int(a["settle_tick"][c]) == expected, f"clip {c}"

    assert any(t != -1 for t in computed), (
        "premise failed: no clip settled, so this test compared -1 with -1 and "
        "would pass against a hardcoded answer")
