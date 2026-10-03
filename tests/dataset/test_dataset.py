import json
import subprocess

import numpy as np
import pytest

from boids import Physics, velocity_capper
from world import PHYSICS_V1

generate = pytest.importorskip("generate")

pytestmark = pytest.mark.skipif(
    not hasattr(generate, "build_dataset"),
    reason="generate.build_dataset() does not exist yet, this suite is its "
           "specification")

# sim-v1's world: its flocks settle soon enough for the settle_tick tests
# (the capped world's settle far later, and the premise guard fails)
PHYSICS = PHYSICS_V1
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
    return build(tmp_path_factory.mktemp("full"))


@pytest.fixture(scope="module")
def settled(tmp_path_factory):
    return build(tmp_path_factory.mktemp("settled"), n_ticks=SETTLED_TICKS)


@pytest.fixture(scope="module")
def partial(tmp_path_factory):
    return build(tmp_path_factory.mktemp("partial"), n_live=4)


def arrays(out_dir):
    with np.load(out_dir / "clips.npz") as z:
        return {name: z[name] for name in z.files}


def config(out_dir):
    return json.loads((out_dir / "config.json").read_text())


def polarisation(velocities):
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    return np.linalg.norm((velocities / np.where(speeds == 0, 1, speeds)).mean(axis=0))


def upcast(a):
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
    a = arrays(full)

    for rule in ("separation", "alignment", "cohesion"):
        assert a[rule].shape == (N_CLIPS, N_TICKS, N_SLOTS, 2), rule


def test_the_stored_dtypes(full):
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
    assert Physics(**config(full)["physics"]) == PHYSICS


def test_the_generation_parameters_are_recorded(full):
    gen = config(full)["generation"]

    assert gen["n_clips"] == N_CLIPS
    assert gen["n_slots"] == N_SLOTS
    assert gen["n_live"] == N_SLOTS
    assert gen["n_ticks"] == N_TICKS
    assert gen["base_seed"] == BASE_SEED


def test_provenance_is_recorded(full):
    prov = config(full)["provenance"]

    assert prov["commit"], "no commit SHA recorded"
    assert prov["commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=True).stdout.strip()
    assert "tag" in prov
    assert isinstance(prov["dirty"], bool)


def test_the_allow_dirty_flag_is_recorded_even_when_the_tree_is_clean(full):
    assert config(full)["generation"]["allow_dirty"] is True


def test_a_dirty_tree_is_refused_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(generate, "git_is_dirty", lambda: True)

    with pytest.raises(RuntimeError):
        generate.build_dataset(PHYSICS, n_clips=1, n_slots=N_SLOTS,
                               n_live=N_SLOTS, n_ticks=2, base_seed=0,
                               out_dir=tmp_path)

    generate.build_dataset(PHYSICS, n_clips=1, n_slots=N_SLOTS, n_live=N_SLOTS,
                           n_ticks=2, base_seed=0, out_dir=tmp_path,
                           allow_dirty=True)
    assert config(tmp_path)["provenance"]["dirty"] is True


# --- the git helpers ---------------------------------------------------------

def test_git_is_dirty_agrees_with_git_itself():
    porcelain = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()

    assert generate.git_is_dirty() == bool(porcelain)


def test_git_is_dirty_returns_a_real_bool():
    assert isinstance(generate.git_is_dirty(), bool)


def test_provenance_describes_the_code_not_the_callers_directory(tmp_path, monkeypatch):
    expected = generate.git_provenance()["commit"]
    monkeypatch.chdir(tmp_path)          #not a git repository at all

    assert generate.git_provenance()["commit"] == expected


def test_provenance_commit_is_a_full_sha():
    commit = generate.git_provenance()["commit"]

    assert len(commit) == 40, commit
    assert all(c in "0123456789abcdef" for c in commit), commit


def test_provenance_tag_is_never_empty():
    assert generate.git_provenance()["tag"].strip()


def test_provenance_dirty_matches_the_helper():
    assert generate.git_provenance()["dirty"] == generate.git_is_dirty()


# --- the seeds ---------------------------------------------------------------

def test_clip_seed_reproduces_the_spawned_child_and_does_not_depend_on_n():
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
    positions = arrays(full)["positions"]

    for i in range(1, N_CLIPS):
        assert not np.array_equal(positions[0], positions[i]), f"clip {i} == clip 0"


# --- the invariants, restated against what is actually on disk ---------------

def test_forces_reconstruct_velocities_after_the_float32_round_trip(full):
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

    assert worst < 1e-6, f"round trip lost {worst:.3e}, more than the cast alone"


def test_mask_and_nan_are_exact_complements_on_disk(partial):
    a = arrays(partial)
    dead = ~a["mask"]

    assert dead.any(), "premise failed: nothing was masked off"
    np.testing.assert_array_equal(np.isnan(a["positions"][..., 0]), dead)
    np.testing.assert_array_equal(np.isnan(a["velocities"][..., 0]), dead)
    for rule in ("separation", "alignment", "cohesion"):
        np.testing.assert_array_equal(np.isnan(a[rule][..., 0]), dead[:, :-1], rule)


def test_no_nan_in_live_slots(partial):
    a = arrays(partial)
    live = a["mask"]

    assert not np.isnan(a["positions"][live]).any()
    assert not np.isnan(a["velocities"][live]).any()
    for rule in ("separation", "alignment", "cohesion"):
        assert not np.isnan(a[rule][live[:, :-1]]).any(), rule


def test_positions_and_speeds_are_legal_including_tick_0(full):
    a = arrays(full)
    positions = upcast(a["positions"])[a["mask"]]
    speeds = np.linalg.norm(upcast(a["velocities"])[a["mask"]], axis=-1)

    assert positions.min() >= 0.0
    assert positions.max() <= PHYSICS.bounds
    assert speeds.min() >= PHYSICS.min_speed - 1e-5, f"slowest {speeds.min()}"
    assert speeds.max() <= PHYSICS.max_speed + 1e-5, f"fastest {speeds.max()}"


def test_a_clip_with_no_ticks_still_has_a_legal_shape(tmp_path):
    out = build(tmp_path, n_clips=1, n_ticks=0)
    a = arrays(out)

    assert a["positions"].shape == (1, 1, N_SLOTS, 2)
    for rule in ("separation", "alignment", "cohesion"):
        assert a[rule].shape == (1, 0, N_SLOTS, 2), rule
    assert int(a["settle_tick"][0]) in (-1, 0)


def test_thirty_clips_are_all_legal(tmp_path):
    out = build(tmp_path, n_clips=30, n_ticks=25)
    a = arrays(out)
    live = a["mask"]

    assert not np.isnan(a["positions"][live]).any()
    positions = upcast(a["positions"])[live]
    speeds = np.linalg.norm(upcast(a["velocities"])[live], axis=-1)

    assert positions.min() >= 0.0 and positions.max() <= PHYSICS.bounds
    assert speeds.min() >= PHYSICS.min_speed - 1e-5
    assert speeds.max() <= PHYSICS.max_speed + 1e-5
    assert np.all(a["settle_tick"] >= -1) and np.all(a["settle_tick"] <= 25)
    np.testing.assert_array_equal(np.isnan(a["positions"][..., 0]), ~live)


def test_settle_tick_matches_a_recomputation_from_the_stored_velocities(settled):
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
