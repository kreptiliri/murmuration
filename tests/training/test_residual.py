import inspect
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")

if not hasattr(train, "clamped_copier_baseline"):
    pytest.skip("train.clamped_copier_baseline does not exist yet, this suite "
                "is its specification", allow_module_level=True)


CONTEXT = 4
N_SLOTS = 4
DT = 0.1
FLOOR = 1.0
CEILING = 5.0


def window_of(velocities, *, context=CONTEXT, dt=DT, start=50.0):
    """One row in which every bird holds a constant velocity.

    The copier reads zero on such a row, so anything the clamped baseline
    returns is the clamp and nothing else.
    """
    velocities = np.asarray(velocities, dtype=float)
    ticks = np.arange(context, dtype=float)[:, None, None]
    positions = start + velocities[None] * ticks * dt
    return positions.reshape(1, context, -1)


def per_bird(flat):
    return np.asarray(flat).reshape(np.asarray(flat).shape[0], -1, 2)


def speeds(flat):
    return np.linalg.norm(per_bird(flat), axis=-1)


# --- the baseline as a function ------------------------------------------------

def test_inside_the_band_it_is_exactly_the_copier():
    """A bird already flying at a legal speed is never touched, so the clamped
    baseline has to agree with the plain copier to the last bit."""
    window = window_of([[3.0, 0.0], [0.0, -2.5], [2.0, 2.0], [-1.5, 1.0]])

    plain = train.copier_baseline(window, dt=DT)
    clamped = train.clamped_copier_baseline(window, dt=DT, min_speed=FLOOR,
                                            max_speed=CEILING)

    np.testing.assert_allclose(clamped, plain, atol=1e-12)


def test_the_floor_is_applied_to_the_speed_not_to_each_component():
    """The floor, applied to the speed. A bird at (0.6, 0.6) has speed 0.849, under the
    floor. Scaling to length 1.0 gives (0.7071, 0.7071); clamping each component
    to 1.0 gives (1.0, 1.0), which is speed 1.414 - a different bird."""
    window = window_of([[0.6, 0.6]] * N_SLOTS)

    change = train.clamped_copier_baseline(window, dt=DT, min_speed=FLOOR,
                                           max_speed=CEILING)
    new_velocity = per_bird(change) + 0.6

    np.testing.assert_allclose(speeds(new_velocity.reshape(1, -1)),
                               np.full((1, N_SLOTS), FLOOR), atol=1e-9)
    assert not np.allclose(new_velocity, 1.0), (
        "every component came back as 1.0 - the clamp was applied per component "
        "instead of to the speed")


def test_the_ceiling_is_applied_to_the_speed_not_to_each_component():
    """The ceiling, where a componentwise clamp does nothing at all.
    (4.0, 4.0) is speed 5.657, over the ceiling, but neither component is."""
    window = window_of([[4.0, 4.0]] * N_SLOTS)

    change = train.clamped_copier_baseline(window, dt=DT, min_speed=FLOOR,
                                           max_speed=CEILING)
    new_velocity = per_bird(change) + 4.0

    np.testing.assert_allclose(speeds(new_velocity.reshape(1, -1)),
                               np.full((1, N_SLOTS), CEILING), atol=1e-9)
    assert not np.allclose(change, 0.0), (
        "nothing moved - a componentwise clamp leaves (4, 4) alone because "
        "neither component exceeds 5.0")


def test_it_returns_a_change_in_velocity_not_a_velocity():
    """Same units as `copier_baseline` and as the training target. Returning the
    clamped velocity itself is off by the current velocity everywhere, which is
    large and completely plausible."""
    window = window_of([[3.0, 0.0]] * N_SLOTS)

    change = train.clamped_copier_baseline(window, dt=DT, min_speed=FLOOR,
                                           max_speed=CEILING)

    np.testing.assert_allclose(change, 0.0, atol=1e-12)


def test_each_bird_is_clamped_on_its_own():
    """One speed computed over the whole flattened row would drag every bird
    toward one correction and still return the right shape."""
    window = window_of([[0.6, 0.6], [3.0, 0.0], [4.0, 4.0], [0.0, -2.0]])

    change = per_bird(train.clamped_copier_baseline(
        window, dt=DT, min_speed=FLOOR, max_speed=CEILING))[0]

    assert not np.allclose(change[0], 0.0), "the slow bird was not raised"
    np.testing.assert_allclose(change[1], 0.0, atol=1e-9,
                               err_msg="a legal bird was moved")
    assert not np.allclose(change[2], 0.0), "the fast bird was not slowed"
    np.testing.assert_allclose(change[3], 0.0, atol=1e-9,
                               err_msg="a legal bird was moved")


def test_the_limits_come_from_the_caller():
    """The same window under two floors has to give two answers, or the
    limits are hard-coded and the argument is decoration."""
    window = window_of([[1.5, 0.0]] * N_SLOTS)

    at_one = train.clamped_copier_baseline(window, dt=DT, min_speed=1.0,
                                           max_speed=CEILING)
    at_two = train.clamped_copier_baseline(window, dt=DT, min_speed=2.0,
                                           max_speed=CEILING)

    np.testing.assert_allclose(at_one, 0.0, atol=1e-9)
    assert not np.allclose(at_two, 0.0), (
        "a bird at speed 1.5 is legal under a floor of 1.0 and illegal under "
        "2.0; both came back the same, so min_speed is not being read")


def test_it_needs_three_ticks_like_the_copier():
    """Two ticks give one velocity and no way to extrapolate a second."""
    window = window_of([[3.0, 0.0]] * N_SLOTS, context=2)

    with pytest.raises(ValueError):
        train.clamped_copier_baseline(window, dt=DT, min_speed=FLOOR,
                                      max_speed=CEILING)


def test_it_cannot_be_handed_a_target():
    """Pinned by signature. Subtracting a function of the input is a
    rewording of the question. Subtracting anything built from the answer is the
    leak this project has already been caught by once."""
    taken = set(inspect.signature(train.clamped_copier_baseline).parameters)

    assert not taken & {"targets", "y", "truth", "velocities", "forces",
                        "next_velocities"}, (
        f"the baseline accepts {taken} - anything derived from the target makes "
        "every residual score meaningless")


def test_a_bird_at_rest_does_not_come_back_as_nan():
    """Zero velocity has no direction to scale, so the obvious implementation
    divides by zero. A NaN here would ride into the training target and poison
    the loss silently."""
    window = window_of([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [2.0, 2.0]])

    change = train.clamped_copier_baseline(window, dt=DT, min_speed=FLOOR,
                                           max_speed=CEILING)

    assert np.isfinite(change).all(), "a stationary bird produced NaN or inf"


def test_it_is_shaped_like_the_copier_on_real_windows(floor_one_dataset):
    """Both baselines have to be drop-in substitutes for each other in `main()`."""
    import windows
    x, _ = windows.build_windows(str(floor_one_dataset), context=CONTEXT,
                                 clips=range(2))

    plain = train.copier_baseline(x, dt=DT)
    clamped = train.clamped_copier_baseline(x, dt=DT, min_speed=FLOOR,
                                            max_speed=CEILING)

    assert clamped.shape == plain.shape
    assert clamped.dtype == plain.dtype


def test_the_two_baselines_actually_differ_on_real_data(floor_one_dataset):
    """The control for every test below. If the clamp never fires on this
    dataset then `--residual clamped` and `--residual copier` are the same run
    and nothing downstream is being checked."""
    import windows
    x, _ = windows.build_windows(str(floor_one_dataset), context=CONTEXT,
                                 clips=range(2))

    plain = train.copier_baseline(x, dt=DT)
    clamped = train.clamped_copier_baseline(x, dt=DT, min_speed=FLOOR,
                                            max_speed=CEILING)

    assert not np.allclose(plain, clamped), (
        "the clamp never fires on this dataset, so these tests prove nothing")


# --- the flag on main() --------------------------------------------------------

@pytest.fixture(scope="module")
def floor_one_dataset(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("residual_floor_one")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=N_SLOTS,
                           n_live=N_SLOTS, n_ticks=40, base_seed=20260920,
                           out_dir=out, allow_dirty=True)
    return out


@pytest.fixture(scope="module")
def floor_two_dataset(tmp_path_factory):
    """A dataset at the *old* floor, so a hard-coded 1.0 is visible."""
    import sys
    from dataclasses import replace
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("residual_floor_two")
    generate.build_dataset(replace(PHYSICS_CRUISING, min_speed=2.0), n_clips=6,
                           n_slots=N_SLOTS, n_live=N_SLOTS, n_ticks=40,
                           base_seed=20260920, out_dir=out, allow_dirty=True)
    return out


def scores(out):
    """Everything main() prints except the wall-clock line, which never repeats."""
    return [line for line in out.splitlines()
            if line.startswith("R2 ") or line.startswith("skill ")]


def flags(dataset, *extra):
    return ["--data", str(dataset), "--clips", "6", "--context", str(CONTEXT),
            "--passes", "2", "--batch", "64", "--report-every", "0", *extra]


def test_the_default_is_no_residual(floor_one_dataset, capsys):
    """Every score already recorded was trained on the plain target. `none` and
    no flag at all have to be the same run, to the digit."""
    train.main(flags(floor_one_dataset))
    without = scores(capsys.readouterr().out)
    train.main(flags(floor_one_dataset, "--residual", "none"))
    with_none = scores(capsys.readouterr().out)

    assert without and without == with_none


def test_an_unknown_residual_is_refused(floor_one_dataset):
    """A typo must not quietly train against the plain target for an hour."""
    with pytest.raises(SystemExit):
        train.main(flags(floor_one_dataset, "--residual", "coppier"))


def spy_on(monkeypatch, name):
    seen = []
    real = getattr(train, name)

    def watched(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append({"args": args, "kwargs": kwargs, "out": out})
        return out

    monkeypatch.setattr(train, name, watched)
    return seen


def test_the_clamped_choice_reads_the_datasets_own_limits(floor_two_dataset,
                                                          monkeypatch):
    """End to end. This dataset was generated at `min_speed` 2.0. A
    baseline that assumes 1.0 clamps nothing and looks like the plain copier."""
    seen = spy_on(monkeypatch, "clamped_copier_baseline")

    train.main(flags(floor_two_dataset, "--residual", "clamped"))

    assert seen, "--residual clamped never called clamped_copier_baseline"
    physics = json.loads((floor_two_dataset / "config.json").read_text())["physics"]
    for call in seen:
        assert call["kwargs"]["min_speed"] == physics["min_speed"]
        assert call["kwargs"]["max_speed"] == physics["max_speed"]


def test_the_target_the_model_trains_on_is_the_residual(floor_one_dataset,
                                                        monkeypatch):
    """The heart of it. The Y scaler is fitted on whatever the model
    is asked to learn, so watching what it is handed says which target won."""
    import windows
    y_learn = windows.build_windows(str(floor_one_dataset), context=CONTEXT,
                                    clips=range(4))[1]
    handed = []
    real = train.squasher
    monkeypatch.setattr(train, "squasher",
                        lambda values: (handed.append(np.array(values)),
                                        real(values))[1])

    train.main(flags(floor_one_dataset, "--residual", "copier"))

    assert not any(a.shape == y_learn.shape and np.allclose(a, y_learn)
                   for a in handed), (
        "the scaler was fitted on the raw target, so the model is learning the "
        "full velocity change while being trained on a residual")


def test_the_batch_the_model_learns_from_is_the_residual(floor_one_dataset,
                                                         monkeypatch):
    """The other half, and the one that survives every check above.

    Fitting the scaler on the residual and adding the baseline back at scoring
    can both be right while the loop still hands the model the raw target. The
    model then learns the full velocity change, has it scaled as though it were
    a leftover, and gets a baseline added on top. Nothing raises and the score
    is merely bad.

    Every two-axis array handed to `apply_squash` is a target: the held-out set
    once, then one per batch. Over a single pass those batches partition the
    training rows, so the values seen must be exactly the two residual arrays.
    """
    import windows
    x_learn, y_learn = windows.build_windows(str(floor_one_dataset),
                                             context=CONTEXT, clips=range(4))
    x_check, y_check = windows.build_windows(str(floor_one_dataset),
                                             context=CONTEXT, clips=range(4, 6))
    expected = np.concatenate([y_check - train.copier_baseline(x_check, dt=DT),
                               y_learn - train.copier_baseline(x_learn, dt=DT)])

    handed = []
    real = train.apply_squash

    def watched(scaler, values):
        if np.asarray(values).ndim == 2:
            handed.append(np.asarray(values))
        return real(scaler, values)

    monkeypatch.setattr(train, "apply_squash", watched)

    train.main(flags(floor_one_dataset, "--residual", "copier", "--passes", "1"))

    seen = np.concatenate(handed)
    assert seen.shape == expected.shape, (
        f"saw {seen.shape} rows of target, expected {expected.shape}")
    np.testing.assert_allclose(
        np.sort(seen, axis=0), np.sort(expected, axis=0), atol=1e-5,
        err_msg="the targets the model was trained on are not the residual")


def test_the_baseline_is_added_back_before_scoring(floor_one_dataset,
                                                   monkeypatch):
    """With the model's unscaled output forced to zero, the prediction
    that reaches `skill` must be the baseline itself - nothing else."""
    import windows
    x_check = windows.build_windows(str(floor_one_dataset), context=CONTEXT,
                                    clips=range(4, 6))[0]
    monkeypatch.setattr(train, "cancel_squash",
                        lambda scaler, values: np.zeros_like(np.asarray(values)))
    scored = []
    monkeypatch.setattr(train, "skill",
                        lambda prediction, truth, baseline: scored.append(
                            np.array(prediction)) or 0.0)

    train.main(flags(floor_one_dataset, "--residual", "copier"))

    assert len(scored) >= 2, "main() no longer reports a skill against the copier"
    np.testing.assert_allclose(
        scored[-1], train.copier_baseline(x_check, dt=DT), atol=1e-9,
        err_msg="the scored prediction is not model + baseline - the baseline "
                "was never added back")


def test_nothing_is_added_back_when_the_residual_is_off(floor_one_dataset,
                                                        monkeypatch):
    """The control for the test above: the same forced-zero model with
    `--residual none` has to score exactly zero, not the copier."""
    monkeypatch.setattr(train, "cancel_squash",
                        lambda scaler, values: np.zeros_like(np.asarray(values)))
    scored = []
    monkeypatch.setattr(train, "skill",
                        lambda prediction, truth, baseline: scored.append(
                            np.array(prediction)) or 0.0)

    train.main(flags(floor_one_dataset, "--residual", "none"))

    np.testing.assert_allclose(scored[-1], 0.0, atol=1e-12)


def test_the_held_out_baseline_is_built_from_the_held_out_windows(
        floor_one_dataset, monkeypatch):
    """Two clips held out of six give a different row count from the
    four trained on, so the wrong one is a shape error - until a split happens
    to be symmetric, and then it is silent."""
    import windows
    n_learn = len(windows.build_windows(str(floor_one_dataset),
                                        context=CONTEXT, clips=range(4))[0])
    n_check = len(windows.build_windows(str(floor_one_dataset),
                                        context=CONTEXT, clips=range(4, 6))[0])
    assert n_learn != n_check, "this fixture no longer discriminates"
    seen = spy_on(monkeypatch, "copier_baseline")

    train.main(flags(floor_one_dataset, "--residual", "copier"))

    lengths = {len(call["out"]) for call in seen}
    assert n_learn in lengths and n_check in lengths, (
        f"the copier was only ever built on rows of {lengths}; both the "
        "training and the held-out windows need their own baseline")


def test_the_held_out_target_is_never_shifted(floor_one_dataset, monkeypatch):
    """The residual is a training device. Scoring still compares a full
    prediction against the real velocity change, or the R² is against a target
    nobody cares about."""
    import windows
    y_check = windows.build_windows(str(floor_one_dataset), context=CONTEXT,
                                    clips=range(4, 6))[1]
    truths = []
    monkeypatch.setattr(train, "skill",
                        lambda prediction, truth, baseline: truths.append(
                            np.array(truth)) or 0.0)

    train.main(flags(floor_one_dataset, "--residual", "clamped"))

    np.testing.assert_allclose(truths[-1], y_check, atol=1e-12)


def test_a_residual_run_is_repeatable_under_a_seed(floor_one_dataset, capsys):
    """Nothing in the residual path may reach for unseeded randomness."""
    train.main(flags(floor_one_dataset, "--residual", "clamped", "--seed", "3"))
    first = scores(capsys.readouterr().out)
    train.main(flags(floor_one_dataset, "--residual", "clamped", "--seed", "3"))

    assert first and scores(capsys.readouterr().out) == first
