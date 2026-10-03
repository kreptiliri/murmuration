import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip(
    "train",
    reason="train.py does not exist yet, this suite is its specification")


CONTEXT = 8
N_SLOTS = 12
N_FEATURES = N_SLOTS * 2


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    generate = pytest.importorskip("generate")
    out = tmp_path_factory.mktemp("train_defaults")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=4, n_live=4,
                           n_ticks=40, base_seed=20260920, out_dir=out,
                           allow_dirty=True)
    return out


def test_the_default_learning_rate_is_the_one_every_score_was_measured_at(
        tiny_dataset, monkeypatch):
    seen = []
    real = torch.optim.Adam
    monkeypatch.setattr(torch.optim, "Adam",
                        lambda params, **kw: seen.append(kw.get("lr")) or real(params, **kw))

    train.main(["--data", str(tiny_dataset), "--clips", "6", "--context", "4",
                "--passes", "1", "--batch", "64", "--report-every", "0"])

    assert seen == [pytest.approx(1e-3)], (
        f"the optimiser was built with lr={seen}, not 1e-3")


# --- trap 1: the split ---------------------------------------------------------

def test_the_splits_are_disjoint():
    train_clips, test_clips = train.split_by_clip(200, train_fraction=0.7)

    assert not (set(train_clips) & set(test_clips))


def test_the_splits_cover_every_clip():
    train_clips, test_clips = train.split_by_clip(200, train_fraction=0.7)

    assert sorted(set(train_clips) | set(test_clips)) == list(range(200))


def test_the_train_fraction_is_respected():
    train_clips, _ = train.split_by_clip(200, train_fraction=0.7)

    assert len(list(train_clips)) == 140


# --- trap 2: the units ---------------------------------------------------------

def test_scaling_is_invertible():
    values = np.random.default_rng(0).normal(3.0, 7.0, size=(500, N_FEATURES))
    scaler = train.squasher(values)

    restored = train.cancel_squash(scaler, train.apply_squash(scaler, values))

    np.testing.assert_allclose(restored, values, atol=1e-5)


def test_scaled_values_are_actually_scaled():
    values = np.random.default_rng(1).normal(3.0, 7.0, size=(500, N_FEATURES))
    scaler = train.squasher(values)

    scaled = train.apply_squash(scaler, values)

    assert abs(scaled.mean()) < 0.05, "not centred"
    assert abs(scaled.std() - 1.0) < 0.05, "not unit scale"


def test_the_scaler_uses_only_what_it_was_fitted_on():
    rng = np.random.default_rng(2)
    reference = rng.normal(0.0, 1.0, size=(500, N_FEATURES))
    other = rng.normal(50.0, 1.0, size=(500, N_FEATURES))

    scaler = train.squasher(reference)
    scaled_other = train.apply_squash(scaler, other)

    assert abs(scaled_other.mean()) > 1.0, "the scaler was refitted on the new data"


def test_a_zero_variance_column_does_not_divide_by_zero():
    values = np.ones((50, N_FEATURES))
    scaler = train.squasher(values)

    scaled = train.apply_squash(scaler, values)

    assert np.isfinite(scaled).all()


# --- trap 3: the wiring --------------------------------------------------------

def test_the_model_maps_a_window_to_one_tick_of_prediction():
    model = train.build_model(context=CONTEXT, n_features=N_FEATURES)
    batch = torch.zeros(4, CONTEXT, N_FEATURES)

    output = model(batch)

    assert tuple(output.shape) == (4, N_FEATURES)


def test_the_model_can_memorise_eight_windows():
    torch.manual_seed(0)
    inputs = torch.randn(8, CONTEXT, N_FEATURES)
    targets = torch.randn(8, N_FEATURES)

    model = train.build_model(context=CONTEXT, n_features=N_FEATURES)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()

    first = loss_function(model(inputs), targets).item()
    for _ in range(400):
        optimiser.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimiser.step()
    last = loss.item()

    assert last < first * 0.05, (
        f"loss went {first:.4f} -> {last:.4f}. A model that cannot memorise "
        "eight examples is wired wrong, not undertrained")


def test_two_models_built_under_the_same_seed_are_identical():
    torch.manual_seed(7)
    first = train.build_model(context=CONTEXT, n_features=N_FEATURES)
    torch.manual_seed(7)
    second = train.build_model(context=CONTEXT, n_features=N_FEATURES)

    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


# --- trap 4: the score ---------------------------------------------------------

def test_skill_is_zero_against_the_baseline_itself():
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(100, N_FEATURES))
    baseline = truth + rng.normal(scale=0.1, size=truth.shape)

    assert abs(train.skill(baseline, truth, baseline)) < 1e-9


def test_skill_is_one_for_a_perfect_prediction():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(100, N_FEATURES))
    baseline = truth + rng.normal(scale=0.1, size=truth.shape)

    assert abs(train.skill(truth, truth, baseline) - 1.0) < 1e-9


def test_skill_is_negative_when_worse_than_the_baseline():
    rng = np.random.default_rng(5)
    truth = rng.normal(size=(100, N_FEATURES))
    baseline = truth + rng.normal(scale=0.1, size=truth.shape)
    worse = truth + rng.normal(scale=0.5, size=truth.shape)

    assert train.skill(worse, truth, baseline) < 0


# --- trap 5: the input arrangement ---------------------------------------------

def test_to_moves_keeps_the_shape():
    rng = np.random.default_rng(6)
    positions = rng.normal(size=(20, CONTEXT, N_FEATURES))

    assert train.to_moves(positions).shape == positions.shape


def test_to_moves_leaves_the_first_tick_alone():
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(20, CONTEXT, N_FEATURES))

    np.testing.assert_array_equal(train.to_moves(positions)[:, 0], positions[:, 0])


def test_to_moves_throws_nothing_away():
    rng = np.random.default_rng(8)
    positions = rng.normal(size=(20, CONTEXT, N_FEATURES))

    restored = np.cumsum(train.to_moves(positions), axis=1)

    np.testing.assert_allclose(restored, positions, atol=1e-12)


def test_to_moves_puts_the_step_between_ticks_in_row_one():
    rng = np.random.default_rng(9)
    positions = rng.normal(size=(20, CONTEXT, N_FEATURES))

    np.testing.assert_allclose(train.to_moves(positions)[:, 1],
                               positions[:, 1] - positions[:, 0], atol=1e-12)


# --- trap 6: the baseline every score is reported against -----------------------

def test_the_copier_is_the_previous_velocity_change():
    rng = np.random.default_rng(10)
    positions = rng.normal(size=(20, CONTEXT, N_FEATURES))

    expected = (positions[:, -1] - 2 * positions[:, -2] + positions[:, -3]) / 0.1

    np.testing.assert_allclose(train.copier_baseline(positions, dt=0.1),
                               expected, atol=1e-12)


def test_the_copier_predicts_no_change_at_constant_velocity():
    steady = np.arange(CONTEXT)[None, :, None] * np.ones((5, 1, N_FEATURES)) * 0.4

    predicted = train.copier_baseline(steady, dt=0.1)

    np.testing.assert_allclose(predicted, 0.0, atol=1e-12)


def test_the_copier_refuses_a_context_it_cannot_read():
    with pytest.raises(ValueError, match="3 ticks"):
        train.copier_baseline(np.zeros((5, 2, N_FEATURES)), dt=0.1)


# --- trap 7: the run itself -----------------------------------------------------

generate = pytest.importorskip("generate")


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("training")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=4, n_live=4,
                           n_ticks=40, base_seed=20260915, out_dir=out,
                           allow_dirty=True)
    return out


def test_main_runs_end_to_end_and_reports(tiny_dataset, capsys):
    code = train.main(["--data", str(tiny_dataset), "--clips", "6", "--context", "4",
                       "--passes", "3", "--batch", "64", "--report-every", "0"])

    printed = capsys.readouterr().out
    assert code == 0
    assert "skill against the copier" in printed
    assert "R2 copier" in printed


def test_main_is_repeatable_under_a_seed(tiny_dataset, capsys):
    flags = ["--data", str(tiny_dataset), "--clips", "6", "--context", "4",
             "--passes", "3", "--batch", "64", "--report-every", "0", "--seed", "1"]

    train.main(flags)
    first = [line for line in capsys.readouterr().out.splitlines() if "R2 model" in line]
    train.main(flags)
    second = [line for line in capsys.readouterr().out.splitlines() if "R2 model" in line]

    assert first == second, "same seed, same data, different score"


def test_main_never_fits_the_scaler_on_the_held_out_clips(tiny_dataset, monkeypatch):
    seen = []
    real = train.squasher
    monkeypatch.setattr(train, "squasher", lambda values: seen.append(len(values)) or real(values))

    train.main(["--data", str(tiny_dataset), "--clips", "6", "--context", "4",
                "--passes", "1", "--batch", "64", "--report-every", "0"])

    n_windows_per_clip = 41 - 4
    assert seen, "squasher was never called"
    assert all(rows == 4 * n_windows_per_clip for rows in seen), (
        f"squasher saw {seen} rows; the training half is "
        f"{4 * n_windows_per_clip} (4 of 6 clips)")
