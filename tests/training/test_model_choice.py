import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
flocker = pytest.importorskip("flocker")
generate = pytest.importorskip("generate")

if "kind" not in inspect.signature(train.build_model).parameters:
    pytest.skip("train.build_model does not take a `kind` yet, this suite is "
                "its specification", allow_module_level=True)


CONTEXT = 8
N_SLOTS = 12
N_FEATURES = N_SLOTS * 2


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """Small enough to train in a second, long enough for a real split."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("model_choice")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=4, n_live=4,
                           n_ticks=40, base_seed=20260916, out_dir=out,
                           allow_dirty=True)
    return out


def flags(dataset, *extra):
    return ["--data", str(dataset), "--clips", "6", "--context", "4",
            "--passes", "2", "--batch", "64", "--report-every", "0", *extra]


# --- which model gets built ----------------------------------------------------

def test_the_default_is_still_the_linear_model():
    """Every score already recorded was the linear model. The default moving
    would silently change what those numbers mean."""
    model = train.build_model(context=CONTEXT, n_features=N_FEATURES)

    assert isinstance(model, train.Model)


def test_the_transformer_is_the_one_in_flocker():
    """Not a second copy built inside train.py. `flocker.Transformer` is the
    thing 34 tests in tests/transformer/ are about."""
    model = train.build_model(context=CONTEXT, n_features=N_FEATURES,
                              kind="transformer")

    assert isinstance(model, flocker.Transformer)


def test_the_transformer_maps_a_window_to_one_tick_of_prediction():
    """Same contract as the linear model: a whole window in, one tick out."""
    model = train.build_model(context=CONTEXT, n_features=N_FEATURES,
                              kind="transformer")

    output = model(torch.zeros(4, CONTEXT, N_FEATURES))

    assert tuple(output.shape) == (4, N_FEATURES)


def test_an_unknown_kind_is_refused():
    """A typo must not fall back to the linear model and train for an hour."""
    with pytest.raises(ValueError):
        train.build_model(context=CONTEXT, n_features=N_FEATURES, kind="lienar")


def test_two_transformers_under_the_same_seed_are_identical():
    """A score has to be repeatable, and the transformer has 403,096 ways not
    to be."""
    torch.manual_seed(7)
    first = train.build_model(context=CONTEXT, n_features=N_FEATURES,
                              kind="transformer")
    torch.manual_seed(7)
    second = train.build_model(context=CONTEXT, n_features=N_FEATURES,
                               kind="transformer")

    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_the_transformer_can_memorise_eight_windows():
    """The wiring check the linear model already gets.

    A correctly wired model can overfit a handful of examples into the ground.
    If it cannot, an axis is transposed or the context is mis-sliced, and no
    amount of real training will fix it - it converges to something plausible
    and wrong instead.
    """
    torch.manual_seed(0)
    inputs = torch.randn(8, CONTEXT, N_FEATURES)
    targets = torch.randn(8, N_FEATURES)

    model = train.build_model(context=CONTEXT, n_features=N_FEATURES,
                              kind="transformer")
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()

    first = loss_function(model(inputs), targets).item()
    for _ in range(400):
        optimiser.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimiser.step()

    assert loss.item() < first * 0.05, (
        f"loss went {first:.4f} -> {loss.item():.4f}. A model that cannot "
        "memorise eight examples is wired wrong, not undertrained")


def test_main_trains_the_transformer_end_to_end(tiny_dataset, capsys):
    """The whole point: one command, from a clone, reproduces a transformer
    number."""
    code = train.main(flags(tiny_dataset, "--model", "transformer"))

    printed = capsys.readouterr().out
    assert code == 0
    assert "skill against the copier" in printed


def test_the_model_flag_actually_reaches_build_model(tiny_dataset, monkeypatch):
    """`--model` being dropped on the floor is invisible from the outside.

    The run completes, the receipt prints, the score is plausible - and the
    one-layer Linear trained for an hour under the transformer's name.
    Nothing else in this file catches it, because everything else a linear model
    also does.
    """
    asked = []
    real = train.build_model
    monkeypatch.setattr(train, "build_model",
                        lambda **kw: asked.append(kw.get("kind")) or real(**kw))

    train.main(flags(tiny_dataset, "--model", "transformer"))

    assert asked == ["transformer"], (
        f"main() asked build_model for {asked}; --model transformer never "
        "reached it")


# --- where the augmentation is applied -----------------------------------------

def spy_on_permute(monkeypatch):
    """Record every call to permute_slots without changing what it does."""
    seen = []
    real = train.permute_slots

    def watched(position_windows, targets, rng):
        seen.append((np.asarray(position_windows).copy(),
                     np.asarray(targets).copy()))
        return real(position_windows, targets, rng)

    monkeypatch.setattr(train, "permute_slots", watched)
    return seen


def test_augment_off_never_calls_it(tiny_dataset, monkeypatch):
    """Every number in NOTES.md was measured without this. If it runs when the
    flag is off, none of them reproduce and nothing says so."""
    seen = spy_on_permute(monkeypatch)

    train.main(flags(tiny_dataset))

    assert seen == [], f"permute_slots ran {len(seen)} times with --augment off"


def test_augment_on_calls_it_every_batch(tiny_dataset, monkeypatch):
    """One draw per row per batch, not once per run."""
    seen = spy_on_permute(monkeypatch)

    train.main(flags(tiny_dataset, "--augment"))

    assert len(seen) > 1, (
        f"permute_slots ran {len(seen)} time(s); a per-batch augmentation over "
        "2 passes must run at least once per pass")


def test_it_is_handed_raw_positions_not_moves_and_not_scaled(tiny_dataset,
                                                             monkeypatch):
    """A permutation applied after the scaler, and the only way to catch it.

    Raw windows are absolute coordinates in [0, bounds) - large and positive.
    `to_moves` output has a first row that size and seven rows near zero.
    Scaled output is centred on zero. Only the first is acceptable, and the
    other two raise nothing.
    """
    seen = spy_on_permute(monkeypatch)

    train.main(flags(tiny_dataset, "--augment"))

    assert seen, "permute_slots was never called"
    for windows, _ in seen:
        later_ticks = np.abs(windows[:, 1:]).mean()
        assert later_ticks > 1.0, (
            f"mean magnitude after the first tick is {later_ticks:.3f}. Absolute "
            "coordinates average tens; `to_moves` output averages about 0.15 "
            "there and scaled data about 0.8. So this is not raw positions - "
            "the permutation ran after the scaler")


def test_the_window_and_the_target_arrive_together(tiny_dataset, monkeypatch):
    """Trap: permuting the window and leaving the target alone. The labels are
    then wrong for eleven of twelve birds and the loss merely plateaus higher."""
    seen = spy_on_permute(monkeypatch)

    train.main(flags(tiny_dataset, "--augment"))

    for windows, targets in seen:
        assert len(windows) == len(targets), (
            f"{len(windows)} windows handed in with {len(targets)} targets")
        assert windows.shape[-1] == targets.shape[-1], (
            "the window and the target must describe the same birds")


def test_the_permuted_target_is_the_one_that_gets_trained_on(tiny_dataset,
                                                             monkeypatch):
    """`raw_batch, _ = permute_slots(...)`, calling it and throwing the target
    away.

    Every test above still passes when that happens: permute_slots really was
    called, with raw positions, with both arrays. The labels are simply wrong
    for eleven of twelve birds afterwards, and the only symptom is a loss that
    plateaus higher than it should. Caught by watching what the scaler is
    actually handed.
    """
    returned = []
    real_permute = train.permute_slots
    monkeypatch.setattr(train, "permute_slots",
                        lambda w, t, rng: returned.append(real_permute(w, t, rng))
                                          or returned[-1])
    scaled = []
    real_squash = train.apply_squash
    monkeypatch.setattr(train, "apply_squash",
                        lambda scaler, values: scaled.append(np.asarray(values))
                                               or real_squash(scaler, values))

    train.main(flags(tiny_dataset, "--augment"))

    assert returned, "permute_slots was never called"
    two_d = [a for a in scaled if a.ndim == 2]
    for _, permuted_target in returned[:3]:
        assert any(a.shape == permuted_target.shape
                   and np.array_equal(a, permuted_target) for a in two_d), (
            "a target that permute_slots returned was never handed to the "
            "scaler, so the batch trained on the unpermuted labels")


def test_the_held_out_windows_are_never_augmented(tiny_dataset, monkeypatch):
    """The held-out set is the only honest thing in the run."""
    seen = spy_on_permute(monkeypatch)

    train.main(flags(tiny_dataset, "--augment"))

    n_windows_per_clip = 41 - 4
    held_out = 2 * n_windows_per_clip
    for windows, _ in seen:
        assert len(windows) != held_out, (
            f"permute_slots was handed {held_out} rows, which is exactly the "
            "held-out set")


def test_the_scaler_still_never_sees_the_held_out_clips(tiny_dataset,
                                                        monkeypatch):
    """The leak that no loss curve shows, re-checked with augmentation on
    because the augmentation moves where the scaler sits in the chain."""
    seen = []
    real = train.squasher
    monkeypatch.setattr(train, "squasher",
                        lambda values: seen.append(len(values)) or real(values))

    train.main(flags(tiny_dataset, "--augment"))

    n_windows_per_clip = 41 - 4
    assert seen, "squasher was never called"
    assert all(rows == 4 * n_windows_per_clip for rows in seen), (
        f"squasher saw {seen} rows; the training half is "
        f"{4 * n_windows_per_clip} (4 of 6 clips)")


def test_an_augmented_run_is_repeatable_under_a_seed(tiny_dataset, capsys):
    """The augmentation draws 12! per row. If it draws from anything but the
    run's own seed, no augmented score can be compared to another."""
    argv = flags(tiny_dataset, "--augment", "--seed", "3")

    train.main(argv)
    first = [l for l in capsys.readouterr().out.splitlines() if "R2 model" in l]
    train.main(argv)
    second = [l for l in capsys.readouterr().out.splitlines() if "R2 model" in l]

    assert first == second, "same seed, same data, different augmented score"
