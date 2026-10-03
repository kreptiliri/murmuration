import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
windows = pytest.importorskip("windows")
generate = pytest.importorskip("generate")

if not hasattr(train, "shared_squasher"):
    pytest.skip("train.shared_squasher does not exist yet, this suite is its "
                "specification", allow_module_level=True)


N_BIRDS = 12
N_FEATURES = N_BIRDS * 2


def pooled_by_hand(values):
    # one mean and one std per (tick, coordinate), over every row and every
    # bird. columns run x0 y0 x1 y1 ..., so column j is coordinate j % 2
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1, values.shape[-1])
    mean = np.empty(flat.shape[1:])
    std = np.empty(flat.shape[1:])
    for tick in range(flat.shape[1]):
        for coordinate in (0, 1):
            pool = flat[:, tick, coordinate::2]
            mean[tick, coordinate::2] = pool.mean()
            std[tick, coordinate::2] = pool.std()
    std[std == 0] = 1.0
    return mean.reshape(values.shape[1:]), std.reshape(values.shape[1:])


def uneven_windows(rng, n_rows=400, context=3):
    # every slot on a different ruler, x and y on different rulers, and tick 0
    # absolute positions while the later ticks are small moves - the three
    # things a wrong pooling would blur together
    slot_scale = rng.uniform(0.5, 3.0, size=N_BIRDS)
    slot_shift = rng.uniform(-5.0, 5.0, size=N_BIRDS)
    birds = rng.normal(size=(n_rows, context, N_BIRDS, 2))
    birds = birds * slot_scale[:, None] + slot_shift[:, None]
    birds[..., 1] *= 4.0
    birds[:, 0] = birds[:, 0] * 10.0 + 50.0
    return birds.reshape(n_rows, context, N_FEATURES)


def is_pooled(stat):
    stat = np.asarray(stat)
    birds = stat.reshape(*stat.shape[:-1], -1, 2)
    return bool((birds == birds[..., :1, :]).all())


# --- the function ----------------------------------------------------------------

def test_it_returns_what_squasher_returns_in_shape():
    values = uneven_windows(np.random.default_rng(0))

    for got, per_slot in zip(train.shared_squasher(values), train.squasher(values)):
        assert np.shape(got) == np.shape(per_slot)


def test_the_statistics_are_pooled_over_the_birds():
    values = uneven_windows(np.random.default_rng(1))
    expected_mean, expected_std = pooled_by_hand(values)

    mean, std = train.shared_squasher(values)

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(std, expected_std, rtol=1e-9, atol=1e-9)


def test_every_bird_gets_the_same_ruler():
    mean, std = train.shared_squasher(uneven_windows(np.random.default_rng(2)))

    assert is_pooled(mean), "the mean differs between bird slots"
    assert is_pooled(std), "the std differs between bird slots"


def test_x_and_y_keep_their_own_ruler():
    mean, std = train.shared_squasher(uneven_windows(np.random.default_rng(3)))

    for stat, name in ((mean, "mean"), (std, "std")):
        stat = np.asarray(stat)
        assert not np.allclose(stat[..., 0::2], stat[..., 1::2]), (
            f"x and y share one {name}, the coordinates were pooled together")


def test_each_tick_keeps_its_own_ruler():
    mean, std = train.shared_squasher(uneven_windows(np.random.default_rng(4)))

    assert abs(mean[0, 0] - mean[1, 0]) > 10.0, "tick 0 and tick 1 share a mean"
    assert std[0, 0] > 5 * std[1, 0], "tick 0 and tick 1 share a std"


def test_a_one_tick_target_is_pooled_too():
    rng = np.random.default_rng(5)
    targets = uneven_windows(rng, context=1)[:, 0]
    expected_mean, expected_std = pooled_by_hand(targets)

    mean, std = train.shared_squasher(targets)

    assert np.shape(mean) == (N_FEATURES,)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(std, expected_std, rtol=1e-9, atol=1e-9)


def test_the_number_of_birds_comes_from_the_width():
    values = np.random.default_rng(6).normal(size=(50, 2, 8))
    expected_mean, _ = pooled_by_hand(values)

    mean, _ = train.shared_squasher(values)

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-9, atol=1e-9)


def test_scaling_is_invertible():
    values = uneven_windows(np.random.default_rng(7))
    scaler = train.shared_squasher(values)

    restored = train.cancel_squash(scaler, train.apply_squash(scaler, values))

    np.testing.assert_allclose(restored, values, atol=1e-9)


def test_a_zero_variance_pool_does_not_divide_by_zero():
    values = np.ones((50, 3, N_FEATURES))

    scaled = train.apply_squash(train.shared_squasher(values), values)

    assert np.isfinite(scaled).all()


def test_it_does_not_modify_what_it_is_handed():
    values = uneven_windows(np.random.default_rng(8))
    before = values.copy()

    train.shared_squasher(values)

    np.testing.assert_array_equal(values, before)


# --- the wiring in main() --------------------------------------------------------

CONTEXT = 3


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("shared_squasher")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=4, n_live=4,
                           n_ticks=40, base_seed=20260924, out_dir=out,
                           allow_dirty=True)
    return out


def flags(dataset, model, *extra):
    return ["--data", str(dataset), "--clips", "6", "--context", str(CONTEXT),
            "--passes", "1", "--batch", "64", "--report-every", "0",
            "--model", model, *extra]


def learn_split(dataset, residual="none"):
    # what main() fits its scalers on: clips 0-3 of 6, as moves, and the
    # target with the baseline taken off
    x, y = windows.build_windows(dataset, context=CONTEXT, clips=range(4))
    if residual == "clamped":
        physics = json.loads((dataset / "config.json").read_text())["physics"]
        y = y - train.clamped_copier_baseline(x, dt=physics["dt"],
                                              min_speed=physics["min_speed"],
                                              max_speed=physics["max_speed"])
    return train.to_moves(x), y


def checkpoint_of(dataset, model, tmp_path, *extra):
    train.main(flags(dataset, model, "--checkpoint-every", "1",
                     "--checkpoint-dir", str(tmp_path), *extra))
    return train.load_checkpoint(tmp_path / "pass_0001.pt")


@pytest.mark.parametrize("residual", ["none", "clamped"])
def test_a_per_bird_run_fits_both_scalers_pooled(tiny_dataset, tmp_path, residual):
    moves, target = learn_split(tiny_dataset, residual)

    got = checkpoint_of(tiny_dataset, "per_bird", tmp_path, "--residual", residual)

    for name, saved, values in (("x", got["x_scaler"], moves),
                                ("y", got["y_scaler"], target)):
        expected_mean, expected_std = pooled_by_hand(values)
        np.testing.assert_allclose(saved[0], expected_mean, rtol=1e-5, atol=1e-6,
                                   err_msg=f"{name} mean is not the pooled one")
        np.testing.assert_allclose(saved[1], expected_std, rtol=1e-5, atol=1e-6,
                                   err_msg=f"{name} std is not the pooled one")


@pytest.mark.parametrize("model", ["linear", "transformer"])
def test_the_other_two_models_keep_the_per_slot_scaler_exactly(tiny_dataset,
                                                               tmp_path, model):
    moves, target = learn_split(tiny_dataset)

    got = checkpoint_of(tiny_dataset, model, tmp_path)

    for name, saved, values in (("x", got["x_scaler"], moves),
                                ("y", got["y_scaler"], target)):
        per_slot = train.squasher(values)
        np.testing.assert_array_equal(saved[0], per_slot[0], err_msg=f"{name} mean")
        np.testing.assert_array_equal(saved[1], per_slot[1], err_msg=f"{name} std")


@pytest.mark.parametrize("model", ["linear", "transformer"])
def test_the_other_two_models_never_call_it(tiny_dataset, monkeypatch, model):
    called = []
    real = train.shared_squasher
    monkeypatch.setattr(train, "shared_squasher",
                        lambda values: called.append(len(values)) or real(values))

    train.main(flags(tiny_dataset, model))

    assert called == [], f"{model} run called shared_squasher {len(called)} times"


def test_every_batch_is_scaled_with_the_pooled_scaler(tiny_dataset, monkeypatch):
    seen = []
    real = train.apply_squash
    monkeypatch.setattr(train, "apply_squash",
                        lambda scaler, values: seen.append(scaler) or real(scaler, values))

    train.main(flags(tiny_dataset, "per_bird"))

    assert len(seen) > 2, "apply_squash was barely called"
    for mean, std in seen:
        assert is_pooled(mean) and is_pooled(std), (
            "a batch or the held-out set was scaled with a per-slot scaler")


def test_the_score_is_unscaled_with_the_pooled_target_scaler(tiny_dataset, monkeypatch):
    seen = []
    real = train.cancel_squash
    monkeypatch.setattr(train, "cancel_squash",
                        lambda scaler, values: seen.append(scaler) or real(scaler, values))

    train.main(flags(tiny_dataset, "per_bird"))

    assert seen, "cancel_squash was never called"
    for mean, std in seen:
        assert is_pooled(mean) and is_pooled(std), (
            "the predictions went back to world units through a per-slot scaler")


def test_it_never_sees_the_held_out_clips(tiny_dataset, monkeypatch):
    seen = []
    real = train.shared_squasher
    monkeypatch.setattr(train, "shared_squasher",
                        lambda values: seen.append(len(values)) or real(values))

    train.main(flags(tiny_dataset, "per_bird"))

    windows_per_clip = 41 - CONTEXT
    assert len(seen) == 2, f"shared_squasher ran {len(seen)} times, expected x and y"
    assert all(rows == 4 * windows_per_clip for rows in seen), (
        f"shared_squasher saw {seen} rows; the training half is "
        f"{4 * windows_per_clip} (4 of 6 clips)")
