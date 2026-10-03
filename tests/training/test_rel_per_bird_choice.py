import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
flocker = pytest.importorskip("flocker")
generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")

if not hasattr(flocker, "RelPerBirdTransformer") or not hasattr(train, "pair_geometry"):
    pytest.skip("the relative-position model is not wired in yet, this suite is "
                "its specification", allow_module_level=True)

CLIPS = 6
BATCH = 64


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("rel_per_bird")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=CLIPS, n_slots=4, n_live=4,
                           n_ticks=40, base_seed=20260926, out_dir=out,
                           allow_dirty=True)
    return out


def flags(dataset, *extra, context="2"):
    return ["--data", str(dataset), "--clips", str(CLIPS), "--context", context,
            "--passes", "1", "--batch", str(BATCH), "--report-every", "0",
            "--model", "rel_per_bird", *extra]


def learn_batches(dataset, context=2):
    learn, _ = train.split_by_clip(CLIPS, train_fraction=0.7)
    x, _ = windows.build_windows(dataset, context=context, clips=learn)
    return math.ceil(len(x) / BATCH)


def recording(monkeypatch, name):
    calls = []
    real = getattr(train, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(train, name, wrapper)
    return calls


# --- which model gets built ------------------------------------------------------------

def test_rel_per_bird_builds_the_one_in_flocker():
    model = train.build_model(context=2, n_features=24, kind="rel_per_bird")
    assert isinstance(model, flocker.RelPerBirdTransformer)


def test_per_bird_is_still_the_old_model():
    model = train.build_model(context=2, n_features=24, kind="per_bird")
    assert isinstance(model, flocker.PerBirdTransformer)
    assert not isinstance(model, flocker.RelPerBirdTransformer)


def test_an_unknown_kind_is_still_refused():
    with pytest.raises(ValueError):
        train.build_model(context=2, n_features=24, kind="rel_perbird")


def test_the_flag_reaches_build_model(tiny_dataset, monkeypatch):
    asked = []
    real = train.build_model
    monkeypatch.setattr(train, "build_model",
                        lambda **kw: asked.append(kw.get("kind")) or real(**kw))
    train.main(flags(tiny_dataset))
    assert asked == ["rel_per_bird"]


# --- the run ----------------------------------------------------------------------------

def test_main_trains_it_end_to_end_at_context_2(tiny_dataset, capsys):
    assert train.main(flags(tiny_dataset)) == 0
    assert "R2 model" in capsys.readouterr().out


def test_context_1_is_refused(tiny_dataset):
    with pytest.raises(ValueError):
        train.main(flags(tiny_dataset, context="1"))


def test_it_uses_the_shared_scaler(tiny_dataset, monkeypatch):
    calls = recording(monkeypatch, "shared_squasher")
    train.main(flags(tiny_dataset))
    assert len(calls) == 2


@pytest.fixture(scope="module")
def odd_world(tmp_path_factory):
    import sys
    from dataclasses import replace
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("rel_odd_world")
    physics = replace(PHYSICS_CRUISING, bounds=90.0, separation_radius=15.0)
    generate.build_dataset(physics, n_clips=CLIPS, n_slots=4, n_live=4,
                           n_ticks=20, base_seed=20260927, out_dir=out,
                           allow_dirty=True)
    return out


def test_the_geometry_uses_the_datasets_bounds_and_the_separation_radius(odd_world, monkeypatch):
    calls = recording(monkeypatch, "pair_geometry")
    train.main(flags(odd_world, "--symmetries"))
    assert calls, "pair_geometry was never called"
    for _, kwargs in calls:
        assert kwargs.get("bounds") == 90.0
        assert kwargs.get("scale") == 15.0


def test_the_geometry_comes_from_the_same_window_as_the_moves(tiny_dataset, monkeypatch):
    moves = recording(monkeypatch, "to_moves")
    geo = recording(monkeypatch, "pair_geometry")
    train.main(flags(tiny_dataset, "--augment", "--symmetries"))
    seen = [args[0] for args, _ in moves]
    for args, _ in geo:
        assert any(a.shape == args[0].shape and np.array_equal(a, args[0]) for a in seen), (
            "pair_geometry was handed a window to_moves never saw")
    assert len(geo) >= learn_batches(tiny_dataset) + 1


def test_the_held_out_windows_get_their_own_geometry(tiny_dataset, monkeypatch):
    calls = recording(monkeypatch, "pair_geometry")
    train.main(flags(tiny_dataset))
    _, check = train.split_by_clip(CLIPS, train_fraction=0.7)
    x_check, _ = windows.build_windows(tiny_dataset, context=2, clips=check)
    assert any(np.array_equal(args[0], x_check) for args, _ in calls)


# --- --symmetries ---------------------------------------------------------------------------

def test_without_the_flag_nothing_is_turned(tiny_dataset, monkeypatch):
    calls = recording(monkeypatch, "symmetry_augment")
    train.main(flags(tiny_dataset))
    assert calls == []


def test_with_the_flag_every_learn_batch_is_turned_and_nothing_else(tiny_dataset, monkeypatch):
    calls = recording(monkeypatch, "symmetry_augment")
    train.main(flags(tiny_dataset, "--symmetries"))
    assert len(calls) == learn_batches(tiny_dataset)


def test_it_turns_raw_windows_in_world_units(tiny_dataset, monkeypatch):
    calls = recording(monkeypatch, "symmetry_augment")
    train.main(flags(tiny_dataset, "--symmetries"))
    for args, _ in calls:
        assert np.abs(args[0]).max() > 10, "these look scaled, not world positions"


def test_it_turns_in_the_datasets_world(odd_world, monkeypatch):
    calls = recording(monkeypatch, "symmetry_augment")
    train.main(flags(odd_world, "--symmetries"))
    assert calls and all(kwargs.get("bounds") == 90.0 for _, kwargs in calls)


def test_symmetries_work_for_the_old_per_bird_model_too(tiny_dataset, capsys):
    argv = flags(tiny_dataset, "--symmetries")
    argv[argv.index("rel_per_bird")] = "per_bird"
    assert train.main(argv) == 0


# --- checkpoints ----------------------------------------------------------------------------

def test_its_checkpoint_loads_back_as_the_same_model(tiny_dataset, tmp_path):
    train.main(flags(tiny_dataset, "--checkpoint-every", "1", "--checkpoint-dir", str(tmp_path)))
    loaded = train.load_checkpoint(tmp_path / "pass_0001.pt")
    assert isinstance(loaded["model"], flocker.RelPerBirdTransformer)
    x = torch.randn(3, 2, 8)
    pair = torch.randn(3, 4, 4, 3)
    assert loaded["model"](x, pair).shape == (3, 8)
