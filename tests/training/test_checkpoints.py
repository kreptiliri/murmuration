import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")

if not hasattr(train, "save_checkpoint"):
    pytest.skip("train.save_checkpoint does not exist yet, this suite is its "
                "specification", allow_module_level=True)


CONTEXT = 8
N_SLOTS = 12
N_FEATURES = N_SLOTS * 2


@pytest.fixture
def scalers():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 100.0, size=(64, CONTEXT, N_FEATURES))
    y = rng.normal(0.0, 0.02, size=(64, N_FEATURES))
    return train.squasher(x), train.squasher(y)


@pytest.fixture
def config():
    return {"kind": "transformer", "context": CONTEXT, "n_features": N_FEATURES,
            "d_model": 128, "n_heads": 4, "n_blocks": 2}


def a_model(config, seed=0):
    torch.manual_seed(seed)
    return train.build_model(context=config["context"],
                             n_features=config["n_features"],
                             kind=config["kind"], d_model=config["d_model"],
                             n_heads=config["n_heads"], n_blocks=config["n_blocks"])


def nudge(model):
    """Move every weight, the way one pass of training would."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.full_like(p, 0.01))


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("checkpoints")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=4, n_live=4,
                           n_ticks=40, base_seed=20260917, out_dir=out,
                           allow_dirty=True)
    return out


# --- what a single checkpoint has to carry ------------------------------------

def test_a_loaded_model_predicts_exactly_what_the_saved_one_did(tmp_path,
                                                                config, scalers):
    """The whole contract in one line. Everything else here is a way of failing
    this without noticing."""
    xs, ys = scalers
    model = a_model(config)
    model.eval()
    batch = torch.randn(4, CONTEXT, N_FEATURES)
    with torch.no_grad():
        before = model(batch)

    train.save_checkpoint(tmp_path / "c.pt", model=model, pass_number=7,
                          config=config, x_scaler=xs, y_scaler=ys)
    loaded = train.load_checkpoint(tmp_path / "c.pt")

    with torch.no_grad():
        after = loaded["model"](batch)
    assert torch.equal(before, after), "the reloaded model is a different model"


def test_the_pass_number_survives(tmp_path, config, scalers):
    """It is the x-axis of the headline plot."""
    xs, ys = scalers
    train.save_checkpoint(tmp_path / "c.pt", model=a_model(config),
                          pass_number=137, config=config, x_scaler=xs, y_scaler=ys)

    assert train.load_checkpoint(tmp_path / "c.pt")["pass_number"] == 137


def test_both_scalers_survive_exactly(tmp_path, config, scalers):
    """A checkpoint without the scaler that produced it cannot be fed
    the data it was trained on, and other data goes in without raising."""
    xs, ys = scalers
    train.save_checkpoint(tmp_path / "c.pt", model=a_model(config), pass_number=0,
                          config=config, x_scaler=xs, y_scaler=ys)
    got = train.load_checkpoint(tmp_path / "c.pt")

    for name, saved, back in [("x", xs, got["x_scaler"]), ("y", ys, got["y_scaler"])]:
        np.testing.assert_array_equal(saved[0], back[0], err_msg=f"{name} mean")
        np.testing.assert_array_equal(saved[1], back[1], err_msg=f"{name} std")


def test_the_loaded_model_is_in_eval_mode(tmp_path, config, scalers):
    """A probe run in training mode is a probe of a different function."""
    xs, ys = scalers
    train.save_checkpoint(tmp_path / "c.pt", model=a_model(config), pass_number=0,
                          config=config, x_scaler=xs, y_scaler=ys)

    assert not train.load_checkpoint(tmp_path / "c.pt")["model"].training


def test_the_architecture_is_recorded_not_guessed(tmp_path, config, scalers):
    """Loading must not depend on the caller knowing what was saved."""
    xs, ys = scalers
    train.save_checkpoint(tmp_path / "c.pt", model=a_model(config), pass_number=0,
                          config=config, x_scaler=xs, y_scaler=ys)
    got = train.load_checkpoint(tmp_path / "c.pt")["config"]

    for key, value in config.items():
        assert got[key] == value, f"{key} came back as {got.get(key)!r}"


def test_a_linear_checkpoint_round_trips_too(tmp_path, scalers):
    """The default model has to survive this as well, or every score already
    recorded becomes unreproducible from a checkpoint."""
    xs, ys = scalers
    cfg = {"kind": "linear", "context": CONTEXT, "n_features": N_FEATURES,
           "d_model": 128, "n_heads": 4, "n_blocks": 2}
    model = a_model(cfg)
    model.eval()
    batch = torch.randn(4, CONTEXT, N_FEATURES)
    with torch.no_grad():
        before = model(batch)

    train.save_checkpoint(tmp_path / "c.pt", model=model, pass_number=3,
                          config=cfg, x_scaler=xs, y_scaler=ys)

    with torch.no_grad():
        assert torch.equal(before, train.load_checkpoint(tmp_path / "c.pt")["model"](batch))


def test_a_score_can_ride_along_and_is_optional(tmp_path, config, scalers):
    """The probe curve is read against how good the model was at that point."""
    xs, ys = scalers
    train.save_checkpoint(tmp_path / "with.pt", model=a_model(config), pass_number=1,
                          config=config, x_scaler=xs, y_scaler=ys, score=0.87)
    train.save_checkpoint(tmp_path / "without.pt", model=a_model(config), pass_number=1,
                          config=config, x_scaler=xs, y_scaler=ys)

    assert train.load_checkpoint(tmp_path / "with.pt")["score"] == pytest.approx(0.87)
    assert train.load_checkpoint(tmp_path / "without.pt")["score"] is None


# --- trap 1: the one that would waste a week ----------------------------------

def test_two_checkpoints_of_a_changing_model_are_not_the_same_model(tmp_path,
                                                                    config, scalers):
    """`state_dict()` hands back tensors that share storage with the live model.

    Save one, train on, save another, and both files can hold the *final*
    weights. Every probe in the sweep then reads the same model, the readability
    curve comes out flat, and the flat curve is reported as a finding. Nothing
    raises anywhere.
    """
    xs, ys = scalers
    model = a_model(config)
    model.eval()
    batch = torch.randn(4, CONTEXT, N_FEATURES)

    train.save_checkpoint(tmp_path / "early.pt", model=model, pass_number=0,
                          config=config, x_scaler=xs, y_scaler=ys)
    nudge(model)
    train.save_checkpoint(tmp_path / "late.pt", model=model, pass_number=1,
                          config=config, x_scaler=xs, y_scaler=ys)

    with torch.no_grad():
        early = train.load_checkpoint(tmp_path / "early.pt")["model"](batch)
        late = train.load_checkpoint(tmp_path / "late.pt")["model"](batch)

    assert not torch.equal(early, late), (
        "the two checkpoints predict identically, so the earlier one was "
        "overwritten by later training")


def test_the_early_checkpoint_still_holds_its_own_weights(tmp_path, config, scalers):
    """Not just 'they differ' but 'the early one is
    still what it was at the time'."""
    xs, ys = scalers
    model = a_model(config)
    model.eval()
    batch = torch.randn(4, CONTEXT, N_FEATURES)
    with torch.no_grad():
        at_save_time = model(batch)

    train.save_checkpoint(tmp_path / "early.pt", model=model, pass_number=0,
                          config=config, x_scaler=xs, y_scaler=ys)
    nudge(model)

    with torch.no_grad():
        assert torch.equal(at_save_time,
                           train.load_checkpoint(tmp_path / "early.pt")["model"](batch))


# --- the flags on main() -------------------------------------------------------

def run(dataset, *extra):
    return ["--data", str(dataset), "--clips", "6", "--context", "4",
            "--passes", "6", "--batch", "64", "--report-every", "0", *extra]


def test_no_checkpoints_are_written_by_default(tiny_dataset, tmp_path):
    """Checkpointing a 1000-pass sweep by accident fills the disk quietly."""
    train.main(run(tiny_dataset, "--checkpoint-dir", str(tmp_path)))

    assert list(tmp_path.glob("*.pt")) == []


def test_the_interval_is_respected(tiny_dataset, tmp_path):
    """6 passes every 2 is 3 checkpoints, whatever the off-by-one convention."""
    train.main(run(tiny_dataset, "--checkpoint-every", "2",
                   "--checkpoint-dir", str(tmp_path)))

    written = sorted(tmp_path.glob("*.pt"))
    assert 3 <= len(written) <= 4, f"6 passes every 2 gave {len(written)} files"


def test_checkpoints_do_not_overwrite_each_other(tiny_dataset, tmp_path):
    """One filename for the whole run leaves only the last model, and a plot
    with one point."""
    train.main(run(tiny_dataset, "--checkpoint-every", "1",
                   "--checkpoint-dir", str(tmp_path)))

    assert len(list(tmp_path.glob("*.pt"))) >= 6


def test_the_files_sort_chronologically(tiny_dataset, tmp_path):
    """A probe sweep globs the directory. `pass_9` before `pass_10` puts the
    x-axis out of order and the plot still looks plausible."""
    train.main(run(tiny_dataset, "--checkpoint-every", "1",
                   "--checkpoint-dir", str(tmp_path)))

    names = sorted(p.name for p in tmp_path.glob("*.pt"))
    passes = [train.load_checkpoint(tmp_path / n)["pass_number"] for n in names]
    assert passes == sorted(passes), (
        f"lexical order gives pass numbers {passes}; zero-pad the filenames")


def test_a_checkpoint_from_a_real_run_carries_that_runs_scalers(tiny_dataset,
                                                                tmp_path):
    """End to end: the scaler in the file must be the one main() fitted
    on the training half, not a fresh or default one."""
    train.main(run(tiny_dataset, "--checkpoint-every", "3",
                   "--checkpoint-dir", str(tmp_path)))

    got = train.load_checkpoint(sorted(tmp_path.glob("*.pt"))[0])
    mean, std = got["x_scaler"]
    assert np.asarray(mean).shape == (4, 8), (
        f"x_scaler mean has shape {np.asarray(mean).shape}, expected one column "
        "per (tick, slot, axis) of a context-4 four-bird window")
    assert (np.asarray(std) > 0).all(), "a zero or negative std cannot unscale"


def test_a_transformer_run_checkpoints_a_transformer(tiny_dataset, tmp_path):
    """`--model` and `--checkpoint-every` have to agree with each other."""
    train.main(run(tiny_dataset, "--model", "transformer",
                   "--checkpoint-every", "3", "--checkpoint-dir", str(tmp_path)))

    got = train.load_checkpoint(sorted(tmp_path.glob("*.pt"))[0])
    assert got["config"]["kind"] == "transformer"
    assert isinstance(got["model"], __import__("flocker").Transformer)


def test_the_finished_model_is_checkpointed_too(tiny_dataset, tmp_path):
    """Checkpointing at the top of the pass loop stops one interval short. With
    `--checkpoint-every 2` over 6 passes the last file is pass 4, and the model
    that was actually trained, the one every score in the run describes, is not
    on disk anywhere."""
    train.main(run(tiny_dataset, "--checkpoint-every", "2",
                   "--checkpoint-dir", str(tmp_path)))

    passes = sorted(train.load_checkpoint(p)["pass_number"]
                    for p in tmp_path.glob("*.pt"))

    assert passes[-1] == 6, (
        f"the last checkpoint is pass {passes[-1]}; the finished model after 6 "
        "passes was never written")
