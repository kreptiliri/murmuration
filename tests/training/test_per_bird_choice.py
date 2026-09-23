import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
flocker = pytest.importorskip("flocker")
generate = pytest.importorskip("generate")


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("per_bird")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=4, n_live=4,
                           n_ticks=40, base_seed=20260923, out_dir=out,
                           allow_dirty=True)
    return out


def flags(dataset, *extra):
    return ["--data", str(dataset), "--clips", "6", "--context", "2",
            "--passes", "1", "--batch", "64", "--report-every", "0",
            "--model", "per_bird", *extra]


def test_per_bird_builds_the_one_in_flocker():
    model = train.build_model(context=2, n_features=24, kind="per_bird")

    assert isinstance(model, flocker.PerBirdTransformer)


def test_the_flag_reaches_build_model(tiny_dataset, monkeypatch):
    """A dropped flag is invisible: the run finishes and prints a score, and the
    linear model trained under the per-bird name."""
    asked = []
    real = train.build_model
    monkeypatch.setattr(train, "build_model",
                        lambda **kw: asked.append(kw.get("kind")) or real(**kw))

    train.main(flags(tiny_dataset))

    assert asked == ["per_bird"], f"main() asked build_model for {asked}"


def test_main_trains_it_end_to_end_at_context_2(tiny_dataset, capsys):
    """The configuration it exists for, from one command."""
    assert train.main(flags(tiny_dataset)) == 0
    assert "R2 model" in capsys.readouterr().out


def test_its_checkpoint_loads_back_as_the_same_model(tiny_dataset, tmp_path):
    """load_checkpoint rebuilds from the saved config. If that config cannot
    describe a per-bird model, every checkpoint of the real run is unreadable,
    and the probes in step 5 read checkpoints, not live models."""
    train.main(flags(tiny_dataset, "--checkpoint-every", "1",
                     "--checkpoint-dir", str(tmp_path)))

    loaded = train.load_checkpoint(tmp_path / "pass_0000.pt")

    assert isinstance(loaded["model"], flocker.PerBirdTransformer)
    window = torch.randn(3, 2, 8)
    assert loaded["model"](window).shape == (3, 8)
