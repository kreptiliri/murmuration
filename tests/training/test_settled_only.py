import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
windows = pytest.importorskip("windows")
generate = pytest.importorskip("generate")


N_TICKS = 40  # 41 states, so 38 windows per clip at context 3
CONTEXT = 3
PER_CLIP = N_TICKS + 1 - CONTEXT
# written into the dataset by hand, so every case is present: learn clips 0-3
# (never settles, settled from the start, settles at 10, settles on the final
# state) and held-out clips 4-5 (settles on the last input state, at 20)
SETTLE = [-1, 0, 10, 40, 39, 20]
# windows whose last input state is at or after settle_tick
SETTLED_LEARN = 0 + 38 + 30 + 0
SETTLED_CHECK = 1 + 20


def make_dataset(tmp_path_factory, name, settle):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp(name)
    generate.build_dataset(PHYSICS_CRUISING, n_clips=len(settle), n_slots=4,
                           n_live=4, n_ticks=N_TICKS, base_seed=20260924,
                           out_dir=out, allow_dirty=True)
    path = out / "clips.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["settle_tick"] = np.array(settle, dtype=np.int32)
    np.savez(path, **arrays)
    return out


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return make_dataset(tmp_path_factory, "settled_only", SETTLE)


def flags(dataset, *extra, model="per_bird"):
    return ["--data", str(dataset), "--clips", "6", "--context", str(CONTEXT),
            "--passes", "1", "--batch", "64", "--report-every", "0",
            "--model", model, *extra]


def settled_by_hand(clips):
    # written from the rule, not imported: window `offset` is settled when its
    # last input state offset+CONTEXT-1 is at or after the clip's settle_tick
    return np.array([SETTLE[c] >= 0 and offset + CONTEXT - 1 >= SETTLE[c]
                     for c in clips for offset in range(PER_CLIP)])


def row_counter(monkeypatch, name):
    seen = []
    real = getattr(train, name)
    monkeypatch.setattr(train, name, lambda values: seen.append(len(values)) or real(values))
    return seen


# --- off by default --------------------------------------------------------------

def test_without_the_flag_every_window_is_used(dataset, monkeypatch, capsys):
    """Every score recorded so far was trained on all windows."""
    seen = row_counter(monkeypatch, "shared_squasher")

    train.main(flags(dataset))

    assert seen == [4 * PER_CLIP] * 2
    assert f"{4 * PER_CLIP} windows" in capsys.readouterr().out


# --- what the flag keeps -----------------------------------------------------------

@pytest.mark.parametrize("model, scaler", [("per_bird", "shared_squasher"),
                                           ("linear", "squasher")])
def test_the_scalers_are_fitted_on_settled_learn_windows_only(dataset, monkeypatch,
                                                              capsys, model, scaler):
    """Fitted on every window, the target scaler describes the big turns before
    settling, squashes every settled change to almost nothing, and the loss
    stops caring about them again - the thing this flag exists to undo."""
    seen = row_counter(monkeypatch, scaler)

    train.main(flags(dataset, "--settled-only", model=model))

    assert seen == [SETTLED_LEARN] * 2, (
        f"the scalers saw {seen} rows; the settled learn windows are "
        f"{SETTLED_LEARN} (never-settling clip 0 and final-state clip 3 have none)")
    assert f"{SETTLED_LEARN} windows" in capsys.readouterr().out


@pytest.mark.parametrize("residual", ["none", "clamped"])
def test_the_saved_scalers_are_the_settled_ones(dataset, tmp_path, residual):
    """Checked through the checkpoint, the file the probes will load. With a
    baseline, it has to come off the same settled rows as the target."""
    x, y = windows.build_windows(dataset, context=CONTEXT, clips=range(4))
    keep = settled_by_hand(range(4))
    x, y = x[keep], y[keep]
    if residual == "clamped":
        import json
        physics = json.loads((dataset / "config.json").read_text())["physics"]
        y = y - train.clamped_copier_baseline(x, dt=physics["dt"],
                                              min_speed=physics["min_speed"],
                                              max_speed=physics["max_speed"])

    train.main(flags(dataset, "--settled-only", "--residual", residual,
                     "--checkpoint-every", "1", "--checkpoint-dir", str(tmp_path)))
    got = train.load_checkpoint(tmp_path / "pass_0001.pt")

    for name, saved, values in (("x", got["x_scaler"], train.to_moves(x)),
                                ("y", got["y_scaler"], y)):
        expected = train.shared_squasher(values)
        np.testing.assert_allclose(saved[0], expected[0], rtol=1e-5, atol=1e-7,
                                   err_msg=f"{name} mean is not the settled one")
        np.testing.assert_allclose(saved[1], expected[1], rtol=1e-5, atol=1e-7,
                                   err_msg=f"{name} std is not the settled one")


def test_training_and_scoring_see_only_settled_windows(dataset, monkeypatch):
    """Everything that goes through the scaler: the held-out set once for the
    input and once for the target, then every training batch twice."""
    seen = []
    real = train.apply_squash
    monkeypatch.setattr(train, "apply_squash",
                        lambda scaler, values: seen.append(len(values)) or real(scaler, values))

    train.main(flags(dataset, "--settled-only"))

    assert sum(seen) == 2 * (SETTLED_LEARN + SETTLED_CHECK), (
        f"{sum(seen)} rows were scaled; settled learn {SETTLED_LEARN} and settled "
        f"held-out {SETTLED_CHECK}, twice each")


def test_the_printed_score_is_over_settled_held_out_windows(dataset, monkeypatch):
    """Otherwise the R2 it prints cannot be set against the -0.13 the old model
    scores on settled windows."""
    seen = []
    real = train.cancel_squash
    monkeypatch.setattr(train, "cancel_squash",
                        lambda scaler, values: seen.append(len(values)) or real(scaler, values))

    train.main(flags(dataset, "--settled-only"))

    assert seen == [SETTLED_CHECK]


# --- nothing left to train or score on ------------------------------------------------

@pytest.mark.parametrize("settle, where", [([-1, -1, -1, -1, 0, 0], "learn"),
                                          ([0, 0, 0, 0, -1, -1], "held-out")])
def test_no_settled_window_is_refused(tmp_path_factory, settle, where):
    """Zero rows gives a scaler of NaNs or an R2 of NaN, printed like a result."""
    empty = make_dataset(tmp_path_factory, f"none_settled_{where}", settle)

    with pytest.raises(ValueError, match="settled"):
        train.main(flags(empty, "--settled-only"))
