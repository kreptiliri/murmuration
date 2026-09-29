import dataclasses
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")
probe = pytest.importorskip("probe")

if not hasattr(probe, "readability"):
    pytest.skip("probe.readability does not exist yet, this suite is its specification",
                allow_module_level=True)

from world import PHYSICS_CRUISING

PHYSICS = dataclasses.asdict(PHYSICS_CRUISING)
RULES = ("separation", "alignment", "cohesion")
N_BIRDS, N_BLOCKS, N_CLIPS = 6, 2, 10


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    data = tmp_path_factory.mktemp("readability_data")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=N_CLIPS, n_slots=N_BIRDS, n_live=N_BIRDS,
                           n_ticks=60, base_seed=20260930, out_dir=data, allow_dirty=True)
    ckpts = tmp_path_factory.mktemp("readability_checkpoints")
    train.main(["--data", str(data), "--clips", str(N_CLIPS), "--context", "2", "--passes", "1",
                "--batch", "64", "--report-every", "0", "--model", "rel_per_bird",
                "--log-checkpoints", "1", "--checkpoint-dir", str(ckpts)])
    return data, ckpts


def windows_and_labels(data, clips):
    x, _ = windows.build_windows(data, context=2, clips=clips)
    return x, windows.build_force_labels(data, context=2, clips=clips)


def last_snapshot(ckpts):
    return train.load_checkpoint(max(ckpts.glob("step_*.pt")))


# --- pick_clips: a sample to learn from and a sample to score on ------------------------

def test_the_two_samples_never_share_a_clip():
    learn, check = probe.pick_clips(1000, learn_clips=50, check_clips=20)
    assert not set(learn) & set(check)


def test_the_learn_sample_comes_from_the_training_clips_and_the_check_sample_from_the_held_out_ones():
    """The same 70 / 30 split as train.py. Scoring the probe on clips the model
    trained on would let memorised flights pass for understanding."""
    learn_all, check_all = train.split_by_clip(1000, train_fraction=0.7)
    learn, check = probe.pick_clips(1000, learn_clips=50, check_clips=20)
    assert set(learn) <= set(learn_all)
    assert set(check) <= set(check_all)
    assert len(learn) == 50 and len(check) == 20


def test_asking_for_more_clips_than_there_are_is_refused():
    with pytest.raises(ValueError):
        probe.pick_clips(10, learn_clips=8, check_clips=2)
    with pytest.raises(ValueError):
        probe.pick_clips(10, learn_clips=2, check_clips=5)


# --- readability: one snapshot, every rule, every draft ------------------------------------

def test_a_score_for_every_rule_and_every_draft(run):
    data, ckpts = run
    learn, check = windows_and_labels(data, range(0, 7)), windows_and_labels(data, range(7, 10))
    scores = probe.readability(last_snapshot(ckpts), *learn, *check, physics=PHYSICS, alpha=1.0)
    assert set(scores) == set(RULES)
    for rule in RULES:
        assert len(scores[rule]) == N_BLOCKS + 1, rule
        assert all(isinstance(s, float) and s <= 1.0 for s in scores[rule]), rule


def test_each_score_is_the_probe_fitted_on_learn_and_scored_on_check(run):
    """Spelled out by hand for one rule and one draft: fit on the learn clips'
    pages, score on the check clips' pages. Fitting and scoring on the same
    pages always looks better, and nothing would raise."""
    data, ckpts = run
    ck = last_snapshot(ckpts)
    (x_l, lab_l), (x_c, lab_c) = windows_and_labels(data, range(0, 7)), windows_and_labels(data, range(7, 10))
    scores = probe.readability(ck, x_l, lab_l, x_c, lab_c, physics=PHYSICS, alpha=1.0)

    pages_l = probe.read_drafts(ck, x_l, physics=PHYSICS)
    pages_c = probe.read_drafts(ck, x_c, physics=PHYSICS)
    for rule, k in [("separation", 2), ("alignment", 0), ("cohesion", 1)]:
        fitted = probe.fit_probe(pages_l[k], lab_l[rule].reshape(-1, 2), alpha=1.0)
        expected = probe.probe_r2(fitted, pages_c[k], lab_c[rule].reshape(-1, 2))
        assert scores[rule][k] == pytest.approx(expected), f"{rule}, draft {k}"


def test_each_birds_label_meets_that_birds_page(run):
    """Swap the real pushes for each bird's own move, which draft 0 holds in a
    straight line. Draft 0 then has to score about 1. If labels and pages were
    paired to the wrong birds or windows, it could not."""
    data, ckpts = run
    x_l, _ = windows_and_labels(data, range(0, 7))
    x_c, _ = windows_and_labels(data, range(7, 10))

    def own_moves(x):
        move = (x[:, 1] - x[:, 0]).reshape(len(x), N_BIRDS, 2)
        return {rule: move for rule in RULES}

    scores = probe.readability(last_snapshot(ckpts), x_l, own_moves(x_l), x_c, own_moves(x_c),
                               physics=PHYSICS, alpha=1e-6)
    assert scores["separation"][0] > 0.999


# --- main: both snapshots and the gain -------------------------------------------------------

def args(data, ckpts, *extra):
    return ["--checkpoint-dir", str(ckpts), "--data", str(data), "--clips", str(N_CLIPS),
            "--learn-clips", "5", "--check-clips", "3", *extra]


def test_main_compares_the_untrained_and_the_finished_model_by_default(run):
    data, ckpts = run
    result = probe.main(args(data, ckpts))
    batches = sorted(int(p.stem.split("_")[1]) for p in ckpts.glob("step_*.pt"))
    assert result["untrained_batch"] == batches[0] == 0
    assert result["trained_batch"] == batches[-1]


def test_the_gain_is_trained_minus_untrained(run):
    """The number that matters. What a random network already shows is free."""
    data, ckpts = run
    result = probe.main(args(data, ckpts))
    for rule in RULES:
        for k in range(N_BLOCKS + 1):
            assert result["gain"][rule][k] == pytest.approx(
                result["trained"][rule][k] - result["untrained"][rule][k]), f"{rule}, draft {k}"


def test_main_can_be_pointed_at_any_snapshot(run):
    data, ckpts = run
    batches = sorted(int(p.stem.split("_")[1]) for p in ckpts.glob("step_*.pt"))
    result = probe.main(args(data, ckpts, "--trained-batch", str(batches[1])))
    assert result["trained_batch"] == batches[1]


def test_main_writes_what_it_found_when_asked(run, tmp_path):
    data, ckpts = run
    out = tmp_path / "readability.json"
    result = probe.main(args(data, ckpts, "--out", str(out)))
    assert json.loads(out.read_text()) == json.loads(json.dumps(result))
