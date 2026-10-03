import dataclasses
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")
ablate = pytest.importorskip("ablate")

if not hasattr(ablate, "ablation_report"):
    pytest.skip("ablate.ablation_report does not exist yet, this suite is its specification",
                allow_module_level=True)

from world import PHYSICS_CRUISING

PHYSICS = dataclasses.asdict(PHYSICS_CRUISING)
DT = PHYSICS["dt"]
RULES = ("separation", "alignment", "cohesion")
N_BIRDS, N_BLOCKS, N_CLIPS, CONTEXT = 6, 2, 10, 2


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    data = tmp_path_factory.mktemp("report_data")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=N_CLIPS, n_slots=N_BIRDS, n_live=N_BIRDS,
                           n_ticks=60, base_seed=20260930, out_dir=data, allow_dirty=True)
    ckpts = tmp_path_factory.mktemp("report_checkpoints")
    train.main(["--data", str(data), "--clips", str(N_CLIPS), "--context", str(CONTEXT), "--passes", "2",
                "--batch", "64", "--report-every", "0", "--model", "rel_per_bird",
                "--log-checkpoints", "1", "--checkpoint-dir", str(ckpts)])
    ck = train.load_checkpoint(max(ckpts.glob("step_*.pt")))
    return data, ckpts, ck


def clips_of(data, clips):
    x, y = windows.build_windows(data, context=CONTEXT, clips=clips)
    return x, y, windows.build_force_labels(data, context=CONTEXT, clips=clips)


def model_inputs(ck, x):
    moves = torch.from_numpy(train.apply_squash(ck["x_scaler"], train.to_moves(x))).float()
    pair = torch.from_numpy(train.pair_geometry(x, bounds=PHYSICS["bounds"],
                                                scale=PHYSICS["separation_radius"]))
    return moves, pair


def fake_pushes(seed, n_rows=4000):
    #three made-up pushes of very different sizes, like the real ones
    rng = np.random.default_rng(seed)
    sizes = {"separation": 0.3, "alignment": 3.0, "cohesion": 0.5}
    labels = {}
    for rule in RULES:
        labels[rule] = rng.normal(size=(n_rows, N_BIRDS, 2)) * sizes[rule]
    return labels


def push(labels, weights):
    #what a model that uses each rule by that fraction would answer, per bird
    total = 0
    for rule in RULES:
        total = total + weights[rule] * labels[rule] * DT
    return total


# --- predict: the model's answers, in real units ------------------------------------------

def test_predict_without_a_block_is_the_normal_model_in_real_units(run):
    data, _, ck = run
    x, _, _ = clips_of(data, range(7, 10))
    moves, pair = model_inputs(ck, x)
    with torch.no_grad():
        expected = train.cancel_squash(ck["y_scaler"], ck["model"](moves, pair).numpy())
    out = ablate.predict(ck, x, physics=PHYSICS)
    assert out.shape == (len(x), 2 * N_BIRDS)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)


def test_predict_with_a_block_is_the_ablated_model_in_real_units(run):
    data, _, ck = run
    x, _, _ = clips_of(data, range(7, 10))
    moves, pair = model_inputs(ck, x)
    fixed = torch.linspace(-1, 1, ck["model"].head.in_features)
    with torch.no_grad():
        ablated = ablate.ablated_forward(ck["model"], moves, pair, block=1, note=fixed)
    expected = train.cancel_squash(ck["y_scaler"], ablated.numpy())
    out = ablate.predict(ck, x, physics=PHYSICS, block=1, note=fixed)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)


def test_predict_reads_more_than_one_batch_in_order(run):
    data, _, ck = run
    x, _, _ = clips_of(data, range(N_CLIPS))
    assert len(x) > 256
    whole = ablate.predict(ck, x, physics=PHYSICS)
    for start in [0, 250, 256, 400]:
        part = ablate.predict(ck, x[start:start + 10], physics=PHYSICS)
        np.testing.assert_allclose(whole[start:start + 10], part, rtol=1e-4, atol=1e-6)


def test_predict_needs_a_block_and_a_note_together(run):
    data, _, ck = run
    x, _, _ = clips_of(data, range(7, 8))
    with pytest.raises(ValueError):
        ablate.predict(ck, x, physics=PHYSICS, block=1)
    with pytest.raises(ValueError):
        ablate.predict(ck, x, physics=PHYSICS, note=torch.zeros(ck["model"].head.in_features))


# --- score: one number for how right the answers are ------------------------------------------

def test_score_is_1_for_perfect_and_0_for_guessing_the_average():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(200, 2 * N_BIRDS))
    assert ablate.score(truth, truth) == pytest.approx(1.0)
    average = np.tile(truth.mean(axis=0), (200, 1))
    assert ablate.score(average, truth) == pytest.approx(0.0, abs=1e-12)


def test_score_is_the_one_train_py_prints(run):
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(300, 4))
    guess = truth + rng.normal(size=(300, 4)) * 0.3
    expected = 1 - ((guess - truth) ** 2).sum() / ((truth.mean(axis=0) - truth) ** 2).sum()
    assert ablate.score(guess, truth) == pytest.approx(expected)


# --- rules_explain: which bird-windows the three pushes fully describe ------------------------

def test_rules_explain_marks_the_birds_whose_change_is_the_three_pushes():
    labels = fake_pushes(2, n_rows=50)
    targets = push(labels, {"separation": 1, "alignment": 1, "cohesion": 1}).reshape(50, -1)
    #bird 3 in windows 10-19 hit a speed limit, so its change is not the pushes
    targets[10:20, 6:8] += 0.2
    keep = ablate.rules_explain(targets, labels, dt=DT)
    assert keep.shape == (50 * N_BIRDS,)
    assert keep.dtype == bool
    for w in range(50):
        for b in range(N_BIRDS):
            clamped = 10 <= w < 20 and b == 3
            assert keep[w * N_BIRDS + b] == (not clamped), f"window {w}, bird {b}"


def test_on_real_clips_most_birds_are_the_three_pushes_and_those_rebuild_the_answer(run):
    data, _, _ = run
    x, y, labels = clips_of(data, range(N_CLIPS))
    keep = ablate.rules_explain(y, labels, dt=DT)
    assert keep.mean() > 0.5
    rebuilt = push(labels, {"separation": 1, "alignment": 1, "cohesion": 1}).reshape(-1, 2)
    np.testing.assert_allclose(rebuilt[keep], y.reshape(-1, 2)[keep], atol=1e-4)


# --- push_lost: how much of each rule the switched-off model stopped doing ----------------------

def test_push_lost_reads_back_the_fraction_of_each_rule_that_went_missing():
    labels = fake_pushes(3)
    lost = push(labels, {"separation": 0.5, "alignment": 1.0, "cohesion": 0.0}).reshape(4000, -1)
    fractions = ablate.push_lost(lost, labels, dt=DT)
    assert set(fractions) == set(RULES)
    assert fractions["separation"] == pytest.approx(0.5, abs=1e-6)
    assert fractions["alignment"] == pytest.approx(1.0, abs=1e-6)
    assert fractions["cohesion"] == pytest.approx(0.0, abs=1e-6)


def test_push_lost_is_a_fraction_so_a_big_push_does_not_drown_a_small_one():
    labels = fake_pushes(4)
    lost = push(labels, {"separation": 1.0, "alignment": 0.0, "cohesion": 0.0}).reshape(4000, -1)
    bigger = dict(labels)
    bigger["alignment"] = labels["alignment"] * 10
    fractions = ablate.push_lost(lost, bigger, dt=DT)
    assert fractions["separation"] == pytest.approx(1.0, abs=1e-6)
    assert fractions["alignment"] == pytest.approx(0.0, abs=1e-6)


def test_push_lost_tells_apart_pushes_that_point_against_each_other():
    labels = fake_pushes(5)
    rng = np.random.default_rng(6)
    labels["cohesion"] = -0.6 * labels["separation"] + rng.normal(size=labels["separation"].shape) * 0.2
    lost = push(labels, {"separation": 0.0, "alignment": 0.0, "cohesion": 1.0}).reshape(4000, -1)
    fractions = ablate.push_lost(lost, labels, dt=DT)
    assert fractions["cohesion"] == pytest.approx(1.0, abs=1e-6)
    assert fractions["separation"] == pytest.approx(0.0, abs=1e-6)


def test_push_lost_uses_only_the_kept_birds():
    labels = fake_pushes(7)
    lost = push(labels, {"separation": 0.0, "alignment": 1.0, "cohesion": 0.0}).reshape(4000, -1)
    keep = np.ones(4000 * N_BIRDS, dtype=bool)
    keep[::3] = False
    wrecked = lost.reshape(-1, 2).copy()
    wrecked[~keep] = 50.0          #nonsense on the birds that are left out
    fractions = ablate.push_lost(wrecked.reshape(4000, -1), labels, dt=DT, keep=keep)
    assert fractions["alignment"] == pytest.approx(1.0, abs=1e-6)


def test_nothing_lost_is_zero_of_everything():
    labels = fake_pushes(8)
    fractions = ablate.push_lost(np.zeros((4000, 2 * N_BIRDS)), labels, dt=DT)
    for rule in RULES:
        assert fractions[rule] == pytest.approx(0.0, abs=1e-12), rule


# --- push_explained: how much of what was lost the three pushes account for ------------------------

def test_push_explained_is_1_when_the_loss_is_only_pushes():
    labels = fake_pushes(9)
    lost = push(labels, {"separation": 0.3, "alignment": 0.8, "cohesion": 0.5}).reshape(4000, -1)
    assert ablate.push_explained(lost, labels, dt=DT) == pytest.approx(1.0, abs=1e-9)


def test_push_explained_is_near_0_when_the_loss_has_nothing_to_do_with_the_pushes():
    labels = fake_pushes(10)
    lost = np.random.default_rng(11).normal(size=(4000, 2 * N_BIRDS))
    assert ablate.push_explained(lost, labels, dt=DT) < 0.01


def test_a_shift_that_is_no_push_is_not_explained():
    labels = fake_pushes(14)
    lost = np.full((4000, 2 * N_BIRDS), 0.05)
    assert ablate.push_explained(lost, labels, dt=DT) == pytest.approx(0.0, abs=0.01)


def test_push_explained_counts_only_the_kept_birds():
    labels = fake_pushes(12)
    lost = push(labels, {"separation": 0.0, "alignment": 1.0, "cohesion": 0.0}).reshape(-1, 2).copy()
    keep = np.ones(len(lost), dtype=bool)
    keep[::4] = False
    lost[~keep] = 50.0
    assert ablate.push_explained(lost.reshape(4000, -1), labels, dt=DT, keep=keep) == pytest.approx(1.0, abs=1e-9)


def test_push_explained_is_0_when_nothing_was_lost():
    labels = fake_pushes(13)
    assert ablate.push_explained(np.zeros((4000, 2 * N_BIRDS)), labels, dt=DT) == 0.0


# --- ablation_report: every block, every note -----------------------------------------------------

def report_for(run):
    data, _, ck = run
    x_learn, _, _ = clips_of(data, range(0, 7))
    x_check, y_check, labels_check = clips_of(data, range(7, 10))
    return ablate.ablation_report(ck, x_learn, x_check, y_check, labels_check, physics=PHYSICS)


def test_the_report_has_the_normal_score_and_a_row_per_block_and_note(run):
    report = report_for(run)
    assert isinstance(report["normal"], float)
    assert 0 < report["kept"] <= 1
    seen = []
    for row in report["rows"]:
        seen.append((row["block"], row["note"]))
        assert isinstance(row["score"], float)
        assert set(row["lost"]) == set(RULES)
        assert isinstance(row["explained"], float) and row["explained"] <= 1.0
    assert sorted(seen) == [(0, "average"), (0, "self"), (0, "zero"),
                            (1, "average"), (1, "self"), (1, "zero")]


def test_the_normal_score_is_the_normal_model_on_the_check_clips(run):
    data, _, ck = run
    x_check, y_check, _ = clips_of(data, range(7, 10))
    expected = ablate.score(ablate.predict(ck, x_check, physics=PHYSICS), y_check)
    assert report_for(run)["normal"] == pytest.approx(expected, rel=1e-6)


def test_the_average_note_comes_from_the_learn_clips(run):
    data, _, ck = run
    x_learn, _, _ = clips_of(data, range(0, 7))
    x_check, y_check, _ = clips_of(data, range(7, 10))
    fixed = ablate.average_listening(ck, x_learn, physics=PHYSICS, block=1)
    expected = ablate.score(ablate.predict(ck, x_check, physics=PHYSICS, block=1, note=fixed), y_check)
    for row in report_for(run)["rows"]:
        if row["block"] == 1 and row["note"] == "average":
            assert row["score"] == pytest.approx(expected, rel=1e-6)


def test_a_model_that_never_listens_loses_nothing(run):
    data, ckpts, _ = run
    ck = train.load_checkpoint(max(ckpts.glob("step_*.pt")))
    for block in ck["model"].blocks:
        torch.nn.init.zeros_(block.attention.proj.weight)
    x_learn, _, _ = clips_of(data, range(0, 7))
    x_check, y_check, labels_check = clips_of(data, range(7, 10))
    report = ablate.ablation_report(ck, x_learn, x_check, y_check, labels_check, physics=PHYSICS)
    for row in report["rows"]:
        assert row["score"] == pytest.approx(report["normal"], abs=1e-6)
        for rule in RULES:
            assert row["lost"][rule] == pytest.approx(0.0, abs=1e-6)


# --- main --------------------------------------------------------------------------------------------

def test_main_scores_the_last_snapshot_and_saves_it(run, tmp_path, capsys):
    data, ckpts, _ = run
    out = tmp_path / "ablation.json"
    result = ablate.main(["--checkpoint-dir", str(ckpts), "--data", str(data), "--clips", str(N_CLIPS),
                          "--learn-clips", "5", "--check-clips", "3", "--out", str(out)])
    saved = json.loads(out.read_text())
    assert saved["batch"] == max(int(p.stem.split("_")[1]) for p in ckpts.glob("step_*.pt"))
    assert saved["learn_clips"] == 5 and saved["check_clips"] == 3
    assert len(saved["rows"]) == 3 * N_BLOCKS
    assert saved["normal"] == pytest.approx(result["normal"])
    printed = capsys.readouterr().out
    for rule in RULES:
        assert rule in printed


def test_main_can_score_a_chosen_snapshot(run, tmp_path):
    data, ckpts, _ = run
    first = min(int(p.stem.split("_")[1]) for p in ckpts.glob("step_*.pt") if p.stem != "step_00000000")
    result = ablate.main(["--checkpoint-dir", str(ckpts), "--data", str(data), "--clips", str(N_CLIPS),
                          "--learn-clips", "5", "--check-clips", "3", "--batch", str(first)])
    assert result["batch"] == first
