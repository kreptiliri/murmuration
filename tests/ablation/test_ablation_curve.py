import dataclasses
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")
ablate = pytest.importorskip("ablate")

if not hasattr(ablate, "ablation_curve"):
    pytest.skip("ablate.ablation_curve does not exist yet, this suite is its specification",
                allow_module_level=True)

from world import PHYSICS_CRUISING

PHYSICS = dataclasses.asdict(PHYSICS_CRUISING)
DT = PHYSICS["dt"]
RULES = ("separation", "alignment", "cohesion")
N_BIRDS, N_BLOCKS, N_CLIPS, CONTEXT = 6, 2, 10, 2


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    data = tmp_path_factory.mktemp("curve_data")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=N_CLIPS, n_slots=N_BIRDS, n_live=N_BIRDS,
                           n_ticks=60, base_seed=20260930, out_dir=data, allow_dirty=True)
    ckpts = tmp_path_factory.mktemp("curve_checkpoints")
    train.main(["--data", str(data), "--clips", str(N_CLIPS), "--context", str(CONTEXT), "--passes", "2",
                "--batch", "64", "--report-every", "0", "--model", "rel_per_bird",
                "--log-checkpoints", "1", "--checkpoint-dir", str(ckpts)])
    x, y = windows.build_windows(data, context=CONTEXT, clips=range(7, 10))
    labels = windows.build_force_labels(data, context=CONTEXT, clips=range(7, 10))
    return data, ckpts, x, y, labels


# --- self_ablation: one snapshot ------------------------------------------------------------

def test_one_snapshot_has_the_normal_score_what_the_model_does_and_a_row_per_block(run):
    _, ckpts, x, y, labels = run
    ck = train.load_checkpoint(max(ckpts.glob("step_*.pt")))
    row = ablate.self_ablation(ck, x, y, labels, physics=PHYSICS)
    assert isinstance(row["normal"], float)
    assert set(row["done"]) == set(RULES)
    assert [b["block"] for b in row["blocks"]] == list(range(N_BLOCKS))
    for b in row["blocks"]:
        assert set(b["lost"]) == set(RULES)
        assert isinstance(b["score"], float) and isinstance(b["explained"], float)


def test_done_is_how_much_of_each_push_the_normal_answer_holds(run):
    """The same fit as push_lost, on the model's whole answer instead of on what
    was lost: 1 = the model does all of that rule, 0 = none of it."""
    _, ckpts, x, y, labels = run
    ck = train.load_checkpoint(max(ckpts.glob("step_*.pt")))
    keep = ablate.rules_explain(y, labels, dt=DT)
    normal = ablate.predict(ck, x, physics=PHYSICS)
    expected = ablate.push_lost(normal, labels, dt=DT, keep=keep)
    row = ablate.self_ablation(ck, x, y, labels, physics=PHYSICS)
    for rule in RULES:
        assert row["done"][rule] == pytest.approx(expected[rule], rel=1e-6), rule


def test_each_block_row_is_that_block_hearing_only_itself(run):
    _, ckpts, x, y, labels = run
    ck = train.load_checkpoint(max(ckpts.glob("step_*.pt")))
    keep = ablate.rules_explain(y, labels, dt=DT)
    normal = ablate.predict(ck, x, physics=PHYSICS)
    row = ablate.self_ablation(ck, x, y, labels, physics=PHYSICS)
    for k in range(N_BLOCKS):
        ablated = ablate.predict(ck, x, physics=PHYSICS, block=k, note="self")
        b = row["blocks"][k]
        assert b["score"] == pytest.approx(ablate.score(ablated, y), rel=1e-6)
        expected = ablate.push_lost(normal - ablated, labels, dt=DT, keep=keep)
        for rule in RULES:
            assert b["lost"][rule] == pytest.approx(expected[rule], rel=1e-6, abs=1e-9), f"block {k}, {rule}"
        assert b["explained"] == pytest.approx(ablate.push_explained(normal - ablated, labels, dt=DT, keep=keep))


# --- ablation_curve: every snapshot ----------------------------------------------------------------

def test_one_row_per_snapshot_in_training_order(run):
    _, ckpts, x, y, labels = run
    curve = ablate.ablation_curve(ckpts, x, y, labels, physics=PHYSICS)
    batches = sorted(int(p.stem.split("_")[1]) for p in ckpts.glob("step_*.pt"))
    assert [row["batch"] for row in curve] == batches
    assert curve[0]["batch"] == 0
    for row in curve:
        assert isinstance(row["pass"], float)


def test_each_row_is_that_snapshot(run):
    _, ckpts, x, y, labels = run
    curve = ablate.ablation_curve(ckpts, x, y, labels, physics=PHYSICS)
    for row in [curve[0], curve[len(curve) // 2], curve[-1]]:
        ck = train.load_checkpoint(ckpts / f"step_{row['batch']:08d}.pt")
        alone = ablate.self_ablation(ck, x, y, labels, physics=PHYSICS)
        assert row["normal"] == pytest.approx(alone["normal"], rel=1e-6)
        for rule in RULES:
            assert row["blocks"][1]["lost"][rule] == pytest.approx(alone["blocks"][1]["lost"][rule],
                                                                   rel=1e-6, abs=1e-9)


# --- main --sweep --------------------------------------------------------------------------------------

def test_sweep_saves_the_curve(run, tmp_path):
    data, ckpts, _, _, _ = run
    out = tmp_path / "ablation_curve.json"
    ablate.main(["--checkpoint-dir", str(ckpts), "--data", str(data), "--clips", str(N_CLIPS),
                 "--learn-clips", "5", "--check-clips", "3", "--sweep", "--out", str(out)])
    saved = json.loads(out.read_text())
    assert saved["check_clips"] == 3
    assert len(saved["curve"]) == len(list(ckpts.glob("step_*.pt")))
    assert saved["curve"][-1]["batch"] == max(int(p.stem.split("_")[1]) for p in ckpts.glob("step_*.pt"))


def test_sweep_without_out_is_refused(run):
    data, ckpts, _, _, _ = run
    with pytest.raises(SystemExit):
        ablate.main(["--checkpoint-dir", str(ckpts), "--data", str(data), "--clips", str(N_CLIPS),
                     "--learn-clips", "5", "--check-clips", "3", "--sweep"])
