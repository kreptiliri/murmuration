import dataclasses
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")
probe = pytest.importorskip("probe")

if not hasattr(probe, "readability_curve"):
    pytest.skip("probe.readability_curve does not exist yet, this suite is its specification",
                allow_module_level=True)

from world import PHYSICS_CRUISING

PHYSICS = dataclasses.asdict(PHYSICS_CRUISING)
RULES = ("separation", "alignment", "cohesion")
N_BIRDS, N_BLOCKS, N_CLIPS = 6, 2, 10


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    data = tmp_path_factory.mktemp("curve_data")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=N_CLIPS, n_slots=N_BIRDS, n_live=N_BIRDS,
                           n_ticks=60, base_seed=20261001, out_dir=data, allow_dirty=True)
    ckpts = tmp_path_factory.mktemp("curve_checkpoints")
    train.main(["--data", str(data), "--clips", str(N_CLIPS), "--context", "2", "--passes", "1",
                "--batch", "64", "--report-every", "0", "--model", "rel_per_bird",
                "--checkpoint-every", "1", "--log-checkpoints", "1", "--checkpoint-dir", str(ckpts)])
    x_l, _ = windows.build_windows(data, context=2, clips=range(0, 5))
    x_c, _ = windows.build_windows(data, context=2, clips=range(7, 10))
    lab_l = windows.build_force_labels(data, context=2, clips=range(0, 5))
    lab_c = windows.build_force_labels(data, context=2, clips=range(7, 10))
    curve = probe.readability_curve(ckpts, x_l, lab_l, x_c, lab_c, physics=PHYSICS, alpha=1.0)
    return data, ckpts, (x_l, lab_l, x_c, lab_c), curve


def step_files(ckpts):
    return sorted(ckpts.glob("step_*.pt"))


# --- one row per snapshot ------------------------------------------------------------

def test_one_row_per_in_pass_snapshot_in_training_order(run):
    _, ckpts, _, curve = run
    want = [int(p.stem.split("_")[1]) for p in step_files(ckpts)]
    assert [row["batch"] for row in curve] == want
    assert want[0] == 0


def test_only_the_in_pass_snapshots_not_the_pass_files(run):
    """The folder also holds pass_*.pt. Mixed in, pass 1 would sit next to
    batch 1 on the x-axis."""
    _, ckpts, _, curve = run
    assert list(ckpts.glob("pass_*.pt")), "the fixture should have both kinds"
    assert len(curve) == len(step_files(ckpts))


def test_each_row_carries_how_far_through_training_it_was(run):
    _, ckpts, _, curve = run
    for path, row in zip(step_files(ckpts), curve):
        assert row["pass"] == pytest.approx(train.load_checkpoint(path)["pass_number"]), path.name


def test_each_row_is_that_snapshots_own_readability(run):
    """Row for batch N must be batch N's scores, not the last model's or the
    first's, spelled out by hand for the first, a middle and the last one."""
    _, ckpts, (x_l, lab_l, x_c, lab_c), curve = run
    paths = step_files(ckpts)
    for i in [0, len(paths) // 2, len(paths) - 1]:
        expected = probe.readability(train.load_checkpoint(paths[i]), x_l, lab_l, x_c, lab_c,
                                     physics=PHYSICS, alpha=1.0)
        for rule in RULES:
            assert curve[i]["scores"][rule] == pytest.approx(expected[rule]), f"{paths[i].name}, {rule}"


def test_the_snapshots_do_not_all_score_the_same(run):
    _, _, _, curve = run
    first = curve[0]["scores"]["separation"][1]
    last = curve[-1]["scores"]["separation"][1]
    assert first != last


# --- main --sweep writes it down ------------------------------------------------------

def test_main_sweep_writes_the_curve_to_a_file(run, tmp_path):
    data, ckpts, _, _ = run
    out = tmp_path / "curve.json"
    probe.main(["--checkpoint-dir", str(ckpts), "--data", str(data), "--clips", str(N_CLIPS),
                "--learn-clips", "5", "--check-clips", "3", "--sweep", "--out", str(out)])
    saved = json.loads(out.read_text())
    assert [row["batch"] for row in saved["curve"]] == [int(p.stem.split("_")[1]) for p in step_files(ckpts)]
    assert saved["learn_clips"] == 5 and saved["check_clips"] == 3
    for row in saved["curve"]:
        assert set(row["scores"]) == set(RULES)
        assert all(len(row["scores"][rule]) == N_BLOCKS + 1 for rule in RULES)


def test_main_sweep_without_somewhere_to_save_it_is_refused(run):
    """Minutes of work printed to a terminal and gone."""
    data, ckpts, _, _ = run
    with pytest.raises(SystemExit):
        probe.main(["--checkpoint-dir", str(ckpts), "--data", str(data), "--clips", str(N_CLIPS),
                    "--learn-clips", "5", "--check-clips", "3", "--sweep"])
