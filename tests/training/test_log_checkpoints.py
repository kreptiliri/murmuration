import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")

if not hasattr(train, "log_steps"):
    pytest.skip("train.log_steps does not exist yet, this suite is its "
                "specification", allow_module_level=True)


# --- log_steps: which batches get a snapshot ----------------------------------------

def test_one_per_doubling_is_plain_doubling():
    assert train.log_steps(1, 100) == [1, 2, 4, 8, 16, 32, 64]


def test_two_per_doubling_puts_one_in_between():
    assert train.log_steps(2, 32) == [1, 2, 3, 4, 6, 8, 11, 16, 23, 32]


def test_the_steps_are_whole_distinct_and_in_order():
    steps = train.log_steps(8, 40950)
    assert all(isinstance(s, int) for s in steps)
    assert steps == sorted(set(steps))


def test_it_starts_at_the_first_batch_and_never_runs_past_the_last():
    steps = train.log_steps(4, 1000)
    assert steps[0] == 1
    assert steps[-1] <= 1000


def test_every_doubling_gets_about_the_same_number():
    steps = np.array(train.log_steps(8, 2 ** 15))
    for low in [2 ** 6, 2 ** 10, 2 ** 14]:
        inside = ((steps >= low) & (steps < 2 * low)).sum()
        assert inside == 8, f"{inside} snapshots between batch {low} and {2 * low}"


def test_a_30_pass_run_gets_about_a_hundred():
    assert 90 <= len(train.log_steps(8, 30 * 1365)) <= 120


# --- the flag on main() ---------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("log_checkpoints")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=4, n_live=4,
                           n_ticks=40, base_seed=20260917, out_dir=out,
                           allow_dirty=True)
    return out


PASSES, BATCH, CONTEXT = 6, 64, 4


def run(dataset, *extra):
    return ["--data", str(dataset), "--clips", "6", "--context", str(CONTEXT),
            "--passes", str(PASSES), "--batch", str(BATCH), "--report-every", "0", *extra]


def batches_per_pass(dataset):
    learn, _ = train.split_by_clip(6, train_fraction=0.7)
    x_learn, _ = windows.build_windows(dataset, context=CONTEXT, clips=learn)
    return len(range(0, len(x_learn), BATCH))


def step_of(path):
    return int(path.stem.split("_")[1])


def predict(path):
    torch.manual_seed(0)
    batch = torch.randn(4, CONTEXT, 8)
    with torch.no_grad():
        return train.load_checkpoint(path)["model"](batch)


def test_no_step_snapshots_by_default(tiny_dataset, tmp_path):
    train.main(run(tiny_dataset, "--checkpoint-every", "1", "--checkpoint-dir", str(tmp_path)))
    assert list(tmp_path.glob("step_*.pt")) == []


def test_it_writes_the_untrained_model_the_schedule_and_the_finished_model(tiny_dataset,
                                                                           tmp_path):
    train.main(run(tiny_dataset, "--log-checkpoints", "1", "--checkpoint-dir", str(tmp_path)))

    total = PASSES * batches_per_pass(tiny_dataset)
    assert total > 16, "the tiny run is too short for this test to mean anything"
    want = sorted({0, *train.log_steps(1, total), total})
    got = sorted(step_of(p) for p in tmp_path.glob("step_*.pt"))
    assert got == want


def test_it_makes_its_own_folder(tiny_dataset, tmp_path):
    where = tmp_path / "not" / "there" / "yet"
    train.main(run(tiny_dataset, "--log-checkpoints", "1", "--checkpoint-dir", str(where)))
    assert len(list(where.glob("step_*.pt"))) > 0


def test_the_files_sort_chronologically(tiny_dataset, tmp_path):
    train.main(run(tiny_dataset, "--log-checkpoints", "2", "--checkpoint-dir", str(tmp_path)))

    names = sorted(tmp_path.glob("step_*.pt"))
    steps = [step_of(p) for p in names]
    assert steps == sorted(steps), f"lexical order gives batches {steps}; zero-pad the filenames"


def test_the_pass_number_is_how_far_through_training_it_was(tiny_dataset, tmp_path):
    train.main(run(tiny_dataset, "--log-checkpoints", "1", "--checkpoint-dir", str(tmp_path)))

    per_pass = batches_per_pass(tiny_dataset)
    for path in tmp_path.glob("step_*.pt"):
        got = train.load_checkpoint(path)["pass_number"]
        assert got == pytest.approx(step_of(path) / per_pass), path.name
    last = max(tmp_path.glob("step_*.pt"), key=step_of)
    assert train.load_checkpoint(last)["pass_number"] == pytest.approx(PASSES)


def test_the_snapshot_is_taken_after_the_batch_not_before(tiny_dataset, tmp_path):
    train.main(run(tiny_dataset, "--checkpoint-every", "1", "--log-checkpoints", "8",
                   "--checkpoint-dir", str(tmp_path)))

    assert torch.equal(predict(tmp_path / "step_00000000.pt"), predict(tmp_path / "pass_0000.pt"))
    assert not torch.equal(predict(tmp_path / "step_00000001.pt"),
                           predict(tmp_path / "step_00000000.pt"))
    per_pass = batches_per_pass(tiny_dataset)
    assert torch.equal(predict(tmp_path / f"step_{per_pass:08d}.pt"),
                       predict(tmp_path / "pass_0001.pt")), (
        "the snapshot after the last batch of pass 1 is not the model that starts pass 2")


def test_the_last_snapshot_is_the_finished_model(tiny_dataset, tmp_path):
    train.main(run(tiny_dataset, "--checkpoint-every", "1", "--log-checkpoints", "1",
                   "--checkpoint-dir", str(tmp_path)))

    last = max(tmp_path.glob("step_*.pt"), key=step_of)
    assert torch.equal(predict(last), predict(tmp_path / f"pass_{PASSES:04d}.pt"))


def test_taking_snapshots_does_not_change_the_training(tiny_dataset, tmp_path):
    plain, snapped = tmp_path / "plain", tmp_path / "snapped"
    train.main(run(tiny_dataset, "--checkpoint-every", str(PASSES), "--checkpoint-dir", str(plain)))
    train.main(run(tiny_dataset, "--checkpoint-every", str(PASSES), "--log-checkpoints", "2",
                   "--checkpoint-dir", str(snapped)))

    name = f"pass_{PASSES:04d}.pt"
    assert torch.equal(predict(plain / name), predict(snapped / name))
