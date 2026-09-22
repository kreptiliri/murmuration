import pytest

train = pytest.importorskip("train")
generate = pytest.importorskip("generate")


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from world import PHYSICS_CRUISING

    out = tmp_path_factory.mktemp("short_context")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=4, n_live=4,
                           n_ticks=40, base_seed=20260922, out_dir=out,
                           allow_dirty=True)
    return out


def run_args(dataset, context, *extra):
    return ["--data", str(dataset), "--clips", "6", "--context", str(context),
            "--passes", "1", "--report-every", "0", *extra]


# context 2 is the largest window in which the copier cannot be expressed, so
# it is a real configuration, not an edge case

def test_a_context_2_run_finishes_and_reports_the_model(tiny_dataset, capsys):
    assert train.main(run_args(tiny_dataset, 2)) == 0
    out = capsys.readouterr().out
    assert "R2 model" in out
    assert "copier" not in out, "a copier score was printed for a window too short to have one"


def test_a_context_2_run_keeps_its_finished_checkpoint(tiny_dataset, tmp_path):
    train.main(run_args(tiny_dataset, 2, "--checkpoint-every", "1",
                        "--checkpoint-dir", str(tmp_path)))
    assert (tmp_path / "pass_0001.pt").exists()


def test_the_copier_is_still_reported_where_it_exists(tiny_dataset, capsys):
    assert train.main(run_args(tiny_dataset, 3)) == 0
    out = capsys.readouterr().out
    assert "R2 copier" in out
    assert "skill against the copier" in out
