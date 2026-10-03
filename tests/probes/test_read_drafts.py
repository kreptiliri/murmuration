import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")
probe = pytest.importorskip("probe")

if not hasattr(probe, "read_drafts"):
    pytest.skip("probe.read_drafts does not exist yet, this suite is its specification",
                allow_module_level=True)

from world import PHYSICS_CRUISING

PHYSICS = dataclasses.asdict(PHYSICS_CRUISING)
N_BIRDS, D_MODEL, N_BLOCKS = 6, 128, 2


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    data = tmp_path_factory.mktemp("drafts_data")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=10, n_slots=N_BIRDS, n_live=N_BIRDS,
                           n_ticks=60, base_seed=20260929, out_dir=data, allow_dirty=True)
    ckpts = tmp_path_factory.mktemp("drafts_checkpoints")
    train.main(["--data", str(data), "--clips", "10", "--context", "2", "--passes", "1",
                "--batch", "64", "--report-every", "0", "--model", "rel_per_bird",
                "--checkpoint-every", "1", "--checkpoint-dir", str(ckpts)])
    ck = train.load_checkpoint(ckpts / "pass_0001.pt")
    x, _ = windows.build_windows(data, context=2, clips=range(10))
    return data, ckpts, ck, x


def by_hand(ck, x):
    #what the model's drafts are, worked out directly for a few windows
    moves = torch.from_numpy(train.apply_squash(ck["x_scaler"], train.to_moves(x))).float()
    pair = torch.from_numpy(train.pair_geometry(x, bounds=PHYSICS["bounds"],
                                                scale=PHYSICS["separation_radius"]))
    with torch.no_grad():
        return [d.numpy() for d in ck["model"].hidden_states(moves, pair)]


# --- what comes back ---------------------------------------------------------------

def test_one_set_of_pages_per_draft(run):
    _, _, ck, x = run
    pages = probe.read_drafts(ck, x[:50], physics=PHYSICS)
    assert len(pages) == N_BLOCKS + 1


def test_one_page_per_bird_per_window_128_boxes_each(run):
    _, _, ck, x = run
    for k, page_set in enumerate(probe.read_drafts(ck, x[:50], physics=PHYSICS)):
        assert page_set.shape == (50 * N_BIRDS, D_MODEL), f"draft {k}"


def test_the_pages_are_numpy_so_the_probe_can_use_them(run):
    _, _, ck, x = run
    for page_set in probe.read_drafts(ck, x[:20], physics=PHYSICS):
        assert isinstance(page_set, np.ndarray)


def test_it_does_not_change_the_windows(run):
    _, _, ck, x = run
    part = x[:30].copy()
    probe.read_drafts(ck, part, physics=PHYSICS)
    np.testing.assert_array_equal(part, x[:30])


# --- the pages are the model's real drafts ----------------------------------------------

def test_each_page_is_that_birds_draft_in_that_window(run):
    _, _, ck, x = run
    part = x
    pages = probe.read_drafts(ck, part, physics=PHYSICS)
    for w in [0, 1, 255, 256, 300, len(part) - 1]:
        expected = by_hand(ck, part[w:w + 1])
        for b in [0, 3, N_BIRDS - 1]:
            for k in range(N_BLOCKS + 1):
                np.testing.assert_allclose(pages[k][w * N_BIRDS + b], expected[k][0, b],
                                           atol=1e-5, err_msg=f"window {w}, bird {b}, draft {k}")


def test_bird_bs_own_move_can_be_read_off_bird_bs_page(run):
    _, _, ck, x = run
    pages = probe.read_drafts(ck, x, physics=PHYSICS)[0]
    own_moves = (x[:, 1] - x[:, 0]).reshape(len(x) * N_BIRDS, 2)
    half = len(pages) // 2
    fitted = probe.fit_probe(pages[:half], own_moves[:half], alpha=1e-6)
    assert probe.probe_r2(fitted, pages[half:], own_moves[half:]) > 0.999


def test_the_pages_line_up_with_the_force_labels(run):
    data, _, ck, x = run
    labels = windows.build_force_labels(data, context=2, clips=range(10))
    pages = probe.read_drafts(ck, x, physics=PHYSICS)
    for k in range(N_BLOCKS + 1):
        assert len(pages[k]) == len(labels["separation"].reshape(-1, 2))


def test_it_refuses_a_model_without_drafts(run, tmp_path):
    data, _, _, x = run
    train.main(["--data", str(data), "--clips", "10", "--context", "2", "--passes", "1",
                "--batch", "64", "--report-every", "0", "--model", "per_bird",
                "--checkpoint-every", "1", "--checkpoint-dir", str(tmp_path)])
    per_bird = train.load_checkpoint(tmp_path / "pass_0001.pt")
    with pytest.raises(ValueError):
        probe.read_drafts(per_bird, x[:10], physics=PHYSICS)
