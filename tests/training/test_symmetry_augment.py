import numpy as np
import pytest

train = pytest.importorskip("train")

if not hasattr(train, "symmetry_augment"):
    pytest.skip("train.symmetry_augment does not exist yet, this suite is its "
                "specification", allow_module_level=True)

BOUNDS = 100.0
N_BIRDS = 6


def data(seed, n_rows=64, context=3):
    """Seam-repaired windows, as build_windows makes them: first tick inside
    the box, later ticks follow each bird's path even past the edge."""
    rng = np.random.default_rng(seed)
    first = rng.uniform(0, BOUNDS, (n_rows, 1, N_BIRDS, 2))
    moves = rng.normal(0, 0.4, (n_rows, context - 1, N_BIRDS, 2))
    ticks = np.concatenate([first, first + np.cumsum(moves, axis=1)], axis=1)
    targets = rng.normal(0, 0.2, (n_rows, N_BIRDS * 2))
    return ticks.reshape(n_rows, context, N_BIRDS * 2), targets


def run(seed=0, rng_seed=1, **kw):
    x, y = data(seed, **kw)
    new_x, new_y = train.symmetry_augment(x, y, np.random.default_rng(rng_seed), bounds=BOUNDS)
    return x, y, new_x, new_y


def birds(windows):
    return windows.reshape(windows.shape[0], windows.shape[1], -1, 2)


def moves_of(windows):
    b = birds(windows)
    return b[:, 1:] - b[:, :-1]


def torus(delta, bounds=BOUNDS):
    return (delta + bounds / 2) % bounds - bounds / 2


def the_turn(x, new_x):
    """Per window, the 2x2 matrix that takes the old moves to the new ones."""
    old = moves_of(x).reshape(len(x), -1, 2)
    new = moves_of(new_x).reshape(len(x), -1, 2)
    return np.stack([np.linalg.lstsq(o, n, rcond=None)[0].T for o, n in zip(old, new)])


# --- the contract ------------------------------------------------------------------

def test_shapes_come_back_unchanged():
    x, y, new_x, new_y = run()
    assert new_x.shape == x.shape and new_y.shape == y.shape


def test_it_does_not_change_what_it_is_handed():
    x, y = data(0)
    x0, y0 = x.copy(), y.copy()
    train.symmetry_augment(x, y, np.random.default_rng(1), bounds=BOUNDS)
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(y, y0)


def test_the_same_seed_gives_the_same_batch():
    """A run has to be repeatable, so the randomness comes from the rng handed in."""
    _, _, a_x, a_y = run(rng_seed=5)
    _, _, b_x, b_y = run(rng_seed=5)
    np.testing.assert_array_equal(a_x, b_x)
    np.testing.assert_array_equal(a_y, b_y)


def test_a_different_seed_gives_a_different_batch():
    _, _, a_x, _ = run(rng_seed=5)
    _, _, b_x, _ = run(rng_seed=6)
    assert not np.allclose(a_x, b_x)


# --- it is a symmetry of the physics ---------------------------------------------------

def test_every_torus_distance_between_birds_is_kept_at_every_tick():
    """The rules only read distances and directions between birds. Keep every
    distance and the augmented window is one the simulator could have made."""
    x, _, new_x, _ = run()
    old, new = birds(x), birds(new_x)
    d_old = np.linalg.norm(torus(old[:, :, :, None] - old[:, :, None]), axis=-1)
    d_new = np.linalg.norm(torus(new[:, :, :, None] - new[:, :, None]), axis=-1)
    np.testing.assert_allclose(d_new, d_old, atol=1e-6)


def test_each_window_is_turned_by_one_of_the_eight_square_symmetries():
    """A quarter turn, a half turn, a flip: the eight ways a square maps onto
    itself. Any other angle would not map the square torus onto itself."""
    x, _, new_x, _ = run()
    turns = np.round(the_turn(x, new_x), 6)
    for t in turns:
        assert set(np.abs(t).ravel()) == {0.0, 1.0}, f"not a square symmetry:\n{t}"
        assert abs(abs(np.linalg.det(t)) - 1) < 1e-9


def test_the_targets_turn_with_the_window():
    """The bug this suite exists for. Turn the flock but not the answers, and
    every augmented example teaches the model a wrong rule. The loss still
    falls, and nothing raises."""
    x, y, new_x, new_y = run()
    turns = the_turn(x, new_x)
    for r in range(len(x)):
        np.testing.assert_allclose(new_y[r].reshape(-1, 2), y[r].reshape(-1, 2) @ turns[r].T,
                                   atol=1e-6, err_msg=f"window {r}")


def test_all_eight_symmetries_turn_up():
    """Only rotating, or only flipping, leaves the model half the symmetry."""
    x, _, new_x, _ = run(n_rows=400)
    seen = {tuple(np.round(t, 6).ravel()) for t in the_turn(x, new_x)}
    assert len(seen) == 8, f"{len(seen)} distinct symmetries in 400 windows"


def test_each_window_is_slid_by_its_own_random_amount():
    """Undo the turn and what is left is a slide around the torus: the same for
    every bird in a window, different from window to window. The slide is
    compared with the turn undone, since one slide seen through eight
    different turns already looks like eight different slides."""
    x, _, new_x, _ = run(n_rows=200)
    turns = the_turn(x, new_x)
    old_first = birds(x)[:, 0]
    new_unturned = birds(new_x)[:, 0] @ turns          # each bird's first position, turned back
    slides = torus(new_unturned - old_first)
    same_in_window = torus(slides - slides[:, :1])
    np.testing.assert_allclose(same_in_window, 0.0, atol=1e-6)
    per_window = slides[:, 0]
    assert per_window.std(axis=0).min() > BOUNDS / 5, "the slides barely vary"


def test_the_birds_keep_their_slots():
    """Bird 3 in the window is still bird 3 in the targets. Relabelling is a
    separate augmentation, and doing it here without the targets breaks the pairing."""
    x, y, new_x, new_y = run()
    turns = the_turn(x, new_x)
    for r in range(5):
        move = moves_of(x)[r, -1]
        np.testing.assert_allclose(moves_of(new_x)[r, -1], move @ turns[r].T, atol=1e-6)


# --- the windows stay the kind build_windows makes -----------------------------------------

def test_the_first_tick_is_back_inside_the_box():
    _, _, new_x, _ = run()
    first = birds(new_x)[:, 0]
    assert (first >= 0).all() and (first < BOUNDS).all()


def test_each_birds_path_stays_unbroken():
    """Wrapping every tick into the box would make a bird crossing the seam
    jump 100 units in one tick. Only the first tick is wrapped; the rest follow."""
    x, _, new_x, _ = run()
    old = np.linalg.norm(moves_of(x), axis=-1)
    new = np.linalg.norm(moves_of(new_x), axis=-1)
    np.testing.assert_allclose(new, old, atol=1e-6)


def test_it_works_for_context_two():
    x, y, new_x, new_y = run(context=2)
    assert new_x.shape == x.shape
    np.testing.assert_allclose(np.linalg.norm(new_y.reshape(len(y), -1, 2), axis=-1),
                               np.linalg.norm(y.reshape(len(y), -1, 2), axis=-1), atol=1e-6)


def test_it_reads_the_bounds_it_is_given():
    x, y = data(0)
    x = x * 0.5
    new_x, _ = train.symmetry_augment(x, y, np.random.default_rng(1), bounds=50.0)
    first = birds(new_x)[:, 0]
    assert (first >= 0).all() and (first < 50.0).all()
    assert first.max() > 40.0, "slides only reach part of a 50-wide world"
