import numpy as np
import pytest

train = pytest.importorskip("train")

if not hasattr(train, "permute_slots"):
    pytest.skip("train.permute_slots does not exist yet, this suite is its "
                "specification", allow_module_level=True)


CONTEXT = 8
N_SLOTS = 12
N_FEATURES = N_SLOTS * 2
N = 40


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    windows = rng.uniform(0.0, 100.0, size=(N, CONTEXT, N_FEATURES))
    targets = rng.normal(0.0, 0.02, size=(N, N_FEATURES))
    return windows, targets


def as_birds(flat):
    return flat.reshape(*flat.shape[:-1], N_SLOTS, 2)


# --- shape and dtype ----------------------------------------------------------

def test_the_shapes_come_back_unchanged(batch):
    windows, targets = batch

    out_w, out_t = train.permute_slots(windows, targets,
                                       np.random.default_rng(1))

    assert out_w.shape == windows.shape
    assert out_t.shape == targets.shape


def test_the_inputs_are_not_modified_in_place(batch):
    windows, targets = batch
    before_w, before_t = windows.copy(), targets.copy()

    train.permute_slots(windows, targets, np.random.default_rng(1))

    np.testing.assert_array_equal(windows, before_w)
    np.testing.assert_array_equal(targets, before_t)


# --- trap 3: it is a permutation of birds, not of numbers ---------------------

def test_every_bird_survives_somewhere(batch):
    windows, _ = batch
    out_w, _ = train.permute_slots(windows, _dummy(windows),
                                   np.random.default_rng(2))

    for row in range(N):
        for tick in range(CONTEXT):
            before = sorted(map(tuple, as_birds(windows)[row, tick]))
            after = sorted(map(tuple, as_birds(out_w)[row, tick]))
            assert before == after, (
                f"row {row} tick {tick}: the set of birds changed, so this is "
                "shuffling 24 numbers rather than 12 birds")


def test_the_x_and_y_of_a_bird_stay_together(batch):
    windows, _ = batch
    out_w, _ = train.permute_slots(windows, _dummy(windows),
                                   np.random.default_rng(3))

    birds_in = as_birds(windows)[0, 0]
    birds_out = as_birds(out_w)[0, 0]
    for bird in birds_out:
        assert any(np.array_equal(bird, original) for original in birds_in), (
            f"{bird} is not one of the twelve (x, y) pairs that went in")


def test_every_bird_to_bird_distance_is_preserved(batch):
    windows, _ = batch
    out_w, _ = train.permute_slots(windows, _dummy(windows),
                                   np.random.default_rng(4))

    for row in range(N):
        before = np.sort(_pairwise(as_birds(windows)[row, 0]))
        after = np.sort(_pairwise(as_birds(out_w)[row, 0]))
        np.testing.assert_allclose(before, after, atol=1e-9)


# --- traps 1 and 2: the target follows, and follows the SAME draw -------------

def test_the_target_gets_the_same_permutation_as_the_window(batch):
    windows, targets = batch
    out_w, out_t = train.permute_slots(windows, targets,
                                       np.random.default_rng(5))

    for row in range(N):
        order = _recover_permutation(as_birds(windows)[row, 0],
                                     as_birds(out_w)[row, 0])
        np.testing.assert_allclose(as_birds(out_t)[row],
                                   as_birds(targets)[row][order], atol=1e-12,
                                   err_msg=f"row {row}: the target did not "
                                           "follow the window's permutation")


def test_the_same_permutation_is_used_at_every_tick_of_a_window(batch):
    windows, targets = batch
    out_w, _ = train.permute_slots(windows, targets, np.random.default_rng(6))

    for row in range(N):
        first = _recover_permutation(as_birds(windows)[row, 0],
                                     as_birds(out_w)[row, 0])
        for tick in range(1, CONTEXT):
            here = _recover_permutation(as_birds(windows)[row, tick],
                                        as_birds(out_w)[row, tick])
            np.testing.assert_array_equal(here, first,
                                          err_msg=f"row {row} tick {tick} used a "
                                                  "different permutation")


# --- it actually permutes, and differently per row ----------------------------

def test_it_does_something(batch):
    windows, targets = batch

    out_w, _ = train.permute_slots(windows, targets, np.random.default_rng(7))

    assert not np.array_equal(out_w, windows), (
        "nothing moved. Every other test in this file passes on the identity")


def test_rows_get_independent_draws(batch):
    windows, targets = batch

    out_w, _ = train.permute_slots(windows, targets, np.random.default_rng(8))

    orders = {tuple(_recover_permutation(as_birds(windows)[row, 0],
                                         as_birds(out_w)[row, 0]))
              for row in range(N)}
    assert len(orders) > 1, (
        f"all {N} rows got the same permutation")


# --- repeatability ------------------------------------------------------------

def test_two_calls_with_the_same_seed_agree(batch):
    windows, targets = batch

    first = train.permute_slots(windows, targets, np.random.default_rng(9))
    second = train.permute_slots(windows, targets, np.random.default_rng(9))

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_two_calls_with_different_seeds_differ(batch):
    windows, targets = batch

    first = train.permute_slots(windows, targets, np.random.default_rng(10))
    second = train.permute_slots(windows, targets, np.random.default_rng(11))

    assert not np.array_equal(first[0], second[0])


def test_consecutive_calls_on_one_generator_differ(batch):
    windows, targets = batch
    rng = np.random.default_rng(12)

    first = train.permute_slots(windows, targets, rng)
    second = train.permute_slots(windows, targets, rng)

    assert not np.array_equal(first[0], second[0])


# --- it composes with what comes after it ------------------------------------

def test_the_copier_baseline_permutes_with_the_window(batch):
    windows, targets = batch
    out_w, _ = train.permute_slots(windows, targets, np.random.default_rng(13))

    before = train.copier_baseline(windows, dt=0.1)
    after = train.copier_baseline(out_w, dt=0.1)

    for row in range(N):
        order = _recover_permutation(as_birds(windows)[row, 0],
                                     as_birds(out_w)[row, 0])
        np.testing.assert_allclose(as_birds(after)[row],
                                   as_birds(before)[row][order], atol=1e-9)


def test_to_moves_commutes_with_it(batch):
    windows, targets = batch
    out_w, _ = train.permute_slots(windows, targets, np.random.default_rng(14))

    moves_after = train.to_moves(out_w)
    moves_before = train.to_moves(windows)

    for row in range(N):
        order = _recover_permutation(as_birds(windows)[row, 0],
                                     as_birds(out_w)[row, 0])
        np.testing.assert_allclose(as_birds(moves_after)[row],
                                   as_birds(moves_before)[row][:, order],
                                   atol=1e-9)


# --- helpers ------------------------------------------------------------------

def _dummy(windows):
    return np.zeros((len(windows), N_FEATURES))


def _pairwise(birds):
    offsets = birds[:, None, :] - birds[None, :, :]
    return np.linalg.norm(offsets, axis=-1)[np.triu_indices(N_SLOTS, k=1)]


def _recover_permutation(before, after):
    order = np.empty(N_SLOTS, dtype=int)
    for i, bird in enumerate(after):
        matches = np.flatnonzero((before == bird).all(axis=1))
        assert len(matches) == 1, (
            f"bird {i} of the permuted window matched {len(matches)} birds of "
            "the original; it is not a permutation of them")
        order[i] = matches[0]
    return order
