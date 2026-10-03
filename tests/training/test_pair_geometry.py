import numpy as np
import pytest

train = pytest.importorskip("train")

if not hasattr(train, "pair_geometry"):
    pytest.skip("train.pair_geometry does not exist yet, this suite is its "
                "specification", allow_module_level=True)

BOUNDS = 100.0
SCALE = 20.0


def windows_from(last_tick, context=2, drift=(0.3, -0.2)):
    last = np.asarray(last_tick, dtype=np.float64)
    ticks = [last - (context - 1 - t) * np.asarray(drift) for t in range(context)]
    return np.stack(ticks)[None].reshape(1, context, -1)


def random_windows(seed, n_rows=5, context=2, n_birds=6):
    rng = np.random.default_rng(seed)
    first = rng.uniform(0, BOUNDS, (n_rows, 1, n_birds, 2))
    moves = rng.normal(0, 0.3, (n_rows, context - 1, n_birds, 2))
    ticks = np.concatenate([first, first + np.cumsum(moves, axis=1)], axis=1)
    return ticks.reshape(n_rows, context, n_birds * 2)


# --- the contract ------------------------------------------------------------------

def test_one_row_of_numbers_per_pair_of_birds():
    out = train.pair_geometry(random_windows(0, n_rows=5, n_birds=6), bounds=BOUNDS, scale=SCALE)
    assert out.shape == (5, 6, 6, 3)


def test_it_comes_back_as_float32():
    out = train.pair_geometry(random_windows(1), bounds=BOUNDS, scale=SCALE)
    assert out.dtype == np.float32


def test_the_number_of_birds_comes_from_the_windows():
    out = train.pair_geometry(random_windows(2, n_birds=4), bounds=BOUNDS, scale=SCALE)
    assert out.shape[1:] == (4, 4, 3)


def test_it_does_not_change_what_it_is_handed():
    w = random_windows(3)
    before = w.copy()
    train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)
    np.testing.assert_array_equal(w, before)


# --- what the numbers mean -------------------------------------------------------------

def test_the_offset_points_from_the_listener_to_the_speaker():
    w = windows_from([[50.0, 50.0], [53.0, 46.0]])
    out = train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)[0]
    np.testing.assert_allclose(out[0, 1, :2], np.array([3.0, -4.0]) / SCALE, rtol=1e-6)
    np.testing.assert_allclose(out[1, 0, :2], np.array([-3.0, 4.0]) / SCALE, rtol=1e-6)


def test_the_third_number_is_the_distance():
    w = windows_from([[50.0, 50.0], [53.0, 46.0]])
    out = train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)[0]
    assert out[0, 1, 2] == pytest.approx(5.0 / SCALE, rel=1e-6)
    assert out[1, 0, 2] == pytest.approx(5.0 / SCALE, rel=1e-6)


def test_everything_is_divided_by_the_scale():
    w = random_windows(4)
    a = train.pair_geometry(w, bounds=BOUNDS, scale=20.0)
    b = train.pair_geometry(w, bounds=BOUNDS, scale=10.0)
    np.testing.assert_allclose(b, 2 * a, rtol=1e-5)


def test_a_bird_sees_itself_at_zero():
    out = train.pair_geometry(random_windows(5), bounds=BOUNDS, scale=SCALE)
    n = out.shape[1]
    for i in range(n):
        np.testing.assert_array_equal(out[:, i, i], 0.0)


def test_offsets_flip_and_distances_match_between_the_two_birds_of_a_pair():
    out = train.pair_geometry(random_windows(6), bounds=BOUNDS, scale=SCALE)
    np.testing.assert_allclose(out[..., :2], -out.transpose(0, 2, 1, 3)[..., :2], atol=1e-6)
    np.testing.assert_allclose(out[..., 2], out[..., 2].transpose(0, 2, 1), atol=1e-6)


def test_distance_is_the_length_of_the_offset():
    out = train.pair_geometry(random_windows(7), bounds=BOUNDS, scale=SCALE)
    np.testing.assert_allclose(out[..., 2], np.linalg.norm(out[..., :2], axis=-1), rtol=1e-5, atol=1e-7)


def test_it_reads_the_last_tick():
    w = windows_from([[50.0, 50.0], [53.0, 46.0]], context=3,
                     drift=[[1.0, 0.0], [-0.5, 2.0]])     # the birds were elsewhere before
    assert not np.allclose(w[0, 0, 2:] - w[0, 0, :2], w[0, -1, 2:] - w[0, -1, :2]), (
        "premise: the first and last ticks must disagree about the offset")
    out = train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)[0]
    np.testing.assert_allclose(out[0, 1, :2], np.array([3.0, -4.0]) / SCALE, rtol=1e-6)


def test_each_window_gets_its_own_geometry():
    w = random_windows(8, n_rows=4)
    together = train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)
    for r in range(4):
        np.testing.assert_array_equal(together[r], train.pair_geometry(w[r:r + 1], bounds=BOUNDS,
                                                                       scale=SCALE)[0])


# --- the seam ---------------------------------------------------------------------------

def test_two_birds_either_side_of_the_seam_are_close():
    w = windows_from([[99.0, 50.0], [1.0, 50.0]])
    out = train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)[0]
    np.testing.assert_allclose(out[0, 1], np.array([2.0, 0.0, 2.0]) / SCALE, rtol=1e-6, atol=1e-7)


def test_the_seam_in_both_directions_at_once():
    w = windows_from([[98.0, 1.0], [1.0, 97.0]])
    out = train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)[0]
    np.testing.assert_allclose(out[0, 1, :2], np.array([3.0, -4.0]) / SCALE, rtol=1e-6)
    assert out[0, 1, 2] == pytest.approx(5.0 / SCALE, rel=1e-6)


def test_positions_outside_the_box_are_wrapped_too():
    w = windows_from([[104.0, 50.0], [-3.0, 50.0]])
    out = train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)[0]
    np.testing.assert_allclose(out[0, 1], np.array([-7.0, 0.0, 7.0]) / SCALE, rtol=1e-6, atol=1e-7)


def test_it_reads_the_bounds_it_is_given():
    w = windows_from([[49.0, 10.0], [1.0, 10.0]])
    small = train.pair_geometry(w, bounds=50.0, scale=SCALE)[0]
    np.testing.assert_allclose(small[0, 1, :2], np.array([2.0, 0.0]) / SCALE, rtol=1e-6, atol=1e-7)


def test_no_distance_is_longer_than_half_the_diagonal():
    out = train.pair_geometry(random_windows(9, n_rows=50, n_birds=12), bounds=BOUNDS, scale=SCALE)
    assert out[..., 2].max() * SCALE <= BOUNDS / 2 * np.sqrt(2) + 1e-4
    assert np.abs(out[..., :2]).max() * SCALE <= BOUNDS / 2 + 1e-4


# --- the symmetries it has to respect ---------------------------------------------------

def test_sliding_the_whole_flock_changes_nothing():
    w = random_windows(10)
    moved = w + np.tile([37.5, -61.25], w.shape[-1] // 2)
    np.testing.assert_allclose(train.pair_geometry(moved, bounds=BOUNDS, scale=SCALE),
                               train.pair_geometry(w, bounds=BOUNDS, scale=SCALE), atol=1e-5)


def test_turning_the_flock_turns_the_offsets_and_keeps_the_distances():
    w = random_windows(11)
    birds = w.reshape(w.shape[0], w.shape[1], -1, 2)
    turned = np.stack([-birds[..., 1], birds[..., 0]], axis=-1).reshape(w.shape)
    a = train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)
    b = train.pair_geometry(turned, bounds=BOUNDS, scale=SCALE)
    np.testing.assert_allclose(b[..., 0], -a[..., 1], atol=1e-5)
    np.testing.assert_allclose(b[..., 1], a[..., 0], atol=1e-5)
    np.testing.assert_allclose(b[..., 2], a[..., 2], atol=1e-5)


def test_relabelling_the_birds_relabels_the_rows_and_columns():
    w = random_windows(12, n_birds=5)
    order = np.array([3, 0, 4, 1, 2])
    birds = w.reshape(w.shape[0], w.shape[1], 5, 2)
    shuffled = birds[:, :, order].reshape(w.shape)
    a = train.pair_geometry(w, bounds=BOUNDS, scale=SCALE)
    b = train.pair_geometry(shuffled, bounds=BOUNDS, scale=SCALE)
    np.testing.assert_allclose(b, a[:, order][:, :, order], atol=1e-6)
