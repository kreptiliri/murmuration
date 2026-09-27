import numpy as np
import pytest

probe = pytest.importorskip("probe")

if not hasattr(probe, "fit_probe"):
    pytest.skip("probe.fit_probe does not exist yet, this suite is its specification",
                allow_module_level=True)

D = 16      #numbers per bird in these tests (the real model has 128)


def pages(n, seed):
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, D))


def hidden_rule(features):
    #a push that really is written on the page, in a straight line
    rng = np.random.default_rng(99)
    weights = rng.normal(0.0, 1.0, (D, 2))
    return features @ weights + np.array([5.0, -3.0])


# --- what it gives back -------------------------------------------------------------

def test_one_guess_per_row_two_numbers_each():
    x = pages(200, 0)
    fitted = probe.fit_probe(x, hidden_rule(x), alpha=1.0)
    assert probe.probe_predict(fitted, pages(30, 1)).shape == (30, 2)


def test_it_does_not_change_what_it_is_handed():
    x = pages(200, 0)
    y = hidden_rule(x)
    x0, y0 = x.copy(), y.copy()
    fitted = probe.fit_probe(x, y, alpha=1.0)
    probe.probe_r2(fitted, x, y)
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(y, y0)


# --- it reads what is there ------------------------------------------------------------------

def test_a_push_written_in_a_straight_line_is_read_back_on_new_pages():
    train_x, test_x = pages(500, 0), pages(500, 1)
    fitted = probe.fit_probe(train_x, hidden_rule(train_x), alpha=1e-3)
    assert probe.probe_r2(fitted, test_x, hidden_rule(test_x)) > 0.999


def test_the_average_push_is_learned_too_not_only_the_multipliers():
    """The pushes do not average to zero. Without a bias the probe has to fake
    the offset with the multipliers and scores worse for a reason unrelated
    to the model."""
    train_x, test_x = pages(500, 0), pages(500, 1)
    y_train = hidden_rule(train_x) + 100.0
    fitted = probe.fit_probe(train_x, y_train, alpha=1e-3)
    assert probe.probe_r2(fitted, test_x, hidden_rule(test_x) + 100.0) > 0.999


def test_a_push_that_is_not_there_scores_about_zero_on_new_pages():
    """Noise can be fitted on the training pages, never on new ones. Scoring on
    the pages it was fitted on would report this as a finding."""
    rng = np.random.default_rng(5)
    train_x, test_x = pages(300, 0), pages(300, 1)
    noise_train, noise_test = rng.normal(size=(300, 2)), rng.normal(size=(300, 2))
    fitted = probe.fit_probe(train_x, noise_train, alpha=1.0)
    on_train = probe.probe_r2(fitted, train_x, noise_train)
    on_new = probe.probe_r2(fitted, test_x, noise_test)
    assert on_new < 0.05
    assert on_train > on_new + 0.02


def test_a_box_with_the_same_number_on_every_page_does_not_break_it():
    """A draft can have a box the model never uses, the same on every page.
    Dividing by its spread is dividing by zero."""
    train_x, test_x = pages(300, 0), pages(300, 1)
    train_x[:, 3] = 7.0
    test_x[:, 3] = 7.0
    fitted = probe.fit_probe(train_x, hidden_rule(train_x), alpha=1e-3)
    guess = probe.probe_predict(fitted, test_x)
    assert np.isfinite(guess).all()


def test_the_size_of_a_box_does_not_decide_how_much_it_counts():
    """One box in the thousands and the rest near 1: ridge would squash the
    small ones harder. Every box is put on the same scale before fitting."""
    train_x, test_x = pages(500, 0), pages(500, 1)
    y_train, y_test = hidden_rule(train_x), hidden_rule(test_x)
    plain = probe.probe_r2(probe.fit_probe(train_x, y_train, alpha=1.0), test_x, y_test)
    train_x[:, 0] *= 1000.0
    test_x[:, 0] *= 1000.0
    stretched = probe.probe_r2(probe.fit_probe(train_x, y_train, alpha=1.0), test_x, y_test)
    assert stretched == pytest.approx(plain, abs=1e-6)


def test_a_bigger_penalty_means_smaller_multipliers():
    x = pages(300, 0)
    y = hidden_rule(x)
    small = probe.fit_probe(x, y, alpha=0.1)["weights"]
    big = probe.fit_probe(x, y, alpha=1000.0)["weights"]
    assert np.linalg.norm(big) < np.linalg.norm(small)


# --- the score -------------------------------------------------------------------------------

def test_the_score_is_r2_over_both_numbers_against_their_own_average():
    """1 - (the probe's squared miss) / (the squared miss of always guessing the
    average of these targets), both numbers of the push pooled, the same R2
    as train.py reports."""
    train_x, test_x = pages(200, 0), pages(100, 1)
    rng = np.random.default_rng(3)
    y_train = hidden_rule(train_x) + rng.normal(0, 3.0, (200, 2))
    y_test = hidden_rule(test_x) + rng.normal(0, 3.0, (100, 2))
    fitted = probe.fit_probe(train_x, y_train, alpha=1.0)
    guess = probe.probe_predict(fitted, test_x)
    expected = 1 - ((guess - y_test) ** 2).sum() / ((y_test - y_test.mean(axis=0)) ** 2).sum()
    assert probe.probe_r2(fitted, test_x, y_test) == pytest.approx(expected)


def test_an_enormous_penalty_just_guesses_the_average():
    """Multipliers squashed to zero leave only the bias, the training average.
    On the training pages that scores 0."""
    x = pages(300, 0)
    y = hidden_rule(x)
    fitted = probe.fit_probe(x, y, alpha=1e12)
    assert probe.probe_r2(fitted, x, y) == pytest.approx(0.0, abs=1e-6)
