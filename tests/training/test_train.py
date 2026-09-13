"""The contract for `train.py` — written before it exists.

Step 4 is a throwaway run whose only products are a wall-clock number and a
first score next to the copier. The code does not need to be good. It does need
to be honest, and these are the four ways a training script lies:

1. **It leaks.** A clip in both train and test makes any score meaningless.
2. **It reports in the wrong units.** The target gets scaled to train on, and if
   the scaling is not undone the number printed is not comparable to anything.
3. **It is wired wrong and trains anyway.** A transposed axis or a mis-sliced
   tick produces a loss that falls smoothly while the model learns nothing
   useful. Nothing raises.
4. **It is graded against nothing.** A score with no baseline beside it cannot
   be read - see `NOTES.md`, where a do-nothing predictor scored a perfect
   result against the original target.

Five functions are pinned here. Say so and I will change any of them:

    split_by_clip(n_clips, *, train_fraction)   -> (range, range)
    squasher(values)                          -> dict
    apply_squash(scaler, values)                -> array, centred and scaled
    cancel_squash(scaler, values)               -> array, back in real units
    build_model(*, context, n_features, ...)    -> torch.nn.Module
    skill(prediction, truth, baseline)          -> float

`build_model` returns a module mapping `(batch, context, n_features)` to
`(batch, n_features)` - a whole window in, one tick of prediction out.

`skill` is the honest score: 0 means the prediction is no better than the
baseline handed to it, 1 means perfect, negative means worse than the baseline.
It is deliberately *not* R², because R² against the mean flatters everything on
this target.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip(
    "train",
    reason="train.py does not exist yet — this suite is its specification")


CONTEXT = 8
N_SLOTS = 12
N_FEATURES = N_SLOTS * 2


# --- trap 1: the split ---------------------------------------------------------

def test_the_splits_are_disjoint():
    """One clip in both halves and the test score is partly a memory test."""
    train_clips, test_clips = train.split_by_clip(200, train_fraction=0.7)

    assert not (set(train_clips) & set(test_clips))


def test_the_splits_cover_every_clip():
    """Silently dropping clips wastes data and hides itself."""
    train_clips, test_clips = train.split_by_clip(200, train_fraction=0.7)

    assert sorted(set(train_clips) | set(test_clips)) == list(range(200))


def test_the_train_fraction_is_respected():
    train_clips, _ = train.split_by_clip(200, train_fraction=0.7)

    assert len(list(train_clips)) == 140


# --- trap 2: the units ---------------------------------------------------------

def test_scaling_is_invertible():
    """A score printed in scaled units is not comparable to anything in NOTES."""
    values = np.random.default_rng(0).normal(3.0, 7.0, size=(500, N_FEATURES))
    scaler = train.squasher(values)

    restored = train.cancel_squash(scaler, train.apply_squash(scaler, values))

    np.testing.assert_allclose(restored, values, atol=1e-5)


def test_scaled_values_are_actually_scaled():
    """Guards against fit/apply being silently the identity."""
    values = np.random.default_rng(1).normal(3.0, 7.0, size=(500, N_FEATURES))
    scaler = train.squasher(values)

    scaled = train.apply_squash(scaler, values)

    assert abs(scaled.mean()) < 0.05, "not centred"
    assert abs(scaled.std() - 1.0) < 0.05, "not unit scale"


def test_the_scaler_uses_only_what_it_was_fitted_on():
    """The test set must not influence the numbers used to scale it.

    Fitting on train and applying to test is correct. Fitting on everything is
    a leak that is invisible in the loss curve, so it is pinned here: applying a
    train-fitted scaler to differently-distributed data must NOT come out
    centred.
    """
    rng = np.random.default_rng(2)
    reference = rng.normal(0.0, 1.0, size=(500, N_FEATURES))
    other = rng.normal(50.0, 1.0, size=(500, N_FEATURES))

    scaler = train.squasher(reference)
    scaled_other = train.apply_squash(scaler, other)

    assert abs(scaled_other.mean()) > 1.0, "the scaler was refitted on the new data"


def test_a_zero_variance_column_does_not_divide_by_zero():
    """A dead slot, or a bird that never moves, is a constant column."""
    values = np.ones((50, N_FEATURES))
    scaler = train.squasher(values)

    scaled = train.apply_squash(scaler, values)

    assert np.isfinite(scaled).all()


# --- trap 3: the wiring --------------------------------------------------------

def test_the_model_maps_a_window_to_one_tick_of_prediction():
    """A whole window in, one tick of velocity change out."""
    model = train.build_model(context=CONTEXT, n_features=N_FEATURES)
    batch = torch.zeros(4, CONTEXT, N_FEATURES)

    output = model(batch)

    assert tuple(output.shape) == (4, N_FEATURES)


def test_the_model_can_memorise_eight_windows():
    """The wiring check, and the only test here that actually trains.

    A correctly wired model can overfit a handful of examples into the ground.
    If it cannot, an axis is transposed or the target is misaligned, and no
    amount of real training will fix it - it will simply converge to something
    plausible and wrong. Run this before any long run, not after.
    """
    torch.manual_seed(0)
    inputs = torch.randn(8, CONTEXT, N_FEATURES)
    targets = torch.randn(8, N_FEATURES)

    model = train.build_model(context=CONTEXT, n_features=N_FEATURES)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()

    first = loss_function(model(inputs), targets).item()
    for _ in range(400):
        optimiser.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimiser.step()
    last = loss.item()

    assert last < first * 0.05, (
        f"loss went {first:.4f} -> {last:.4f}. A model that cannot memorise "
        "eight examples is wired wrong, not undertrained")


def test_two_models_built_under_the_same_seed_are_identical():
    """A timing number and a score both need the run to be repeatable."""
    torch.manual_seed(7)
    first = train.build_model(context=CONTEXT, n_features=N_FEATURES)
    torch.manual_seed(7)
    second = train.build_model(context=CONTEXT, n_features=N_FEATURES)

    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


# --- trap 4: the score ---------------------------------------------------------

def test_skill_is_zero_against_the_baseline_itself():
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(100, N_FEATURES))
    baseline = truth + rng.normal(scale=0.1, size=truth.shape)

    assert abs(train.skill(baseline, truth, baseline)) < 1e-9


def test_skill_is_one_for_a_perfect_prediction():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(100, N_FEATURES))
    baseline = truth + rng.normal(scale=0.1, size=truth.shape)

    assert abs(train.skill(truth, truth, baseline) - 1.0) < 1e-9


def test_skill_is_negative_when_worse_than_the_baseline():
    """The result that matters. A model below its baseline learned nothing."""
    rng = np.random.default_rng(5)
    truth = rng.normal(size=(100, N_FEATURES))
    baseline = truth + rng.normal(scale=0.1, size=truth.shape)
    worse = truth + rng.normal(scale=0.5, size=truth.shape)

    assert train.skill(worse, truth, baseline) < 0
