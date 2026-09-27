import numpy as np
import pytest

generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")

if not hasattr(windows, "build_force_labels"):
    pytest.skip("windows.build_force_labels does not exist yet, this suite is its "
                "specification", allow_module_level=True)

from world import PHYSICS_CRUISING

RULES = ("separation", "alignment", "cohesion")
N_BIRDS = 12


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("force_labels")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=3, n_slots=N_BIRDS, n_live=N_BIRDS,
                           n_ticks=60, base_seed=20260928, out_dir=out, allow_dirty=True)
    return out


def recorded(dataset):
    with np.load(dataset / "clips.npz") as clips:
        return {rule: clips[rule] for rule in RULES}, clips["velocities"]


# --- what comes back ----------------------------------------------------------------

def test_one_array_per_rule(dataset):
    labels = windows.build_force_labels(dataset, context=2)
    assert set(labels) == set(RULES)


def test_one_row_per_window_the_same_count_as_build_windows(dataset):
    """Label i has to belong to window i. A different count means they cannot."""
    for context in (2, 5):
        x, _ = windows.build_windows(dataset, context=context)
        labels = windows.build_force_labels(dataset, context=context)
        for rule in RULES:
            assert labels[rule].shape == (len(x), N_BIRDS, 2), f"{rule}, context {context}"


def test_no_missing_values(dataset):
    labels = windows.build_force_labels(dataset, context=2)
    for rule in RULES:
        assert np.isfinite(labels[rule]).all(), rule


# --- the right tick: the trap this suite exists for -------------------------------------

@pytest.mark.parametrize("context", [2, 3, 5])
def test_the_three_pushes_add_up_to_the_target(dataset, context):
    """The window ends at tick T and its target is v(T+1) - v(T). The simulator
    made that change from the three pushes it recorded while stepping T -> T+1,
    times dt. So the labels, added up, must give back the target exactly.

    Take the pushes one tick early or late and they still look right: pushes
    change slowly, the probes still score, and every curve is quietly off.
    This is the only test that can tell.

    A bird that hit the speed limit had its velocity trimmed after the pushes,
    so those birds are left out."""
    _, targets = windows.build_windows(dataset, context=context)
    labels = windows.build_force_labels(dataset, context=context)
    targets = targets.reshape(len(targets), N_BIRDS, 2)
    total = (labels["separation"] + labels["alignment"] + labels["cohesion"]) * PHYSICS_CRUISING.dt

    _, velocities = recorded(dataset)
    n_windows_clip = velocities.shape[1] - context
    speed_after = []
    for c in range(velocities.shape[0]):
        for offset in range(n_windows_clip):
            speed_after.append(np.linalg.norm(velocities[c, offset + context], axis=-1))
    speed_after = np.array(speed_after)
    free = ((speed_after > PHYSICS_CRUISING.min_speed + 1e-3) &
            (speed_after < PHYSICS_CRUISING.max_speed - 1e-3))
    assert free.mean() > 0.3, "too few birds away from the speed limits for this test to mean anything"

    np.testing.assert_allclose(total[free], targets[free], atol=1e-5)


def test_window_zero_of_each_clip_reads_the_push_at_its_last_tick(dataset):
    """Spelled out on one window, so the rule is readable: context 4 means ticks
    0 to 3 go in, and the pushes recorded at index 3 (the step 3 -> 4) come out."""
    forces, _ = recorded(dataset)
    labels = windows.build_force_labels(dataset, context=4)
    n_windows_clip = forces["separation"].shape[1] + 1 - 4
    for c in range(3):
        row = c * n_windows_clip
        for rule in RULES:
            np.testing.assert_array_equal(labels[rule][row], forces[rule][c, 3])


# --- the clips argument, the same as build_windows ---------------------------------------

def test_a_single_clip_is_the_same_rows_as_in_the_full_set(dataset):
    full = windows.build_force_labels(dataset, context=2)
    only = windows.build_force_labels(dataset, context=2, clips=1)
    n = len(only["separation"])
    for rule in RULES:
        np.testing.assert_array_equal(only[rule], full[rule][n:2 * n], err_msg=rule)


def test_a_list_of_clips_keeps_that_order(dataset):
    one = windows.build_force_labels(dataset, context=2, clips=[2])
    two = windows.build_force_labels(dataset, context=2, clips=[2, 0])
    n = len(one["separation"])
    np.testing.assert_array_equal(two["alignment"][:n], one["alignment"])


def test_a_context_longer_than_the_clip_is_refused(dataset):
    with pytest.raises(ValueError):
        windows.build_force_labels(dataset, context=500)
