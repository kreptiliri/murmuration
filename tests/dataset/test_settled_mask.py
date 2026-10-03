import numpy as np
import pytest

from world import PHYSICS_CRUISING

generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")

if not hasattr(windows, "settled_mask"):
    pytest.skip("windows.settled_mask does not exist yet, this suite is its "
                "specification", allow_module_level=True)


N_TICKS = 40  # 41 states, 0-40
CONTEXT = 3
# one clip per case: never settles, settled from the first state, settles
# mid-clip, settles on the final state (which no window's input reaches),
# settles on the last state a window's input does reach, settles mid-clip again
SETTLE = [-1, 0, 10, 40, 39, 20]


def with_settle_ticks(dataset_dir, settle):
    # the simulator decides settle_tick from the flight; here it is written by
    # hand, so every edge case is present whatever the flocks happened to do
    path = dataset_dir / "clips.npz"
    with np.load(path) as z:
        arrays = {name: z[name] for name in z.files}
    arrays["settle_tick"] = np.array(settle, dtype=np.int32)
    np.savez(path, **arrays)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("settled_mask")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=len(SETTLE), n_slots=4,
                           n_live=4, n_ticks=N_TICKS, base_seed=20260924,
                           out_dir=out, allow_dirty=True)
    with_settle_ticks(out, SETTLE)
    return out


def by_hand(settle, context):
    # window `offset` is fed states offset .. offset+context-1 and scored on the
    # change into state offset+context. It is settled when its last INPUT state
    # is at or after settle_tick, and never when settle_tick is -1
    flags = []
    for s in settle:
        for offset in range(N_TICKS + 1 - context):
            last_input = offset + context - 1
            flags.append(s >= 0 and last_input >= s)
    return np.array(flags, dtype=bool)


def clip_rows(mask, clip, context=CONTEXT):
    per_clip = N_TICKS + 1 - context
    return mask[clip * per_clip:(clip + 1) * per_clip]


# --- the shape contract ------------------------------------------------------

def test_one_flag_per_window_in_build_windows_order(dataset):
    inputs, _ = windows.build_windows(dataset, context=CONTEXT)

    mask = windows.settled_mask(dataset, context=CONTEXT)

    assert mask.shape == (len(inputs),)
    assert mask.dtype == bool


def test_it_follows_the_rule_by_hand(dataset):
    mask = windows.settled_mask(dataset, context=CONTEXT)

    np.testing.assert_array_equal(mask, by_hand(SETTLE, CONTEXT))


# --- the edge cases, one clip each ----------------------------------------------

def test_a_clip_that_never_settles_has_no_settled_window(dataset):
    mask = windows.settled_mask(dataset, context=CONTEXT)

    assert not clip_rows(mask, 0).any()


def test_a_clip_settled_from_the_first_state_is_settled_throughout(dataset):
    mask = windows.settled_mask(dataset, context=CONTEXT)

    assert clip_rows(mask, 1).all()


def test_the_boundary_is_the_last_input_state(dataset):
    rows = clip_rows(windows.settled_mask(dataset, context=CONTEXT), 2)

    assert not rows[7], "window 7 is fed states 7-9, all before settling"
    assert rows[8], "window 8 is fed states 8-10, and state 10 is settled"
    assert not rows[:8].any() and rows[8:].all()


def test_settling_on_the_final_state_settles_no_window(dataset):
    mask = windows.settled_mask(dataset, context=CONTEXT)

    assert not clip_rows(mask, 3).any()


def test_settling_on_the_last_input_state_settles_only_the_last_window(dataset):
    rows = clip_rows(windows.settled_mask(dataset, context=CONTEXT), 4)

    assert rows[-1] and not rows[:-1].any()


@pytest.mark.parametrize("context", [1, 5])
def test_the_context_moves_the_boundary(dataset, context):
    mask = windows.settled_mask(dataset, context=context)

    np.testing.assert_array_equal(mask, by_hand(SETTLE, context))


# --- choosing clips, the same way build_windows does ------------------------------

@pytest.mark.parametrize("clips", [range(2, 5), [5, 1], 4],
                         ids=["a range", "a list out of order", "one int"])
def test_clips_are_chosen_like_build_windows(dataset, clips):
    chosen = [clips] if isinstance(clips, int) else list(clips)
    inputs, _ = windows.build_windows(dataset, context=CONTEXT, clips=clips)

    mask = windows.settled_mask(dataset, context=CONTEXT, clips=clips)

    assert len(mask) == len(inputs)
    np.testing.assert_array_equal(mask, by_hand([SETTLE[c] for c in chosen], CONTEXT))


@pytest.mark.parametrize("context", [0, N_TICKS + 1])
def test_a_context_build_windows_refuses_is_refused(dataset, context):
    with pytest.raises(ValueError):
        windows.settled_mask(dataset, context=context)
