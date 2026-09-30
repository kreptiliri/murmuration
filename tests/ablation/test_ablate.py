import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")
windows = pytest.importorskip("windows")
flocker = pytest.importorskip("flocker")
ablate = pytest.importorskip("ablate")

from world import PHYSICS_CRUISING

PHYSICS = dataclasses.asdict(PHYSICS_CRUISING)
N_BIRDS, CONTEXT = 6, 2
D_MODEL, N_HEADS = 32, 4


def small_model(n_blocks):
    torch.manual_seed(0)
    model = flocker.RelPerBirdTransformer(n_features=2 * N_BIRDS, d_model=D_MODEL, n_heads=N_HEADS,
                                          n_blocks=n_blocks, context=CONTEXT)
    model.eval()
    return model


def fake_input(seed, batch=5):
    #random moves and random geometry, the shapes the model takes
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, CONTEXT, 2 * N_BIRDS, generator=gen)
    pair = torch.randn(batch, N_BIRDS, N_BIRDS, 3, generator=gen)
    return x, pair


def note(seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(D_MODEL, generator=gen)


def answers_from(model, draft, x):
    #what the head does with the last draft, the same as the end of model.forward
    batch = x.shape[0]
    return model.head(model.norm(draft)).reshape(batch, 2 * N_BIRDS)


def think_alone(block, draft):
    #the second line of a block, the one ablation leaves alone
    return draft + block.feedforward(block.norm_two(draft))


# --- ablated_forward: what comes back ---------------------------------------------------

def test_it_answers_in_the_same_shape_as_the_model():
    model = small_model(2)
    x, pair = fake_input(1)
    with torch.no_grad():
        out = ablate.ablated_forward(model, x, pair, block=1, note=note(2))
        assert out.shape == model(x, pair).shape


def test_it_does_not_change_the_model():
    """The trained weights are the experiment's control. Switching a step off
    for one call must leave the real model exactly as it was."""
    model = small_model(2)
    x, pair = fake_input(1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        normal_before = model(x, pair)
        ablate.ablated_forward(model, x, pair, block=1, note=note(2))
        normal_after = model(x, pair)
    torch.testing.assert_close(normal_before, normal_after, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)


def test_it_refuses_a_block_that_does_not_exist():
    model = small_model(2)
    x, pair = fake_input(1)
    for bad in [2, 5, -1]:
        with pytest.raises(ValueError):
            ablate.ablated_forward(model, x, pair, block=bad, note=note(2))


def test_the_limit_comes_from_the_model_not_a_fixed_number():
    """A one-block model has only block 0. Asking for block 1 would switch
    nothing off and run the normal model without a word."""
    model = small_model(1)
    x, pair = fake_input(1)
    with pytest.raises(ValueError):
        ablate.ablated_forward(model, x, pair, block=1, note=note(2))


# --- ablated_forward: exactly the listening is replaced, nothing else ---------------------

def test_the_last_block_listens_to_nothing_and_still_thinks():
    """Block 1 switched off by hand: the draft coming in (draft 1) gets the note
    instead of what the neighbours said, then the thinking line runs as normal,
    then the head reads it."""
    model = small_model(2)
    x, pair = fake_input(3)
    fixed = note(4)
    with torch.no_grad():
        draft_1 = model.hidden_states(x, pair)[1]
        expected = answers_from(model, think_alone(model.blocks[1], draft_1 + fixed), x)
        out = ablate.ablated_forward(model, x, pair, block=1, note=fixed)
    torch.testing.assert_close(out, expected)


def test_the_first_block_switched_off_and_the_second_still_listens():
    """Block 0 switched off, block 1 untouched: the blocks after the switched
    off one run normally, pair and all."""
    model = small_model(2)
    x, pair = fake_input(5)
    fixed = note(6)
    with torch.no_grad():
        draft_0 = model.hidden_states(x, pair)[0]
        draft_1 = think_alone(model.blocks[0], draft_0 + fixed)
        draft_2 = model.blocks[1](draft_1, pair)
        expected = answers_from(model, draft_2, x)
        out = ablate.ablated_forward(model, x, pair, block=0, note=fixed)
    torch.testing.assert_close(out, expected)


def test_if_the_listening_already_wrote_nothing_switching_it_off_changes_nothing():
    """A block whose listening output is always zero, switched off with a zero
    note, is the same model. If this fails, something besides the listening
    line was changed."""
    model = small_model(2)
    for block in model.blocks:
        torch.nn.init.zeros_(block.attention.proj.weight)
    x, pair = fake_input(7)
    zero = torch.zeros(D_MODEL)
    with torch.no_grad():
        for k in range(2):
            out = ablate.ablated_forward(model, x, pair, block=k, note=zero)
            torch.testing.assert_close(out, model(x, pair), msg=f"block {k}")


def test_the_note_is_used():
    model = small_model(2)
    x, pair = fake_input(8)
    with torch.no_grad():
        one = ablate.ablated_forward(model, x, pair, block=1, note=note(9))
        other = ablate.ablated_forward(model, x, pair, block=1, note=note(10))
    assert not torch.allclose(one, other)


# --- ablated_forward: what switching off means for the birds -------------------------------

def test_with_the_only_block_switched_off_a_bird_cannot_hear_its_neighbours():
    """The point of the whole experiment. One block, its listening replaced by
    a fixed note: bird 0's answer must not move when every other bird's moves
    and every distance change."""
    model = small_model(1)
    x, pair = fake_input(11)
    changed_x = x.clone()
    changed_x[:, :, 2:] = torch.randn_like(changed_x[:, :, 2:])  #everyone but bird 0
    changed_pair = torch.randn_like(pair)
    with torch.no_grad():
        out = ablate.ablated_forward(model, x, pair, block=0, note=note(12))
        changed = ablate.ablated_forward(model, changed_x, changed_pair, block=0, note=note(12))
    torch.testing.assert_close(out[:, :2], changed[:, :2])


def test_with_one_of_two_blocks_switched_off_a_bird_still_hears_through_the_other():
    """Only one round goes quiet. The other still listens, so the neighbours
    still reach bird 0. If this fails, both rounds were switched off."""
    model = small_model(2)
    x, pair = fake_input(13)
    changed_x = x.clone()
    changed_x[:, :, 2:] = torch.randn_like(changed_x[:, :, 2:])
    with torch.no_grad():
        for k in range(2):
            out = ablate.ablated_forward(model, x, pair, block=k, note=note(14))
            changed = ablate.ablated_forward(model, changed_x, pair, block=k, note=note(14))
            assert not torch.allclose(out[:, :2], changed[:, :2]), f"block {k}"


# --- average_listening: the fixed note, from real windows ------------------------------------

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    data = tmp_path_factory.mktemp("ablate_data")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=10, n_slots=N_BIRDS, n_live=N_BIRDS,
                           n_ticks=60, base_seed=20260930, out_dir=data, allow_dirty=True)
    ckpts = tmp_path_factory.mktemp("ablate_checkpoints")
    train.main(["--data", str(data), "--clips", "10", "--context", str(CONTEXT), "--passes", "1",
                "--batch", "64", "--report-every", "0", "--model", "rel_per_bird",
                "--checkpoint-every", "1", "--checkpoint-dir", str(ckpts)])
    ck = train.load_checkpoint(ckpts / "pass_0001.pt")
    x, _ = windows.build_windows(data, context=CONTEXT, clips=range(10))
    return data, ck, x


def listening_by_hand(ck, x, block):
    #every bird's listening output in block k, all windows at once, no batches
    moves = torch.from_numpy(train.apply_squash(ck["x_scaler"], train.to_moves(x))).float()
    pair = torch.from_numpy(train.pair_geometry(x, bounds=PHYSICS["bounds"],
                                                scale=PHYSICS["separation_radius"]))
    model = ck["model"]
    with torch.no_grad():
        draft = model.hidden_states(moves, pair)[block]
        heard = model.blocks[block].attention(model.blocks[block].norm_one(draft), pair)
    return heard.reshape(-1, heard.shape[-1])


def test_the_note_is_one_number_per_box(run):
    _, ck, x = run
    d_model = ck["model"].head.in_features
    average = ablate.average_listening(ck, x[:50], physics=PHYSICS, block=1)
    assert isinstance(average, torch.Tensor)
    assert average.shape == (d_model,)


def test_the_note_is_the_average_over_every_bird_in_every_window(run):
    """More than 256 windows, so reading in batches is covered too."""
    _, ck, x = run
    assert len(x) > 256
    for k in range(2):
        expected = listening_by_hand(ck, x, k).mean(dim=0)
        average = ablate.average_listening(ck, x, physics=PHYSICS, block=k)
        torch.testing.assert_close(average, expected, rtol=1e-4, atol=1e-5, msg=f"block {k}")


def test_a_short_last_batch_counts_as_much_as_a_full_one_per_window(run):
    """300 windows is a batch of 256 and a batch of 44. The average of the two
    batch averages would count each of the 44 windows about six times as much
    as the others. The right note weighs every window the same."""
    _, ck, x = run
    part = x[:300]
    expected = listening_by_hand(ck, part, 1).mean(dim=0)
    average = ablate.average_listening(ck, part, physics=PHYSICS, block=1)
    torch.testing.assert_close(average, expected, rtol=1e-4, atol=1e-5)


def test_the_note_comes_from_the_windows_it_is_given(run):
    """The note is fitted on the learn clips and the damage scored on the check
    clips, so it has to depend on which windows go in."""
    _, ck, x = run
    first = ablate.average_listening(ck, x[:100], physics=PHYSICS, block=1)
    last = ablate.average_listening(ck, x[-100:], physics=PHYSICS, block=1)
    assert not torch.allclose(first, last)


def test_it_does_not_change_the_windows(run):
    _, ck, x = run
    part = x[:30].copy()
    ablate.average_listening(ck, part, physics=PHYSICS, block=0)
    np.testing.assert_array_equal(part, x[:30])


def test_the_average_refuses_a_block_that_does_not_exist(run):
    _, ck, x = run
    for bad in [2, -1]:
        with pytest.raises(ValueError):
            ablate.average_listening(ck, x[:10], physics=PHYSICS, block=bad)


def test_it_refuses_a_model_that_does_not_listen_by_pairs(run, tmp_path):
    data, _, x = run
    train.main(["--data", str(data), "--clips", "10", "--context", str(CONTEXT), "--passes", "1",
                "--batch", "64", "--report-every", "0", "--model", "per_bird",
                "--checkpoint-every", "1", "--checkpoint-dir", str(tmp_path)])
    per_bird = train.load_checkpoint(tmp_path / "pass_0001.pt")
    with pytest.raises(ValueError):
        ablate.average_listening(per_bird, x[:10], physics=PHYSICS, block=0)


# --- the two together, on real windows -------------------------------------------------------

def test_the_average_note_plugs_into_the_ablated_model(run):
    _, ck, x = run
    average = ablate.average_listening(ck, x, physics=PHYSICS, block=1)
    moves = torch.from_numpy(train.apply_squash(ck["x_scaler"], train.to_moves(x[:40]))).float()
    pair = torch.from_numpy(train.pair_geometry(x[:40], bounds=PHYSICS["bounds"],
                                                scale=PHYSICS["separation_radius"]))
    with torch.no_grad():
        out = ablate.ablated_forward(ck["model"], moves, pair, block=1, note=average)
    assert out.shape == (40, 2 * N_BIRDS)
    assert torch.isfinite(out).all()


# --- listening only to yourself -----------------------------------------------------------------

def test_self_listening_is_one_note_per_bird():
    model = small_model(2)
    draft, pair = torch.randn(5, N_BIRDS, D_MODEL), torch.randn(5, N_BIRDS, N_BIRDS, 3)
    with torch.no_grad():
        heard = ablate.self_listening(model.blocks[0].attention, draft, pair)
    assert heard.shape == (5, N_BIRDS, D_MODEL)


def test_with_one_bird_self_listening_is_the_normal_listening():
    """A bird alone has nobody else to hear, so the real attention already puts
    all its weight on itself. The two have to agree exactly there."""
    model = small_model(2)
    draft, pair = torch.randn(5, 1, D_MODEL), torch.randn(5, 1, 1, 3)
    attention = model.blocks[1].attention
    with torch.no_grad():
        torch.testing.assert_close(ablate.self_listening(attention, draft, pair), attention(draft, pair))


def test_self_listening_does_not_hear_the_neighbours():
    """Bird 0's note must not move when every other bird's draft changes and
    every distance except its own (bird 0 to bird 0) changes."""
    model = small_model(2)
    draft, pair = torch.randn(5, N_BIRDS, D_MODEL), torch.randn(5, N_BIRDS, N_BIRDS, 3)
    changed_draft = draft.clone()
    changed_draft[:, 1:] = torch.randn_like(changed_draft[:, 1:])
    changed_pair = torch.randn_like(pair)
    changed_pair[:, 0, 0] = pair[:, 0, 0]
    attention = model.blocks[0].attention
    with torch.no_grad():
        heard = ablate.self_listening(attention, draft, pair)
        changed = ablate.self_listening(attention, changed_draft, changed_pair)
    torch.testing.assert_close(heard[:, 0], changed[:, 0])


def test_self_listening_still_hears_its_own_draft_and_its_own_place():
    model = small_model(2)
    draft, pair = torch.randn(5, N_BIRDS, D_MODEL), torch.randn(5, N_BIRDS, N_BIRDS, 3)
    attention = model.blocks[0].attention
    changed_draft = draft.clone()
    changed_draft[:, 0] = torch.randn_like(changed_draft[:, 0])
    changed_pair = pair.clone()
    changed_pair[:, 0, 0] = torch.randn_like(changed_pair[:, 0, 0])
    with torch.no_grad():
        heard = ablate.self_listening(attention, draft, pair)
        own_draft = ablate.self_listening(attention, changed_draft, pair)
        own_place = ablate.self_listening(attention, draft, changed_pair)
    assert not torch.allclose(heard[:, 0], own_draft[:, 0])
    assert not torch.allclose(heard[:, 0], own_place[:, 0])


def test_note_self_is_the_listening_line_with_only_yourself():
    """Block 1 switched off with note="self": the listening line runs, but
    each bird hears only itself, then the thinking line and the head as usual."""
    model = small_model(2)
    x, pair = fake_input(15)
    block = model.blocks[1]
    with torch.no_grad():
        draft_1 = model.hidden_states(x, pair)[1]
        heard = ablate.self_listening(block.attention, block.norm_one(draft_1), pair)
        expected = answers_from(model, think_alone(block, draft_1 + heard), x)
        out = ablate.ablated_forward(model, x, pair, block=1, note="self")
    torch.testing.assert_close(out, expected)


def test_with_the_only_block_on_self_a_bird_cannot_hear_its_neighbours():
    model = small_model(1)
    x, pair = fake_input(16)
    changed_x = x.clone()
    changed_x[:, :, 2:] = torch.randn_like(changed_x[:, :, 2:])
    changed_pair = torch.randn_like(pair)
    changed_pair[:, 0, 0] = pair[:, 0, 0]
    with torch.no_grad():
        out = ablate.ablated_forward(model, x, pair, block=0, note="self")
        changed = ablate.ablated_forward(model, changed_x, changed_pair, block=0, note="self")
    torch.testing.assert_close(out[:, :2], changed[:, :2])


def test_note_self_is_not_the_blank_note():
    model = small_model(2)
    x, pair = fake_input(17)
    with torch.no_grad():
        own = ablate.ablated_forward(model, x, pair, block=1, note="self")
        blank = ablate.ablated_forward(model, x, pair, block=1, note=torch.zeros(D_MODEL))
    assert not torch.allclose(own, blank)


def test_a_note_named_anything_else_is_refused():
    model = small_model(2)
    x, pair = fake_input(18)
    with pytest.raises(ValueError):
        ablate.ablated_forward(model, x, pair, block=1, note="average")


def test_every_bird_hears_its_own_place_and_no_one_elses():
    """Checked for every bird, not just bird 0: changing every distance except
    the diagonal (each bird to itself) changes nobody's note, and changing
    one bird's own place changes that bird's note."""
    model = small_model(2)
    draft, pair = torch.randn(5, N_BIRDS, D_MODEL), torch.randn(5, N_BIRDS, N_BIRDS, 3)
    attention = model.blocks[0].attention
    others_changed = torch.randn_like(pair)
    for i in range(N_BIRDS):
        others_changed[:, i, i] = pair[:, i, i]
    with torch.no_grad():
        heard = ablate.self_listening(attention, draft, pair)
        torch.testing.assert_close(ablate.self_listening(attention, draft, others_changed), heard)
        for b in range(N_BIRDS):
            own_changed = pair.clone()
            own_changed[:, b, b] = pair[:, b, b] + 1.0
            moved = ablate.self_listening(attention, draft, own_changed)
            assert not torch.allclose(moved[:, b], heard[:, b]), f"bird {b}"
