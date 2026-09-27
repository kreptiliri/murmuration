import pytest

torch = pytest.importorskip("torch")
flocker = pytest.importorskip("flocker")

if not hasattr(flocker.RelPerBirdTransformer, "hidden_states"):
    pytest.skip("RelPerBirdTransformer.hidden_states does not exist yet, this suite is "
                "its specification", allow_module_level=True)

N_BIRDS, CONTEXT, D_MODEL, N_HEADS = 12, 2, 32, 4


def a_model(n_blocks=2, seed=0):
    torch.manual_seed(seed)
    model = flocker.RelPerBirdTransformer(n_features=N_BIRDS * 2, d_model=D_MODEL,
                                          n_heads=N_HEADS, n_blocks=n_blocks, context=CONTEXT)
    model.eval()
    return model


def some_input(batch=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, CONTEXT, N_BIRDS * 2, generator=g)
    pair = torch.randn(batch, N_BIRDS, N_BIRDS, 3, generator=g)
    return x, pair


def drafts_of(model, x, pair):
    with torch.no_grad():
        return model.hidden_states(x, pair)


# --- what comes back ---------------------------------------------------------------

@pytest.mark.parametrize("n_blocks", [1, 2, 3])
def test_one_draft_after_the_embedding_and_one_after_each_block(n_blocks):
    x, pair = some_input()
    drafts = drafts_of(a_model(n_blocks), x, pair)
    assert len(drafts) == n_blocks + 1


def test_every_draft_is_one_row_of_d_model_numbers_per_bird():
    x, pair = some_input(batch=5)
    for k, draft in enumerate(drafts_of(a_model(), x, pair)):
        assert tuple(draft.shape) == (5, N_BIRDS, D_MODEL), f"draft {k}"


# --- each draft is the right one ---------------------------------------------------------

def test_draft_0_is_each_birds_own_move_after_the_embedding():
    """With context 2 a bird's token is its one move, x[:, 1] (x[:, 0] is where it
    started, which the relative model never reads)."""
    model = a_model()
    x, pair = some_input()
    own_moves = x[:, 1].reshape(len(x), N_BIRDS, 2)
    with torch.no_grad():
        expected = model.embedding(own_moves)
    torch.testing.assert_close(drafts_of(model, x, pair)[0], expected)


def test_each_later_draft_is_the_block_applied_to_the_draft_before():
    model = a_model()
    x, pair = some_input()
    drafts = drafts_of(model, x, pair)
    for k, block in enumerate(model.blocks):
        with torch.no_grad():
            expected = block(drafts[k], pair)
        torch.testing.assert_close(drafts[k + 1], expected, msg=f"draft {k + 1}")


def test_the_answer_is_written_from_the_last_draft():
    """The drafts have to belong to the model that flies. If they came from a
    second copy of the code, the probes would read a model nobody trained on."""
    model = a_model()
    x, pair = some_input()
    last = drafts_of(model, x, pair)[-1]
    with torch.no_grad():
        from_last = model.head(model.norm(last)).reshape(len(x), N_BIRDS * 2)
        answer = model(x, pair)
    torch.testing.assert_close(answer, from_last)


def test_forward_goes_through_hidden_states():
    """One copy of the code, not two. Change what hidden_states hands back and the
    answer has to change with it."""
    model = a_model()
    x, pair = some_input()
    with torch.no_grad():
        before = model(x, pair)
    real = model.hidden_states

    def doubled(x, pair):
        drafts = real(x, pair)
        return drafts[:-1] + [drafts[-1] * 2]

    model.hidden_states = doubled
    with torch.no_grad():
        after = model(x, pair)
    assert not torch.equal(before, after), "forward ignores hidden_states, the code is doubled"


def test_the_drafts_do_not_all_come_out_the_same():
    """Appending the same tensor and then changing it in place leaves a list of
    identical drafts. Every layer would then probe the same."""
    drafts = drafts_of(a_model(), *some_input())
    for k in range(len(drafts) - 1):
        assert not torch.equal(drafts[k], drafts[k + 1]), f"draft {k} and {k + 1} are the same"


# --- the rows are the birds, which is what a probe relies on ---------------------------------

def test_draft_0_knows_only_its_own_bird_and_draft_1_has_listened():
    """Before the first listening round a bird knows nothing about the others."""
    model = a_model()
    x, pair = some_input()
    nudged = x.clone()
    nudged[:, 1, 3 * 2] += 1.0      #bird 3's move, x part
    before = drafts_of(model, x, pair)
    after = drafts_of(model, nudged, pair)

    changed_0 = (after[0] - before[0]).abs().amax(dim=(0, 2)) > 1e-6
    assert changed_0.tolist() == [b == 3 for b in range(N_BIRDS)]
    changed_1 = (after[1] - before[1]).abs().amax(dim=(0, 2)) > 1e-6
    assert changed_1.sum() > 1, "nobody else heard bird 3 in the first round"


def test_row_i_is_bird_i_in_every_draft():
    """The probe for bird 5 reads row 5. Shuffle the birds' order going in, and
    every draft's rows must come out shuffled the same way."""
    model = a_model()
    x, pair = some_input()
    order = torch.randperm(N_BIRDS, generator=torch.Generator().manual_seed(7))
    x_shuffled = x.view(len(x), CONTEXT, N_BIRDS, 2)[:, :, order].reshape(len(x), CONTEXT, N_BIRDS * 2)
    pair_shuffled = pair[:, order][:, :, order]
    plain = drafts_of(model, x, pair)
    shuffled = drafts_of(model, x_shuffled, pair_shuffled)
    for k in range(len(plain)):
        torch.testing.assert_close(shuffled[k], plain[k][:, order], msg=f"draft {k}")
