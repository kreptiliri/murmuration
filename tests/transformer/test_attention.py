import pytest

torch = pytest.importorskip("torch")
flocker = pytest.importorskip(
    "flocker",
    reason="flocker.py does not exist yet, this suite is its specification")

D_MODEL = 32
N_HEADS = 4
CONTEXT = 8
BATCH = 3


@pytest.fixture
def attention():
    torch.manual_seed(0)
    return flocker.CausalSelfAttention(d_model=D_MODEL, n_heads=N_HEADS)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, CONTEXT, D_MODEL)


def test_a_window_comes_back_the_same_shape(attention, x):
    """Attention re-mixes the ticks; it does not change how many there are."""
    assert attention(x).shape == x.shape


def test_it_matches_pytorchs_own_attention(attention, x):
    """SPEC §5's reference check, run on every commit instead of once by hand.

    Rebuilds the same computation out of `torch.nn.functional`, using this
    module's own weights, and demands agreement. Catches a wrong scale factor,
    a head reshape that interleaves instead of splitting, and a mask applied to
    the wrong axis - none of which show up as an error at training time.
    """
    import torch.nn.functional as F

    batch, context, _ = x.shape
    head_dim = D_MODEL // N_HEADS

    query, key, value = attention.q(x), attention.k(x), attention.v(x)
    shape_into_heads = lambda t: t.view(batch, context, N_HEADS, head_dim).transpose(1, 2)

    reference = F.scaled_dot_product_attention(
        shape_into_heads(query), shape_into_heads(key), shape_into_heads(value),
        is_causal=True)
    reference = reference.transpose(1, 2).contiguous().view(batch, context, D_MODEL)
    reference = attention.proj(reference)

    torch.testing.assert_close(attention(x), reference, atol=1e-5, rtol=1e-5)


def test_a_tick_cannot_see_the_future(attention, x):
    """The one that matters, and it is independent of the reference above.

    Change the last tick of the input. Every earlier position's output must be
    bit-for-bit identical. If it moves, the mask leaks and the model is reading
    the answer it is being asked to predict.
    """
    before = attention(x)

    tampered = x.clone()
    tampered[:, -1] = torch.randn(BATCH, D_MODEL)
    after = attention(tampered)

    torch.testing.assert_close(before[:, :-1], after[:, :-1], atol=0, rtol=0)


def test_every_earlier_tick_is_protected_not_just_the_last(attention, x):
    """A mask off by one passes the test above and fails this one."""
    for tick in range(1, CONTEXT):
        tampered = x.clone()
        tampered[:, tick] = torch.randn(BATCH, D_MODEL)

        torch.testing.assert_close(
            attention(x)[:, :tick], attention(tampered)[:, :tick], atol=0, rtol=0,
            msg=f"changing tick {tick} moved an output at or before it")


def test_the_first_tick_only_ever_sees_itself(attention):
    """With nothing to attend to but itself, tick 0 is a function of tick 0 alone."""
    torch.manual_seed(2)
    one = torch.randn(1, CONTEXT, D_MODEL)
    other = one.clone()
    other[:, 1:] = torch.randn(1, CONTEXT - 1, D_MODEL)

    torch.testing.assert_close(attention(one)[:, 0], attention(other)[:, 0],
                               atol=0, rtol=0)


def test_the_heads_have_to_divide_the_width(attention):
    """5 heads into 32 channels is a silent truncation, not an arrangement."""
    with pytest.raises(ValueError):
        flocker.CausalSelfAttention(d_model=32, n_heads=5)


def test_two_calls_agree(attention, x):
    """No dropout, no randomness at inference. A score has to be repeatable."""
    torch.testing.assert_close(attention(x), attention(x), atol=0, rtol=0)


def test_two_built_under_the_same_seed_are_identical():
    torch.manual_seed(7)
    first = flocker.CausalSelfAttention(d_model=D_MODEL, n_heads=N_HEADS)
    torch.manual_seed(7)
    second = flocker.CausalSelfAttention(d_model=D_MODEL, n_heads=N_HEADS)

    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
