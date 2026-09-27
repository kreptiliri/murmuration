import pytest

torch = pytest.importorskip("torch")
flocker = pytest.importorskip("flocker")

if not hasattr(flocker, "RelAttention"):
    pytest.skip("flocker.RelAttention does not exist yet, this suite is its "
                "specification", allow_module_level=True)

D_MODEL = 32
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
N_BIRDS = 6
BATCH = 3


@pytest.fixture
def attention():
    torch.manual_seed(0)
    return flocker.RelAttention(d_model=D_MODEL, n_heads=N_HEADS)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, N_BIRDS, D_MODEL)


@pytest.fixture
def pair():
    """Stand-in geometry: [b, i, j] is (dx, dy, distance) from bird i to bird j."""
    torch.manual_seed(2)
    offsets = torch.randn(BATCH, N_BIRDS, 1, 2) - torch.randn(BATCH, 1, N_BIRDS, 2)
    offsets = offsets - offsets.transpose(1, 2)
    return torch.cat([offsets, offsets.norm(dim=-1, keepdim=True)], dim=-1)


class Zeros(torch.nn.Module):
    """Stands in for a pair network that has learned nothing."""

    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, pair):
        return torch.zeros(*pair.shape[:-1], self.width)


def by_hand(attention, x, pair):
    """Relative-position attention written out one step at a time, from the
    module's own weights. Each line is one idea from the lesson."""
    b, n, d = x.shape
    heads = lambda t: t.view(b, n, N_HEADS, D_HEAD).transpose(1, 2)        # (b, h, n, dh)
    q, k, v = heads(attention.q(x)), heads(attention.k(x)), heads(attention.v(x))

    scores = q @ k.transpose(-2, -1) / D_HEAD ** 0.5                         # (b, h, i, j)
    bias = attention.pair_bias(pair)                                         # (b, i, j, h)
    scores = scores + bias.permute(0, 3, 1, 2)                               # one number per head per pair
    weights = scores.softmax(dim=-1)                                         # each listener's row sums to 1

    heard = weights @ v                                                      # what the speakers said
    extra = attention.pair_value(pair).view(b, n, n, N_HEADS, D_HEAD)       # (b, i, j, h, dh)
    extra = torch.einsum("bhij,bijhd->bhid", weights, extra)                # where they said it from

    out = (heard + extra).transpose(1, 2).reshape(b, n, d)
    return attention.proj(out)


# --- the contract --------------------------------------------------------------------

def test_birds_come_back_the_same_shape(attention, x, pair):
    assert attention(x, pair).shape == x.shape


def test_it_has_the_parts_of_ordinary_attention(attention):
    for name in ("q", "k", "v", "proj"):
        assert isinstance(getattr(attention, name), torch.nn.Linear), name


def test_the_pair_networks_give_one_number_per_head_and_one_row_per_width(attention, pair):
    assert attention.pair_bias(pair).shape == (BATCH, N_BIRDS, N_BIRDS, N_HEADS)
    assert attention.pair_value(pair).shape == (BATCH, N_BIRDS, N_BIRDS, D_MODEL)


def test_the_heads_have_to_divide_the_width():
    with pytest.raises(ValueError):
        flocker.RelAttention(d_model=32, n_heads=5)


# --- the computation ------------------------------------------------------------------

def test_it_matches_the_computation_written_out_by_hand(attention, x, pair):
    """The reference check. Catches the bias landing on [j, i] instead of
    [i, j], the pair value added after the softmax weights instead of with
    them, the bias added after the softmax, and a missing 1/sqrt(d_head)."""
    torch.testing.assert_close(attention(x, pair), by_hand(attention, x, pair), atol=1e-5, rtol=1e-5)


def test_with_the_pair_networks_silenced_it_is_your_attention_without_the_mask(attention, x, pair):
    """The upgrade adds to what you built; it does not replace it. Zero both
    pair terms and what is left is CausalSelfAttention(causal=False) with the
    same four weight matrices."""
    attention.pair_bias = Zeros(N_HEADS)
    attention.pair_value = Zeros(D_MODEL)
    plain = flocker.CausalSelfAttention(d_model=D_MODEL, n_heads=N_HEADS, causal=False)
    plain.load_state_dict({f"{name}.weight": getattr(attention, name).weight
                           for name in ("q", "k", "v", "proj")})
    torch.testing.assert_close(attention(x, pair), plain(x), atol=1e-5, rtol=1e-5)


def test_the_bias_alone_changes_who_listens_to_whom(attention, x, pair):
    """Silence only the value term: the output must still depend on the pair,
    through the attention weights."""
    attention.pair_value = Zeros(D_MODEL)
    assert not torch.allclose(attention(x, pair), attention(x, pair * 2.0), atol=1e-6)


def test_the_value_term_alone_changes_what_is_heard(attention, x, pair):
    """Silence only the bias: the output must still depend on the pair,
    through what each speaker sends."""
    attention.pair_bias = Zeros(N_HEADS)
    assert not torch.allclose(attention(x, pair), attention(x, pair * 2.0), atol=1e-6)


def test_every_bird_hears_every_other_bird(attention, x, pair):
    """No mask. Bird 0 must hear bird 5 as well as bird 5 hears bird 0."""
    one, geo = x[:1], pair[:1]
    jac = torch.autograd.functional.jacobian(lambda t: attention(t, geo), one)
    per_pair = jac.reshape(N_BIRDS, D_MODEL, N_BIRDS, D_MODEL).abs().sum(dim=(1, 3))
    deaf = [(i, j) for i in range(N_BIRDS) for j in range(N_BIRDS) if per_pair[i, j] < 1e-8]
    assert not deaf, f"(listener, speaker) pairs that cannot hear each other: {deaf[:4]}"


def test_a_birds_pair_row_only_reaches_that_bird(attention, x, pair):
    """Change the geometry bird 2 sees (row 2) and only bird 2's output may move.
    A bias transposed onto [j, i] moves everyone else instead."""
    changed = pair.clone()
    changed[:, 2] = changed[:, 2] + torch.randn(BATCH, N_BIRDS, 3)
    before, after = attention(x, pair), attention(x, changed)
    others = [i for i in range(N_BIRDS) if i != 2]
    torch.testing.assert_close(before[:, others], after[:, others], atol=0, rtol=0)
    assert not torch.allclose(before[:, 2], after[:, 2], atol=1e-6)


def test_relabelling_the_birds_relabels_the_answers(attention, x, pair):
    order = torch.tensor([4, 0, 5, 2, 1, 3])
    shuffled = attention(x[:, order], pair[:, order][:, :, order])
    torch.testing.assert_close(shuffled, attention(x, pair)[:, order], atol=1e-5, rtol=1e-5)


def test_windows_in_a_batch_never_see_each_other(attention, x, pair):
    torch.testing.assert_close(attention(x[:1], pair[:1]), attention(x, pair)[:1], atol=1e-5, rtol=1e-5)


def test_every_parameter_gets_a_gradient(attention, x, pair):
    attention(x, pair).sum().backward()
    for name, parameter in attention.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, f"{name} got no gradient"


def test_two_built_under_the_same_seed_are_identical():
    torch.manual_seed(7)
    first = flocker.RelAttention(d_model=D_MODEL, n_heads=N_HEADS)
    torch.manual_seed(7)
    second = flocker.RelAttention(d_model=D_MODEL, n_heads=N_HEADS)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
