import pytest

torch = pytest.importorskip("torch")
flocker = pytest.importorskip("flocker")

if not hasattr(flocker, "RelBlock"):
    pytest.skip("flocker.RelBlock does not exist yet, this suite is its "
                "specification", allow_module_level=True)

D_MODEL = 32
N_HEADS = 4
N_BIRDS = 6
BATCH = 3


@pytest.fixture
def block():
    torch.manual_seed(0)
    return flocker.RelBlock(d_model=D_MODEL, n_heads=N_HEADS)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, N_BIRDS, D_MODEL)


@pytest.fixture
def pair():
    torch.manual_seed(2)
    return torch.randn(BATCH, N_BIRDS, N_BIRDS, 3)


def test_it_comes_back_the_same_shape(block, x, pair):
    assert block(x, pair).shape == x.shape


def test_it_is_your_block_with_the_attention_swapped(block):
    assert isinstance(block.norm_one, torch.nn.LayerNorm)
    assert isinstance(block.attention, flocker.RelAttention)
    assert isinstance(block.norm_two, torch.nn.LayerNorm)
    assert isinstance(block.feedforward, flocker.FeedForward)


def test_it_matches_pre_norm_residual_written_out(block, x, pair):
    h = x + block.attention(block.norm_one(x), pair)
    expected = h + block.feedforward(block.norm_two(h))
    torch.testing.assert_close(block(x, pair), expected, atol=1e-6, rtol=1e-6)


def test_the_residual_is_real(block, x, pair):
    with torch.no_grad():
        block.attention.proj.weight.zero_()
        block.feedforward.narrow.weight.zero_()
        block.feedforward.narrow.bias.zero_()
    torch.testing.assert_close(block(x, pair), x, atol=0, rtol=0)


def test_the_geometry_reaches_the_attention(block, x, pair):
    assert not torch.allclose(block(x, pair), block(x, pair * 2.0), atol=1e-6)
