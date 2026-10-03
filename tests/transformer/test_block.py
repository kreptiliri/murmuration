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
def block():
    torch.manual_seed(0)
    return flocker.Block(d_model=D_MODEL, n_heads=N_HEADS)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, CONTEXT, D_MODEL)


def test_it_comes_back_the_same_shape(block, x):
    assert block(x).shape == x.shape


def test_the_residual_is_real(block, x):
    with torch.no_grad():
        block.attention.proj.weight.zero_()
        block.feedforward.narrow.weight.zero_()
        block.feedforward.narrow.bias.zero_()

    torch.testing.assert_close(block(x), x, atol=0, rtol=0)


def test_a_tick_still_cannot_see_the_future(block, x):
    before = block(x)

    tampered = x.clone()
    tampered[:, -1] = torch.randn(BATCH, D_MODEL)

    torch.testing.assert_close(before[:, :-1], block(tampered)[:, :-1],
                               atol=0, rtol=0)


def test_every_parameter_gets_a_gradient(block, x):
    block(x).sum().backward()

    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"


def test_two_calls_agree(block, x):
    torch.testing.assert_close(block(x), block(x), atol=0, rtol=0)


def test_the_whole_block_matches_pytorchs_own(block, x):
    reference = torch.nn.TransformerEncoderLayer(
        d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=4 * D_MODEL,
        activation="gelu", dropout=0.0, batch_first=True, norm_first=True)

    with torch.no_grad():
        reference.self_attn.in_proj_weight.copy_(torch.cat(
            [block.attention.q.weight, block.attention.k.weight,
             block.attention.v.weight], dim=0))
        reference.self_attn.in_proj_bias.zero_()      # ours are bias=False
        reference.self_attn.out_proj.weight.copy_(block.attention.proj.weight)
        reference.self_attn.out_proj.bias.zero_()
        reference.linear1.weight.copy_(block.feedforward.wide.weight)
        reference.linear1.bias.copy_(block.feedforward.wide.bias)
        reference.linear2.weight.copy_(block.feedforward.narrow.weight)
        reference.linear2.bias.copy_(block.feedforward.narrow.bias)
        reference.norm1.weight.copy_(block.norm_one.weight)
        reference.norm1.bias.copy_(block.norm_one.bias)
        reference.norm2.weight.copy_(block.norm_two.weight)
        reference.norm2.bias.copy_(block.norm_two.bias)

    block.eval()
    reference.eval()
    causal = torch.triu(torch.full((CONTEXT, CONTEXT), float("-inf")), diagonal=1)

    with torch.no_grad():
        torch.testing.assert_close(block(x), reference(x, src_mask=causal),
                                   atol=1e-5, rtol=1e-5)
