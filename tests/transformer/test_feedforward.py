import pytest

torch = pytest.importorskip("torch")
flocker = pytest.importorskip(
    "flocker",
    reason="flocker.py does not exist yet, this suite is its specification")

D_MODEL = 32
CONTEXT = 8
BATCH = 3


@pytest.fixture
def feedforward():
    torch.manual_seed(0)
    return flocker.FeedForward(d_model=D_MODEL)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, CONTEXT, D_MODEL)


def test_it_comes_back_the_same_shape(feedforward, x):
    assert feedforward(x).shape == x.shape


def test_it_is_actually_nonlinear(feedforward, x):
    torch.manual_seed(2)
    h = torch.randn(BATCH, CONTEXT, D_MODEL)

    curved = feedforward(x + h) + feedforward(x - h)
    straight = 2 * feedforward(x)

    assert not torch.allclose(curved, straight, atol=1e-4), (
        "f(x+h) + f(x-h) == 2f(x): this map is affine. The nonlinearity "
        "between the two Linears is missing or is not being applied.")


def test_nothing_mixes_across_ticks(feedforward, x):
    before = feedforward(x)

    tampered = x.clone()
    tampered[:, 5] = torch.randn(BATCH, D_MODEL)
    after = feedforward(tampered)

    for tick in (0, 1, 2, 3, 4, 6, 7):
        torch.testing.assert_close(before[:, tick], after[:, tick],
                                   atol=0, rtol=0,
                                   msg=f"changing tick 5 moved tick {tick}")


def test_it_does_something(feedforward, x):
    assert not torch.allclose(feedforward(x), x, atol=1e-6)


def test_every_parameter_gets_a_gradient(feedforward, x):
    feedforward(x).sum().backward()

    for name, parameter in feedforward.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert parameter.grad.abs().sum() > 0, f"{name}'s gradient is all zero"


def test_two_calls_agree(feedforward, x):
    torch.testing.assert_close(feedforward(x), feedforward(x), atol=0, rtol=0)
