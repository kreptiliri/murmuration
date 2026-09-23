import pytest

torch = pytest.importorskip("torch")
flocker = pytest.importorskip(
    "flocker",
    reason="flocker.py does not exist yet, this suite is its specification")

N_FEATURES = 24
CONTEXT = 8
BATCH = 3
SMALL = dict(n_features=N_FEATURES, d_model=32, n_heads=4, n_blocks=2,
             max_context=CONTEXT)


@pytest.fixture
def transformer():
    torch.manual_seed(0)
    return flocker.Transformer(**SMALL)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, CONTEXT, N_FEATURES)


def test_a_window_becomes_one_tick_of_prediction(transformer, x):
    """Whole window in, 24 numbers out - the same contract as the Linear."""
    assert transformer(x).shape == (BATCH, N_FEATURES)


def test_a_shorter_window_is_fine(transformer):
    short = torch.randn(BATCH, 3, N_FEATURES)

    assert transformer(short).shape == (BATCH, N_FEATURES)


def test_the_blocks_actually_stack(transformer):
    """A plain list instead of ModuleList hides the blocks from parameters().

    The model still runs and still uses them; the optimiser simply never sees
    them, so most of the network never learns. It looks like a model that
    trains badly rather than one that is wired wrong.
    """
    torch.manual_seed(0)
    one = flocker.Transformer(**{**SMALL, "n_blocks": 1})

    assert sum(p.numel() for p in transformer.parameters()) > \
           sum(p.numel() for p in one.parameters())


def test_the_default_configuration_is_within_the_design_budget():
    """The design budget: d_model 128-256, 4 heads, ~10^5-10^6 parameters."""
    torch.manual_seed(0)
    model = flocker.Transformer(n_features=N_FEATURES, d_model=128, n_heads=4,
                                n_blocks=2, max_context=CONTEXT)

    total = sum(p.numel() for p in model.parameters())

    assert 1e5 <= total <= 1e6, f"{total:,} parameters, outside the design budget"


def test_it_can_memorise_eight_windows():
    """The wiring check. Run this before any long run, not after."""
    torch.manual_seed(0)
    inputs = torch.randn(8, CONTEXT, N_FEATURES)
    targets = torch.randn(8, N_FEATURES)

    model = flocker.Transformer(**SMALL)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()

    first = loss_function(model(inputs), targets).item()
    for _ in range(400):
        optimiser.zero_grad()
        loss = loss_function(model(inputs), targets)
        loss.backward()
        optimiser.step()

    assert loss.item() < first * 0.05, (
        f"loss went {first:.4f} -> {loss.item():.4f}. A model that cannot "
        "memorise eight examples is wired wrong, not undertrained")


def test_every_parameter_gets_a_gradient(transformer, x):
    transformer(x).sum().backward()

    for name, parameter in transformer.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"


def test_two_built_under_the_same_seed_are_identical():
    """A timing number and a score both need the run to be repeatable."""
    torch.manual_seed(7)
    first = flocker.Transformer(**SMALL)
    torch.manual_seed(7)
    second = flocker.Transformer(**SMALL)

    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_every_tick_in_the_window_reaches_the_output(transformer, x):
    """Pins which position the head reads, which nothing else here does.

    Reading `x[:, 0]` instead of `x[:, -1]` survives every other test in this
    file - eight random windows are still memorisable from their first ticks
    alone - while throwing away seven eighths of the input. The model would
    train, converge, and read as "the transformer is not very good".

    With a causal mask only the last position has seen the whole window, so a
    change anywhere in it must be able to move the output.
    """
    before = transformer(x)

    for tick in range(CONTEXT):
        tampered = x.clone()
        tampered[:, tick] = torch.randn(BATCH, N_FEATURES)

        assert not torch.allclose(before, transformer(tampered), atol=1e-6), (
            f"changing tick {tick} left the output identical - the head is not "
            "reading the last position")
