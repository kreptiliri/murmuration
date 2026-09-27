import pytest

torch = pytest.importorskip("torch")
flocker = pytest.importorskip("flocker")

if not hasattr(flocker, "RelPerBirdTransformer"):
    pytest.skip("flocker.RelPerBirdTransformer does not exist yet, this suite is "
                "its specification", allow_module_level=True)

N_BIRDS = 12
N_FEATURES = N_BIRDS * 2
CONTEXT = 2
BATCH = 3
SMALL = dict(n_features=N_FEATURES, d_model=32, n_heads=4, n_blocks=2, context=CONTEXT)


def geometry(batch, n_birds, seed):
    torch.manual_seed(seed)
    offsets = torch.randn(batch, n_birds, 1, 2) - torch.randn(batch, 1, n_birds, 2)
    offsets = offsets - offsets.transpose(1, 2)
    return torch.cat([offsets, offsets.norm(dim=-1, keepdim=True)], dim=-1)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return flocker.RelPerBirdTransformer(**SMALL)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, CONTEXT, N_FEATURES)


@pytest.fixture
def pair():
    return geometry(BATCH, N_BIRDS, 2)


def shuffle_birds(values, order):
    leading = values.shape[:-1]
    birds = values.reshape(*leading, len(order), 2)
    return birds[..., order, :].reshape(*leading, len(order) * 2)


# --- the contract train.py relies on --------------------------------------------------

def test_a_window_and_its_geometry_become_one_tick_of_prediction(model, x, pair):
    """Same output as PerBirdTransformer; one extra input."""
    assert model(x, pair).shape == (BATCH, N_FEATURES)


def test_the_number_of_birds_comes_from_n_features():
    torch.manual_seed(0)
    four = flocker.RelPerBirdTransformer(n_features=8, d_model=32, n_heads=4, n_blocks=2, context=3)
    assert four(torch.randn(BATCH, 3, 8), geometry(BATCH, 4, 3)).shape == (BATCH, 8)


@pytest.mark.parametrize("context", [0, 1])
def test_it_needs_at_least_one_move(context):
    """Tick 0 is thrown away (see below), so context 1 leaves a bird with
    nothing to say about itself. Refuse it at construction, not at the first batch."""
    with pytest.raises(ValueError):
        flocker.RelPerBirdTransformer(**{**SMALL, "context": context})


def test_it_is_built_from_rel_blocks(model):
    assert isinstance(model.blocks, torch.nn.ModuleList)
    assert len(model.blocks) == SMALL["n_blocks"]
    for block in model.blocks:
        assert isinstance(block, flocker.RelBlock)


# --- what makes it relative ------------------------------------------------------------

def test_the_absolute_position_is_ignored(model, x, pair):
    """The point of the upgrade. train.py hands every model the same window:
    tick 0 is the birds' positions, the ticks after it are moves. Positions
    now reach the model only through the pair geometry, so tick 0 must not
    move the answer at all. If it does, the seam is back."""
    moved = x.clone()
    moved[:, 0] = torch.randn(BATCH, N_FEATURES) * 5
    torch.testing.assert_close(model(moved, pair), model(x, pair), atol=0, rtol=0)


def test_the_moves_are_used(model, x, pair):
    moved = x.clone()
    moved[:, 1] = torch.randn(BATCH, N_FEATURES)
    assert not torch.allclose(model(moved, pair), model(x, pair), atol=1e-6)


def test_the_geometry_is_used(model, x, pair):
    assert not torch.allclose(model(x, pair), model(x, geometry(BATCH, N_BIRDS, 9)), atol=1e-6)


def test_every_block_is_handed_the_geometry(model, x, pair):
    """Easy to pass it to the first block and forget the rest."""
    seen = []
    def hook(module, args, kwargs, output):
        handed = list(args[1:]) + list(kwargs.values())
        seen.append(any(h is pair for h in handed))

    for block in model.blocks:
        block.register_forward_hook(hook, with_kwargs=True)
    model(x, pair)
    assert seen == [True] * len(model.blocks)


def test_a_birds_moves_stay_in_order():
    """With context 3 a bird has two moves, and which came first is how it is
    turning. Summing them into one token throws that away."""
    torch.manual_seed(0)
    model = flocker.RelPerBirdTransformer(**{**SMALL, "context": 3})
    torch.manual_seed(4)
    x = torch.randn(BATCH, 3, N_FEATURES)
    swapped = x.clone()
    swapped[:, 1], swapped[:, 2] = x[:, 2], x[:, 1]
    pair = geometry(BATCH, N_BIRDS, 5)
    assert not torch.allclose(model(x, pair), model(swapped, pair), atol=1e-6)


@pytest.mark.parametrize("context", [2, 3])
def test_relabelling_the_birds_relabels_the_answers(context, pair):
    """The per-bird property, now with the geometry relabelled on both axes.
    Context 3 as well: at context 2 a bird has one move, and a bare reshape
    into tokens happens to give the right answer. At 3 it mixes birds."""
    torch.manual_seed(0)
    model = flocker.RelPerBirdTransformer(**{**SMALL, "context": context})
    torch.manual_seed(1)
    x = torch.randn(BATCH, context, N_FEATURES)
    order = torch.tensor([5, 0, 11, 3, 8, 1, 10, 2, 7, 4, 9, 6])
    shuffled_in = model(shuffle_birds(x, order), pair[:, order][:, :, order])
    torch.testing.assert_close(shuffled_in, shuffle_birds(model(x, pair), order), atol=1e-5, rtol=1e-5)


def test_every_bird_hears_every_other_bird(model, x, pair):
    one, geo = x[:1], pair[:1]
    jac = torch.autograd.functional.jacobian(lambda t: model(t, geo), one)
    per_pair = jac.reshape(N_BIRDS, 2, CONTEXT, N_BIRDS, 2).abs().sum(dim=(1, 2, 4))
    deaf = [(i, j) for i in range(N_BIRDS) for j in range(N_BIRDS) if per_pair[i, j] < 1e-8]
    assert not deaf, f"{len(deaf)} (listener, speaker) pairs cannot hear each other, e.g. {deaf[:4]}"


def test_windows_in_a_batch_never_see_each_other(model, x, pair):
    torch.testing.assert_close(model(x[:1], pair[:1]), model(x, pair)[:1], atol=1e-5, rtol=1e-5)


# --- wiring checks every model gets ----------------------------------------------------------

def test_it_can_memorise_eight_windows():
    torch.manual_seed(0)
    inputs = torch.randn(8, CONTEXT, N_FEATURES)
    geo = geometry(8, N_BIRDS, 6)
    targets = torch.randn(8, N_FEATURES)
    model = flocker.RelPerBirdTransformer(**SMALL)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()

    first = loss_function(model(inputs, geo), targets).item()
    for _ in range(400):
        optimiser.zero_grad()
        loss = loss_function(model(inputs, geo), targets)
        loss.backward()
        optimiser.step()

    assert loss.item() < first * 0.05, f"loss went {first:.4f} -> {loss.item():.4f}"


def test_every_parameter_gets_a_gradient(model, x, pair):
    model(x, pair).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"


def test_two_built_under_the_same_seed_are_identical():
    torch.manual_seed(7)
    first = flocker.RelPerBirdTransformer(**SMALL)
    torch.manual_seed(7)
    second = flocker.RelPerBirdTransformer(**SMALL)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_the_run_configuration_is_within_the_design_budget():
    torch.manual_seed(0)
    model = flocker.RelPerBirdTransformer(n_features=N_FEATURES, d_model=128, n_heads=4,
                                          n_blocks=2, context=CONTEXT)
    total = sum(p.numel() for p in model.parameters())
    assert 1e5 <= total <= 1e6, f"{total:,} parameters, outside the design budget"
