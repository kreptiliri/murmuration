import pytest

torch = pytest.importorskip("torch")
flocker = pytest.importorskip("flocker")

N_BIRDS = 12
N_FEATURES = N_BIRDS * 2
CONTEXT = 2
BATCH = 3
SMALL = dict(n_features=N_FEATURES, d_model=32, n_heads=4, n_blocks=2,
             context=CONTEXT)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return flocker.PerBirdTransformer(**SMALL)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, CONTEXT, N_FEATURES)


def shuffle_birds(values, order):
    # values end in 2 * n_birds numbers, bird by bird: x0 y0 x1 y1 ...
    leading = values.shape[:-1]
    birds = values.reshape(*leading, len(order), 2)
    return birds[..., order, :].reshape(*leading, len(order) * 2)


# --- the contract train.py relies on ------------------------------------------

def test_a_window_becomes_one_tick_of_prediction(model, x):
    assert model(x).shape == (BATCH, N_FEATURES)


def test_the_number_of_birds_comes_from_n_features():
    torch.manual_seed(0)
    four = flocker.PerBirdTransformer(n_features=8, d_model=32, n_heads=4,
                                      n_blocks=2, context=3)

    assert four(torch.randn(BATCH, 3, 8)).shape == (BATCH, 8)


# --- what makes it per bird ----------------------------------------------------

def test_shuffling_the_birds_shuffles_the_answers_the_same_way(model, x):
    order = torch.tensor([5, 0, 11, 3, 8, 1, 10, 2, 7, 4, 9, 6])

    shuffled_in = model(shuffle_birds(x, order))
    shuffled_out = shuffle_birds(model(x), order)

    torch.testing.assert_close(shuffled_in, shuffled_out, atol=1e-5, rtol=1e-5)


def test_every_bird_hears_every_other_bird(model, x):
    one = x[:1]
    jacobian = torch.autograd.functional.jacobian(model, one)
    # (1, 24 out, 1, CONTEXT, 24 in) -> (12 birds out, 2, CONTEXT, 12 birds in, 2)
    per_pair = jacobian.reshape(N_BIRDS, 2, CONTEXT, N_BIRDS, 2).abs().sum(dim=(1, 2, 4))

    deaf = [(i, j) for i in range(N_BIRDS) for j in range(N_BIRDS)
            if per_pair[i, j] < 1e-8]
    assert not deaf, (
        f"{len(deaf)} of {N_BIRDS * N_BIRDS} (listener, speaker) pairs cannot hear "
        f"each other, e.g. {deaf[:4]}. If only i < j pairs are listed, the causal "
        "mask came along with the attention")


def test_a_birds_ticks_stay_in_order(model, x):
    backwards = x.flip(dims=[1])

    assert not torch.allclose(model(x), model(backwards), atol=1e-6)


def test_windows_in_a_batch_never_see_each_other(model, x):
    alone = model(x[:1])
    together = model(x)[:1]

    torch.testing.assert_close(alone, together, atol=1e-5, rtol=1e-5)


# --- wiring checks the other models get too --------------------------------------

def test_it_can_memorise_eight_windows():
    torch.manual_seed(0)
    inputs = torch.randn(8, CONTEXT, N_FEATURES)
    targets = torch.randn(8, N_FEATURES)

    model = flocker.PerBirdTransformer(**SMALL)
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


def test_every_parameter_gets_a_gradient(model, x):
    model(x).sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"


def test_the_blocks_actually_stack(model):
    torch.manual_seed(0)
    one = flocker.PerBirdTransformer(**{**SMALL, "n_blocks": 1})

    assert sum(p.numel() for p in model.parameters()) > \
           sum(p.numel() for p in one.parameters())


def test_the_run_configuration_is_within_the_design_budget():
    torch.manual_seed(0)
    model = flocker.PerBirdTransformer(n_features=N_FEATURES, d_model=128,
                                       n_heads=4, n_blocks=2, context=CONTEXT)

    total = sum(p.numel() for p in model.parameters())

    assert 1e5 <= total <= 1e6, f"{total:,} parameters, outside the design budget"
