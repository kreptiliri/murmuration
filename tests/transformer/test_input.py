import pytest

torch = pytest.importorskip("torch")
flocker = pytest.importorskip(
    "flocker",
    reason="flocker.py does not exist yet, this suite is its specification")

N_FEATURES = 24
D_MODEL = 32
CONTEXT = 8
BATCH = 3


@pytest.fixture
def embedding():
    torch.manual_seed(0)
    return flocker.InputEmbedding(n_features=N_FEATURES, d_model=D_MODEL,
                                max_context=CONTEXT)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(BATCH, CONTEXT, N_FEATURES)


def test_a_tick_comes_out_d_model_wide(embedding, x):
    assert embedding(x).shape == (BATCH, CONTEXT, D_MODEL)


def test_a_shorter_window_is_fine(embedding):
    """Only the first rows of the position table are used."""
    short = torch.randn(BATCH, 3, N_FEATURES)

    assert embedding(short).shape == (BATCH, 3, D_MODEL)


def test_a_window_longer_than_the_table_fails_loudly(embedding):
    """There is no row for position 9. It must raise, not wrap to row 0."""
    too_long = torch.randn(BATCH, CONTEXT + 1, N_FEATURES)

    with pytest.raises((ValueError, IndexError, RuntimeError)):
        embedding(too_long)


def test_the_same_numbers_at_two_positions_come_out_different(embedding):
    """The position stamp is actually applied, and differs between positions.

    Hand every tick the identical 24 numbers. If the outputs are also identical,
    either the position table was never added or it is all zeros - and the model
    cannot tell tick 0 from tick 7.
    """
    torch.manual_seed(2)
    one_tick = torch.randn(1, 1, N_FEATURES)
    repeated = one_tick.repeat(1, CONTEXT, 1)

    out = embedding(repeated)

    for tick in range(1, CONTEXT):
        assert not torch.equal(out[0, 0], out[0, tick]), (
            f"tick 0 and tick {tick} came out identical from identical numbers")


def test_nothing_mixes_across_ticks(embedding, x):
    """Tick 3 must not be able to see tick 5. That is attention's job, later.

    Mixing here would leave the probes unable to say at which layer a
    concept first became available, because it would already be everywhere.
    """
    before = embedding(x)

    tampered = x.clone()
    tampered[:, 5] = torch.randn(BATCH, N_FEATURES)
    after = embedding(tampered)

    for tick in (0, 1, 2, 3, 4, 6, 7):
        torch.testing.assert_close(before[:, tick], after[:, tick],
                                   atol=0, rtol=0,
                                   msg=f"changing tick 5 moved tick {tick}")


def test_two_built_under_the_same_seed_are_identical():
    torch.manual_seed(7)
    first = flocker.InputEmbedding(n_features=N_FEATURES, d_model=D_MODEL,
                                 max_context=CONTEXT)
    torch.manual_seed(7)
    second = flocker.InputEmbedding(n_features=N_FEATURES, d_model=D_MODEL,
                                  max_context=CONTEXT)

    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
