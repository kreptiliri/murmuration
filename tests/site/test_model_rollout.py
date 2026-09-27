import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
train = pytest.importorskip("train")
generate = pytest.importorskip("generate")
boids = pytest.importorskip("boids")
rollout = pytest.importorskip("rollout")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world import PHYSICS_CRUISING

PHYSICS = dataclasses.asdict(PHYSICS_CRUISING)   #the same dict a dataset's config.json holds
BOUNDS, DT = PHYSICS["bounds"], PHYSICS["dt"]
N_BIRDS = 6


# --- fake models, so every expected answer is known exactly -------------------------

class Constant(torch.nn.Module):
    """Always predicts the same change for every bird."""
    def __init__(self, change):
        super().__init__()
        self.change = change

    def forward(self, x, pair=None):
        n_features = x.shape[-1]
        return torch.full((x.shape[0], n_features), float(self.change))


class Recorder(Constant):
    """Predicts zero, and keeps what it was handed."""
    def __init__(self):
        super().__init__(0.0)
        self.inputs, self.pairs = [], []

    def forward(self, x, pair=None):
        self.inputs.append(x.clone())
        self.pairs.append(None if pair is None else pair.clone())
        return super().forward(x, pair)


class Simulator(torch.nn.Module):
    """The perfect model: reads the window back into positions and a velocity,
    runs boids.step, and answers with the change step made. Only works with
    identity scalers and a model kind that sees the first tick (not rel)."""
    def __init__(self, physics):
        super().__init__()
        self.physics = physics

    def forward(self, x, pair=None):
        window = x[0].double().numpy()                       #(context, birds*2), moves after the first tick
        ticks = np.cumsum(window, axis=0).reshape(len(window), -1, 2)
        now = ticks[-1] % self.physics.bounds
        velocity = (ticks[-1] - ticks[-2]) / self.physics.dt
        _, new_velocity, _ = boids.step(now, velocity, self.physics)
        return torch.from_numpy((new_velocity - velocity).reshape(1, -1))


def checkpoint(model, *, kind="linear", context=2, y_mean=0.0, y_std=1.0, x_std=1.0):
    n_features = N_BIRDS * 2
    return {"model": model,
            "config": {"kind": kind, "context": context, "n_features": n_features},
            "x_scaler": (np.zeros(n_features), np.full(n_features, x_std)),
            "y_scaler": (np.full(n_features, y_mean), np.full(n_features, y_std))}


def real_start(context=2, seed=3, n_ticks=40):
    clip = generate.generate_clip(PHYSICS_CRUISING, n_slots=N_BIRDS, n_live=N_BIRDS,
                                  n_ticks=n_ticks, seed=seed)
    return clip["positions"]


# --- the contract --------------------------------------------------------------------

def test_the_shape_is_the_start_plus_the_new_ticks():
    start = real_start()[:2]
    out = rollout.rollout(checkpoint(Constant(0.0)), start, n_ticks=15, physics=PHYSICS)
    assert out.shape == (2 + 15, N_BIRDS, 2)


def test_it_begins_with_the_real_ticks_it_was_given():
    start = real_start()[:2]
    out = rollout.rollout(checkpoint(Constant(0.0)), start, n_ticks=5, physics=PHYSICS)
    np.testing.assert_array_equal(out[:2], start)


def test_it_does_not_change_the_start_it_is_handed():
    start = real_start()[:2]
    before = start.copy()
    rollout.rollout(checkpoint(Constant(0.3)), start, n_ticks=5, physics=PHYSICS)
    np.testing.assert_array_equal(start, before)


def test_every_bird_stays_inside_the_box():
    """The page draws positions straight onto a 100 x 100 canvas."""
    start = real_start()[:2]
    out = rollout.rollout(checkpoint(Constant(0.7)), start, n_ticks=60, physics=PHYSICS)
    assert (out >= 0).all() and (out < BOUNDS).all()


def test_the_start_has_to_match_the_context():
    start = real_start()[:3]
    with pytest.raises(ValueError):
        rollout.rollout(checkpoint(Constant(0.0), context=2), start, n_ticks=5, physics=PHYSICS)


def test_the_same_call_gives_the_same_flight():
    start = real_start()[:2]
    a = rollout.rollout(checkpoint(Constant(0.2)), start, n_ticks=10, physics=PHYSICS)
    b = rollout.rollout(checkpoint(Constant(0.2)), start, n_ticks=10, physics=PHYSICS)
    np.testing.assert_array_equal(a, b)


# --- it flies the way the simulator moves birds ---------------------------------------

def test_predicting_no_change_flies_straight_on():
    """Zero change keeps the last velocity, so each tick adds the last move again."""
    start = real_start()[:2]
    out = rollout.rollout(checkpoint(Constant(0.0)), start, n_ticks=10, physics=PHYSICS)
    move = boids.get_torus(start[1], start[0], BOUNDS)
    for t in range(10):
        expected = (start[1] + (t + 1) * move) % BOUNDS
        np.testing.assert_allclose(out[2 + t], expected, atol=1e-9, err_msg=f"tick {t}")


def test_a_constant_change_speeds_up_by_that_much_every_tick():
    """The model's answer is a change of velocity, not a position or a velocity.
    Each tick the velocity grows by the change, so each move grows by change * dt."""
    start = real_start()[:2]
    out = rollout.rollout(checkpoint(Constant(0.5)), start, n_ticks=6, physics=PHYSICS)
    moves = boids.get_torus(out[1:], out[:-1], BOUNDS)
    growth = moves[1:] - moves[:-1]
    np.testing.assert_allclose(growth, 0.5 * DT, atol=1e-9)


def test_the_perfect_model_reproduces_the_simulator():
    """The whole contract in one test. A model that answers exactly what boids.step
    would do has to retrace the real clip. If this drifts, the rollout moves birds
    in a way the simulator never did, and every frame of the slider is wrong
    while still looking like birds."""
    clip = real_start(n_ticks=40)
    out = rollout.rollout(checkpoint(Simulator(PHYSICS_CRUISING)), clip[:2], n_ticks=30,
                          physics=PHYSICS)
    offset = boids.get_torus(out, clip[:32], BOUNDS)
    assert np.abs(offset).max() < 1e-5, f"off the real flight by {np.abs(offset).max():.2e}"


def test_the_birds_cross_the_seam_without_jumping():
    """A bird at x = 99.8 moving right reaches 0.3 next tick. Read naively, that
    move is -99.5, and the model is handed a bird teleporting across the world."""
    start = np.full((2, N_BIRDS, 2), 50.0)
    start[0, 0] = [99.4, 50.0]
    start[1, 0] = [99.8, 50.0]
    model = Recorder()
    rollout.rollout(checkpoint(model), start, n_ticks=4, physics=PHYSICS)
    for x in model.inputs:
        moves = x[0, 1:].numpy()
        assert np.abs(moves).max() < 1.0, f"the model was handed a move of {np.abs(moves).max():.1f}"


# --- it talks to the model the way training did -----------------------------------------

def test_the_model_gets_the_window_squashed_with_the_checkpoints_own_scaler():
    """Training fed the model moves divided by the x scaler. Raw positions go in
    without raising and every prediction is nonsense."""
    start = real_start()[:2]
    model = Recorder()
    rollout.rollout(checkpoint(model, x_std=4.0), start, n_ticks=1, physics=PHYSICS)
    move = boids.get_torus(start[1], start[0], BOUNDS)
    expected = np.stack([start[0], move]).reshape(1, 2, N_BIRDS * 2) / 4.0
    np.testing.assert_allclose(model.inputs[0].numpy(), expected, atol=1e-5)


def test_the_answer_is_unsquashed_with_the_checkpoints_own_scaler():
    """A model answering 0 in squashed units means 'the average change', which is
    the y scaler's mean, not zero."""
    start = real_start()[:2]
    moved = rollout.rollout(checkpoint(Constant(0.0), y_mean=0.5, y_std=3.0), start,
                            n_ticks=3, physics=PHYSICS)
    reference = rollout.rollout(checkpoint(Constant(0.5)), start, n_ticks=3, physics=PHYSICS)
    np.testing.assert_allclose(moved, reference, atol=1e-9)


def test_the_relative_model_gets_the_pair_geometry_of_the_latest_window():
    start = real_start()[:2]
    model = Recorder()
    out = rollout.rollout(checkpoint(model, kind="rel_per_bird"), start, n_ticks=3,
                          physics=PHYSICS)
    for t, pair in enumerate(model.pairs):
        window = out[t:t + 2].reshape(1, 2, N_BIRDS * 2)
        expected = train.pair_geometry(window, bounds=BOUNDS, scale=PHYSICS["separation_radius"])
        np.testing.assert_allclose(pair.numpy(), expected, atol=1e-5, err_msg=f"tick {t}")


def test_other_models_get_no_pair_geometry():
    start = real_start()[:2]
    model = Recorder()
    rollout.rollout(checkpoint(model, kind="per_bird"), start, n_ticks=2, physics=PHYSICS)
    assert all(p is None for p in model.pairs)


def test_the_model_sees_its_own_flight_not_the_real_one():
    """After the start, the only thing a rollout has is what the model did. Feeding
    it real ticks would make every snapshot fly perfectly and the slider pointless."""
    start = real_start()[:2]
    model = Recorder()
    out = rollout.rollout(checkpoint(model), start, n_ticks=4, physics=PHYSICS)
    last = model.inputs[-1][0].numpy()
    first_tick = last[0].reshape(N_BIRDS, 2)
    #the last call made tick 5 from ticks 3 and 4, so its window starts at tick 3
    np.testing.assert_allclose(first_tick, out[3], atol=1e-4)


# --- end to end, with a real checkpoint -------------------------------------------------

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    data = tmp_path_factory.mktemp("rollout_data")
    generate.build_dataset(PHYSICS_CRUISING, n_clips=6, n_slots=N_BIRDS, n_live=N_BIRDS,
                           n_ticks=40, base_seed=20260927, out_dir=data, allow_dirty=True)
    out = tmp_path_factory.mktemp("rollout_checkpoints")
    train.main(["--data", str(data), "--clips", "6", "--context", "2", "--passes", "1",
                "--batch", "64", "--report-every", "0", "--model", "rel_per_bird",
                "--checkpoint-every", "1", "--checkpoint-dir", str(out)])
    return data, out


def test_a_saved_relative_model_flies(trained):
    data, out = trained
    with np.load(data / "clips.npz") as clips:
        start = clips["positions"][5, :2]
    ck = train.load_checkpoint(out / "pass_0001.pt")
    flight = rollout.rollout(ck, start, n_ticks=30, physics=PHYSICS)
    assert flight.shape == (32, N_BIRDS, 2)
    assert np.isfinite(flight).all()
    assert (flight >= 0).all() and (flight < BOUNDS).all()
