import numpy as np
import pytest

from boids import Physics, get_torus, step


PHYSICS = Physics()
RULES = ("separation", "alignment", "cohesion")


def a_flock(seed, *, n=12, physics=PHYSICS):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, physics.bounds, size=(n, 2))
    heading = rng.uniform(0.0, 2 * np.pi, size=n)
    speed = rng.uniform(physics.min_speed, physics.max_speed, size=n)
    velocities = np.column_stack((speed * np.cos(heading), speed * np.sin(heading)))
    return positions, velocities


# --- 1. the bird order is a label, not a fact ---------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_relabelling_the_birds_permutes_the_answer_and_changes_nothing_else(seed):
    positions, velocities = a_flock(seed)
    order = np.random.default_rng(seed + 900).permutation(len(positions))

    plain_p, plain_v, plain_f = step(positions, velocities, PHYSICS)
    mixed_p, mixed_v, mixed_f = step(positions[order], velocities[order], PHYSICS)

    np.testing.assert_allclose(mixed_p, plain_p[order], atol=1e-12)
    np.testing.assert_allclose(mixed_v, plain_v[order], atol=1e-12)
    for rule in RULES:
        np.testing.assert_allclose(mixed_f[rule], plain_f[rule][order], atol=1e-12,
                                   err_msg=f"{rule} did not follow the relabelling")


def test_the_permutation_test_can_fail():
    positions, velocities = a_flock(0)
    order = np.roll(np.arange(len(positions)), 1)

    _, _, plain_f = step(positions, velocities, PHYSICS)
    _, _, mixed_f = step(positions[order], velocities[order], PHYSICS)

    assert not np.allclose(mixed_f["separation"], plain_f["separation"]), (
        "a rolled flock produced identical unpermuted forces, so this suite "
        "would pass even if `step` ignored its arguments")


# --- 2. where the flock is does not matter, only where it is relative to itself --

@pytest.mark.parametrize("seed", range(5))
def test_sliding_the_whole_flock_leaves_the_forces_alone(seed):
    positions, velocities = a_flock(seed)
    offset = np.random.default_rng(seed + 800).uniform(0, PHYSICS.bounds, size=2)

    plain_p, plain_v, plain_f = step(positions, velocities, PHYSICS)
    slid_p, slid_v, slid_f = step((positions + offset) % PHYSICS.bounds,
                                  velocities, PHYSICS)

    np.testing.assert_allclose(slid_v, plain_v, atol=1e-12)
    for rule in RULES:
        np.testing.assert_allclose(slid_f[rule], plain_f[rule], atol=1e-12,
                                   err_msg=f"{rule} moved when the flock slid")
    #every bird moved by the same offset, and by exactly the offset
    moved_by = get_torus(slid_p, plain_p, PHYSICS.bounds)
    expected = np.broadcast_to(get_torus(offset, np.zeros(2), PHYSICS.bounds),
                               moved_by.shape)
    np.testing.assert_allclose(moved_by, expected, atol=1e-9)


def test_sliding_a_flock_across_the_seam_still_leaves_it_alone():
    positions = np.array([[0.5, 50.0], [1.5, 50.0], [2.5, 51.0], [1.0, 49.0]])
    velocities = np.array([[2.0, 0.0], [-2.0, 0.5], [0.0, 2.0], [1.4, -1.4]])

    _, plain_v, plain_f = step(positions, velocities, PHYSICS)
    _, wrapped_v, wrapped_f = step((positions - 5.0) % PHYSICS.bounds,
                                   velocities, PHYSICS)

    np.testing.assert_allclose(wrapped_v, plain_v, atol=1e-12)
    for rule in RULES:
        np.testing.assert_allclose(wrapped_f[rule], plain_f[rule], atol=1e-12,
                                   err_msg=f"{rule} changed when the flock wrapped")


# --- 3. the eight symmetries of a square ---------------------------------------

def square_maps(bounds):
    def wrap(p):
        return p % bounds

    return [
        ("identity",        lambda p: wrap(p.copy())),
        ("rotate 90",       lambda p: wrap(np.stack([-p[..., 1], p[..., 0]], -1))),
        ("rotate 180",      lambda p: wrap(-p)),
        ("rotate 270",      lambda p: wrap(np.stack([p[..., 1], -p[..., 0]], -1))),
        ("flip x",          lambda p: wrap(np.stack([-p[..., 0], p[..., 1]], -1))),
        ("flip y",          lambda p: wrap(np.stack([p[..., 0], -p[..., 1]], -1))),
        ("swap axes",       lambda p: wrap(np.stack([p[..., 1], p[..., 0]], -1))),
        ("anti-swap axes",  lambda p: wrap(np.stack([-p[..., 1], -p[..., 0]], -1))),
    ]


def linear_part(name):
    return {
        "identity":       lambda v: v.copy(),
        "rotate 90":      lambda v: np.stack([-v[..., 1], v[..., 0]], -1),
        "rotate 180":     lambda v: -v,
        "rotate 270":     lambda v: np.stack([v[..., 1], -v[..., 0]], -1),
        "flip x":         lambda v: np.stack([-v[..., 0], v[..., 1]], -1),
        "flip y":         lambda v: np.stack([v[..., 0], -v[..., 1]], -1),
        "swap axes":      lambda v: np.stack([v[..., 1], v[..., 0]], -1),
        "anti-swap axes": lambda v: np.stack([-v[..., 1], -v[..., 0]], -1),
    }[name]


@pytest.mark.parametrize("name", [n for n, _ in square_maps(100.0)])
@pytest.mark.parametrize("seed", range(3))
def test_the_square_symmetries_carry_the_forces_with_them(name, seed):
    bounds = PHYSICS.bounds
    on_positions = dict(square_maps(bounds))[name]
    on_vectors = linear_part(name)
    positions, velocities = a_flock(seed)

    plain_p, plain_v, plain_f = step(positions, velocities, PHYSICS)
    moved_p, moved_v, moved_f = step(on_positions(positions),
                                     on_vectors(velocities), PHYSICS)

    np.testing.assert_allclose(moved_v, on_vectors(plain_v), atol=1e-12)
    np.testing.assert_allclose(get_torus(moved_p, on_positions(plain_p), bounds),
                               0.0, atol=1e-9)
    for rule in RULES:
        np.testing.assert_allclose(moved_f[rule], on_vectors(plain_f[rule]),
                                   atol=1e-12,
                                   err_msg=f"{name} did not carry {rule}")


def test_an_arbitrary_rotation_is_not_a_symmetry():
    bounds = PHYSICS.bounds
    angle = np.deg2rad(30.0)
    rotate = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    positions, velocities = a_flock(3, n=30)

    _, _, plain_f = step(positions, velocities, PHYSICS)
    _, _, turned_f = step((positions @ rotate.T) % bounds,
                          velocities @ rotate.T, PHYSICS)

    assert not np.allclose(turned_f["separation"], plain_f["separation"] @ rotate.T,
                           atol=1e-6), (
        "a 30-degree rotation came out exact. If that is really true the "
        "augmentation group is larger than this file claims, and the claim "
        "should be widened deliberately rather than by accident")
