import numpy as np
import pytest

import boids
from boids import calculate_alignment, calculate_cohesion, calculate_separation
from world import ALIGNMENT_RADIUS, BOUNDS, COHESION_RADIUS, SEPARATION_RADIUS, random_flock

if not hasattr(boids, "edge_weight"):
    pytest.skip("boids.edge_weight does not exist yet, this suite is its "
                "specification", allow_module_level=True)

from boids import edge_weight

SOFT = 0.25
RADII = {"separation": SEPARATION_RADIUS, "alignment": ALIGNMENT_RADIUS,
         "cohesion": COHESION_RADIUS}
RULES = tuple(RADII)


def force(rule, positions, velocities, softness=None, boid_id=0):
    kw = {} if softness is None else {"softness": softness}
    if rule == "separation":
        return calculate_separation(boid_id, positions, SEPARATION_RADIUS, BOUNDS, 1.0, **kw)
    if rule == "alignment":
        return calculate_alignment(boid_id, positions, velocities, ALIGNMENT_RADIUS, BOUNDS, **kw)
    return calculate_cohesion(boid_id, positions, COHESION_RADIUS, BOUNDS, **kw)


def pair(distance, angle=0.3):
    here = np.array([50.0, 50.0])
    there = here + distance * np.array([np.cos(angle), np.sin(angle)])
    return np.array([here, there]), np.array([[1.0, 0.2], [-0.4, 1.3]])


def fading(rule, share):
    r = RADII[rule]
    return r - share * SOFT * r


# --- the weight --------------------------------------------------------------------

@pytest.mark.parametrize("radius", [20.0, 30.0, 40.0])
def test_full_weight_up_to_the_start_of_the_fade(radius):
    for d in (0.0, 0.3 * radius, (1 - SOFT) * radius):
        assert edge_weight(d, radius, SOFT) == 1.0, f"distance {d}"


@pytest.mark.parametrize("radius", [20.0, 30.0, 40.0])
def test_zero_at_the_radius_and_beyond(radius):
    for d in (radius, radius + 1e-9, radius + 0.5, 2 * radius):
        assert edge_weight(d, radius, SOFT) == 0.0, f"distance {d}"


@pytest.mark.parametrize("radius", [20.0, 30.0, 40.0])
def test_the_fade_is_a_straight_line(radius):
    for share in (0.25, 0.5, 0.75):
        d = radius - share * SOFT * radius
        assert edge_weight(d, radius, SOFT) == pytest.approx(share, abs=1e-12)


def test_softness_zero_is_the_hard_edge():
    assert edge_weight(19.999999, 20.0, 0.0) == 1.0
    assert edge_weight(20.0, 20.0, 0.0) == 0.0
    assert edge_weight(0.0, 20.0, 0.0) == 1.0


def test_softness_one_fades_from_the_centre():
    assert edge_weight(0.0, 40.0, 1.0) == 1.0
    assert edge_weight(20.0, 40.0, 1.0) == pytest.approx(0.5)
    assert edge_weight(36.0, 40.0, 1.0) == pytest.approx(0.1)


@pytest.mark.parametrize("softness", [0.1, 0.25, 1.0])
def test_the_weight_never_grows_with_distance_and_stays_in_zero_to_one(softness):
    ds = np.linspace(0.0, 45.0, 901)
    w = np.array([edge_weight(d, 40.0, softness) for d in ds])
    assert ((w >= 0.0) & (w <= 1.0)).all()
    assert (np.diff(w) <= 1e-15).all(), "the weight rose somewhere as distance grew"


# --- the rules, with softness -----------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_default_is_the_hard_edge(rule, seed):
    positions, velocities = random_flock(seed=seed, n_birds=12)
    for i in range(12):
        np.testing.assert_array_equal(force(rule, positions, velocities, boid_id=i),
                                      force(rule, positions, velocities, 0.0, boid_id=i))


@pytest.mark.parametrize("rule", RULES)
def test_a_lone_neighbour_before_the_fade_is_untouched(rule):
    positions, velocities = pair(0.5 * RADII[rule])
    np.testing.assert_allclose(force(rule, positions, velocities, SOFT),
                               force(rule, positions, velocities, 0.0), rtol=1e-12, atol=0)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("share", [0.2, 0.5, 0.9])
def test_a_lone_fading_neighbour_pushes_in_proportion_to_its_weight(rule, share):
    positions, velocities = pair(fading(rule, share))
    hard = force(rule, positions, velocities, 0.0)
    assert np.linalg.norm(hard) > 1e-3, "premise: the hard rule must push here"
    np.testing.assert_allclose(force(rule, positions, velocities, SOFT), share * hard,
                               rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("rule", RULES)
def test_two_neighbours_average_by_weight(rule):
    r = RADII[rule]
    here = np.array([50.0, 50.0])
    a = here + 0.5 * r * np.array([1.0, 0.0])
    b = here + fading(rule, 0.5) * np.array([0.0, 1.0])
    positions = np.array([here, a, b])
    velocities = np.array([[1.0, 0.2], [-0.4, 1.3], [0.7, -0.9]])

    got = force(rule, positions, velocities, SOFT)

    if rule == "separation":
        def push(p):
            off = here - p
            d = np.linalg.norm(off)
            return off / d * min(1.0 / d, 1.0)
        expected = (push(a) + 0.5 * push(b)) / 1.5
    elif rule == "alignment":
        expected = (velocities[1] + 0.5 * velocities[2]) / 1.5 - velocities[0]
    else:
        expected = ((a - here) + 0.5 * (b - here)) / 1.5 / r
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize("rule", RULES)
def test_a_neighbour_leaving_the_radius_fades_to_nothing(rule):
    r = RADII[rule]
    here = np.array([50.0, 50.0])
    inside_plateau = here + 0.4 * r * np.array([1.0, 0.0])
    velocities = np.array([[1.0, 0.2], [-0.4, 1.3], [0.7, -0.9]])

    def with_third_at(d, softness):
        third = here + d * np.array([-0.6, 0.8])
        return force(rule, np.array([here, inside_plateau, third]), velocities, softness)

    eps = 1e-6 * r
    soft_step = np.linalg.norm(with_third_at(r - eps, SOFT) - with_third_at(r + eps, SOFT))
    hard_step = np.linalg.norm(with_third_at(r - eps, 0.0) - with_third_at(r + eps, 0.0))
    assert hard_step > 1e-3, "premise: the hard edge must jump here"
    assert soft_step < 1e-5 * hard_step, (
        f"crossing the radius moved the {rule} force by {soft_step:.2e} "
        f"(hard edge: {hard_step:.2e})")


@pytest.mark.parametrize("rule", RULES)
def test_a_lone_neighbour_leaving_the_radius_fades_too(rule):
    positions, velocities = pair(RADII[rule] * (1 - 1e-6))
    hard = np.linalg.norm(force(rule, positions, velocities, 0.0))
    assert hard > 1e-3
    assert np.linalg.norm(force(rule, positions, velocities, SOFT)) < 1e-4 * hard


@pytest.mark.parametrize("rule", RULES)
def test_a_bird_beyond_the_radius_changes_nothing(rule):
    r = RADII[rule]
    here = np.array([50.0, 50.0])
    near = here + fading(rule, 0.5) * np.array([0.0, 1.0])
    far = here + (r + 0.5) * np.array([1.0, 0.0])
    v = np.array([[1.0, 0.2], [-0.4, 1.3], [0.7, -0.9]])
    np.testing.assert_array_equal(
        force(rule, np.array([here, near, far]), v, SOFT),
        force(rule, np.array([here, near]), v[:2], SOFT))


def test_each_rule_fades_at_its_own_radius():
    positions, velocities = pair(28.0)
    assert np.all(force("separation", positions, velocities, SOFT) == 0.0)
    np.testing.assert_allclose(force("alignment", positions, velocities, SOFT),
                               (2.0 / 7.5) * force("alignment", positions, velocities, 0.0), rtol=1e-9)
    np.testing.assert_allclose(force("cohesion", positions, velocities, SOFT),
                               force("cohesion", positions, velocities, 0.0), rtol=1e-12)


def test_the_weight_is_measured_around_the_seam():
    positions = np.array([[2.0, 50.0], [67.0, 50.0]])
    velocities = np.zeros((2, 2))
    hard = force("cohesion", positions, velocities, 0.0)
    assert hard[0] < 0, "premise: the pull must point across the seam, to the left"
    np.testing.assert_allclose(force("cohesion", positions, velocities, SOFT), 0.5 * hard, rtol=1e-9)


def test_separation_still_ignores_a_bird_sitting_on_top():
    here = np.array([50.0, 50.0])
    near = here + np.array([6.0, 8.0])
    v = np.zeros((3, 2))
    np.testing.assert_array_equal(
        force("separation", np.array([here, here.copy(), near]), v, SOFT),
        force("separation", np.array([here, near]), v[:2], SOFT))


def test_alignment_still_counts_a_bird_sitting_on_top():
    here = np.array([50.0, 50.0])
    positions = np.array([here, here.copy()])
    velocities = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(force("alignment", positions, velocities, SOFT), [-1.0, 1.0])


@pytest.mark.parametrize("rule", RULES)
def test_a_bird_alone_feels_nothing(rule):
    positions, velocities = pair(0.9 * BOUNDS / 2)
    np.testing.assert_array_equal(force(rule, positions[:1], velocities[:1], SOFT), [0.0, 0.0])
